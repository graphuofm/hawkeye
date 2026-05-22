"""GraphEagleVision — the Structural-Evolution module for temporal link prediction.

Components
----------
* Incremental structural maintenance (CPU/NumPy): degree, k-core, triangle,
  k-truss, clustering — kept up to date over the edge stream.
* Rolling statistics per node × indicator: current / ema(trend) / variance /
  delta / max_change  → a 5K-dim per-node feature.
* Structural encoder (MLP by default) → per-node structural embedding.
* Pairwise structural features (computed on the fly from the current graph):
  common neighbours / Adamic-Adar / Jaccard / 2-hop CN / cohesiveness of the
  common neighbours / how many of them have rising coreness, ... — the signal
  needed to *rank* candidate destinations.
* Fusion with an optional interaction embedding from any base TG model
  (struct_only / concat / additive / gated).
* Link predictor MLP over [h_src ; h_dst ; (pairwise feats)].

The structural side is non-parametric; encoder / fusion / predictor are torch
modules trained end-to-end (BCE on positive vs negative edges).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from gev.encoder import build_encoder
from gev.features import PAIRWISE_FEATURE_NAMES, compute_pairs, pairwise_mode_columns
from gev.features.sparse_pairwise import build_csr, compute_pairs_csr
from gev.fusion import build_fusion
from gev.graph import DynamicGraph
from gev.indicators import BaseIndicator, build_indicators
from gev.stats import RollingStatistics


@dataclass
class GEVConfig:
    indicators: List[str] = field(default_factory=lambda: ["degree", "core"])
    stats_decay: float = 0.95
    # extra (slower) decay rates -> "trend at scale d" = ema - ema_d features.
    # e.g. [0.99, 0.999, 0.9999] gives short/medium/long-window trend signals.
    trend_decays: List[float] = field(default_factory=list)
    # which rolling-stat groups feed the encoder. Names: current / ema / std / delta /
    # max_change / trend_<d> / recency. Special tokens: "all" (default), "static"
    # (= ["current"]), "dynamic" (= everything except "current").
    stat_groups: List[str] = field(default_factory=lambda: ["all"])
    encoder_type: str = "mlp"          # mlp | identity | gru
    hidden_dim: int = 128
    struct_dim: int = 64
    encoder_layers: int = 2
    dropout: float = 0.1
    fusion_mode: str = "struct_only"   # struct_only | concat | additive | gated
    output_dim: int = 128
    inter_dim: int = 0                 # dim of interaction embeddings (for fusion)
    predictor_hidden: int = 128
    truss_recompute_every: int = 64
    feature_clip: float = 0.0          # if >0, log1p-compress raw node features beyond this magnitude
    # sliding-window graph: edges older than ``window_abs`` time units are pruned
    # at each batch boundary; if 0, the graph is cumulative (the default). The
    # runner typically computes window_abs = window_fraction * (t_max - t_min).
    window_abs: float = 0.0
    # pairwise structural features
    #   "all"      -> generic CN/AA/RA/Jaccard/2-hop + k-family-derived feats
    #   "cohesion" -> only the k-family-derived pairwise feats (the paper's signal)
    #   "generic"  -> only the classic common-neighbour heuristics (a baseline)
    #   "none"     -> no pairwise features
    pairwise_mode: str = "all"
    use_pairwise: bool = True          # backward-compat: False overrides pairwise_mode -> "none"
    pairwise_max_2hop: int = 20000
    pairwise_dropout: float = 0.0


class GraphEagleVision(nn.Module):
    def __init__(self, config: Union[GEVConfig, dict, None] = None, **kw) -> None:
        super().__init__()
        if config is None:
            config = GEVConfig(**kw)
        elif isinstance(config, dict):
            config = GEVConfig(**{**config, **kw})
        self.cfg = config

        # --- structural side (non-parametric) ---
        self.indicators: List[BaseIndicator] = build_indicators(
            config.indicators, truss_recompute_every=config.truss_recompute_every
        )
        self.indicator_names: List[str] = [getattr(i, "name", f"ind{j}") for j, i in enumerate(self.indicators)]
        self.K = len(self.indicators)
        self._idx = {n: i for i, n in enumerate(self.indicator_names)}
        self.stats = RollingStatistics(self.K, decay=config.stats_decay, trend_decays=config.trend_decays)
        self.graph = DynamicGraph()

        # which rolling-stat groups (columns) to keep
        all_groups = list(self.stats.group_names)   # e.g. [current, ema, std, delta, max_change, trend_0.99, ..., recency]
        req = list(config.stat_groups)
        if req == ["all"] or not req:
            keep = all_groups
        elif req == ["static"]:
            keep = ["current"]
        elif req == ["dynamic"]:
            keep = [g for g in all_groups if g != "current"]
        else:
            bad = [g for g in req if g not in all_groups]
            if bad:
                raise KeyError(f"unknown stat group(s) {bad}; choose from {all_groups} (or 'all'/'static'/'dynamic')")
            keep = req
        self.stat_groups_used = keep
        cols: List[int] = []
        for g in keep:
            o = self.stats._off[g]
            cols.extend(range(o, o + self.K))
        self._stat_cols = np.asarray(sorted(set(cols)), dtype=np.int64)
        self._use_all_stats = len(self._stat_cols) == self.stats.F

        # --- parametric side ---
        feat_dim = int(len(self._stat_cols))
        self.encoder = build_encoder(
            config.encoder_type, input_dim=feat_dim, num_indicators=self.K,
            hidden_dim=config.hidden_dim, output_dim=config.struct_dim,
            num_layers=config.encoder_layers, dropout=config.dropout,
        )
        struct_dim = self.encoder.output_dim
        self.fusion = build_fusion(
            config.fusion_mode, struct_dim=struct_dim, output_dim=config.output_dim, inter_dim=config.inter_dim,
        )
        h = config.output_dim
        pw_mode = "none" if not config.use_pairwise else config.pairwise_mode
        self._pw_cols = np.asarray(pairwise_mode_columns(pw_mode), dtype=np.int64)
        self.pairwise_dim = int(len(self._pw_cols))
        self.pairwise_feature_names = [PAIRWISE_FEATURE_NAMES[i] for i in self._pw_cols]
        if self.pairwise_dim:
            self.pairwise_norm = nn.LayerNorm(self.pairwise_dim)
            self.pairwise_drop = nn.Dropout(config.pairwise_dropout)
        self.link_predictor = nn.Sequential(
            nn.Linear(2 * h + self.pairwise_dim, config.predictor_hidden),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.predictor_hidden, 1),
        )

    # ================================================================== #
    # structural maintenance
    # ================================================================== #
    def reset_structure(self) -> None:
        self.graph = DynamicGraph()
        for ind in self.indicators:
            try:
                ind.reset()
            except NotImplementedError:
                ind.__init__()  # type: ignore[misc]
        self.stats.reset()

    def update_structure(self, u: int, v: int, t: float = 0.0) -> bool:
        """Add one edge; update all indicators + rolling stats. Returns True iff new edge."""
        u, v, t = int(u), int(v), float(t)
        if not self.graph.add_edge(u, v, t):
            return False
        affected = {u, v}
        for ind in self.indicators:
            affected |= ind.update(self.graph, u, v, t)
        for n in affected:
            vals = np.fromiter((ind.get_value(n) for ind in self.indicators), dtype=np.float32, count=self.K)
            self.stats.update(n, vals)
        return True

    def update_structure_batch(self, srcs, dsts, ts) -> None:
        """Advance the structure by a batch of edges. If ``window_abs`` > 0, first
        prune edges older than (max-time-in-batch − window_abs); if any were
        pruned, re-initialise the indicators on the resulting graph and force a
        rolling-stats refresh for the affected nodes (capturing the indicator
        drop as a negative delta). Then add the batch's edges incrementally."""
        if self.cfg.window_abs and len(ts):
            cutoff = float(max(ts)) - float(self.cfg.window_abs)
            n_removed = self.graph.prune_before(cutoff)
            if n_removed:
                # full reinit of indicators on the post-prune graph
                for ind in self.indicators:
                    ind.initialize(self.graph)
                # refresh rolling stats for every currently-known node (captures drops as deltas)
                for n in list(self.graph.nodes()):
                    vals = np.fromiter((ind.get_value(n) for ind in self.indicators),
                                       dtype=np.float32, count=self.K)
                    self.stats.update(int(n), vals)
        for u, v, t in zip(srcs, dsts, ts):
            self.update_structure(int(u), int(v), float(t))

    def set_window(self, window_abs: float) -> None:
        """Configure the sliding-window size at runtime (overrides ``cfg.window_abs``)."""
        self.cfg.window_abs = float(window_abs)

    # ================================================================== #
    # node-level features
    # ================================================================== #
    def _maybe_clip(self, feats: np.ndarray) -> np.ndarray:
        c = self.cfg.feature_clip
        if c and c > 0:
            return np.sign(feats) * np.log1p(np.abs(feats) / c) * c
        return feats

    def node_features(self, nodes: Sequence[int]) -> np.ndarray:
        feats = self.stats.get_batch_features(nodes)
        if not self._use_all_stats:
            feats = feats[:, self._stat_cols]
        return self._maybe_clip(feats)

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def encode_nodes(self, nodes: Sequence[int]) -> torch.Tensor:
        x = torch.as_tensor(self.node_features(nodes), dtype=torch.float32, device=self._device())
        return self.encoder(x)

    # ================================================================== #
    # pairwise structural features (on the fly, from current graph state)
    # ================================================================== #
    def _value_lookup(self, name: str):
        if name not in self._idx:
            return None
        return self.indicators[self._idx[name]].get_value  # callable node -> value

    def _dyn_lookup(self, name: str):
        """node -> (ema, delta, std, max_change) of the named indicator."""
        if name not in self._idx:
            return None
        k = self._idx[name]
        cur, ema, esq, dl, mx = (self.stats.current, self.stats.ema, self.stats.ema_sq,
                                 self.stats.delta, self.stats.max_change)

        def f(n: int):
            if n not in cur:
                return (0.0, 0.0, 0.0, 0.0)
            e = float(ema[n][k])
            var = float(esq[n][k]) - e * e
            return (e, float(dl[n][k]), (var ** 0.5) if var > 0 else 0.0, float(mx[n][k]))

        return f

    @property
    def _cohesion_indicator(self) -> str:
        """Which indicator drives the cohesiveness pairwise features (tightest available)."""
        for name in ("truss", "core", "triangle", "clustering", "degree"):
            if name in self._idx:
                return name
        return self.indicator_names[0]

    def build_pairwise_csr(self):
        """Build a CSR adjacency snapshot of the current graph (caller caches per batch)."""
        n = max(self.graph.num_nodes, (max(self.graph.adj) + 1) if self.graph.adj else 1)
        return build_csr(self.graph, n)

    def pairwise_features(self, srcs: Sequence[int], dsts: Sequence[int], csr=None) -> np.ndarray:
        if not self.pairwise_dim:
            return np.zeros((len(srcs), 0), dtype=np.float32)
        xname = self._cohesion_indicator
        if csr is not None:
            full = compute_pairs_csr(
                self.graph, csr.shape[0], srcs, dsts, csr=csr,
                x_lookup=self._value_lookup(xname), dyn_lookup=self._dyn_lookup(xname),
                truss_lookup=self._value_lookup("truss"),
            )
        else:
            full = compute_pairs(
                self.graph, srcs, dsts,
                x=self._value_lookup(xname), x_dyn=self._dyn_lookup(xname),
                truss=self._value_lookup("truss"), max_2hop=self.cfg.pairwise_max_2hop,
            )
        return full[:, self._pw_cols]

    # ================================================================== #
    # fusion + scoring  (end-to-end from node ids, using current state)
    # ================================================================== #
    def _fuse(self, h_struct: torch.Tensor, h_inter: Optional[torch.Tensor]) -> torch.Tensor:
        if self.cfg.fusion_mode == "struct_only":
            return self.fusion(h_struct)
        assert h_inter is not None, "non-struct_only fusion needs interaction embeddings"
        return self.fusion(h_struct, h_inter)

    def predict_scores(
        self,
        srcs: Sequence[int],
        dsts: Sequence[int],
        h_inter_src: Optional[torch.Tensor] = None,
        h_inter_dst: Optional[torch.Tensor] = None,
        pairwise_feat: Optional[np.ndarray] = None,
        pairwise_csr=None,
    ) -> torch.Tensor:
        """Score candidate edges (srcs[i], dsts[i]) with the current structural state.

        If ``pairwise_csr`` is given, the (vectorised) sparse-matrix pairwise path
        is used — recommended on dense graphs; the caller builds one CSR per batch.
        """
        h_s = self._fuse(self.encode_nodes(srcs), h_inter_src)
        h_d = self._fuse(self.encode_nodes(dsts), h_inter_dst)
        parts = [h_s, h_d]
        if self.pairwise_dim:
            pf = pairwise_feat if pairwise_feat is not None else self.pairwise_features(srcs, dsts, csr=pairwise_csr)
            pf_t = torch.as_tensor(pf, dtype=torch.float32, device=self._device())
            parts.append(self.pairwise_drop(self.pairwise_norm(pf_t)))
        return self.link_predictor(torch.cat(parts, dim=-1)).squeeze(-1)

    # quick scoring from precomputed pieces (training over cached features) ---
    def score_from_embeddings(self, h_src: torch.Tensor, h_dst: torch.Tensor,
                              pairwise_feat: Optional[torch.Tensor] = None) -> torch.Tensor:
        parts = [h_src, h_dst]
        if self.pairwise_dim and pairwise_feat is not None:
            parts.append(self.pairwise_drop(self.pairwise_norm(pairwise_feat)))
        return self.link_predictor(torch.cat(parts, dim=-1)).squeeze(-1)

    # ------------------------------------------------------------------ #
    @property
    def feature_dim(self) -> int:
        return int(len(self._stat_cols))

    def structural_memory_bytes(self) -> int:
        return self.stats.memory_usage_bytes

    def extra_repr(self) -> str:
        return (
            f"indicators={self.indicator_names}, K={self.K}, node_feat_dim={self.feature_dim}, "
            f"pairwise_dim={self.pairwise_dim}, fusion={self.cfg.fusion_mode}, encoder={self.cfg.encoder_type}"
        )

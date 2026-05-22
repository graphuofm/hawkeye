"""Per-node rolling statistics over the K structural indicators.

Base groups (each K-dim), in order:
  current     : latest indicator values
  ema         : exponential moving average (short-scale trend)
  std         : sqrt of EMA variance (volatility)
  delta       : last change (current - previous)
  max_change  : decayed running max of |delta| (burstiness)
Then one "trend at scale d" group per extra decay d in ``trend_decays``:
  trend_d     : ema - ema_d   (>0 means recently above the longer-scale mean)
And one "recency" group:
  recency     : log1p(#node-events since this indicator last changed)
                — small = recently moved (captures the 0->1 / emergence aspect).

Feature dim = (5 + len(trend_decays) + 1) * K. A dense (capacity, F) table
caches these features so batched reads are vectorised.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


class RollingStatistics:
    def __init__(self, num_indicators: int, decay: float = 0.95,
                 trend_decays: Sequence[float] = (), capacity: int = 1024) -> None:
        self.K = int(num_indicators)
        self.decay = float(decay)
        self.trend_decays: List[float] = [float(d) for d in trend_decays]
        self.n_trend = len(self.trend_decays)
        self.F = (5 + self.n_trend + 1) * self.K
        # state
        self.current: Dict[int, np.ndarray] = {}
        self.ema: Dict[int, np.ndarray] = {}
        self.ema_sq: Dict[int, np.ndarray] = {}
        self.delta: Dict[int, np.ndarray] = {}
        self.max_change: Dict[int, np.ndarray] = {}
        self.ema_trend: List[Dict[int, np.ndarray]] = [dict() for _ in range(self.n_trend)]
        self.steps_since: Dict[int, np.ndarray] = {}   # int32 K-dim, #events since last change
        # dense feature cache
        self._cap = max(1, int(capacity))
        self._table = np.zeros((self._cap, self.F), dtype=np.float32)
        self._zerosF = np.zeros(self.F, dtype=np.float32)
        # column offsets
        self._off = {}
        names = ["current", "ema", "std", "delta", "max_change"] + \
                [f"trend_{d}" for d in self.trend_decays] + ["recency"]
        for gi, nm in enumerate(names):
            self._off[nm] = gi * self.K
        self.group_names = names

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.current.clear(); self.ema.clear(); self.ema_sq.clear()
        self.delta.clear(); self.max_change.clear(); self.steps_since.clear()
        for d in self.ema_trend:
            d.clear()
        self._table.fill(0.0)

    def _ensure(self, n: int) -> None:
        if n >= self._cap:
            new_cap = self._cap
            while n >= new_cap:
                new_cap *= 2
            tab = np.zeros((new_cap, self.F), dtype=np.float32)
            tab[: self._cap] = self._table
            self._table = tab
            self._cap = new_cap

    # ------------------------------------------------------------------ #
    def update(self, node: int, new_values: np.ndarray) -> None:
        nv = np.asarray(new_values, dtype=np.float32)
        d = self.decay
        if node in self.current:
            old = self.current[node]
            dl = nv - old
            self.delta[node] = dl
            self.ema[node] = d * self.ema[node] + (1.0 - d) * nv
            self.ema_sq[node] = d * self.ema_sq[node] + (1.0 - d) * nv * nv
            self.max_change[node] = np.maximum(self.max_change[node] * d, np.abs(dl))
            for i, td in enumerate(self.trend_decays):
                self.ema_trend[i][node] = td * self.ema_trend[i][node] + (1.0 - td) * nv
            ss = self.steps_since[node]
            ss += 1
            ss[dl != 0.0] = 0
        else:
            self.ema[node] = nv.copy()
            self.ema_sq[node] = (nv * nv).copy()
            self.delta[node] = np.zeros(self.K, dtype=np.float32)
            self.max_change[node] = np.zeros(self.K, dtype=np.float32)
            for i in range(self.n_trend):
                self.ema_trend[i][node] = nv.copy()
            self.steps_since[node] = np.zeros(self.K, dtype=np.int64)
        self.current[node] = nv.copy()
        # refresh the cached feature row
        if node >= 0:
            self._ensure(node)
            ema = self.ema[node]
            var = self.ema_sq[node] - ema * ema
            np.clip(var, 0.0, None, out=var)
            K = self.K
            row = self._table[node]
            row[0 * K:1 * K] = nv
            row[1 * K:2 * K] = ema
            row[2 * K:3 * K] = np.sqrt(var)
            row[3 * K:4 * K] = self.delta[node]
            row[4 * K:5 * K] = self.max_change[node]
            o = 5 * K
            for i in range(self.n_trend):
                row[o:o + K] = ema - self.ema_trend[i][node]
                o += K
            row[o:o + K] = np.log1p(self.steps_since[node].astype(np.float32))

    # ------------------------------------------------------------------ #
    def get_features(self, node: int) -> np.ndarray:
        if node not in self.current or node < 0 or node >= self._cap:
            return self._zerosF
        return self._table[node]

    def get_batch_features(self, nodes: Iterable[int]) -> np.ndarray:
        idx = np.asarray(list(nodes), dtype=np.int64)
        if idx.size == 0:
            return np.zeros((0, self.F), dtype=np.float32)
        if idx.max(initial=0) >= self._cap:
            self._ensure(int(idx.max()))
        return self._table[idx]

    def snapshot_table(self, max_node_id: int) -> np.ndarray:
        if max_node_id + 1 > self._cap:
            self._ensure(max_node_id)
        return self._table[: max_node_id + 1].copy()

    # ------------------------------------------------------------------ #
    @property
    def feature_dim(self) -> int:
        return self.F

    @property
    def memory_usage_bytes(self) -> int:
        return len(self.current) * self.F * 4

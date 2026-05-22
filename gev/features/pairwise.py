"""Pairwise structural features for a candidate edge (u, v), computed on the fly
from the current dynamic graph + cohesiveness indicators + their rolling stats.

These complement the per-node rolling-statistics features: ranking candidate
destinations needs *pairwise* signal (common neighbours, etc.), not just node
embeddings. We include:
  * classic 1-hop heuristics (CN / Adamic-Adar / RA / Jaccard / PA);
  * a 2-hop variant (so features are not degenerate on bipartite graphs);
  * **cohesiveness-aware** variants — how cohesive (high-core/truss) the common
    neighbours are;
  * **temporal / dynamics** variants — aggregates of the *rolling statistics*
    (trend / change / volatility) of the common neighbours' cohesiveness, i.e.
    "is the bridge between u and v structurally strengthening over time?".

All raw counts are log1p-compressed. A node never seen has degree 0 and
contributes nothing.
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from gev.graph import DynamicGraph

_LOG = math.log
_LOG1P = math.log1p

# ordered list of feature names this module produces
STATIC_NAMES: List[str] = [
    "cn",            # |N(u) ∩ N(v)|
    "aa",            # Adamic-Adar over 1-hop common neighbours
    "ra",            # Resource Allocation over 1-hop common neighbours
    "jaccard",       # |N(u)∩N(v)| / |N(u)∪N(v)|
    "pa",            # preferential attachment deg(u)*deg(v)
    "cn2",           # |N2(u) ∩ N(v)|  (2-hop common neighbours)
    "aa2",           # Adamic-Adar over 2-hop common neighbours
]
COHESION_NAMES: List[str] = [
    "cn_x_sum",      # Σ_{w∈CN} indicator(w)        -- cohesiveness of the 1-hop bridge
    "cn_x_max",      # max_{w∈CN} indicator(w)
    "cn_x_mean",     # mean_{w∈CN} indicator(w)
    "cn_truss_sum",  # Σ_{w∈CN} truss(w)            (0 if no truss indicator)
    "cn2_x_sum",     # Σ_{w∈CN2} indicator(w)       -- cohesiveness of the 2-hop bridge (works on sparse/bipartite)
    "cn2_x_mean",    # mean_{w∈CN2} indicator(w)
    "x_u",           # indicator(u)
    "x_v",           # indicator(v)
    "x_min",         # min(indicator(u), indicator(v))
    "x_gap",         # |indicator(u) - indicator(v)|
]
DYNAMIC_NAMES: List[str] = [
    "cn_x_ema_sum",   # Σ_{w∈CN} ema[indicator](w)      -- 1-hop bridge's trend level
    "cn_x_delta_sum", # Σ_{w∈CN} delta[indicator](w)    -- 1-hop bridge's recent change
    "cn_x_dpos",      # #{w∈CN : delta[indicator](w) > 0} (log1p)  -- 1-hop bridge rising
    "cn_x_dneg",      # #{w∈CN : delta[indicator](w) < 0} (log1p)
    "cn_x_std_sum",   # Σ_{w∈CN} std[indicator](w)      -- 1-hop bridge's volatility
    "cn_x_mxch_sum",  # Σ_{w∈CN} max_change[indicator](w)
    "cn2_x_ema_sum",  # Σ_{w∈CN2} ema[indicator](w)     -- 2-hop bridge's trend level
    "cn2_x_delta_sum",# Σ_{w∈CN2} delta[indicator](w)   -- 2-hop bridge's recent change
    "cn2_x_dpos",     # #{w∈CN2 : delta[indicator](w) > 0} (log1p)  -- 2-hop bridge rising
    "u_x_ema", "u_x_delta",   # u's own indicator trend / change
    "v_x_ema", "v_x_delta",
]
FEATURE_NAMES: List[str] = STATIC_NAMES + COHESION_NAMES + DYNAMIC_NAMES
DIM = len(FEATURE_NAMES)

# subsets of the {DIM} features for the `pairwise_mode` knob:
#   "generic"   -> only the classic common-neighbour heuristics (NOT k-family): STATIC
#   "cohesion"  -> only the k-family-derived pairwise features (the paper's signal): COHESION+DYNAMIC
#   "all"       -> everything   |   "none" -> nothing
_GENERIC_IDX = list(range(0, len(STATIC_NAMES)))
_COHESION_IDX = list(range(len(STATIC_NAMES), len(STATIC_NAMES) + len(COHESION_NAMES) + len(DYNAMIC_NAMES)))
PAIRWISE_MODES = {
    "none": [],
    "generic": _GENERIC_IDX,
    "cohesion": _COHESION_IDX,
    "all": list(range(DIM)),
}


def pairwise_mode_columns(mode: str) -> List[int]:
    if mode not in PAIRWISE_MODES:
        raise KeyError(f"unknown pairwise_mode {mode!r}; choose from {sorted(PAIRWISE_MODES)}")
    return list(PAIRWISE_MODES[mode])

# type aliases
Lookup = Union[Dict[int, float], Callable[[int], float], None]
# a "dyn" lookup returns (ema, delta, std, max_change) for the chosen indicator
DynLookup = Optional[Callable[[int], Tuple[float, float, float, float]]]


def _mk(lookup: Lookup) -> Callable[[int], float]:
    if lookup is None:
        return lambda _n: 0.0
    if callable(lookup):
        return lookup  # type: ignore[return-value]
    d = lookup
    return lambda n: float(d.get(n, 0.0))


def _two_hop(graph: DynamicGraph, u: int, cap: int) -> set:
    Nu = graph.adj.get(u)
    if not Nu:
        return set()
    out: set = set()
    for w in Nu:
        for x in graph.adj.get(w, ()):  # type: ignore[arg-type]
            if x != u and x not in Nu:
                out.add(x)
                if len(out) >= cap:
                    return out
    return out


_ZERO4 = (0.0, 0.0, 0.0, 0.0)


def compute_pairs(
    graph: DynamicGraph,
    srcs: Sequence[int],
    dsts: Sequence[int],
    *,
    x: Lookup = None,            # the active cohesiveness indicator value (core / degree / ...)
    x_dyn: DynLookup = None,     # node -> (ema, delta, std, max_change) of that indicator
    truss: Lookup = None,
    max_2hop: int = 20000,
    log_compress: bool = True,
) -> np.ndarray:
    """Return (len(srcs), DIM) float32 array of pairwise features.

    Efficient when many rows share the same source (groups by src, reuses that
    source's neighbour / 2-hop sets) — exactly the eval pattern.
    """
    srcs = np.asarray(srcs, dtype=np.int64)
    dsts = np.asarray(dsts, dtype=np.int64)
    assert len(srcs) == len(dsts)
    M = len(srcs)
    out = np.zeros((M, DIM), dtype=np.float32)
    x_f = _mk(x)
    truss_f = _mk(truss)
    xd_f = x_dyn if x_dyn is not None else (lambda _n: _ZERO4)
    f1p = _LOG1P if log_compress else (lambda v: v)

    order = np.argsort(srcs, kind="stable")
    i = 0
    while i < M:
        j = i
        s = int(srcs[order[i]])
        while j < M and int(srcs[order[j]]) == s:
            j += 1
        rows = order[i:j]
        Ns = graph.adj.get(s, set())
        deg_s = len(Ns)
        x_s = x_f(s)
        sema, sdel, _ss, _sm = xd_f(s)
        N2s = _two_hop(graph, s, max_2hop) if deg_s else set()
        for r in rows:
            d = int(dsts[r])
            Nd = graph.adj.get(d, set())
            deg_d = len(Nd)
            if deg_s and deg_d:
                small, big = (Ns, Nd) if deg_s <= deg_d else (Nd, Ns)
                common = [w for w in small if w in big]
            else:
                common = []
            cn = len(common)
            aa = ra = 0.0
            x_sum = x_max = 0.0
            tr_sum = 0.0
            ema_sum = del_sum = std_sum = mxch_sum = 0.0
            dpos = dneg = 0
            for w in common:
                dw = graph.degree(w)
                if dw > 1:
                    aa += 1.0 / _LOG(dw)
                if dw > 0:
                    ra += 1.0 / dw
                xw = x_f(w)
                x_sum += xw
                if xw > x_max:
                    x_max = xw
                tr_sum += truss_f(w)
                ew, dlw, stw, mxw = xd_f(w)
                ema_sum += ew
                del_sum += dlw
                std_sum += stw
                mxch_sum += mxw
                if dlw > 0.0:
                    dpos += 1
                elif dlw < 0.0:
                    dneg += 1
            union = deg_s + deg_d - cn
            jac = (cn / union) if union > 0 else 0.0
            if N2s and deg_d:
                small2, big2 = (N2s, Nd) if len(N2s) <= deg_d else (Nd, N2s)
                common2 = [a for a in small2 if a in big2]
            else:
                common2 = []
            cn2 = len(common2)
            aa2 = 0.0
            x2_sum = ema2_sum = del2_sum = 0.0
            dpos2 = 0
            for a in common2:
                da = graph.degree(a)
                if da > 1:
                    aa2 += 1.0 / _LOG(da)
                xa = x_f(a)
                x2_sum += xa
                ea, dla, _sta, _mxa = xd_f(a)
                ema2_sum += ea
                del2_sum += dla
                if dla > 0.0:
                    dpos2 += 1
            x_d = x_f(d)
            dema, ddel, _ds, _dm = xd_f(d)
            out[r] = (
                # static (7) -- graph-topology heuristics, no k-family indicator
                f1p(cn), aa, ra, jac, f1p(deg_s * deg_d), f1p(cn2), aa2,
                # cohesion (10) -- uses the k-family indicator at the current step
                f1p(x_sum), x_max, (x_sum / cn) if cn else 0.0, f1p(tr_sum),
                f1p(x2_sum), (x2_sum / cn2) if cn2 else 0.0,
                x_s, x_d, min(x_s, x_d), abs(x_s - x_d),
                # dynamic (13) -- uses the k-family indicator's rolling statistics
                ema_sum, del_sum, f1p(dpos), f1p(dneg), std_sum, mxch_sum,
                ema2_sum, del2_sum, f1p(dpos2),
                sema, sdel, dema, ddel,
            )
        i = j
    return out


def pairwise_feature_names() -> List[str]:
    return list(FEATURE_NAMES)

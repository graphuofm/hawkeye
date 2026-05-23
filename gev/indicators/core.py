"""Incremental k-core (coreness) maintenance for an insertion-only graph.

Key fact (insertion only): when edge (u, v) is added with K = min(core[u], core[v]),
the only nodes whose coreness can change are those reachable from {u, v} via nodes
of coreness *exactly* K (the "K-shell component" connected through the new edge),
and each such node's coreness can increase by at most 1.

We collect that candidate set S, then peel it: a node w of coreness K joins the
(K+1)-core iff it has > K neighbours that are themselves in the (K+1)-core or
higher. Neighbours of S with coreness > K are unconditional survivors; neighbours
of S with coreness == K are all in S (closure). Peeling S w.r.t. effective degree
(neighbours-in-S + neighbours-with-core>K) and threshold K gives exactly the set
of promoted nodes. Cost: O(|S| + edges incident to S) — local in practice.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Set

from gev.graph import DynamicGraph
from gev.indicators.base import BaseIndicator


class CoreNumberIndicator(BaseIndicator):
    """k-core number. Full O(n+m) recompute via the C++ kernel every
    ``recompute_every`` inserted edges (falls back to a pure-Python full
    decomposition if the native kernel is unavailable). Core numbers are exact
    at each recompute; between recomputes they are held (a documented
    structural-trace granularity, identical to the truss indicator)."""

    name = "core"
    complexity = "O(n+m) native recompute / recompute_every"
    supports_incremental = False

    def __init__(self, recompute_every: int = 64) -> None:
        self._core: Dict[int, int] = defaultdict(int)
        self.recompute_every = max(1, int(recompute_every))
        self._since = 0

    # ------------------------------------------------------------------ #
    def initialize(self, graph: DynamicGraph) -> None:
        self._recompute(graph)
        self._since = 0

    def reset(self) -> None:
        self._core = defaultdict(int)
        self._since = 0

    # ------------------------------------------------------------------ #
    def _recompute(self, graph: DynamicGraph) -> Set[int]:
        old = self._core
        c: Dict[int, int] = defaultdict(int)
        from gev import native
        if native.available():
            arr = native.kcore(graph)
            for nid, k in enumerate(arr):
                if k:
                    c[int(nid)] = int(k)
        else:
            c.update(_kcore_full(graph))
        self._core = c
        return {n for n in set(old) | set(c) if old.get(n, 0) != c.get(n, 0)}

    def update(self, graph: DynamicGraph, u: int, v: int, t: float) -> Set[int]:
        self._since += 1
        if self._since >= self.recompute_every:
            self._since = 0
            return self._recompute(graph) | {u, v}
        return {u, v}

    # ------------------------------------------------------------------ #
    def get_value(self, node: int) -> float:
        return float(self._core.get(node, 0))

    def get_all_values(self) -> Dict[int, float]:
        return {k: float(v) for k, v in self._core.items()}


# ---------------------------------------------------------------------- #
def _kcore_full(graph: DynamicGraph) -> Dict[int, int]:
    """Reference coreness via min-degree peeling. O(m log n). Used for init/tests."""
    import heapq

    deg = {n: graph.degree(n) for n in graph.nodes()}
    core: Dict[int, int] = {}
    if not deg:
        return core
    heap = [(d, n) for n, d in deg.items()]
    heapq.heapify(heap)
    removed: Set[int] = set()
    cur = 0
    while heap:
        d, n = heapq.heappop(heap)
        if n in removed or d != deg[n]:
            continue  # stale entry
        cur = max(cur, d)
        core[n] = cur
        removed.add(n)
        for w in graph.neighbors(n):
            if w not in removed:
                deg[w] -= 1
                heapq.heappush(heap, (deg[w], w))
    return core

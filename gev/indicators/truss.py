"""k-truss (trussness) indicator.

Trussness is defined on *edges*: tau(e) is the largest k such that e belongs to a
k-truss (every edge supported by >= k-2 triangles). We map it to nodes via
tau_node(v) = max trussness over v's incident edges (0 if isolated).

A truly incremental k-truss maintenance algorithm is involved; for v1 we recompute
the full edge-trussness every ``recompute_every`` inserted edges (truss is only
used on small/medium graphs, where O(m^1.5) recompute is cheap). Between
recomputes ``update`` reports no change. This is an explicit, documented
approximation of the temporal trace granularity for the truss indicator.

TODO(perf): replace with a proper incremental truss-maintenance algorithm.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Set, Tuple

from gev.graph import DynamicGraph
from gev.indicators.base import BaseIndicator


def _ekey(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a < b else (b, a)


def truss_decomposition(graph: DynamicGraph) -> Dict[Tuple[int, int], int]:
    """Edge trussness via the standard support-peeling algorithm.

    tau(e) returned as k (a k-truss means each edge has >= k-2 triangles). Edges
    in no triangle get trussness 2 (belong to a 2-truss trivially).
    """
    # support[e] = number of triangles containing e
    support: Dict[Tuple[int, int], int] = {}
    for u in graph.nodes():
        for v in graph.neighbors(u):
            if v <= u:
                continue
            e = (u, v)
            s = 0
            su, sv = graph.adj[u], graph.adj[v]
            small, big = (su, sv) if len(su) <= len(sv) else (sv, su)
            for w in small:
                if w in big:
                    s += 1
            support[e] = s
    if not support:
        return {}

    # bucket peeling by current support
    import heapq

    trussness: Dict[Tuple[int, int], int] = {}
    sup = dict(support)
    heap = [(s, e) for e, s in sup.items()]
    heapq.heapify(heap)
    removed: Set[Tuple[int, int]] = set()
    k = 2
    while heap:
        s, e = heapq.heappop(heap)
        if e in removed or s != sup[e]:
            continue
        if s + 2 > k:
            k = s + 2
        trussness[e] = k
        removed.add(e)
        u, v = e
        # decrement support of edges that shared a *still-present* triangle with e
        su, sv = graph.adj[u], graph.adj[v]
        small, big_set = (su, sv) if len(su) <= len(sv) else (sv, su)
        for w in small:
            if w not in big_set:
                continue
            e1 = _ekey(u, w)
            e2 = _ekey(v, w)
            if e1 in removed or e2 in removed:
                continue  # that triangle no longer exists in the peeled graph
            sup[e1] -= 1
            heapq.heappush(heap, (sup[e1], e1))
            sup[e2] -= 1
            heapq.heappush(heap, (sup[e2], e2))
    return trussness


class TrussIndicator(BaseIndicator):
    name = "truss"
    complexity = "O(m^1.5) recompute / recompute_every"
    supports_incremental = False
    max_recommended_edges = 1_000_000

    def __init__(self, recompute_every: int = 1) -> None:
        self.recompute_every = max(1, int(recompute_every))
        self._node_truss: Dict[int, int] = defaultdict(int)
        self._since = 0

    def initialize(self, graph: DynamicGraph) -> None:
        self._recompute(graph)
        self._since = 0

    def reset(self) -> None:
        self._node_truss = defaultdict(int)
        self._since = 0

    def _recompute(self, graph: DynamicGraph) -> Set[int]:
        old = dict(self._node_truss)
        edge_truss = truss_decomposition(graph)
        nt: Dict[int, int] = defaultdict(int)
        for (a, b), k in edge_truss.items():
            if k > nt[a]:
                nt[a] = k
            if k > nt[b]:
                nt[b] = k
        self._node_truss = nt
        changed = {n for n in set(old) | set(nt) if old.get(n, 0) != nt.get(n, 0)}
        return changed

    def update(self, graph: DynamicGraph, u: int, v: int, t: float) -> Set[int]:
        self._since += 1
        if self._since >= self.recompute_every:
            self._since = 0
            return self._recompute(graph) | {u, v}
        return {u, v}

    def get_value(self, node: int) -> float:
        return float(self._node_truss.get(node, 0))

    def get_all_values(self) -> Dict[int, float]:
        return {k: float(v) for k, v in self._node_truss.items()}

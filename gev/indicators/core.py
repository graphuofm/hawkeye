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
    name = "core"
    complexity = "O(local)"
    supports_incremental = True

    def __init__(self) -> None:
        self._core: Dict[int, int] = defaultdict(int)

    # ------------------------------------------------------------------ #
    def initialize(self, graph: DynamicGraph) -> None:
        self._core = defaultdict(int)
        self._core.update(_kcore_full(graph))

    def reset(self) -> None:
        self._core = defaultdict(int)

    # ------------------------------------------------------------------ #
    def update(self, graph: DynamicGraph, u: int, v: int, t: float) -> Set[int]:
        core = self._core
        cu, cv = core[u], core[v]
        K = cu if cu <= cv else cv
        root = u if cu <= cv else v

        # 1) collect candidate set S: BFS from root through nodes of coreness == K
        S: Set[int] = set()
        dq = deque([root])
        S.add(root)
        adj = graph.adj
        while dq:
            w = dq.popleft()
            for x in adj.get(w, ()):  # type: ignore[arg-type]
                if x not in S and core[x] == K:
                    S.add(x)
                    dq.append(x)

        if not S:
            return {u, v}

        # 2) effective degree within S (+ neighbours with core > K count as fixed survivors)
        eff: Dict[int, int] = {}
        for w in S:
            d = 0
            for x in adj.get(w, ()):  # type: ignore[arg-type]
                cx = core[x]
                if cx > K or x in S:
                    d += 1
            eff[w] = d

        # 3) peel: repeatedly drop nodes with eff <= K
        alive = set(S)
        dq = deque(w for w in S if eff[w] <= K)
        while dq:
            w = dq.popleft()
            if w not in alive:
                continue
            alive.discard(w)
            for x in adj.get(w, ()):  # type: ignore[arg-type]
                if x in alive and core[x] == K:  # x is in S (closure) by construction
                    eff[x] -= 1
                    if eff[x] <= K:
                        dq.append(x)

        # 4) promote survivors
        for w in alive:
            core[w] = K + 1

        affected = set(alive)
        affected.add(u)
        affected.add(v)
        return affected

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

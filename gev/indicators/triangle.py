"""Triangle participation count per node (a cheap proxy for k-truss).

On inserting edge (u, v): every common neighbour w of u and v closes a new
triangle {u, v, w}, so tri[u] += 1, tri[v] += 1, tri[w] += 1.
Cost: O(min(deg u, deg v)) per edge.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Set

from gev.graph import DynamicGraph
from gev.indicators.base import BaseIndicator


class TriangleIndicator(BaseIndicator):
    name = "triangle"
    complexity = "O(min(d_u, d_v))"
    supports_incremental = True
    max_recommended_edges = 10_000_000

    def __init__(self) -> None:
        self._tri: Dict[int, int] = defaultdict(int)

    def initialize(self, graph: DynamicGraph) -> None:
        self._tri = defaultdict(int)
        for u in graph.nodes():
            for v in graph.neighbors(u):
                if v <= u:
                    continue
                for w in graph.common_neighbors(u, v):
                    self._tri[w] += 1  # counted once per (u<v) edge
        # each triangle {a,b,c} counted: for edge(a,b): +c ; edge(a,c): +b ; edge(b,c): +a
        # so above already gives tri[w] = #triangles containing w. (each triangle
        # contributes to exactly the 3 (edge, apex) pairs, one per vertex of the triangle)

    def reset(self) -> None:
        self._tri = defaultdict(int)

    def update(self, graph: DynamicGraph, u: int, v: int, t: float) -> Set[int]:
        affected: Set[int] = {u, v}
        cn = graph.common_neighbors(u, v)
        if cn:
            n = len(cn)
            self._tri[u] += n
            self._tri[v] += n
            for w in cn:
                self._tri[w] += 1
                affected.add(w)
        return affected

    def get_value(self, node: int) -> float:
        return float(self._tri.get(node, 0))

    def get_all_values(self) -> Dict[int, float]:
        return {k: float(v) for k, v in self._tri.items()}

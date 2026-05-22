"""Local clustering coefficient: cc(v) = 2 * triangles(v) / (deg(v) * (deg(v)-1)).

Derived from a TriangleIndicator + the dynamic graph; cheap to read.
"""
from __future__ import annotations

from typing import Dict, Set

from gev.graph import DynamicGraph
from gev.indicators.base import BaseIndicator
from gev.indicators.triangle import TriangleIndicator


class ClusteringCoefficientIndicator(BaseIndicator):
    name = "clustering"
    complexity = "O(min(d_u, d_v))"  # same as triangle
    supports_incremental = True
    max_recommended_edges = 10_000_000

    def __init__(self, triangle: TriangleIndicator | None = None) -> None:
        self._tri = triangle if triangle is not None else TriangleIndicator()
        self._own_tri = triangle is None
        self._graph: DynamicGraph | None = None

    def initialize(self, graph: DynamicGraph) -> None:
        self._graph = graph
        if self._own_tri:
            self._tri.initialize(graph)

    def reset(self) -> None:
        if self._own_tri:
            self._tri.reset()

    def update(self, graph: DynamicGraph, u: int, v: int, t: float) -> Set[int]:
        self._graph = graph
        if self._own_tri:
            return self._tri.update(graph, u, v, t)
        return {u, v}  # triangle indicator already updated upstream

    def _cc(self, node: int) -> float:
        if self._graph is None:
            return 0.0
        d = self._graph.degree(node)
        if d < 2:
            return 0.0
        return 2.0 * self._tri.get_value(node) / (d * (d - 1))

    def get_value(self, node: int) -> float:
        return self._cc(node)

    def get_all_values(self) -> Dict[int, float]:
        if self._graph is None:
            return {}
        return {n: self._cc(n) for n in self._graph.nodes()}

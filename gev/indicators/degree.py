"""Degree indicator — the weakest cohesiveness constraint. O(1) per edge."""
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Set

from gev.graph import DynamicGraph
from gev.indicators.base import BaseIndicator


class DegreeIndicator(BaseIndicator):
    name = "degree"
    complexity = "O(1)"
    supports_incremental = True

    def __init__(self) -> None:
        self._deg: Dict[int, int] = defaultdict(int)

    def initialize(self, graph: DynamicGraph) -> None:
        self._deg = defaultdict(int)
        for n in graph.nodes():
            self._deg[n] = graph.degree(n)

    def update(self, graph: DynamicGraph, u: int, v: int, t: float) -> Set[int]:
        # graph already updated; just re-read the (possibly) changed degrees.
        self._deg[u] = graph.degree(u)
        self._deg[v] = graph.degree(v)
        return {u, v}

    def get_value(self, node: int) -> float:
        return float(self._deg.get(node, 0))

    def get_all_values(self) -> Dict[int, float]:
        return {k: float(v) for k, v in self._deg.items()}

    def reset(self) -> None:
        self._deg = defaultdict(int)

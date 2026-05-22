"""Base interface for structural cohesiveness indicators."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Iterable, List, Set, Tuple

from gev.graph import DynamicGraph


class BaseIndicator(ABC):
    """A per-node structural indicator maintained incrementally over an edge stream.

    Subclasses must keep an internal mapping ``node -> value`` and update it when
    ``update`` is called. ``update`` returns the set of nodes whose value changed
    *as a result of this edge* (the endpoints u, v are always re-read by the
    framework regardless, so they need not be included, but it is harmless to).
    """

    name: str = "base"
    level: str = "node"          # "node" or "edge" (edge-level indicators map to nodes)
    complexity: str = "?"        # human-readable per-update complexity
    supports_incremental: bool = True
    max_recommended_edges: int = 1 << 62

    # ------------------------------------------------------------------ #
    def initialize(self, graph: DynamicGraph) -> None:
        """(Re)initialise state from an existing graph. Default: assume empty."""
        return None

    @abstractmethod
    def update(self, graph: DynamicGraph, u: int, v: int, t: float) -> Set[int]:
        """Incrementally update after edge (u, v, t) was added to ``graph``.

        ``graph`` already contains the new edge. Returns set of affected nodes.
        """

    @abstractmethod
    def get_value(self, node: int) -> float:
        ...

    @abstractmethod
    def get_all_values(self) -> Dict[int, float]:
        ...

    # ------------------------------------------------------------------ #
    def batch_update(self, graph: DynamicGraph, edges: List[Tuple[int, int, float]]) -> Set[int]:
        affected: Set[int] = set()
        for u, v, t in edges:
            affected |= self.update(graph, u, v, t)
        return affected

    def reset(self) -> None:
        """Clear all state (for re-streaming across epochs)."""
        raise NotImplementedError

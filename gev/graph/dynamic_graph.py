"""Dynamic (insertion-only) simple graph for incremental structural maintenance.

We only track the *simple* undirected graph (distinct neighbor sets). Repeated
edges (same (u, v) at a later timestamp) are no-ops for structure. Timestamps of
the *first* appearance of each edge are kept for optional temporal queries.
"""
from __future__ import annotations

from typing import Dict, Iterable, Set, Tuple


class DynamicGraph:
    __slots__ = ("adj", "edge_time", "_num_edges", "_edge_seq", "_expired_idx")

    def __init__(self) -> None:
        self.adj: Dict[int, Set[int]] = {}
        self.edge_time: Dict[Tuple[int, int], float] = {}
        self._num_edges: int = 0
        # arrival-ordered list of (t, u, v) for the *currently present* edges
        # (set when add_edge succeeds; entries are skipped after remove_edge).
        # Used by prune_before for sliding-window mode.
        self._edge_seq: list = []
        self._expired_idx: int = 0

    # ------------------------------------------------------------------ #
    def add_edge(self, u: int, v: int, t: float = 0.0) -> bool:
        """Add undirected edge. Returns True iff it is a *new* edge."""
        if u == v:
            return False
        au = self.adj.setdefault(u, set())
        av = self.adj.setdefault(v, set())
        if v in au:
            return False  # already present
        au.add(v)
        av.add(u)
        key = (u, v) if u < v else (v, u)
        self.edge_time[key] = t
        self._num_edges += 1
        self._edge_seq.append((t, u, v))
        return True

    # ------------------------------------------------------------------ #
    def remove_edge(self, u: int, v: int) -> bool:
        """Remove undirected edge. Returns True iff it was present."""
        au = self.adj.get(u); av = self.adj.get(v)
        if au is None or av is None or v not in au:
            return False
        au.discard(v); av.discard(u)
        key = (u, v) if u < v else (v, u)
        self.edge_time.pop(key, None)
        self._num_edges -= 1
        return True

    def prune_before(self, cutoff_time: float) -> int:
        """Remove every edge whose insertion time is strictly less than ``cutoff_time``.
        Returns the number of edges removed. O(#expired). Assumes ``_edge_seq``
        is non-decreasing in time (true if you only call add_edge in chronological
        order)."""
        removed = 0
        n = len(self._edge_seq)
        i = self._expired_idx
        while i < n and self._edge_seq[i][0] < cutoff_time:
            t, u, v = self._edge_seq[i]
            key = (u, v) if u < v else (v, u)
            # may have been re-inserted later; check edge_time still equals this t
            if self.edge_time.get(key) == t:
                if self.remove_edge(u, v):
                    removed += 1
            i += 1
        self._expired_idx = i
        return removed

    # ------------------------------------------------------------------ #
    def neighbors(self, u: int) -> Set[int]:
        return self.adj.get(u, _EMPTY)

    def degree(self, u: int) -> int:
        return len(self.adj.get(u, _EMPTY))

    def has_edge(self, u: int, v: int) -> bool:
        s = self.adj.get(u)
        return s is not None and v in s

    def nodes(self) -> Iterable[int]:
        return self.adj.keys()

    @property
    def num_nodes(self) -> int:
        return len(self.adj)

    @property
    def num_edges(self) -> int:
        return self._num_edges

    def common_neighbors(self, u: int, v: int) -> Set[int]:
        su = self.adj.get(u, _EMPTY)
        sv = self.adj.get(v, _EMPTY)
        # iterate the smaller one
        if len(su) > len(sv):
            su, sv = sv, su
        return {w for w in su if w in sv}


_EMPTY: Set[int] = frozenset()  # type: ignore[assignment]

"""Indicator registry + auto-scaling by graph size."""
from __future__ import annotations

from typing import Dict, List, Type, Union

from gev.indicators.base import BaseIndicator
from gev.indicators.clustering import ClusteringCoefficientIndicator
from gev.indicators.core import CoreNumberIndicator
from gev.indicators.degree import DegreeIndicator
from gev.indicators.triangle import TriangleIndicator
from gev.indicators.truss import TrussIndicator

REGISTRY: Dict[str, Type[BaseIndicator]] = {
    "degree": DegreeIndicator,
    "core": CoreNumberIndicator,
    "triangle": TriangleIndicator,
    "truss": TrussIndicator,
    "clustering": ClusteringCoefficientIndicator,
}


def build_indicators(
    names: List[Union[str, BaseIndicator]],
    truss_recompute_every: int = 64,
) -> List[BaseIndicator]:
    out: List[BaseIndicator] = []
    shared_triangle: TriangleIndicator | None = None
    for n in names:
        if isinstance(n, BaseIndicator):
            out.append(n)
            if isinstance(n, TriangleIndicator):
                shared_triangle = n
            continue
        n = n.strip().lower()
        if n not in REGISTRY:
            raise KeyError(f"unknown indicator {n!r}; available: {sorted(REGISTRY)}")
        cls = REGISTRY[n]
        if cls is TrussIndicator:
            out.append(TrussIndicator(recompute_every=truss_recompute_every))
        elif cls is TriangleIndicator:
            shared_triangle = TriangleIndicator()
            out.append(shared_triangle)
        elif cls is ClusteringCoefficientIndicator:
            out.append(ClusteringCoefficientIndicator(triangle=shared_triangle))
        else:
            out.append(cls())
    return out


def auto_scale_indicators(num_edges_hint: int) -> List[str]:
    """Recommended indicator set given the expected number of edges."""
    if num_edges_hint > 10_000_000:
        return ["degree", "core"]
    if num_edges_hint > 1_000_000:
        return ["degree", "core", "triangle"]
    return ["degree", "core", "triangle", "truss"]

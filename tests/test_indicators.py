"""Correctness of incremental indicators vs. full recomputation / networkx."""
import random

import networkx as nx
import pytest

from gev.graph import DynamicGraph
from gev.indicators import (
    ClusteringCoefficientIndicator,
    CoreNumberIndicator,
    DegreeIndicator,
    TriangleIndicator,
    TrussIndicator,
)
from gev.indicators.core import _kcore_full
from gev.indicators.truss import truss_decomposition


def _random_edge_stream(n_nodes, n_edges, seed):
    """Distinct undirected edges in a random order (the indicator update contract
    assumes each call corresponds to a structurally new edge)."""
    rng = random.Random(seed)
    edges = []
    seen = set()
    while len(edges) < n_edges:
        u = rng.randrange(n_nodes)
        v = rng.randrange(n_nodes)
        if u == v:
            continue
        key = (u, v) if u < v else (v, u)
        if key in seen:
            continue
        seen.add(key)
        edges.append((u, v, len(edges)))
    return edges


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_incremental_core_matches_full(seed):
    edges = _random_edge_stream(40, 250, seed)
    g = DynamicGraph()
    ind = CoreNumberIndicator()
    for u, v, t in edges:
        g.add_edge(u, v, t)
        ind.update(g, u, v, t)
    inc = {n: int(ind.get_value(n)) for n in g.nodes()}
    full = _kcore_full(g)
    # networkx ground truth
    G = nx.Graph()
    G.add_nodes_from(g.nodes())
    for u in g.nodes():
        for v in g.neighbors(u):
            G.add_edge(u, v)
    nx_core = nx.core_number(G)
    assert inc == {n: nx_core[n] for n in inc}, "incremental k-core mismatch vs networkx"
    assert full == {n: nx_core[n] for n in full}, "full k-core mismatch vs networkx"


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_incremental_triangle_matches_nx(seed):
    edges = _random_edge_stream(35, 200, seed)
    g = DynamicGraph()
    ind = TriangleIndicator()
    for u, v, t in edges:
        g.add_edge(u, v, t)
        ind.update(g, u, v, t)
    G = nx.Graph()
    for u in g.nodes():
        for v in g.neighbors(u):
            G.add_edge(u, v)
    nx_tri = nx.triangles(G)
    inc = {n: int(ind.get_value(n)) for n in g.nodes()}
    assert inc == {n: nx_tri.get(n, 0) for n in inc}, "incremental triangle count mismatch"


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_truss_decomposition_matches_definition(seed):
    edges = _random_edge_stream(25, 120, seed)
    g = DynamicGraph()
    for u, v, t in edges:
        g.add_edge(u, v, t)
    tau = truss_decomposition(g)
    # brute-force check: for each k, the k-truss = maximal subgraph where every
    # edge has >= k-2 triangles. Verify each edge's reported trussness is exactly
    # the largest k for which it survives the k-truss peeling.
    G = nx.Graph()
    for u in g.nodes():
        for v in g.neighbors(u):
            G.add_edge(u, v)

    def k_truss_edges(k):
        H = G.copy()
        changed = True
        while changed:
            changed = False
            for (a, b) in list(H.edges()):
                sup = len(set(H[a]) & set(H[b]))
                if sup < k - 2:
                    H.remove_edge(a, b)
                    changed = True
        return set(tuple(sorted(e)) for e in H.edges())

    maxk = max(tau.values()) if tau else 2
    surv = {k: k_truss_edges(k) for k in range(2, maxk + 2)}
    for e, k in tau.items():
        ek = tuple(sorted(e))
        assert ek in surv.get(k, set()), f"edge {e} should be in {k}-truss"
        assert ek not in surv.get(k + 1, set()), f"edge {e} should NOT be in {k+1}-truss"


@pytest.mark.parametrize("seed", [0, 1])
def test_truss_indicator_recompute(seed):
    edges = _random_edge_stream(20, 90, seed)
    g = DynamicGraph()
    ind = TrussIndicator(recompute_every=1)
    for u, v, t in edges:
        g.add_edge(u, v, t)
        ind.update(g, u, v, t)
    tau = truss_decomposition(g)
    nt = {}
    for (a, b), k in tau.items():
        nt[a] = max(nt.get(a, 0), k)
        nt[b] = max(nt.get(b, 0), k)
    inc = {n: int(ind.get_value(n)) for n in g.nodes() if ind.get_value(n) > 0}
    assert inc == {n: nt[n] for n in inc}


def test_degree_and_clustering():
    g = DynamicGraph()
    deg = DegreeIndicator()
    tri = TriangleIndicator()
    cc = ClusteringCoefficientIndicator(triangle=tri)
    cc.initialize(g)
    # triangle 0-1-2 plus pendant 3-0
    for (u, v) in [(0, 1), (1, 2), (2, 0), (0, 3)]:
        g.add_edge(u, v, 0)
        deg.update(g, u, v, 0)
        tri.update(g, u, v, 0)
        cc.update(g, u, v, 0)
    assert deg.get_value(0) == 3 and deg.get_value(1) == 2 and deg.get_value(3) == 1
    assert tri.get_value(0) == 1 and tri.get_value(1) == 1 and tri.get_value(2) == 1
    # cc(1) = 2*1 / (2*1) = 1.0 ; cc(0) = 2*1 / (3*2) = 1/3
    assert abs(cc.get_value(1) - 1.0) < 1e-9
    assert abs(cc.get_value(0) - 1.0 / 3.0) < 1e-9

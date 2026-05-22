"""Vectorised pairwise structural features via a CSR adjacency matrix.

Same features as ``gev.features.pairwise.compute_pairs`` (so output columns line
up with ``FEATURE_NAMES``), but computed with a handful of sparse mat-vecs per
*source* instead of Python set ops. This is what makes dense graphs
(tgbl-enron deg≈677, tgbl-lastfm deg≈1306, ...) and bigger graphs tractable.

The caller builds a CSR snapshot of the current graph (``build_csr``) and passes
it in; one CSR per chronological batch is plenty (the structural state used for
that batch's predictions).

Approximation note: ``cn_x_max`` (max indicator over the 1-hop common
neighbours) is approximated by ``max(indicator over N(src)) * 1[cn>0]`` — an
upper bound, exact when the common-neighbour set spans N(src)'s max.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import scipy.sparse as sp

from gev.features.pairwise import DIM
from gev.graph import DynamicGraph

_LOG = np.log


def build_csr(graph: DynamicGraph, num_nodes: int) -> sp.csr_matrix:
    """Symmetric binary CSR adjacency of the current simple graph (rows 0..num_nodes-1)."""
    rows, cols = [], []
    for u, nbrs in graph.adj.items():
        if not nbrs:
            continue
        for v in nbrs:
            rows.append(u); cols.append(v)
    if not rows:
        return sp.csr_matrix((num_nodes, num_nodes), dtype=np.float32)
    data = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((data, (np.asarray(rows), np.asarray(cols))),
                         shape=(num_nodes, num_nodes), dtype=np.float32)


def _arrs(graph: DynamicGraph, num_nodes: int, x_lookup, dyn_lookup, truss_lookup):
    """Dense per-node arrays: degree, x, (ema,delta,std,maxch), truss."""
    deg = np.zeros(num_nodes, dtype=np.float32)
    x = np.zeros(num_nodes, dtype=np.float32)
    truss = np.zeros(num_nodes, dtype=np.float32)
    ema = np.zeros(num_nodes, dtype=np.float32)
    delta = np.zeros(num_nodes, dtype=np.float32)
    std = np.zeros(num_nodes, dtype=np.float32)
    mxch = np.zeros(num_nodes, dtype=np.float32)
    xf = x_lookup or (lambda _n: 0.0)
    tf = truss_lookup or (lambda _n: 0.0)
    df = dyn_lookup or (lambda _n: (0.0, 0.0, 0.0, 0.0))
    for n in graph.adj:
        if not (0 <= n < num_nodes):
            continue
        deg[n] = graph.degree(n)
        x[n] = xf(n)
        truss[n] = tf(n)
        e, dl, s, m = df(n)
        ema[n] = e; delta[n] = dl; std[n] = s; mxch[n] = m
    return deg, x, truss, ema, delta, std, mxch


def compute_pairs_csr(
    graph: DynamicGraph,
    num_nodes: int,
    srcs,
    dsts,
    *,
    csr: Optional[sp.csr_matrix] = None,
    x_lookup=None,
    dyn_lookup=None,
    truss_lookup=None,
    log_compress: bool = True,
) -> np.ndarray:
    srcs = np.asarray(srcs, dtype=np.int64)
    dsts = np.asarray(dsts, dtype=np.int64)
    M = len(srcs)
    out = np.zeros((M, DIM), dtype=np.float32)
    if M == 0:
        return out
    A = csr if csr is not None else build_csr(graph, num_nodes)
    deg, x, truss, ema, delta, std, mxch = _arrs(graph, num_nodes, x_lookup, dyn_lookup, truss_lookup)
    invlog = np.where(deg > 1, 1.0 / np.maximum(_LOG(np.maximum(deg, 2.0)), 1e-9), 0.0).astype(np.float32)
    invdeg = np.where(deg > 0, 1.0 / np.maximum(deg, 1.0), 0.0).astype(np.float32)
    dpos = (delta > 0).astype(np.float32)
    dneg = (delta < 0).astype(np.float32)
    f1p = np.log1p if log_compress else (lambda v: v)

    order = np.argsort(srcs, kind="stable")
    i = 0
    while i < M:
        j = i
        s = int(srcs[order[i]])
        while j < M and int(srcs[order[j]]) == s:
            j += 1
        rows = order[i:j]
        if not (0 <= s < num_nodes):
            i = j; continue
        As = np.asarray(A.getrow(s).todense()).ravel().astype(np.float32)  # N
        Ns_bool = As > 0
        deg_s = float(deg[s])
        # 1-hop common-neighbour aggregates over candidate j: (A @ (weight * As))[j]
        cn = A.dot(As)                  # cn[j] = |N(j) ∩ N(s)|  (also: #common nbrs of j and s)
        aa = A.dot(invlog * As)
        ra = A.dot(invdeg * As)
        cnx = A.dot(x * As)
        cntr = A.dot(truss * As)
        cnema = A.dot(ema * As)
        cndel = A.dot(delta * As)
        cnstd = A.dot(std * As)
        cnmx = A.dot(mxch * As)
        cndpos = A.dot(dpos * As)
        cndneg = A.dot(dneg * As)
        # 2-hop set of s: nodes k with a common neighbour with s, excluding N(s) and s itself
        n2_bool = (cn > 0) & (~Ns_bool)
        n2_bool[s] = False
        n2_f = n2_bool.astype(np.float32)
        cn2 = A.dot(n2_f)
        aa2 = A.dot(invlog * n2_f)
        cn2x = A.dot(x * n2_f)
        cn2ema = A.dot(ema * n2_f)
        cn2del = A.dot(delta * n2_f)
        cn2dpos = A.dot(dpos * n2_f)
        # constants for this source
        x_s = float(x[s]); ema_s = float(ema[s]); del_s = float(delta[s])
        max_x_Ns = float(x[Ns_bool].max()) if Ns_bool.any() else 0.0
        for r in rows:
            d = int(dsts[r])
            if not (0 <= d < num_nodes):
                continue
            cn_d = cn[d]; cn2_d = cn2[d]
            deg_d = float(deg[d])
            union = deg_s + deg_d - cn_d
            jac = (cn_d / union) if union > 0 else 0.0
            x_d = float(x[d])
            out[r] = (
                # static (7)
                f1p(cn_d), aa[d], ra[d], jac, f1p(deg_s * deg_d), f1p(cn2_d), aa2[d],
                # cohesion (10)
                f1p(cnx[d]), (max_x_Ns if cn_d > 0 else 0.0), (cnx[d] / cn_d) if cn_d > 0 else 0.0,
                f1p(cntr[d]), f1p(cn2x[d]), (cn2x[d] / cn2_d) if cn2_d > 0 else 0.0,
                x_s, x_d, min(x_s, x_d), abs(x_s - x_d),
                # dynamic (13)
                cnema[d], cndel[d], f1p(cndpos[d]), f1p(cndneg[d]), cnstd[d], cnmx[d],
                cn2ema[d], cn2del[d], f1p(cn2dpos[d]),
                ema_s, del_s, float(ema[d]), float(delta[d]),
            )
        i = j
    return out

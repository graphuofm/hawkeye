"""ctypes bridge to the C++ k-core / k-truss kernels (libgevnative.so).

k-core and k-truss are exact, deterministic quantities; the C++ kernels just
compute them fast. If the shared lib is missing, ``available()`` returns False
and callers fall back to the pure-Python indicators.
"""
from __future__ import annotations

import ctypes
import os

import numpy as np

_lib = None
_AVAILABLE: bool | None = None
_SO = os.path.join(os.path.dirname(__file__), "libgevnative.so")


def _load():
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(_SO)
        ip = ctypes.POINTER(ctypes.c_int)
        for fn in ("kcore_csr", "truss_csr"):
            f = getattr(lib, fn)
            f.restype = None
            f.argtypes = [ctypes.c_int, ip, ip, ip]
        _lib = lib
    return _lib


def available() -> bool:
    """True iff the compiled kernel can be loaded."""
    global _AVAILABLE
    if _AVAILABLE is None:
        try:
            _load()
            _AVAILABLE = True
        except OSError:
            _AVAILABLE = False
    return _AVAILABLE


def _run(fn_name: str, graph) -> np.ndarray:
    from gev.features.sparse_pairwise import build_csr

    adj = graph.adj
    n = max(int(graph.num_nodes), (max(adj) + 1) if adj else 1)
    csr = build_csr(graph, n)
    indptr = np.ascontiguousarray(csr.indptr, dtype=np.int32)
    indices = np.ascontiguousarray(csr.indices, dtype=np.int32)
    out = np.zeros(n, dtype=np.int32)
    ip = ctypes.POINTER(ctypes.c_int)
    getattr(_load(), fn_name)(
        ctypes.c_int(n),
        indptr.ctypes.data_as(ip),
        indices.ctypes.data_as(ip),
        out.ctypes.data_as(ip),
    )
    return out


def kcore(graph) -> np.ndarray:
    """Core number per node id (index 0..n-1)."""
    return _run("kcore_csr", graph)


def truss(graph) -> np.ndarray:
    """Node trussness per node id (max edge-trussness over incident edges)."""
    return _run("truss_csr", graph)

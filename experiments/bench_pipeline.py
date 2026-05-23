"""Hawkeye CPU-GPU pipeline benchmark.

Faithful but minimal: cache.slot_features() (CPU) + a Transformer-like
attention block (GPU) + optimizer step. The cache is warmed to a realistic
state, then every mode runs on EXACTLY the same fixed batches, same model
init, same number of simulated epochs -- only the pipeline mechanism differs.

Modes
  1) baseline    : sequential CPU then GPU per batch
  2) pinned      : pinned host memory + non_blocking transfer
  3) prefetcher  : 1 background worker pre-computes the next batch's features
  4) precompute  : one-shot precompute of all features, then GPU-only loop
  5) prefetch_x2 : 2 background workers, deeper lookahead
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, "/home/jding/CIKM2026frp")
from gev.cache import CohesionCache  # noqa: E402

# ---- config ----------------------------------------------------------------
N_EPOCHS_SIM = 3
N_BATCHES = 50
BATCH, K = 200, 32
WARMUP = 60
DEVICE = "cuda"
SEED = 0

# ---- data ------------------------------------------------------------------
DS = sys.argv[1] if len(sys.argv) > 1 else "uci"
df = pd.read_csv(f"/home/jding/CIKM2026frp/sota/DyGLib/processed_data/{DS}/ml_{DS}.csv")
print(f"[dataset] {DS}  edges={len(df)}")
src = df["u"].to_numpy().astype("int64")
dst = df["i"].to_numpy().astype("int64")
ts = df["ts"].to_numpy().astype("float64")
n_total = int(max(src.max(), dst.max())) + 1

# ---- cache warmup ----------------------------------------------------------
cache = CohesionCache(
    indicators=["degree", "core", "truss"],
    trend_decays=(0.99, 0.999, 0.9999),
    stat_groups=("current",),
    pairwise_mode="cohesion",
    window_abs=0.0,
    use_csr=True,
    device=None,
)
cache.reset()
for b in range(WARMUP):
    cache.advance(src[b * BATCH:(b + 1) * BATCH],
                  dst[b * BATCH:(b + 1) * BATCH],
                  ts[b * BATCH:(b + 1) * BATCH])
print(f"[warmup] cache advanced {WARMUP} batches  -> graph m={cache.gev.graph.num_edges}")

# ---- fixed test batches (same across all modes) ---------------------------
rng = np.random.default_rng(SEED)
batches = []
for b in range(N_BATCHES):
    s = src[(WARMUP + b) * BATCH:(WARMUP + b + 1) * BATCH]
    d = dst[(WARMUP + b) * BATCH:(WARMUP + b + 1) * BATCH]
    if len(s) == 0:
        break
    hist_s = rng.integers(0, n_total, size=(BATCH, K))
    hist_d = rng.integers(0, n_total, size=(BATCH, K))
    batches.append((hist_s, d, hist_d, s))
print(f"[setup] {len(batches)} fixed batches, BATCH={BATCH}, K={K}")

# ---- model proxy (mimics the attention work after the structure channel) --
class ModelProxy(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(23, 50), nn.ReLU(), nn.Linear(50, 50))
        self.attn = nn.MultiheadAttention(50, 2, batch_first=True)
        self.head = nn.Linear(50, 1)


def fresh_model() -> tuple[ModelProxy, torch.optim.Optimizer]:
    torch.manual_seed(SEED)
    m = ModelProxy().to(DEVICE)
    o = torch.optim.Adam(m.parameters(), lr=1e-4)
    return m, o


def slot(batch):
    hist_s, d, hist_d, s = batch
    fs = cache.slot_features(hist_s, d, device=None)
    fd = cache.slot_features(hist_d, s, device=None)
    return fs, fd


def gpu_step(m: ModelProxy, o, fs: torch.Tensor, fd: torch.Tensor, pinned: bool):
    if pinned and not fs.is_pinned():
        fs = fs.pin_memory()
        fd = fd.pin_memory()
    fs_g = fs.to(DEVICE, non_blocking=pinned)
    fd_g = fd.to(DEVICE, non_blocking=pinned)
    s_emb = m.mlp(fs_g)
    d_emb = m.mlp(fd_g)
    x = torch.cat([s_emb, d_emb], dim=1)
    y, _ = m.attn(x, x, x)
    loss = m.head(y).sum()
    o.zero_grad(set_to_none=True)
    loss.backward()
    o.step()


# ---- mode implementations -------------------------------------------------
def mode_baseline():
    m, o = fresh_model()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_EPOCHS_SIM):
        for b in batches:
            fs, fd = slot(b)
            gpu_step(m, o, fs, fd, pinned=False)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def mode_pinned():
    m, o = fresh_model()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_EPOCHS_SIM):
        for b in batches:
            fs, fd = slot(b)
            gpu_step(m, o, fs, fd, pinned=True)
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def mode_prefetcher():
    m, o = fresh_model()
    ex = ThreadPoolExecutor(max_workers=1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_EPOCHS_SIM):
        fut = ex.submit(slot, batches[0])
        for i, b in enumerate(batches):
            fs, fd = fut.result()
            if i + 1 < len(batches):
                fut = ex.submit(slot, batches[i + 1])
            gpu_step(m, o, fs, fd, pinned=True)
    torch.cuda.synchronize()
    ex.shutdown(wait=True)
    return time.perf_counter() - t0


def mode_precompute():
    m, o = fresh_model()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    precomp = [slot(b) for b in batches]
    t_pre = time.perf_counter() - t0
    for _ in range(N_EPOCHS_SIM):
        for fs, fd in precomp:
            gpu_step(m, o, fs, fd, pinned=True)
    torch.cuda.synchronize()
    total = time.perf_counter() - t0
    return total, t_pre


def mode_prefetch_x2():
    m, o = fresh_model()
    ex = ThreadPoolExecutor(max_workers=2)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(N_EPOCHS_SIM):
        from collections import deque
        q = deque()
        for j in range(min(2, len(batches))):
            q.append(ex.submit(slot, batches[j]))
        for i, b in enumerate(batches):
            fs, fd = q.popleft().result()
            if i + 2 < len(batches):
                q.append(ex.submit(slot, batches[i + 2]))
            gpu_step(m, o, fs, fd, pinned=True)
    torch.cuda.synchronize()
    ex.shutdown(wait=True)
    return time.perf_counter() - t0


# ---- run -----------------------------------------------------------------
print("\n[warmup pass]")
_ = mode_baseline()  # JIT / cache populate / cudnn benchmark

results = {}
print("\n[running]")
for name, fn in [
    ("baseline", mode_baseline),
    ("pinned", mode_pinned),
    ("prefetcher", mode_prefetcher),
    ("prefetch_x2", mode_prefetch_x2),
]:
    t = fn()
    results[name] = t
    print(f"  {name:<14s} {t:>7.2f} s")

t_total, t_pre = mode_precompute()
results["precompute"] = t_total
print(f"  precompute     {t_total:>7.2f} s   (phase1 precompute = {t_pre:.2f} s)")

# ---- table ---------------------------------------------------------------
base = results["baseline"]
print(f"\n=== {DS} ; {N_EPOCHS_SIM} sim-epochs x {len(batches)} batches ===")
print(f"{'Mode':<14s} {'total (s)':>10s} {'speedup':>10s}")
print("-" * 38)
for k, v in results.items():
    print(f"{k:<14s} {v:>10.2f} {base / v:>9.2f}x")

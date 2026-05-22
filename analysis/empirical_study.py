#!/usr/bin/env python
"""Empirical study (measurement only, no model training).

Question: at the moment *just before* a true edge (u, v) appears, are the
k-family cohesiveness indicators — and especially their *temporal change* — of
v / of the (u, v) pair distinguishable from those of the negative destinations?

We stream the dataset chronologically, maintain the structure + rolling stats,
and for each (sub-sampled) val edge with its TGB negatives we record, per
feature:
  * the feature value for the true destination / true pair
  * the feature values for the negative destinations / pairs
Then per feature we report:
  * AUC   = P(feat(pos) > feat(neg))   — averaged over the negatives of each query
  * |d|   = |Cohen's d| between pos and neg populations
  * MRR1  = MRR if you rank candidates *by this single feature alone*
Sorted, this is a "discriminability ranking" of the structural features —
including the static value vs. its ema / delta / variance, so it directly
answers "does the temporal change carry signal beyond the static value?".

Usage:
  python analysis/empirical_study.py --dataset tgbl-uci --indicators degree,core,triangle \
      --val_subsample 2000 --out results/empirical/tgbl-uci.json
"""
from __future__ import annotations

import argparse, json, os, sys, time
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

from gev import GEVConfig, GraphEagleVision  # noqa: E402
from gev.data import load_tgb_linkproppred  # noqa: E402
from gev.features import PAIRWISE_FEATURE_NAMES  # noqa: E402
from gev.utils import set_seed  # noqa: E402


def auc_and_d(pos: np.ndarray, neg_flat: np.ndarray, neg_counts: np.ndarray):
    """pos: [Q] feature value of the true candidate per query.
       neg_flat: concatenation of all negatives' feature values.
       neg_counts: [Q] number of negatives for each query.
    Returns (auc, abs_cohen_d, mrr_by_feature)."""
    Q = len(pos)
    # per-query: fraction of that query's negatives with neg < pos  (+ 0.5 ties)
    aucs = np.empty(Q, dtype=np.float64)
    rr = np.empty(Q, dtype=np.float64)
    off = 0
    for q in range(Q):
        c = int(neg_counts[q])
        segn = neg_flat[off:off + c]; off += c
        p = pos[q]
        lt = np.count_nonzero(segn < p)
        eq = np.count_nonzero(segn == p)
        aucs[q] = (lt + 0.5 * eq) / max(c, 1)
        # rank of pos when sorting candidates by this feature descending
        gt = np.count_nonzero(segn > p)
        rr[q] = 1.0 / (gt + eq * 0.5 + 1.0)
    auc = float(np.mean(aucs))
    # cohen's d (treat all negatives pooled vs all positives)
    mu_p, mu_n = float(np.mean(pos)), float(np.mean(neg_flat))
    sp, sn = float(np.std(pos)), float(np.std(neg_flat))
    n_p, n_n = len(pos), len(neg_flat)
    pooled = np.sqrt(((n_p - 1) * sp * sp + (n_n - 1) * sn * sn) / max(n_p + n_n - 2, 1))
    d = abs(mu_p - mu_n) / pooled if pooled > 0 else 0.0
    return auc, float(d), float(np.mean(rr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="tgbl-uci")
    ap.add_argument("--indicators", default="degree,core,triangle")
    ap.add_argument("--stats_decay", type=float, default=0.95)
    ap.add_argument("--trend_decays", default="", help="comma list of extra (slower) decay rates "
                    "for multi-timescale trend features, e.g. '0.99,0.999,0.9999'")
    ap.add_argument("--feature_clip", type=float, default=0.0,
                    help="keep raw feature scales for interpretability (0 = off)")
    ap.add_argument("--pairwise_mode", default="all", choices=["all", "cohesion", "generic", "none"])
    ap.add_argument("--pairwise_max_2hop", type=int, default=20000)
    ap.add_argument("--pairwise_backend", default="auto", choices=["auto", "loop", "sparse"])
    ap.add_argument("--truss_recompute_every", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=200)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--val_subsample", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no_download", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    data = load_tgb_linkproppred(args.dataset, download=not args.no_download)
    src, dst, t = data.src.astype(np.int64), data.dst.astype(np.int64), data.t
    E, N = data.num_edges, data.num_nodes
    split_mask = data.val_mask if args.split == "val" else data.test_mask
    split_idx = np.where(split_mask)[0]
    if args.val_subsample and args.val_subsample > 0 and len(split_idx) > args.val_subsample:
        stride = len(split_idx) / args.val_subsample
        split_idx = split_idx[(np.arange(args.val_subsample) * stride).astype(np.int64)]
    eval_set = set(int(i) for i in split_idx)
    avg_deg = 2.0 * E / max(N, 1)
    use_csr = (args.pairwise_backend == "sparse") or (args.pairwise_backend == "auto" and avg_deg >= 50 and args.pairwise_mode != "none")
    print(f"[data] {args.dataset} E={E} N={N} eval-{args.split}={len(eval_set)} avg_deg≈{avg_deg:.1f} "
          f"pairwise_backend={'sparse' if use_csr else 'loop'}", flush=True)

    trend_decays = [float(s) for s in args.trend_decays.split(",") if s.strip()]
    cfg = GEVConfig(indicators=[s.strip() for s in args.indicators.split(",") if s.strip()],
                    stats_decay=args.stats_decay, trend_decays=trend_decays, feature_clip=args.feature_clip,
                    pairwise_mode=args.pairwise_mode, pairwise_max_2hop=args.pairwise_max_2hop,
                    truss_recompute_every=args.truss_recompute_every)
    model = GraphEagleVision(cfg)
    K = model.K
    ind_names = model.indicator_names
    # node-feature names follow the RollingStats layout: group-major, indicator-minor
    node_feat_names = [f"{ind_names[k]}.{g}" for g in model.stats.group_names for k in range(K)]
    pw_names = list(model.pairwise_feature_names)
    print(f"[model] indicators={ind_names} stat_groups={model.stats.group_names} "
          f"node_feats={len(node_feat_names)} pairwise={len(pw_names)} ({args.pairwise_mode})", flush=True)

    if args.split == "val":
        data.load_val_ns()
    else:
        data.load_test_ns()
    ns = data.negative_sampler

    # accumulators
    node_pos: List[np.ndarray] = []      # rows: [Q', 5K] feature of true dst
    node_neg: List[np.ndarray] = []      # rows: [sum c, 5K]
    pw_pos: List[np.ndarray] = []
    pw_neg: List[np.ndarray] = []
    neg_counts: List[int] = []

    bounds = list(range(0, E, args.batch_size)) + [E]
    t0 = time.time()
    for bi in range(len(bounds) - 1):
        lo, hi = bounds[bi], bounds[bi + 1]
        bsl, bdl, btl = src[lo:hi], dst[lo:hi], t[lo:hi]
        # eval the split edges in this batch (state = before batch)
        local = [j for j in range(hi - lo) if (lo + j) in eval_set and split_mask[lo + j]]
        if local:
            us = bsl[local].astype(np.int64); vs = bdl[local].astype(np.int64); ts = btl[local].astype(np.float64)
            neg_lists = ns.query_batch(torch.as_tensor(us), torch.as_tensor(vs),
                                       torch.as_tensor(ts), split_mode=args.split)
            csr = model.build_pairwise_csr() if (use_csr and model.pairwise_dim) else None
            # node features (full 5K) for the dst side
            node_pos.append(model.stats.get_batch_features(vs).copy())
            pw_pos.append(model.pairwise_features(us, vs, csr=csr))
            # negatives
            neg_dst_all, neg_src_all = [], []
            for kk, negs in enumerate(neg_lists):
                negs = np.asarray(negs, dtype=np.int64)
                neg_dst_all.append(negs)
                neg_src_all.append(np.full(len(negs), us[kk], dtype=np.int64))
                neg_counts.append(len(negs))
            neg_dst_all = np.concatenate(neg_dst_all); neg_src_all = np.concatenate(neg_src_all)
            node_neg.append(model.stats.get_batch_features(neg_dst_all).copy())
            pw_neg.append(model.pairwise_features(neg_src_all, neg_dst_all, csr=csr))
        # advance structure
        model.update_structure_batch(bsl, bdl, btl)
    elapsed = time.time() - t0
    print(f"[stream] done in {elapsed:.1f}s; collected {sum(len(x) for x in node_pos)} positives", flush=True)

    node_pos = np.concatenate(node_pos, 0) if node_pos else np.zeros((0, 5 * K))
    node_neg = np.concatenate(node_neg, 0) if node_neg else np.zeros((0, 5 * K))
    pw_pos = np.concatenate(pw_pos, 0) if pw_pos else np.zeros((0, len(pw_names)))
    pw_neg = np.concatenate(pw_neg, 0) if pw_neg else np.zeros((0, len(pw_names)))
    neg_counts = np.asarray(neg_counts, dtype=np.int64)
    # consistency: number of queries
    Q = len(node_pos)
    assert len(neg_counts) == Q == len(pw_pos), (Q, len(neg_counts), len(pw_pos))

    rows = []
    for fam, names, P, Ng in (("node", node_feat_names, node_pos, node_neg),
                              ("pair", pw_names, pw_pos, pw_neg)):
        if P.shape[1] == 0:
            continue
        for ci, nm in enumerate(names):
            auc, d, mrr1 = auc_and_d(P[:, ci], Ng[:, ci], neg_counts)
            # "directionless" discriminability: AUC could be < 0.5 if the feature is anti-correlated
            disc_auc = max(auc, 1.0 - auc)
            rows.append({"family": fam, "feature": nm, "auc": auc, "disc_auc": disc_auc,
                         "abs_cohen_d": d, "mrr_single_feature": mrr1})
    rows.sort(key=lambda r: -r["disc_auc"])

    print("\n  rank  family  feature                     AUC    discAUC  |d|     MRR(1feat)")
    print("  " + "-" * 78)
    for i, r in enumerate(rows[:30]):
        print(f"  {i+1:3d}.  {r['family']:5s}  {r['feature']:26s}  {r['auc']:.3f}  {r['disc_auc']:.3f}   "
              f"{r['abs_cohen_d']:.3f}  {r['mrr_single_feature']:.4f}")

    # summary by indicator and by static/dynamic
    by_ind = defaultdict(list); by_kind = defaultdict(list)
    for r in rows:
        if r["family"] == "node":
            ind, grp = r["feature"].split(".", 1)   # group names may contain '.' (e.g. trend_0.99)
            by_ind[ind].append(r["disc_auc"])
            by_kind["static" if grp == "current" else "dynamic"].append(r["disc_auc"])
    print("\n  node-feature discriminability (mean discAUC):")
    for ind, v in sorted(by_ind.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"    {ind:12s}  mean={np.mean(v):.3f}  max={np.max(v):.3f}")
    for kind, v in by_kind.items():
        print(f"    [{kind}]  mean={np.mean(v):.3f}  max={np.max(v):.3f}  (n={len(v)})")

    result = {"dataset": args.dataset, "indicators": ind_names, "split": args.split,
              "pairwise_mode": args.pairwise_mode, "num_queries": int(Q),
              "stream_sec": elapsed, "features": rows,
              "by_indicator_mean_discAUC": {k: float(np.mean(v)) for k, v in by_ind.items()},
              "static_vs_dynamic_mean_discAUC": {k: float(np.mean(v)) for k, v in by_kind.items()},
              "args": vars(args)}
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()

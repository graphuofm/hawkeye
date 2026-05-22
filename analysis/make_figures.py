#!/usr/bin/env python
"""Generate the paper figures from results/. Robust to missing files (skips).

  python analysis/make_figures.py   ->  results/figures/*.png  (+ a brief stdout summary)
"""
from __future__ import annotations

import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)


def jl(name):
    p = os.path.join(RES, name)
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.tight_layout(); fig.savefig(p, dpi=140); plt.close(fig)
    print(f"  [fig] {p}")


# --------------------------------------------------------------------------- #
def fig_hierarchy():
    rows = jl("sweep_hierarchy.jsonl") + jl("struct_matrix.jsonl")
    # keep single-indicator runs on tgbl-uci (the sweep dataset)
    order = ["degree", "core", "triangle", "truss"]
    by_pm = defaultdict(dict)   # pairwise_mode -> {indicator: best_val/test}
    for r in rows:
        ds = r.get("dataset"); ind = r.get("indicators", "")
        pm = r.get("pairwise_mode", r.get("pairwise", "?"))
        if ds != "tgbl-uci" or ind not in order:
            continue
        cur = by_pm[str(pm)].get(ind)
        if cur is None or r["val_mrr"] > cur[0]:
            by_pm[str(pm)][ind] = (r["val_mrr"], r["test_mrr"])
    if not by_pm:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, mi in ((axes[0], "val MRR", 0), (axes[1], "test MRR", 1)):
        for pm in ("cohesion", "all", "generic"):
            if pm not in by_pm:
                continue
            xs = [i for i in order if i in by_pm[pm]]
            ys = [by_pm[pm][i][mi] for i in xs]
            ax.plot(range(len(xs)), ys, "o-", label=f"pairwise={pm}")
            ax.set_xticks(range(len(xs))); ax.set_xticklabels(xs)
        ax.set_title(f"tgbl-uci struct-only — {metric}\n(constraint relax→tighten)")
        ax.set_ylabel(metric); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    save(fig, "fig_hierarchy.png")


def fig_discauc_heatmap():
    paths = sorted(glob.glob(os.path.join(RES, "empirical", "*.json")))
    if not paths:
        return
    data = {}
    for p in paths:
        r = json.load(open(p)); data[r["dataset"]] = {f["feature"]: f["disc_auc"] for f in r["features"]}
    datasets = list(data)
    # curated feature subset (in a sensible order)
    feats = ["cn2", "aa2", "cn2_x_sum", "cn2_x_delta_sum", "cn2_x_ema_sum",
             "degree.current", "degree.ema", "core.current", "core.ema",
             "core.trend_0.99", "core.trend_0.999", "core.trend_0.9999", "core.recency",
             "core.delta", "core.std", "core.max_change",
             "cn", "aa", "jaccard", "cn_x_sum", "triangle.current"]
    feats = [f for f in feats if any(f in data[d] for d in datasets)]
    M = np.full((len(feats), len(datasets)), np.nan)
    for j, d in enumerate(datasets):
        for i, f in enumerate(feats):
            if f in data[d]:
                M[i, j] = data[d][f]
    fig, ax = plt.subplots(figsize=(1.4 * len(datasets) + 3, 0.35 * len(feats) + 1.5))
    im = ax.imshow(M, aspect="auto", cmap="RdYlGn", vmin=0.45, vmax=1.0)
    ax.set_xticks(range(len(datasets))); ax.set_xticklabels(datasets, rotation=30, ha="right")
    ax.set_yticks(range(len(feats))); ax.set_yticklabels(feats, fontsize=8)
    for i in range(len(feats)):
        for j in range(len(datasets)):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="discAUC = max(AUC, 1-AUC)")
    ax.set_title("Single-feature discriminability (true dst vs negatives)")
    save(fig, "fig_discauc_heatmap.png")


def _bar(ax, labels, vals, title, color="tab:blue", rot=30):
    x = np.arange(len(labels))
    ax.bar(x, vals, color=color)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=rot, ha="right", fontsize=8)
    ax.set_title(title, fontsize=9); ax.grid(alpha=0.3, axis="y")
    for xi, v in zip(x, vals):
        ax.text(xi, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7)


def fig_coupling():
    rows = jl("coupling_matrix.jsonl")
    if not rows:
        return
    by_ds = defaultdict(dict)
    for r in rows:
        m = r.get("model", "?")
        d = by_ds[r["dataset"]]
        if m not in d or r["test_mrr"] > d[m]["test_mrr"]:
            d[m] = r
    ds_list = list(by_ds)
    fig, axes = plt.subplots(1, len(ds_list), figsize=(6.5 * len(ds_list), 4.2), squeeze=False)
    for ax, ds in zip(axes[0], ds_list):
        items = sorted(by_ds[ds].items(), key=lambda kv: -kv[1]["test_mrr"])
        labels = [m.replace("TGN+GEV(degree,core,", "").replace(",pw=all)", "").replace("(degree,core, pairwise=all)", "")[:22] for m, _ in items]
        vals = [r["test_mrr"] for _, r in items]
        _bar(ax, labels, vals, f"{ds} — test MRR by coupling", color="tab:green")
    save(fig, "fig_coupling.png")


def fig_window():
    rows = jl("window_matrix.jsonl")
    if not rows:
        return
    # group by (dataset, pairwise_mode), bars over stat_groups
    by_key = defaultdict(dict)
    for r in rows:
        by_key[(r["dataset"], r.get("pairwise_mode", "?"))][r.get("stat_groups", "?")] = r
    keys = sorted(by_key)
    n = len(keys)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.0), squeeze=False)
    sg_order = ["current", "current,ema", "current,trend_0.99", "current,trend_0.9999",
                "current,recency", "ema,std,delta,max_change", "all"]
    for ax, key in zip(axes[0], keys):
        d = by_key[key]
        labels = [sg for sg in sg_order if sg in d] + [sg for sg in d if sg not in sg_order]
        vals = [d[sg]["test_mrr"] for sg in labels]
        _bar(ax, [l.replace(",", "+\n") for l in labels], vals, f"{key[0]} pw={key[1]} — test MRR", color="tab:purple")
    save(fig, "fig_window.png")


def fig_enhance():
    rows = jl("tgn_matrix.jsonl") + jl("tgn_baseline.jsonl")
    if not rows:
        return
    by_ds = defaultdict(dict)
    for r in rows:
        m = r.get("model", "?")
        d = by_ds[r["dataset"]]
        if m not in d or r["test_mrr"] > d[m]["test_mrr"]:
            d[m] = r
    ds_list = [d for d in by_ds if len(by_ds[d]) > 1]
    if not ds_list:
        return
    fig, axes = plt.subplots(1, len(ds_list), figsize=(5.5 * len(ds_list), 4.0), squeeze=False)
    for ax, ds in zip(axes[0], ds_list):
        items = sorted(by_ds[ds].items(), key=lambda kv: kv[1]["test_mrr"])
        labels = [m.replace("TGN+GEV", "+GEV").replace("(degree,core,", "(") for m, _ in items]
        vals = [r["test_mrr"] for _, r in items]
        _bar(ax, labels, vals, f"{ds} — TGN vs TGN+GEV (test MRR)", color="tab:orange")
    save(fig, "fig_enhance.png")


if __name__ == "__main__":
    for fn in (fig_hierarchy, fig_discauc_heatmap, fig_coupling, fig_window, fig_enhance):
        try:
            fn()
        except Exception as e:
            print(f"  [skip {fn.__name__}] {type(e).__name__}: {e}")
    print(f"figures under {FIG}/")

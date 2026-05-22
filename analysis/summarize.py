#!/usr/bin/env python
"""Summarise experiment results into console tables + CSVs (+ a couple figures).

Reads (whatever exists):
  results/empirical/*.json              -> Table 2 (per-feature discriminability across datasets)
  results/sweep_hierarchy.jsonl         -> hierarchy table (struct-only, degree->truss x pairwise_mode)
  results/struct_matrix.jsonl           -> same, the run_matrix.sh struct sweep
  results/tgn_matrix.jsonl, tgn_baseline.jsonl -> TGN vs TGN+GEV (the "enhance" table)

Usage:  python analysis/summarize.py
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "summary")
os.makedirs(OUT, exist_ok=True)


def _load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _w(path, lines):
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  [wrote] {path}")


# --------------------------------------------------------------------------- #
def empirical_table():
    paths = sorted(glob.glob(os.path.join(RES, "empirical", "*.json")))
    if not paths:
        print("(no empirical results yet)")
        return
    data = {}
    for p in paths:
        r = json.load(open(p))
        ds = r["dataset"]
        data[ds] = {f["feature"]: f for f in r["features"]}
    datasets = list(data)
    # union of features, keep a curated order if present
    feats = []
    seen = set()
    for ds in datasets:
        for f in data[ds]:
            if f not in seen:
                seen.add(f); feats.append(f)
    print("\n=== Empirical study: discAUC (max(AUC,1-AUC)) per feature x dataset ===")
    hdr = f"{'feature':24s} " + " ".join(f"{ds:>16s}" for ds in datasets)
    print(hdr); print("-" * len(hdr))
    rows_csv = ["feature," + ",".join(f"{ds}_discAUC,{ds}_MRR1" for ds in datasets)]
    # sort features by mean discAUC desc
    def meandisc(f):
        v = [data[ds][f]["disc_auc"] for ds in datasets if f in data[ds]]
        return -np.mean(v) if v else 0.0
    for f in sorted(feats, key=meandisc):
        cells = []
        csvc = [f]
        for ds in datasets:
            if f in data[ds]:
                d = data[ds][f]; cells.append(f"{d['disc_auc']:.3f}/{d['mrr_single_feature']:.3f}")
                csvc += [f"{d['disc_auc']:.4f}", f"{d['mrr_single_feature']:.4f}"]
            else:
                cells.append(f"{'--':>16s}"); csvc += ["", ""]
        print(f"{f:24s} " + " ".join(f"{c:>16s}" for c in cells))
        rows_csv.append(",".join(csvc))
    _w(os.path.join(OUT, "empirical_discAUC.csv"), rows_csv)
    # by-indicator and static-vs-dynamic summaries
    print("\n  by-indicator / static-vs-dynamic mean discAUC:")
    for p in paths:
        r = json.load(open(p))
        print(f"    {r['dataset']:16s}  byInd={ {k:round(v,3) for k,v in r['by_indicator_mean_discAUC'].items()} }  "
              f"static/dyn={ {k:round(v,3) for k,v in r['static_vs_dynamic_mean_discAUC'].items()} }")


# --------------------------------------------------------------------------- #
def struct_table():
    rows = _load_jsonl(os.path.join(RES, "sweep_hierarchy.jsonl")) + _load_jsonl(os.path.join(RES, "struct_matrix.jsonl"))
    if not rows:
        print("\n(no struct-only results yet)")
        return
    print("\n=== Struct-only: val/test MRR by dataset x indicators x pairwise_mode x stat_groups ===")
    rows_csv = ["dataset,indicators,pairwise_mode,stat_groups,val_mrr,test_mrr,total_min,best_epoch"]
    # group by dataset
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r.get("dataset", "?")].append(r)
    for ds, rr in sorted(by_ds.items()):
        print(f"\n  -- {ds} --")
        rr.sort(key=lambda r: (r.get("indicators", ""), r.get("pairwise_mode", ""), r.get("stat_groups", "")))
        for r in rr:
            pm = r.get("pairwise_mode", r.get("pairwise", "?"))
            sg = r.get("stat_groups", "?")
            print(f"     {r.get('indicators',''):22s} pw={str(pm):9s} stat={str(sg)[:22]:22s} "
                  f"val={r['val_mrr']:.4f} test={r['test_mrr']:.4f}  ({r.get('total_min',0):.0f}m ep{r.get('best_epoch',0)})")
            rows_csv.append(f"{ds},{r.get('indicators','')},{pm},{sg},{r['val_mrr']:.4f},{r['test_mrr']:.4f},{r.get('total_min',0):.1f},{r.get('best_epoch',0)}")
    _w(os.path.join(OUT, "struct_results.csv"), rows_csv)


# --------------------------------------------------------------------------- #
def tgn_table():
    rows = (_load_jsonl(os.path.join(RES, "tgn_matrix.jsonl")) + _load_jsonl(os.path.join(RES, "tgn_baseline.jsonl"))
            + _load_jsonl(os.path.join(RES, "coupling_matrix.jsonl")))
    if not rows:
        print("\n(no TGN results yet)")
        return
    print("\n=== TGN baseline vs TGN+GEV / couplings (the 'enhance' experiment) ===")
    rows_csv = ["dataset,model,coupling,fusion,pairwise_mode,val_mrr,test_mrr,total_min"]
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r.get("dataset", "?")].append(r)
    for ds, rr in sorted(by_ds.items()):
        print(f"\n  -- {ds} --")
        # de-dup on model name, keep the best test
        seen = {}
        for r in rr:
            m = r.get("model", r.get("fusion", "?"))
            if m not in seen or r["test_mrr"] > seen[m]["test_mrr"]:
                seen[m] = r
        for m, r in sorted(seen.items(), key=lambda kv: -kv[1]["test_mrr"]):
            print(f"     {m:42s} val={r['val_mrr']:.4f} test={r['test_mrr']:.4f}  ({r.get('total_min',0):.0f}m)")
            rows_csv.append(f"{ds},{m},{r.get('coupling','')},{r.get('fusion','')},"
                            f"{r.get('pairwise_mode','')},{r['val_mrr']:.4f},{r['test_mrr']:.4f},{r.get('total_min',0):.1f}")
    _w(os.path.join(OUT, "tgn_results.csv"), rows_csv)


# --------------------------------------------------------------------------- #
def window_table():
    rows = _load_jsonl(os.path.join(RES, "window_matrix.jsonl"))
    if not rows:
        print("\n(no window-ablation results yet)")
        return
    print("\n=== RQ2 window-ablation: struct-only val/test MRR by stat_groups ===")
    rows_csv = ["dataset,pairwise_mode,stat_groups,val_mrr,test_mrr"]
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r.get("dataset", "?")].append(r)
    for ds, rr in sorted(by_ds.items()):
        print(f"\n  -- {ds} --")
        for r in sorted(rr, key=lambda r: (r.get("pairwise_mode", ""), r.get("stat_groups", ""))):
            print(f"     pw={str(r.get('pairwise_mode','?')):9s} stat={str(r.get('stat_groups','?'))[:32]:32s} "
                  f"val={r['val_mrr']:.4f} test={r['test_mrr']:.4f}")
            rows_csv.append(f"{ds},{r.get('pairwise_mode','')},{r.get('stat_groups','')},{r['val_mrr']:.4f},{r['test_mrr']:.4f}")
    _w(os.path.join(OUT, "window_results.csv"), rows_csv)


if __name__ == "__main__":
    empirical_table()
    struct_table()
    window_table()
    tgn_table()
    print(f"\nCSVs under {OUT}/")

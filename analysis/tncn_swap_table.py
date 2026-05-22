#!/usr/bin/env python
"""Reframe existing struct_matrix results as a TNCN-style swap-in table.

The struct-only experiments with --pairwise_mode {none, generic, cohesion, all}
are functionally a swap-in ablation against TNCN's structure channel:
    pw=none      ~ no structure channel (popularity prior only)
    pw=generic   ~ TNCN baseline (CN/AA/RA/Jaccard)
    pw=cohesion  ~ TNCN + cohesion-aware swap (our channel)
    pw=all       ~ TNCN + cohesion (both, upper bound)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict


def load_rows(path: str):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def main():
    paths = [
        "results/struct_matrix.jsonl",
        "results/sweep_hierarchy.jsonl",
        "results/window_matrix.jsonl",
    ]
    rows = []
    for p in paths:
        try:
            rows.extend(load_rows(p))
        except FileNotFoundError:
            continue

    # bucket by (dataset, indicators, pairwise_mode) and take latest / highest val
    by_key = defaultdict(list)
    for r in rows:
        a = r.get("args", {}) or {}
        ds = r.get("dataset", "?")
        inds = a.get("indicators", "")
        if isinstance(inds, list):
            inds = ",".join(inds)
        pwm = a.get("pairwise_mode") or r.get("pairwise_mode", "?")
        wf = float(a.get("window_fraction", 0) or 0)
        if wf != 0:
            continue  # exclude window experiments (separate analysis)
        test = r.get("test_mrr")
        val = r.get("val_mrr")
        if test is None:
            continue
        by_key[(ds, inds, pwm)].append({"test": test, "val": val})

    # pick best per key (by val_mrr to match the actual selection criterion)
    best = {}
    for k, lst in by_key.items():
        lst_sorted = sorted(lst, key=lambda d: (d.get("val") or -1, d.get("test") or -1),
                            reverse=True)
        best[k] = lst_sorted[0]

    # focus on indicator set "degree,core" (the canonical pair) + "core" alone
    target_inds = ["degree,core", "core"]
    target_pws = ["none", "generic", "cohesion", "all"]
    datasets = sorted({k[0] for k in best})

    print("=" * 80)
    print("TNCN swap-in reframe: pairwise_mode acts as the structure-channel choice")
    print("=" * 80)
    for inds in target_inds:
        present = any(k[1] == inds for k in best)
        if not present:
            continue
        print(f"\n--- indicators={inds} ---")
        header = f"{'dataset':18s} | " + " | ".join(f"{p:>10s}" for p in target_pws)
        print(header)
        print("-" * len(header))
        for ds in datasets:
            row = [ds]
            any_present = False
            for pwm in target_pws:
                k = (ds, inds, pwm)
                if k in best:
                    row.append(f"{best[k]['test']:.4f}")
                    any_present = True
                else:
                    row.append(" " * 4 + "—   ")
            if any_present:
                print(f"{row[0]:18s} | " + " | ".join(f"{c:>10s}" for c in row[1:]))

    # also dump the row count we have per (dataset, indicators, pwm) for transparency
    print("\n--- coverage (n runs per cell) ---")
    cov = defaultdict(int)
    for k in by_key:
        cov[(k[0], k[1], k[2])] = len(by_key[k])
    for ds in datasets:
        line = f"{ds:18s} | "
        cells = []
        for inds in target_inds:
            for pwm in target_pws:
                n = cov.get((ds, inds, pwm), 0)
                cells.append(f"{inds[:6]}/{pwm[:3]}={n}")
        print(line + " ".join(cells))


if __name__ == "__main__":
    main()

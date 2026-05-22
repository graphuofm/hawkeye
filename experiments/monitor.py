#!/usr/bin/env python3
"""Hawkeye experiment monitor — a live terminal dashboard.

Run it in a spare terminal:  python experiments/monitor.py
It refreshes every REFRESH seconds and shows, per experiment log:
  * current epoch / total
  * seconds since the log was last written  (staleness)
  * STATUS: RUNNING / DONE / CRASHED / STUCK
  * final metric once DONE
plus a GPU summary and a flagged list of problems.

A job is flagged STUCK if its log has not been written for > STUCK_MIN minutes
and it is neither DONE nor CRASHED — that is the "hang" the k-family blow-up
causes on dense graphs.  Ctrl-C to quit.
"""
from __future__ import annotations

import glob
import os
import re
import subprocess
import time

ROOT = "."
LOG_DIR = os.path.join(ROOT, "results", "logs")
REFRESH = 60          # seconds between refreshes
STUCK_MIN = 20        # minutes of no log write => STUCK
ABANDON_H = 12        # non-DONE logs older than this are abandoned (hidden)

# log-file globs to watch (experiment families)
PATTERNS = ["s24_*.log", "win_*.log", "fix_*.log", "sota_*.log",
            "canparl_*.log", "uslegis_*.log", "officialdgf_*.log"]


def sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True,
                              timeout=15).stdout.strip()
    except Exception:
        return ""


def gpu_summary() -> str:
    out = sh("nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu "
             "--format=csv,noheader,nounits")
    procs = sh("nvidia-smi --query-compute-apps=pid --format=csv,noheader")
    n = len([x for x in procs.splitlines() if x.strip()])
    if not out:
        return "GPU: (n/a)"
    used, total, util = [x.strip() for x in out.splitlines()[0].split(",")]
    return f"GPU: {used}/{total} MiB | util {util}% | {n} compute procs"


def parse_log(path: str) -> dict:
    """Return dict(epoch, total, status, metric, age_s)."""
    name = os.path.basename(path)[:-4]
    try:
        age_s = time.time() - os.path.getmtime(path)
    except OSError:
        return dict(name=name, epoch="-", total="-", status="GONE", metric="", age_s=0)

    # read tail (logs can be huge from tqdm)
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 200_000))
            tail = f.read().decode("utf-8", "ignore")
    except OSError:
        tail = ""

    # total epochs from the config line
    m = re.search(r"num_epochs=(\d+)", tail)
    total = m.group(1) if m else "?"

    # current epoch (last "Epoch: N")
    eps = re.findall(r"Epoch:\s*(\d+)", tail)
    epoch = eps[-1] if eps else "0"

    # crash?
    crashed = bool(re.search(r"Traceback \(most recent call last\)|"
                             r"CUDA error|OutOfMemoryError|"
                             r"AssertionError", tail))

    # done?  TGB prints "average test mrr", DGB prints "average test average_precision"
    done = bool(re.search(r"average test (mrr|average_precision)", tail))

    metric = ""
    if done:
        mt = re.findall(r"average test mrr,\s*([0-9.]+)", tail)
        if mt:
            metric = f"test_mrr={mt[-1]}"
        else:
            ap = re.findall(r"average test average_precision,\s*([0-9.]+)", tail)
            if ap:
                metric = f"test_AP={ap[-1]}"

    if done:
        status = "DONE"
    elif age_s > ABANDON_H * 3600:
        # very old, non-DONE log => abandoned/killed run, not a live problem
        status = "ABANDONED"
    elif crashed:
        status = "CRASHED"
    elif age_s > STUCK_MIN * 60:
        status = "STUCK"
    else:
        status = "RUNNING"
    return dict(name=name, epoch=epoch, total=total, status=status,
                metric=metric, age_s=age_s)


def fmt_age(s: float) -> str:
    s = int(s)
    if s < 90:
        return f"{s}s"
    if s < 5400:
        return f"{s // 60}m"
    return f"{s // 3600}h{(s % 3600) // 60}m"


def main():
    while True:
        rows = []
        seen = set()
        abandoned = 0
        for pat in PATTERNS:
            for path in glob.glob(os.path.join(LOG_DIR, pat)):
                if path in seen:
                    continue
                seen.add(path)
                r = parse_log(path)
                if r["status"] == "ABANDONED":
                    abandoned += 1          # count but don't clutter the board
                    continue
                rows.append(r)
        # sort: problems first, then running, then done
        order = {"STUCK": 0, "CRASHED": 1, "RUNNING": 2, "DONE": 3, "GONE": 4}
        rows.sort(key=lambda r: (order.get(r["status"], 9), r["name"]))

        os.system("clear")
        print("=" * 78)
        print(f"  Hawkeye Experiment Monitor   {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  {gpu_summary()}")
        print("=" * 78)
        print(f"  {'JOB':<34}{'EPOCH':>9}{'LAST':>8}  STATUS")
        print("  " + "-" * 74)
        counts = {"RUNNING": 0, "DONE": 0, "STUCK": 0, "CRASHED": 0, "GONE": 0}
        for r in rows:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
            ep = f"{r['epoch']}/{r['total']}"
            flag = {"STUCK": "  <-- STUCK?", "CRASHED": "  <-- CRASHED"}.get(r["status"], "")
            line = (f"  {r['name']:<34}{ep:>9}{fmt_age(r['age_s']):>8}  "
                    f"{r['status']}{flag}")
            if r["metric"]:
                line += f"  {r['metric']}"
            print(line)
        print("  " + "-" * 74)
        print(f"  RUNNING {counts.get('RUNNING',0)}  |  DONE {counts.get('DONE',0)}  "
              f"|  STUCK {counts.get('STUCK',0)}  |  CRASHED {counts.get('CRASHED',0)}"
              f"  |  ({abandoned} old logs hidden)")
        probs = [r for r in rows if r["status"] in ("STUCK", "CRASHED")]
        if probs:
            print("\n  !!! PROBLEMS — need attention:")
            for r in probs:
                print(f"      {r['name']}  ({r['status']}, last write {fmt_age(r['age_s'])} ago)")
        print(f"\n  refresh every {REFRESH}s — Ctrl-C to quit")
        time.sleep(REFRESH)


def check_once():
    """One-shot: print one line per STUCK/CRASHED job (for an external watcher),
    then a RUNNING/DONE summary line. Exit. Prints nothing problematic if all OK."""
    rows = []
    seen = set()
    for pat in PATTERNS:
        for path in glob.glob(os.path.join(LOG_DIR, pat)):
            if path in seen:
                continue
            seen.add(path)
            rows.append(parse_log(path))
    probs = [r for r in rows if r["status"] in ("STUCK", "CRASHED")]
    for r in probs:
        print(f"PROBLEM {r['status']} {r['name']} (last write {fmt_age(r['age_s'])} ago)",
              flush=True)
    run = sum(1 for r in rows if r["status"] == "RUNNING")
    done = sum(1 for r in rows if r["status"] == "DONE")
    print(f"OK summary: RUNNING={run} DONE={done} PROBLEMS={len(probs)}", flush=True)


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        check_once()
    else:
        try:
            main()
        except KeyboardInterrupt:
            print("\nmonitor stopped.")

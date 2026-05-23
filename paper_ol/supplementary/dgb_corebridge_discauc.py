#!/usr/bin/env python
"""Compute the core-weighted 2-hop bridge (2-hop x c) discAUC on the 5 DGB
datasets, to fill the last `---` column of tab:discauc.

Approach:
  1. Build train-set adjacency.
  2. Run a one-shot O(m) Batagelj-Zaversnik k-core decomposition on
     that adjacency (snapshot core number per node).
  3. For each val query (u, v_true, v_neg), compute
       bridge_c(u, v) = sum of c(w) for w in N(v) intersect N_2(u).
     Compare bridge_c(u, v_true) vs bridge_c(u, v_neg).
  4. discAUC = P(true > neg) + 0.5 * P(true == neg).
"""
import csv, random, json, time
from pathlib import Path
from collections import defaultdict

ROOT = Path("/home/jding/CIKM2026frp/sota/DyGLib/processed_data")
OUT  = Path("/home/jding/CIKM2026frp/results/empirical_dgb")
OUT.mkdir(exist_ok=True)
random.seed(0)

DATASETS = ["CanParl", "USLegis", "mooc", "reddit", "UNvote"]


def load_dgb_edges(name):
    fn = ROOT / name / f"ml_{name}.csv"
    edges = []
    with open(fn) as f:
        r = csv.reader(f); next(r)
        for row in r:
            if len(row) < 4: continue
            try:
                u = int(float(row[1])); v = int(float(row[2])); t = float(row[3])
            except ValueError: continue
            edges.append((t, u, v))
    edges.sort(key=lambda x: x[0])
    return edges


def kcore(adj):
    """Batagelj-Zaversnik O(m) k-core decomposition.

    adj: dict node -> set of neighbour nodes.
    Returns dict node -> core_number.
    """
    deg = {v: len(adj[v]) for v in adj}
    nodes_sorted = sorted(adj.keys(), key=lambda v: deg[v])
    pos = {v: i for i, v in enumerate(nodes_sorted)}
    bin_start = [0]
    md = max(deg.values()) if deg else 0
    cnt = [0] * (md + 2)
    for v in deg: cnt[deg[v]] += 1
    for d in range(1, md + 2): cnt[d] += cnt[d - 1]
    # bins: starting position of each degree in nodes_sorted
    bin_start = [0] * (md + 2)
    s = 0
    for d in range(md + 2):
        c = cnt[d] - (cnt[d - 1] if d > 0 else 0)
        bin_start[d] = s
        s += c
    # standard implementation
    core = {}
    # Re-use simpler version: peel
    work = {v: deg[v] for v in deg}
    remaining = set(adj.keys())
    # bucket sort: process node with smallest deg first
    # simple priority via heap
    import heapq
    heap = [(work[v], v) for v in remaining]
    heapq.heapify(heap)
    while heap:
        d, v = heapq.heappop(heap)
        if v not in remaining: continue
        if d != work[v]:
            continue
        core[v] = d
        remaining.discard(v)
        for u in adj[v]:
            if u in remaining and work[u] > d:
                work[u] -= 1
                heapq.heappush(heap, (work[u], u))
    return core


def run_one(name):
    print(f"=== {name} ===", flush=True)
    t0 = time.time()
    edges = load_dgb_edges(name)
    n_edges = len(edges)
    train_end = int(0.70 * n_edges); val_end = int(0.85 * n_edges)
    train, val = edges[:train_end], edges[train_end:val_end]
    nodes = set()
    for _, u, v in edges: nodes.add(u); nodes.add(v)
    nodes_list = list(nodes)

    adj = defaultdict(set)
    for _, u, v in train:
        adj[u].add(v); adj[v].add(u)
    # ensure every node has an entry (so kcore covers them)
    for n in nodes:
        if n not in adj: adj[n] = set()

    t1 = time.time()
    core = kcore(adj)
    print(f"  built adj + kcore: {time.time()-t1:.1f}s  (max core = {max(core.values()):d})")

    val_sample = val if len(val) <= 3000 else random.sample(val, 3000)

    def bridge_c(u, v, snap):
        Nu = snap.get(u, set())
        Nv = snap.get(v, set())
        N2u = set()
        for w in Nu: N2u |= snap.get(w, set())
        N2u.discard(u)
        return sum(core.get(w, 0) for w in (Nv & N2u))

    snap = adj   # snapshot fixed
    tr, ng = [], []
    for _, u, v_true in val_sample:
        v_neg = random.choice(nodes_list)
        while v_neg == u: v_neg = random.choice(nodes_list)
        tr.append(bridge_c(u, v_true, snap))
        ng.append(bridge_c(u, v_neg, snap))

    wins, ties = 0, 0
    for a, b in zip(tr, ng):
        if a > b: wins += 1
        elif a == b: ties += 1
    auc = (wins + 0.5 * ties) / len(tr)
    elapsed = time.time() - t0
    print(f"  2-hop x c discAUC = {auc:.4f}   ({elapsed:.1f}s)")

    # merge into existing JSON
    p = OUT / f"{name}.json"
    if p.exists():
        d = json.loads(p.read_text())
    else:
        d = {"dataset": name}
    d["discauc_2hop_corew"] = auc
    p.write_text(json.dumps(d, indent=2))
    return auc


print(f"{'dataset':<10s} {'2-hop x c':>10s}")
for ds in DATASETS:
    try:
        auc = run_one(ds)
        print(f"{ds:<10s} {auc:>10.4f}")
    except Exception as e:
        print(f"  FAIL {ds}: {e}")

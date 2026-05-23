#!/usr/bin/env python
"""Quick streaming 1-hop CN + 2-hop bridge discAUC on DGB datasets.

For each DGB dataset, stream the edges chronologically, maintain adjacency
sets, and at every validation/test query compute:
  - CN(u, v_true)  = |N(u) ∩ N(v_true)|       (1-hop common neighbour)
  - bridge2(u, v)  = |N(v) ∩ N2(u)|             (2-hop cohesive bridge)
against a random negative dst.  Report discAUC = P(true > neg).
"""
import csv, random, json, sys, time
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
        r = csv.reader(f)
        header = next(r)
        # DGB CSV columns: ,u,i,ts,label,idx
        # so we want indices: u=1, i=2, ts=3
        for row in r:
            if len(row) < 4: continue
            try:
                u = int(float(row[1])); v = int(float(row[2])); t = float(row[3])
            except ValueError: continue
            edges.append((t, u, v))
    edges.sort(key=lambda x: x[0])
    return edges

def run_one(name):
    print(f"=== {name} ===", flush=True)
    t0 = time.time()
    edges = load_dgb_edges(name)
    n_edges = len(edges)
    nodes = set()
    for _, u, v in edges:
        nodes.add(u); nodes.add(v)
    nodes_list = list(nodes)
    print(f"  loaded {n_edges:,} edges, {len(nodes_list):,} nodes  ({time.time()-t0:.1f}s)")

    # 70/15/15 chronological split
    train_end = int(0.70 * n_edges)
    val_end   = int(0.85 * n_edges)
    train = edges[:train_end]
    val   = edges[train_end:val_end]

    # Stream train, then sample val edges and score
    adj = defaultdict(set)
    for _, u, v in train:
        adj[u].add(v); adj[v].add(u)
    print(f"  train adj built, |E_train|={train_end:,}, avg deg≈{2*train_end/len(nodes_list):.1f}")

    # Sample up to 3000 val queries
    val_sample = val if len(val) <= 3000 else random.sample(val, 3000)
    n_q = len(val_sample)

    cn_true, cn_neg = [], []
    b2_true, b2_neg = [], []
    snap_adj = {u: set(s) for u, s in adj.items()}  # frozen snapshot
    for _, u, v_true in val_sample:
        v_neg = random.choice(nodes_list)
        while v_neg == u:
            v_neg = random.choice(nodes_list)

        Nu = snap_adj.get(u, set())
        Nvt = snap_adj.get(v_true, set())
        Nvn = snap_adj.get(v_neg, set())
        cn_true.append(len(Nu & Nvt))
        cn_neg.append(len(Nu & Nvn))

        # 2-hop bridge: |N(v) ∩ N2(u)|
        # N2(u) = neighbours of u's neighbours
        N2u = set()
        for w in Nu:
            N2u |= snap_adj.get(w, set())
        N2u.discard(u)
        b2_true.append(len(Nvt & N2u))
        b2_neg.append(len(Nvn & N2u))

    def discauc(true_scores, neg_scores):
        wins, ties = 0, 0
        for t, n in zip(true_scores, neg_scores):
            if t > n: wins += 1
            elif t == n: ties += 1
        return (wins + 0.5 * ties) / len(true_scores)

    cn_auc  = discauc(cn_true, cn_neg)
    b2_auc  = discauc(b2_true, b2_neg)
    elapsed = time.time() - t0

    res = {
        "dataset": name, "n_edges": n_edges, "n_nodes": len(nodes_list),
        "n_queries": n_q,
        "discauc_1hop_cn": cn_auc, "discauc_2hop_bridge": b2_auc,
        "wall_sec": elapsed,
    }
    print(f"  discAUC: 1-hop CN={cn_auc:.4f}   2-hop bridge={b2_auc:.4f}   ({elapsed:.1f}s)")
    (OUT / f"{name}.json").write_text(json.dumps(res, indent=2))
    return res

results = []
for ds in DATASETS:
    try:
        r = run_one(ds)
        results.append(r)
    except Exception as e:
        print(f"  FAIL {ds}: {e}")

print()
print("=== SUMMARY ===")
print(f"{'dataset':<10s} {'edges':>9s} {'nodes':>8s} {'1-hop CN':>9s} {'2-hop bridge':>13s}")
for r in results:
    print(f"{r['dataset']:<10s} {r['n_edges']:>9d} {r['n_nodes']:>8d} "
          f"{r['discauc_1hop_cn']:>9.4f} {r['discauc_2hop_bridge']:>13.4f}")

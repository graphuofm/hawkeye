#!/usr/bin/env python
"""Schematic 'motivating example' figure: a cohesive sub-community densifying over
time (coreness surge) precedes a burst of new edges — invisible to a model that
only tracks pairwise interaction recency. Output: results/figures/fig_running_example.png"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
FIG = os.path.join(ROOT, "results", "figures")
os.makedirs(FIG, exist_ok=True)

# 5 "ring" addresses a1..a5 + 2 "victims" v1,v2 ; positions
pos = {"a1": (0.0, 1.0), "a2": (0.95, 0.31), "a3": (0.59, -0.81),
       "a4": (-0.59, -0.81), "a5": (-0.95, 0.31), "v1": (2.3, 0.6), "v2": (2.3, -0.6)}
ring = ["a1", "a2", "a3", "a4", "a5"]

E0 = [("a1", "a2"), ("a3", "a4")]                                  # t0: sparse, core≈1
E1 = E0 + [("a2", "a3"), ("a4", "a5"), ("a5", "a1"), ("a1", "a3")]  # t1: densifying, core↑
E2 = E1 + [("a2", "a4"), ("a3", "a5"), ("a1", "a4"), ("a2", "a5"),  # t2: ring complete (core 4)
           ("a1", "v1"), ("a3", "v2"), ("a5", "v1")]                #     + burst of new edges

def coreness_of_ring(edges):
    # crude: core number = (#ring neighbours) capped by peeling -> for a clique on m nodes, core=m-1
    deg = {n: 0 for n in ring}
    for u, v in edges:
        if u in deg: deg[u] += 1
        if v in deg: deg[v] += 1
    # approximate core by min over the ring's induced subgraph after peeling -> just report min deg of the densest core
    # for the schematic just use min(deg) of nodes that survive (good enough)
    return min(deg.values()) if deg else 0

def draw_graph(ax, edges, new_edges, title):
    new = set(tuple(sorted(e)) for e in new_edges)
    for u, v in edges:
        xu, yu = pos[u]; xv, yv = pos[v]
        is_new = tuple(sorted((u, v))) in new
        ax.plot([xu, xv], [yu, yv], "-" if not is_new else "-",
                color="crimson" if is_new else "0.55", lw=2.4 if is_new else 1.2,
                zorder=1, alpha=0.95 if is_new else 0.8)
    for n, (x, y) in pos.items():
        ring_node = n in ring
        ax.scatter([x], [y], s=520 if ring_node else 360,
                   c="steelblue" if ring_node else "darkorange",
                   edgecolors="k", linewidths=1.0, zorder=3)
        ax.text(x, y, n, ha="center", va="center", fontsize=9, color="white", zorder=4, fontweight="bold")
    ax.set_title(title, fontsize=11)
    ax.set_xlim(-1.5, 3.0); ax.set_ylim(-1.5, 1.5); ax.axis("off")


fig = plt.figure(figsize=(13.5, 4.2))
gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.05])
ax0 = fig.add_subplot(gs[0]); ax1 = fig.add_subplot(gs[1]); ax2 = fig.add_subplot(gs[2]); axc = fig.add_subplot(gs[3])
draw_graph(ax0, E0, [], r"$t_0$: a1..a5 sparse, low coreness")
draw_graph(ax1, E1, set(E1) - set(E0), r"$t_1$: sub-community densifies — coreness$\uparrow$")
draw_graph(ax2, E2, set(E2) - set(E1), r"$t_2$: ring complete $\Rightarrow$ burst of new edges")

# coreness trajectory of a ring node
ts = np.array([0, 1, 2, 3, 4, 5, 6])
core_a1 = np.array([1, 1, 2, 3, 4, 4, 4])     # rising as the ring forms
deg_a1  = np.array([1, 1, 2, 3, 4, 5, 6])     # also rising
axc.plot(ts, core_a1, "o-", color="steelblue", lw=2.2, label="coreness $c(a_1,t)$")
axc.plot(ts, deg_a1, "s--", color="0.55", lw=1.6, label="degree $d(a_1,t)$")
axc.axvspan(1.5, 4.5, color="gold", alpha=0.25, label="coreness surge window")
axc.axvline(4.6, color="crimson", ls=":", lw=2, label="new-edge burst")
axc.set_xlabel("time (events)"); axc.set_ylabel("value")
axc.set_title("structural trajectory of a ring address", fontsize=11)
axc.legend(fontsize=8, loc="upper left"); axc.grid(alpha=0.3)
fig.suptitle("A cohesive sub-community forming (coreness surge) precedes a burst of new edges — "
             "a signal orthogonal to pairwise interaction recency", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(FIG, "fig_running_example.png"), dpi=140)
print("  [fig]", os.path.join(FIG, "fig_running_example.png"))

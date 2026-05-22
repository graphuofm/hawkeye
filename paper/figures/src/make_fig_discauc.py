#!/usr/bin/env python3
"""Figure: discriminability AUC of 1-hop common neighbour vs. 2-hop cohesive
bridge across temporal-graph benchmarks (paper §3, Finding F1).

Reproducible: reads ONLY paper/figures/data/discauc.csv (a committed snapshot
extracted from results/empirical/*.json). Writes fig_discauc.pdf (vector, for
Overleaf) and fig_discauc.png (raster, for the markdown draft).

Run:  python paper/figures/src/make_fig_discauc.py
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.dirname(HERE)
DATA = os.path.join(FIG_DIR, "data", "discauc.csv")

# ---- load committed data snapshot ----
rows = list(csv.DictReader(open(DATA)))
labels = [r["dataset"].replace("tgbl-", "") for r in rows]
cn1 = [float(r["cn_1hop"]) for r in rows]
cn2 = [float(r["cn2_2hop"]) for r in rows]

# ---- plot ----
x = np.arange(len(labels))
w = 0.38
fig, ax = plt.subplots(figsize=(3.4, 2.1))
ax.bar(x - w / 2, cn1, w, label="1-hop CN", color="#bdbdbd", edgecolor="#444444", linewidth=0.6)
ax.bar(x + w / 2, cn2, w, label="2-hop cohesive bridge", color="#2c5f8a", edgecolor="#1a3a55", linewidth=0.6)
ax.axhline(0.5, ls="--", lw=0.9, color="#c0392b", label="random (0.50)")

ax.set_ylabel("discriminability AUC", fontsize=8)
ax.set_ylim(0.40, 1.0)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=7, rotation=15)
ax.tick_params(axis="y", labelsize=7)
ax.legend(fontsize=6.5, frameon=False, loc="upper left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout(pad=0.3)

pdf = os.path.join(FIG_DIR, "fig_discauc.pdf")
png = os.path.join(FIG_DIR, "fig_discauc.png")
fig.savefig(pdf, bbox_inches="tight")
fig.savefig(png, dpi=200, bbox_inches="tight")
print("wrote", pdf)
print("wrote", png)

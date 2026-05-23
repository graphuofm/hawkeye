#!/usr/bin/env python
"""Regenerate fig_discauc.pdf as a matplotlib bar chart covering all 11
datasets (6 TGB + 5 DGB) we now have discAUC for, with proper visibility
of 1-hop CN even at the 0.50 baseline.

Also writes a matching fig_discauc.drawio with the same bar layout."""
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_PDF = Path("/home/jding/CIKM2026frp/paper_ol/figures/fig_discauc.pdf")
OUT_DRAWIO = Path("/home/jding/CIKM2026frp/paper_ol/figures/fig_discauc.drawio")

# data: (label, 1-hop CN, 2-hop bridge, regime)
DATA = [
    ("tgbl-uci",       0.50, 0.76, "sparse"),
    ("tgbl-enron",     0.50, 0.73, "sparse"),
    ("tgbl-wiki",      0.50, 0.85, "sparse-bipartite"),
    ("tgbl-subreddit", 0.50, 0.96, "bipartite"),
    ("tgbl-coin",      0.75, 0.77, "dense"),
    ("tgbl-lastfm",    0.50, 0.53, "degen."),
    ("mooc",           0.09, 0.94, "bipartite"),
    ("reddit",         0.43, 0.98, "bipartite"),
    ("CanParl",        0.60, 0.48, "near-cmp"),
    ("USLegis",        0.63, 0.56, "near-cmp"),
    ("UNvote",         0.52, 0.52, "degen."),
]

labels = [d[0] for d in DATA]
cn1 = np.array([d[1] for d in DATA])
cn2 = np.array([d[2] for d in DATA])

# ---------------- matplotlib bar chart ----------------
fig, ax = plt.subplots(figsize=(8.5, 3.6))
x = np.arange(len(labels))
w = 0.38

# 1-hop CN bars (gray) — start from 0 (not 0.4) so 0.50 baseline is clearly visible
ax.bar(x - w/2, cn1, w, color="#BCBCBC", edgecolor="#555555", linewidth=0.6,
       label="1-hop CN")
# 2-hop bridge bars (teal)
ax.bar(x + w/2, cn2, w, color="#1F6F6F", edgecolor="#0C3C3C", linewidth=0.6,
       label="2-hop cohesive bridge")

# value labels on top of each bar (only for non-trivial cases)
for i, (v1, v2) in enumerate(zip(cn1, cn2)):
    ax.text(i - w/2, v1 + 0.015, f"{v1:.2f}", ha="center", va="bottom",
            fontsize=7.5, color="#444444")
    ax.text(i + w/2, v2 + 0.015, f"{v2:.2f}", ha="center", va="bottom",
            fontsize=7.5, color="#0C3C3C", fontweight="bold" if v2 > 0.7 else "normal")

# random-baseline dashed line
ax.axhline(0.50, color="#D7191C", linestyle="--", linewidth=1.0, alpha=0.7,
           label="random (0.50)")

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=22, ha="right", fontsize=9)
ax.set_ylim(0, 1.06)
ax.set_yticks(np.arange(0, 1.05, 0.1))
ax.set_ylabel("discriminability AUC", fontsize=10)
ax.set_axisbelow(True)
ax.grid(axis="y", color="#DDDDDD", linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# legend
ax.legend(loc="upper right", framealpha=0.9, fontsize=9, ncol=1)

# annotate the four regimes underneath the x-axis as a thin band
# colour map for the regimes
regime_color = {
    "sparse": "#FBD0AC", "sparse-bipartite": "#FBD0AC",
    "bipartite": "#A8D5BA",
    "dense": "#C5C9F3",
    "near-cmp": "#F3C5DB",
    "degen.": "#E0E0E0",
}
for i, (lab, _, _, reg) in enumerate(DATA):
    ax.axvspan(i - 0.45, i + 0.45, ymin=-0.08, ymax=-0.02,
               color=regime_color.get(reg, "#FFFFFF"), clip_on=False, zorder=0)

plt.tight_layout()
plt.savefig(OUT_PDF, bbox_inches="tight")
plt.close()
print(f"wrote {OUT_PDF}")

# ---------------- drawio: matching bars ----------------
# Build a simple bar-chart layout in drawio (rounded rectangles).
def add(value, style, x, y, w, h, cid):
    v_esc = value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return (f'<mxCell id="{cid}" value="{v_esc}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>')

PAD = 30
LEFT_LABEL_W = 130
BAR_MAX = 360                # px representing discAUC = 1.00
ROW_H = 30
ROW_GAP = 4
TITLE_H = 26
LEGEND_H = 26
COL_LABEL_W = 50

cells = []
cid = 5000

# title
cid += 1
cells.append(add("Discriminability AUC: 1-hop CN vs 2-hop cohesive bridge",
                 "text;html=1;align=center;fontFamily=Helvetica;fontSize=12;fontStyle=1",
                 PAD, PAD, BAR_MAX + LEFT_LABEL_W + COL_LABEL_W + 100, TITLE_H, cid))

# random baseline rule at 0.50
zero_x = PAD + LEFT_LABEL_W
rand_x = zero_x + int(0.50 * BAR_MAX)
cid += 1
cells.append(add("", "rounded=0;fillColor=#D7191C;strokeColor=none;opacity=70",
                 rand_x - 1, PAD + TITLE_H + 8, 2,
                 len(DATA) * 2 * (ROW_H + ROW_GAP) + 10, cid))
cid += 1
cells.append(add("random=0.50",
                 "text;html=1;align=left;fontFamily=Helvetica;fontSize=9;fontColor=#D7191C",
                 rand_x + 4, PAD + TITLE_H + 2, 90, 14, cid))

# x-axis ticks
for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
    gx = zero_x + int(tick * BAR_MAX)
    cid += 1
    cells.append(add("", "rounded=0;fillColor=#DDDDDD;strokeColor=none",
                     gx - 1, PAD + TITLE_H + 8, 1,
                     len(DATA) * 2 * (ROW_H + ROW_GAP) + 8, cid))
    cid += 1
    cells.append(add(f"{tick:.2f}",
                     "text;html=1;align=center;fontFamily=Helvetica;fontSize=8;fontColor=#888888",
                     gx - 14, PAD + TITLE_H + 8 + len(DATA) * 2 * (ROW_H + ROW_GAP) + 10,
                     28, 14, cid))

y = PAD + TITLE_H + 18
for label, v1, v2, reg in DATA:
    # dataset label row spanning both bars
    cid += 1
    cells.append(add(label,
                     "text;html=1;align=right;verticalAlign=middle;fontFamily=Helvetica;fontSize=10;fontStyle=1",
                     PAD, y + (ROW_H + ROW_GAP) - 4, LEFT_LABEL_W - 8, 2 * ROW_H + ROW_GAP, cid))
    # 1-hop CN bar
    bar1_w = max(2, int(v1 * BAR_MAX))
    cid += 1
    cells.append(add(f"{v1:.2f}",
                     "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#9C9C9C;strokeColor=#555555;strokeWidth=0.5;"
                     "fontColor=#000000;fontFamily=Helvetica;fontSize=9;align=right;verticalAlign=middle;spacingRight=4",
                     zero_x, y, bar1_w, ROW_H, cid))
    cid += 1
    cells.append(add("1-hop CN",
                     "text;html=1;align=left;verticalAlign=middle;fontFamily=Helvetica;fontSize=8;fontColor=#666666",
                     zero_x + bar1_w + 6, y, 70, ROW_H, cid))
    # 2-hop bridge bar
    bar2_w = max(2, int(v2 * BAR_MAX))
    cid += 1
    cells.append(add(f"{v2:.2f}",
                     "rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#1F6F6F;strokeColor=#0C3C3C;strokeWidth=0.5;"
                     "fontColor=#FFFFFF;fontStyle=1;fontFamily=Helvetica;fontSize=9;align=right;verticalAlign=middle;spacingRight=4",
                     zero_x, y + ROW_H + ROW_GAP, bar2_w, ROW_H, cid))
    cid += 1
    cells.append(add("2-hop bridge",
                     "text;html=1;align=left;verticalAlign=middle;fontFamily=Helvetica;fontSize=8;fontColor=#0C3C3C",
                     zero_x + bar2_w + 6, y + ROW_H + ROW_GAP, 90, ROW_H, cid))
    y += 2 * (ROW_H + ROW_GAP) + 6

total_w = PAD * 2 + LEFT_LABEL_W + BAR_MAX + 110
total_h = y + 60

xml = f'''<mxfile host="local" version="22.0.0">
  <diagram name="discAUC-bars-v3" id="discauc_v3">
    <mxGraphModel dx="2000" dy="1200" grid="1" gridSize="10" guides="1" tooltips="1"
      connect="1" arrows="1" fold="1" page="1" pageScale="1"
      pageWidth="{total_w + 60}" pageHeight="{total_h + 60}" math="0" shadow="0">
      <root>
        <mxCell id="0"/>
        <mxCell id="1" parent="0"/>
        {"".join(cells)}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
'''
OUT_DRAWIO.write_text(xml)
print(f"wrote {OUT_DRAWIO}  ({total_w}x{total_h}px, {len(DATA)} datasets x 2 bars)")

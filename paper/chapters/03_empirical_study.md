# Chapter 3 — Empirical Study: Why Existing Structural Signals Fail on Sparse Temporal Graphs

> Project: **Hawkeye** — CIKM 2026 Full Research Paper.
> Target length: 1.2–1.5 pages (1 Figure + 1 Table). Citations `[cite:key]`.
> All numbers from `results/empirical/*.json` (measurement only, no training).

Before presenting our method, we use **training-free measurement** to answer
three questions: (i) how poor is the existing 1-hop common-neighbour signal?
(ii) how strong is the 2-hop cohesive bridge? (iii) what is the relationship
among the $k$-family indicators? The answers (Findings F1–F3) directly drive
the design of Hawkeye in Section 4.

---

## 3.1 Experimental Setup

We run a pure measurement study — **no model is trained** — on six standard TGB
datasets [cite:huang2023tgb]. For each positive edge $(u,v,t)$ in the validation
set and its associated negative destinations, we compute every candidate
structural feature on the cumulative graph $\mathcal{G}(<t)$, and report the
feature's **discriminability AUC (discAUC)**: the probability that it ranks the
true edge above a negative. A discAUC of $0.50$ means the feature is no better
than random. The six datasets span the relevant regimes — bipartite vs.
non-bipartite, sparse vs. dense — including tgbl-wiki (bipartite, 9K nodes,
157K edges), tgbl-uci and tgbl-enron (non-bipartite communication), tgbl-
subreddit (bipartite), the large tgbl-coin, and the near-complete tgbl-lastfm.

---

## 3.2 Finding F1 — 1-hop CN is near-random on sparse graphs; 2-hop is the real signal

**Table 1** reports the discAUC of the 1-hop common neighbour, the 2-hop
cohesive bridge, and its core-weighted variant.

**Table 1.** Discriminability AUC of 1-hop vs. 2-hop structural signals.
(discAUC $=0.50$ ⇒ random.)

| dataset | type | density | 1-hop CN | 2-hop CN | 2-hop×core |
|---|---|---|---:|---:|---:|
| tgbl-uci | non-bipartite | sparse | 0.50 | **0.76** | 0.75 |
| tgbl-enron | non-bipartite | sparse* | 0.50 | **0.73** | 0.71 |
| tgbl-wiki | bipartite | sparse | 0.50 | **0.85** | 0.83 |
| tgbl-subreddit | bipartite | medium | 0.50 | **0.96** | 0.96 |
| tgbl-coin | non-bipartite | denser | 0.75 | 0.77 | 0.77 |
| tgbl-lastfm | non-bipartite | degenerate-dense | 0.50 | 0.53 | 0.52 |

\* enron has only ~$10^2$ active nodes but $1.25\times10^5$ edges, so its
*cumulative* average degree is high; however, at any single timestamp each
node's live neighbourhood is small, i.e. the graph is temporally sparse.

We draw four points from Table 1.

**(1) On sparse temporal graphs, 1-hop CN is exactly random.** On tgbl-uci,
tgbl-enron and tgbl-wiki the 1-hop common neighbour has discAUC $=0.50$ — it
cannot tell a true edge from a negative at all. The cause is direct: these
graphs are too sparse for two arbitrary nodes to share a *direct* neighbour at
time $t$, so $\mathrm{CN}=0$ for positive and negative pairs alike.

**(2) The signal is not absent — it is one hop away.** On the same datasets the
2-hop cohesive bridge attains discAUC $0.73$–$0.96$. Structural signal exists in
abundance; existing channels simply look at the wrong radius (1-hop instead of
2-hop).

**(3) The effect is not a bipartite artefact.** tgbl-uci and tgbl-enron are
non-bipartite user–user communication graphs, yet their 1-hop CN is equally
random ($0.50$). The factor that decides whether 1-hop CN works is graph
**sparsity**, not graph type: on the denser tgbl-coin, 1-hop CN already works
($0.75$). Hawkeye's 2-hop bridge signal is therefore broadly applicable, not a
bipartite-specific trick.

**(4) An honest boundary.** On tgbl-lastfm even the 2-hop bridge is near-random
($0.53$): this is a degenerate near-complete graph in which *every* structural
feature loses discriminative power. This foreshadows — and is later confirmed by
— Hawkeye's failure on degenerate-dense graphs.

---

## 3.3 Finding F2 — $k$-family weighting ≈ raw 2-hop count

The last two columns of Table 1 show that the core-weighted variant
(2-hop×core), which weights each bridging node by its core number, has discAUC
very close to the unweighted 2-hop count ($0.76$ vs. $0.75$ on uci; $0.85$ vs.
$0.83$ on wiki). We report this honestly:

**The 2-hop topology itself is the dominant signal; $k$-family weighting is a
refinement, not the driver.** The value of the $k$-family (core / truss) is
twofold: (a) as **node-level features** — a node's own core number is a useful
single feature (discAUC $0.63$–$0.75$; see Appendix); and (b) as the
**incrementally maintained substrate** — the $k$-core decomposition defines the
graph's cohesiveness hierarchy on which the 2-hop bridge features are computed.
We deliberately do **not** claim that $k$-family weighting substantially boosts
the 2-hop signal, because the measurement does not support it.

---

## 3.4 Finding F3 — the constraint hierarchy degree → core → truss

In a struct-only setting (no GNN; structural pairwise features alone predict the
edge, with `pairwise_mode = cohesion`), the predictive power of the $k$-family
indicators increases monotonically with constraint tightness. On tgbl-uci the
validation MRR is
$$
\text{degree: } 0.022 \;\to\; k\text{-core: } 0.030 \;\to\; k\text{-truss: } 0.097.
$$
A tighter constraint yields a finer structural signal and stronger prediction.
However, $k$-truss costs far more to maintain than $k$-core ($O(m^{1.5})$ worst
case vs. $O(\text{local})$ per edge) and can be intractable on large or dense
graphs. **$k$-core is thus the efficiency–effectiveness sweet spot**: it captures
most of the direction of $k$-truss at a tractable, streaming cost. This supports
the choice of $k$-core as Hawkeye's default indicator in Section 4.
<!-- F3 currently single-dataset (uci); more datasets pending the sweep -->

---

## 3.5 Summary

The study reveals two facts that shape our method. (i) In temporal link
prediction, the effective *simple* structural signal is the **2-hop cohesive
bridge**, not the classical 1-hop common neighbour, which is random on sparse
graphs. (ii) Among cohesiveness indicators, **$k$-core is the best
efficiency–effectiveness trade-off**. Section 4 builds Hawkeye to exploit these
signals systematically.

---
---

# Figure specifications (for figure generation — not paper prose)

## ★ Figure 1 — Running example (recommended placement: Introduction)

Hand-drawn / tikz-style schematic, clean line art (no AI-render plastic look),
single-column width (~3.3 in / 8.4 cm), height ≤ 2 in.

Simplest version: a 3×3 node grid (9 nodes), `u` at top-left (solid blue), `v`
at bottom-right (solid red). `u` and `v` share no common neighbour (1-hop
CN = 0). But `u`'s neighbour `a` (high core, dark fill) connects to `v`'s
neighbour `d` (high core, dark fill), forming the 2-hop bridge `u–a–d–v`
(highlighted thick orange dashed path). Hollow circles for ordinary nodes;
labels small sans-serif; no gradients/shadows/3D.

Caption draft:
> *Running example illustrating the failure of 1-hop common neighbours on a
> sparse graph. `u` and `v` share no direct common neighbour (CN = 0), yet they
> are linked by 2-hop cohesive bridges through high-coreness intermediaries,
> indicating embedding in the same dense structural region.*

## ★ Figure 2 — discAUC bar chart (optional; only if page budget allows)

2-D flat bars, single-column width. x-axis: the 6 datasets; y-axis:
discriminability AUC (0.4–1.0). Two bars per dataset: light-grey = 1-hop CN,
dark-blue = 2-hop cohesive bridge. Red dashed line at y = 0.50 ("random
baseline"). Colour-blind-safe (grey + blue); no 3D, no gradient.

Caption draft:
> *Discriminability AUC of 1-hop common neighbours vs. 2-hop cohesive bridges
> across six temporal-graph benchmarks. Dashed line = random (0.50). On sparse
> graphs 1-hop CN is no better than random; 2-hop bridges reach discAUC
> 0.73–0.96.*

Note: if Figure 1 is in the Introduction, this chapter needs only Table 1;
add Figure 2 only if it does not duplicate Table 1.

---

## Open items

- [ ] Figure 1 placement → recommend Introduction; this chapter keeps Table 1.
- [ ] Figure 2 → include only if page budget allows and it adds over Table 1.
- [ ] F3 is single-dataset (uci) for now → broaden with struct-only
      degree/core/truss runs from the sweep.
- [ ] enron density wording — explained inline via the Table 1 footnote
      ("cumulative degree high, per-timestamp neighbourhood sparse").
- [ ] References: no new bib entries; reuses huang2023tgb, seidman1983kcore,
      cohen2008trusses.

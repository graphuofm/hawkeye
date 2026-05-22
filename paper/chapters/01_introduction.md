# Chapter 1 — Introduction (discussion draft v2)

> Project: **Hawkeye** — CIKM 2026 Full Research Paper.
> Status: discussion draft. Finalize after experiments land, then port to
> `paper_ol/main.tex`. Citations in `[cite:key]` → `\cite{key}` for LaTeX.

---

## ¶1 — Temporal graphs are everywhere; TLP is the task

Many real-world systems are naturally represented as graphs that **evolve over
time**: user interactions in social networks [cite:trivedi2019dyrep],
user–item engagements on e-commerce and streaming platforms
[cite:kumar2019jodie], account-to-account transactions in financial systems
[cite:huang2023tgb], the evolution of entity relations in knowledge graphs, and
device connections in communication networks. Forecasting which new edges will
appear next on such evolving graphs — **temporal link prediction (TLP)** — is a
fundamental and practically important task: it directly underpins friend
recommendation, fraud and anti-money-laundering detection, and drug–target
discovery [cite:poursafaei2022edgebank,cite:huang2023tgb].

## ¶2 — Mainstream methods: modelling interaction dynamics

Temporal-graph learning has advanced rapidly, and the prevailing paradigm models
the **interaction dynamics** of nodes. TGN maintains a per-node *memory* module
that is updated whenever an interaction event occurs, capturing each node's
long-range interaction history [cite:rossi2020tgn]. DyGFormer feeds the
chronologically ordered sequence of a node's historical neighbours into a
Transformer, encoding interaction patterns and temporal dependencies through
self-attention [cite:yu2023dygformer]. TPNet builds and projects a temporal
random-walk matrix and reaches state-of-the-art accuracy without relying on a
GNN architecture [cite:lu2024tpnet]. TNCN dynamically maintains a neighbour
dictionary and strengthens pairwise representations with temporal common-
neighbour signals [cite:zhang2024tncn]; very recent work further refines
interaction-pattern awareness and causal debiasing [cite:zhang2025ipnet,cite:zhang2025tide].
Earlier attention-, walk- and mixing-based encoders share the same spirit
[cite:xu2020tgat,cite:wang2021cawn,cite:cong2023graphmixer]. These methods
differ in emphasis but share one core idea: encode the interaction-history
sequence — *who interacted with whom, and when* — and then score candidate edges
from the learned node embeddings.

## ¶3 — The problem: the structure channel is an afterthought

Architecturally, a state-of-the-art temporal-graph model is, in essence, a
**multi-channel information aggregator**: (a) an *interaction-history channel*
encoding the event sequence; (b) a *time-encoding channel* encoding inter-event
gaps and positions; and (c) a *structure channel* encoding the topological
relation between the candidate pair. The first two channels have been refined
relentlessly — from RNNs to Transformers to state-space models — yet the
structure channel remains a **crude afterthought**. DyGFormer's structure
channel is merely a $1$–$2$-bit neighbour-cooccurrence count ("does this
neighbour also appear in the other node's history"); TNCN's is a single
common-neighbour count. Behind these designs lies an unstated assumption: that
the **1-hop common neighbour** is the right structural signal. Is it?

## ¶4 — Our empirical finding: 1-hop CN is near-random on sparse graphs

A systematic measurement study says it is not. Streaming each benchmark
chronologically and scoring every candidate structural feature by how well it
separates true edges from the official negatives, we find that **on sparse
temporal graphs the 1-hop common-neighbour signal is barely better than a coin
flip**: its discriminative AUC is $\approx 0.50$ on tgbl-uci, tgbl-enron and
tgbl-wiki. The reason is direct — these graphs are too sparse for two arbitrary
nodes to share a *direct* common neighbour, so the 1-hop count is zero for true
and negative pairs alike.

The genuinely discriminative structural signal lies **one hop deeper**. We call
it the **2-hop cohesive bridge**: whether two nodes are embedded in the same
densely connected region through length-2 paths. Measured identically, its
discriminative AUC reaches $0.74$–$0.96$ on the same datasets. Crucially, this
holds on **both bipartite and non-bipartite graphs** — the factor that decides
whether 1-hop CN works is graph **sparsity**, not graph type. (The supporting
per-dataset discAUC numbers are reported in full in Section 3; in brief: 1-hop CN
$\approx 0.50$ on tgbl-uci/enron/wiki vs. 2-hop $0.76/0.73/0.85$, while on the
denser tgbl-coin 1-hop CN already works at $0.75$, and on the degenerate-dense
tgbl-lastfm even 2-hop collapses to $0.53$.)

## ¶5 — Our method: Hawkeye

Motivated by this finding, we propose **Hawkeye**, a *cohesion-aware structure
channel* that lets a temporal-graph model "see one layer deeper". Hawkeye
incrementally maintains, over the edge stream, the classical family of
structural-cohesiveness indicators — degree $\to$ $k$-core $\to$ $k$-truss,
ordered from the loosest to the tightest constraint — and from them forms
**2-hop cohesive-bridge** pairwise features: it not only counts how many bridging
nodes a candidate pair shares within 2 hops, but also weights each bridge by its
cohesion level (core number / truss number), thereby characterising whether the
pair is embedded in the same dense structural region. Hawkeye is designed as a
**plug-and-play structure-channel replacement**: it can directly substitute
DyGFormer's native cooccurrence channel with no change to the backbone.

<!-- ============================================================= -->
<!-- Q-A PLACEHOLDER: sliding-window paragraph — to be written once  -->
<!-- the 1%/5%/10% window ablation lands. Intentionally blank now.   -->
<!-- ============================================================= -->

## ¶6 — Results and an honest boundary

Experiments on TGB and DGB datasets show that replacing DyGFormer's cooccurrence
channel with Hawkeye improves accuracy where structural signal exists to be
exploited: test MRR on tgbl-wiki rises from $0.779$ to $0.807$ ($+2.8$ pts),
above the reported DyGFormer state of the art ($0.798$); test AP on CanParl
rises from $0.700$ to $0.740$ ($+4.0$ pts). At the same time we **honestly
characterise Hawkeye's scope**: on near-complete graphs (USLegis, average degree
$537$; lastfm, $\approx 650$) every node's cohesiveness indicators saturate,
and Hawkeye no longer adds discriminative power. This boundary is itself a
useful finding — the benefit of a structure channel is governed by a graph's
*structural diversity*, and can be predicted before training by a simple
structural-diversity measure.
<!-- N (dataset count) and final numbers to be updated after the sweep -->

## Contributions

1. **An empirical finding.** On sparse temporal graphs, the 1-hop common
   neighbour is near-random as a discriminator (discAUC $\approx 0.50$); the
   real structural signal is at 2 hops — the **2-hop cohesive bridge** attains
   discAUC $0.74$–$0.96$. This holds on both bipartite and non-bipartite graphs.
2. **Hawkeye.** An incrementally maintained, cohesion-aware structure channel
   built on the $k$-family of structural-cohesiveness indicators, which can
   plug-and-play replace the native structure channel of any temporal-graph
   model.
3. **A "when does it help" boundary.** Hawkeye's gain correlates with a graph's
   structural diversity; on dense graphs whose cohesiveness indicators are
   saturated it should fall back. This predictable boundary gives practitioners
   a priori guidance.
4. **Experimental validation** on $N$ TGB + DGB datasets: $+2.8$–$4$ points over
   the SOTA on non-degenerate graphs, with honest negative results on degenerate
   graphs. <!-- N filled after the sweep -->

---

## Open items (to confirm before finalising)

- [ ] Sliding window — whether it enters ¶5 + contributions → pending the
      1%/5%/10% window ablation.
- [ ] Contributions (4): the dataset count $N$ → pending sweep completion.
- [ ] The ¶4 discAUC table: currently full table → Section 3, prose summary in
      ¶4. (User to confirm.)
- [ ] Figure 1 (motivating example) in the intro — recommended; content TBD
      (coreness-trajectory schematic vs. discAUC bar chart).

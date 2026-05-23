# Chapter 4 — Method: Hawkeye

> Project: **Hawkeye** — CIKM 2026 Full Research Paper.
> Target length: 1.8–2.0 pages (architecture Figure 3 + Eqs. 7–12 +
> complexity table). Citations `[cite:key]`.

Hawkeye is a **structure-channel replacement module**: it does not modify the
backbone temporal-graph model — it replaces only the backbone's native
structural channel. We instantiate it on DyGFormer [cite:yu2023dygformer],
swapping out the neighbour-cooccurrence encoder.

---

## 4.1 Framework Overview

Hawkeye follows one design principle: **the structural channel should be
decoupled from the interaction channel**. A SOTA temporal-graph model such as
DyGFormer embeds, inside its Transformer, a small cooccurrence encoder that acts
as its structure channel. Hawkeye keeps the Transformer's interaction modelling
untouched and replaces only that cooccurrence encoder with a **Cohesion Slot
Encoder**.

The framework has three components (Figure 3):

- **(A) Cohesion Cache** — incrementally maintains the $k$-family indicators and
  the graph adjacency over the edge stream (on CPU);
- **(B) Pairwise Feature Extraction** — for a candidate edge $(u,v)$, computes
  2-hop cohesive-bridge features from the cache;
- **(C) Cohesion Slot Encoder** — projects those features into a per-slot
  embedding that is fed to the Transformer in place of the cooccurrence channel.

<!-- FIGURE 3: architecture diagram — spec at end of this chapter. -->

---

## 4.2 Cohesion Cache: Incremental Structural Maintenance

The substrate of Hawkeye is a **Cohesion Cache** that, as the edge stream
arrives, incrementally maintains two kinds of state.

**(a) Graph structure.** A symmetric adjacency structure (CSR-backed) that
supports fast neighbour-set queries. Each arriving edge $(u,v,t)$ inserts
$u\!\to\!v$ and $v\!\to\!u$.

**(b) $k$-family indicators.** For every node, its degree $d(v)$ and core number
$c(v)$ (and, optionally, trussness $\tau(v)$).

**Degree update.** On edge $(u,v,t)$:
$$
d(u)\leftarrow d(u)+1,\qquad d(v)\leftarrow d(v)+1.
\tag{7}
$$
Cost: $O(1)$ per edge.

**Core-number update.** On edge $(u,v,t)$, only nodes "near" $u,v$ at the
current core frontier can change. Let $k=\min(c(u),c(v))$. We traverse outward
from $\{u,v\}$ through nodes of core number exactly $k$, and promote every node
that, restricted to the $k$-core subgraph, attains degree $\ge k+1$:
$$
c(w)\leftarrow c(w)+1 \quad\text{for the affected set } S,
\tag{8}
$$
following the streaming $k$-core maintenance of
[cite:batagelj2003cores,cite:sariyuce2013kcorestream]. The affected set $S$ is
typically far smaller than $|\mathcal{V}|$ (tens to hundreds of nodes), so the
update is millisecond-scale in practice. Trussness is maintained analogously but
at higher cost (Section 4.6); the degree+core pair is the efficient default.

**Sliding window (optional).** The cache can be configured with a time window
$W$: edges older than $t-W$ are evicted, triggering the corresponding
incremental core updates, so the structural signal reflects the *recent* graph
rather than the full cumulative history.
<!-- Q-A: keep in main text only if the window ablation shows a benefit;
     otherwise move to the ablation as a studied design choice. -->

---

## 4.3 Pairwise Feature Extraction: the 2-hop Cohesive Bridge

For each query, DyGFormer processes the source node $u$ together with the
sequence of its historical neighbours $[v_1,\dots,v_L]$. Hawkeye produces, for
every slot $i$, a structural feature vector describing the relation between the
pair $(u,v_i)$.

**Slot feature vector.** Each slot is described by a cohesion feature vector
$$
\mathbf{f}(u,v_i) = \bigl[\;
\mathrm{cn2}(u,v_i),\;
\mathrm{cn2\_x}(u,v_i),\;
\mathrm{cn}(u,v_i),\;
\mathrm{aa}(u,v_i),\;
d(v_i),\;
c(v_i),\;\dots\;\bigr],
\tag{9}
$$
whose central entries are the **2-hop cohesive bridge** and its cohesion-weighted
variant:
$$
\mathrm{cn2}(u,v)=\bigl|\{w\in\mathcal{N}(v):w\in\mathcal{N}_2(u)\}\bigr|,
\quad
\mathrm{cn2\_x}(u,v)=\!\!\sum_{w\in\mathcal{N}(v)\cap\mathcal{N}_2(u)}\!\! c(w),
\tag{10}
$$
together with the 1-hop common neighbour $\mathrm{cn}$, Adamic–Adar $\mathrm{aa}$,
and node-level indicators $d(v_i),c(v_i)$ (Definitions 5–6, Chapter 2).

**Implementation — two backends.** The encoder supports two feature backends.
The **full** backend (used in all reported experiments) computes a
$\sim\!20$-dimensional cohesion feature vector per slot via sparse
matrix–vector products, giving the exact 2-hop bridge and its weighted/temporal
variants. A lightweight **fast** backend computes a compact 6-dimensional vector
by per-pair set intersection, for graphs where the full computation is too
costly. Both expose the same interface to the encoder.

**Cost control.** Computing $\mathcal{N}_2(u)$ exactly is expensive for very
high-degree $u$; the cache caps the 2-hop expansion at `max_2hop` (default
$2\!\times\!10^4$) and falls back to a sampled estimate beyond it.

---

## 4.4 Cohesion Slot Encoder: Embedding

DyGFormer's native cooccurrence channel maps each slot $i$ to a
`channel_dim`-dimensional embedding. Hawkeye replaces it through the **same
interface**:
$$
\mathbf{e}_{\text{struct}}(i)
= \mathrm{MLP}_{d_{\text{feat}}\rightarrow\text{channel\_dim}}
\bigl(\mathbf{f}(u,v_i)\bigr),
\tag{11}
$$
a small projection (a $2$-layer MLP) from the cohesion feature vector to the
channel dimension. The Transformer's per-slot input is then the unchanged sum
of the three channels:
$$
\mathbf{x}(i)=\mathbf{e}_{\text{inter}}(i)+\mathbf{e}_{\text{time}}(i)
+\mathbf{e}_{\text{struct}}(i),
\tag{12}
$$
where $\mathbf{e}_{\text{inter}}$ and $\mathbf{e}_{\text{time}}$ are the
interaction-history and time-encoding channels. This is the entire architectural
change: one encoder swapped for another at the same interface — **the
Transformer backbone itself is untouched**.

---

## 4.5 Training and Inference

Training mirrors DyGFormer exactly. (1) The edge stream is processed in
chronological batches, each pairing positive edges with sampled negatives.
(2) Before scoring a batch, the Cohesion Cache absorbs that batch's positive
edges (incremental structural + $k$-family update). (3) For each $(u,v)$,
Hawkeye extracts the slot features, the Transformer encodes the sequences, and a
link score is produced. (4) The model is trained with binary cross-entropy and
Adam.

At inference, the Cohesion Cache is **reset and replayed** between the
training/validation/test phases so that, when scoring an edge at time $t$, the
cache reflects exactly the graph $\mathcal{G}(<t)$ and never leaks future
structure (cache-management details in Appendix A).

---

## 4.6 Complexity Analysis

**Table 2.** Per-edge / per-query overhead of Hawkeye over the DyGFormer
backbone.

| component | time | space |
|---|---|---|
| adjacency maintenance | $O(1)$ per edge | $O(N+M)$ |
| degree update | $O(1)$ per edge | $O(N)$ |
| core-number update | $O(\lvert S\rvert)$ per edge, $\lvert S\rvert\!\ll\!N$ | $O(N)$ |
| trussness update (optional) | up to $O(m^{1.5})$ | $O(M)$ |
| 2-hop bridge extraction | $O(d_u\cdot d_{\max})$ per query | — |
| Cohesion Slot Encoder | $O(d_{\text{feat}}\cdot\text{channel\_dim})$ per slot | tiny ($\sim\!10^3$ params) |

Three takeaways:

- **No added GPU burden.** All structural computation runs on CPU; the GPU only
  gains a small MLP projection ($\sim\!10^3$ parameters), negligible against the
  Transformer.
- **Streaming-cost maintenance.** Incremental degree+core maintenance is
  millisecond-scale per edge — far below the cost of one Transformer forward
  pass.
- **Light memory.** The Cohesion Cache stores adjacency + $k$-family values on
  CPU; for tgbl-wiki (9K nodes, 157K edges) this is $\sim\!10$–$20$ MB.

Empirically measured per-epoch overhead is reported in §5.7 (Table — uci
3-seed mean): the structure-channel sparse mat-vec dominates and gives
**+79%** overhead for the replacement variant and **+154%** for the additive
variant. The precomputation strategy (§5.7) reduces this overhead by ~3×.

---

## 4.7 Hawkeye vs. DyGFormer's Native Cooccurrence Channel

| dimension | DyGFormer cooccurrence | Hawkeye |
|---|---|---|
| information | $1$–$2$ bits (co-occurs?) | $\sim\!20$-dim continuous cohesion vector |
| structural hierarchy | none | $k$-family (degree / core / truss) |
| field of view | 1-hop | 2-hop cohesive bridge |
| computation | GPU (embedding lookup) | CPU (incremental) + GPU (MLP) |
| pluggability | tied to DyGFormer | any backbone with a structure slot |

In one sentence: Hawkeye upgrades the structure channel from a *1-bit
cooccurrence flag* to a *cohesion-aware continuous feature vector*, while
remaining plug-and-play and low-overhead.

> **Note on the CPU-side design.** Because all structural work is CPU-side
> and a function of the edge stream alone (independent of model weights and
> seed), it admits **offline precomputation**; §5.7 reports a uniform
> **3.0–3.5× training speedup** from this design.

---
---

# Figure 3 specification (for figure generation)

Architecture diagram, two layers. **Top** (dashed box, "not modified"):
DyGFormer's Transformer with three channels — interaction-history, time, and
the structure channel. **Bottom** (solid box, orange border, "our replacement"):
Hawkeye's three modules left-to-right —
(A) Cohesion Cache (grey "CPU" tint): incremental k-core / degree + CSR
adjacency;
(B) Pairwise Feature Extraction: 2-hop cohesive-bridge count, core-weighted
sum, node-level features → a $d_{\text{feat}}$-dim vector per slot;
(C) Cohesion Slot Encoder (light-blue "GPU" tint): MLP → per-slot structure
embedding, feeding the Transformer's structure channel.
Rounded rectangles, plain arrows, no shadow/3D; single-column width (~3.3 in).

Caption draft:
> *Architecture of Hawkeye as a drop-in structure channel for DyGFormer. The
> Cohesion Cache (A) incrementally maintains $k$-family indicators on CPU; for
> each candidate edge, Pairwise Feature Extraction (B) computes 2-hop
> cohesive-bridge features; the Cohesion Slot Encoder (C) projects them into
> per-slot embeddings that replace DyGFormer's native cooccurrence channel. The
> Transformer backbone (dashed box) is unchanged.*

---

## Equation index

| Eq | content |
|---|---|
| (7) | degree incremental update |
| (8) | core-number incremental update |
| (9) | per-slot cohesion feature vector $\mathbf{f}(u,v_i)$ |
| (10) | 2-hop cohesive bridge $\mathrm{cn2}$ and $\mathrm{cn2\_x}$ |
| (11) | Cohesion Slot Encoder projection |
| (12) | integration into the Transformer input |

## Open items

- [ ] Sliding window in §4.2 — keep in main text vs. move to ablation: pending
      the window ablation result.
- [ ] **Feature dimension resolved**: experiments use the *full* backend
      ($\sim\!20$-dim, exact 2-hop). The 6-dim *fast* backend is the lightweight
      alternative. Stated as such in §4.3 — supersedes the "6-dim" in the spec.
- [ ] $k$-truss: maintained and used in experiments (`gev_indicators=degree,
      core,truss`); degree+core is the efficient default. Cost noted in Table 2.
- [ ] Figure 3 second sub-panel (explicit pairwise-feature computation) — add
      only if the page budget allows; Eq. (10) covers it textually otherwise.
- [ ] §4.7 comparison table — drop if pages are tight.

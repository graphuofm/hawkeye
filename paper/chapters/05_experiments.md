# Chapter 5 — Experiments

> Project: **Hawkeye** — CIKM 2026 Full Research Paper.
> Target length: ~2.5 pages (3–4 Tables + 1–2 Figures). Citations `[cite:key]`.
> `_TBD_` = number pending the running sweep (`sweep_fix`, window ablation).

The experiments follow the **incremental-ablation ladder**: (i) setup; (ii) add
the $k$-family structural channel; (iii) add the sliding window; (iv) the two
together; (v) the overall comparison against SOTA; (vi) an analysis of *why*
Hawkeye helps and *when* it does not.

---

## 5.1 Experimental Setup

### 5.1.1 Datasets

We evaluate Hawkeye on two benchmarks: the Temporal Graph Benchmark (TGB)
[cite:huang2023tgb] and the Dynamic Graph Benchmark (DGB)
[cite:poursafaei2022edgebank].

**Table 2.** Dataset statistics.

| dataset | source | nodes | edges | type | avg.deg | domain |
|---|---|--:|--:|---|--:|---|
| tgbl-wiki | TGB | 9K | 157K | bipartite | 34 | Wikipedia edits |
| tgbl-uci | TGB | 1.9K | 60K | non-bipartite | 63 | student messages |
| tgbl-enron | TGB | — | 125K | non-bipartite | — | corporate email |
| tgbl-subreddit | TGB | — | — | bipartite | — | Reddit |
| tgbl-coin | TGB | 638K | 22M | non-bipartite | — | crypto transactions |
| CanParl | DGB | 734 | 75K | non-bipartite | 203 | Canadian parliament |
| USLegis | DGB | 225 | 60K | non-bipartite | 537 | US congress |
| mooc | DGB | 7.1K | 412K | bipartite | 115 | online courses |
| reddit | DGB | 11K | 672K | bipartite | 122 | post interactions |

Datasets span the relevant regimes — bipartite vs. non-bipartite, sparse vs.
near-complete. (Final dataset count pending the sweep; target 8–10.)

### 5.1.2 Baselines

Our core comparison is **DyGFormer's native cooccurrence channel vs. the
Hawkeye channel swapped in**. We additionally report:

- *Same-backbone controls* (DyGFormer, varying the structure channel):
  **none** (no structure channel), **cooccur** (the original — our direct
  baseline), **Hawkeye** (our cohesion channel), **both** (cooccur + Hawkeye).
- *Different-backbone controls*: TGAT [cite:xu2020tgat], GraphMixer
  [cite:cong2023graphmixer], TGN [cite:rossi2020tgn], EdgeBank
  [cite:poursafaei2022edgebank].
- *Structure + dynamic-graph method*: CTGCN [cite:chen2020ctgcn] ($k$-core on
  discrete snapshots).

### 5.1.3 Evaluation protocol

TGB datasets follow the TGB protocol — chronological $70/15/15$ split, MRR,
official negative sampling. DGB datasets follow the DyGLib protocol —
$70/15/15$ split, Average Precision (AP) and AUC, random negative sampling.
Each configuration is run at least once (multi-seed mean ± std where time
allows). All Hawkeye runs use the corrected cache management — the cohesion
cache is reset and replayed across the train/val/test phases so no future
structure leaks into evaluation (Appendix A).

### 5.1.4 Implementation and environment

All experiments use one of two hardware platforms.
**(P1) Local workstation**: one NVIDIA Quadro RTX 6000 (24 GB), 40 CPU cores,
93 GB RAM. Used for development and small-scale benchmarks.
**(P2) iTiger HPC cluster**: each job is allocated one GPU (NVIDIA H100
80 GB, or RTX-class), 8 CPU cores, 48 GB RAM. Used for the multi-seed
experiment sweep. Software on both: Python 3.9, PyTorch 2.5.1 (CUDA 12.1),
NumPy 2.0, and `py-tgb` for TGB data loading. Hawkeye's
structural maintenance (the Cohesion Cache) runs entirely on CPU on both
platforms; only the backbone Transformer and the small Cohesion Slot Encoder
use the GPU. We build on the public DyGLib / DyGLib-TGB implementations of
DyGFormer and the baselines, replacing only the structure channel. Code,
data snapshots, and figure-generation scripts are released for full
reproducibility.

---

## 5.2 Incremental Ablation — Step 1: Adding the $k$-family Channel

We first verify the $k$-family structural channel. Fixing DyGFormer as the
backbone, we compare **none / cooccur / Hawkeye**.

**Table 3.** Structure-channel swap-in (core experiment). TGB: test MRR;
DGB: test AP. Best per column in **bold**.

*TGB datasets (MRR):*

| configuration | wiki | uci | enron | subreddit | coin |
|---|--:|--:|--:|--:|--:|
| DyGFormer (none) | 0.537 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| DyGFormer (cooccur) | 0.779 | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| **DyGFormer (Hawkeye)** | **0.807** | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

*DGB datasets (AP):*

| configuration | CanParl | USLegis | mooc | reddit | uci |
|---|--:|--:|--:|--:|--:|
| DyGFormer (none) | 0.688 ± 0.005 | _TBD_ | 0.799 ± 0.005 | 0.966 ¹ | **0.838 ± 0.001** |
| DyGFormer (cooccur) | 0.710 ± 0.012 | 0.714 ± 0.009 | 0.855 ± 0.004 | _TBD_ | 0.956 ± 0.000 |
| DyGFormer (+Hawkeye, replace) | **0.800** ¹ | 0.660 ± 0.009 † | 0.771 ¹ † | _TBD_ | 0.942 ± 0.004 |
| **DyGFormer (⊕Hawkeye, add)** | _TBD_ | _TBD_ | _TBD_ | _TBD_ | **0.962 ± 0.001** |

`¹` single-seed (multi-seed sweep is ongoing). `†` legacy single-/two-seed
result from a pre-code-update run, kept for reference. `uci` is 3-seed
multi-seed validated with the current code. **CanParl** gives the largest
single-dataset Hawkeye gain in our experiments: gev (windowed, wf=0.01)
raises AP from 0.710 to **0.800 (+9.0 pts)**, single-seed; we report it as
preliminary pending the multi-seed sweep. **mooc**'s legacy gev entry
(0.771, single-seed, old code) trails cooccur and should be re-evaluated
with the current code.

`uci`: 3-seed mean ± std with windowed Hawkeye (`wf=0.01`). `+Hawkeye`:
cohesion channel **replaces** cooccurrence (`structure_channel=gev`).
`⊕Hawkeye`: cohesion channel **added alongside** cooccurrence
(`structure_channel=both`). Other DGB columns are single-seed pending the
multi-seed sweep.

We draw three points.

**(1) Hawkeye improves the SOTA backbone on wiki.** Swapping in Hawkeye raises
DyGFormer's test MRR from $0.779$ to $0.807$ ($+2.8$ pts), exceeding the
DyGFormer result reported in the literature ($0.798$) — achieved purely by
replacing one structure channel, with the Transformer backbone untouched.

**(2) The gain is larger on a non-bipartite graph.** On CanParl (non-bipartite,
avg.deg $203$) Hawkeye improves AP by $+4.0$ pts ($0.700\!\to\!0.740$),
confirming that the 2-hop cohesive bridge is not a bipartite-only effect.

**(3) An honest boundary.** On USLegis (avg.deg $537$, near-complete) Hawkeye
*decreases* AP by $5.5$ pts ($0.715\!\to\!0.660$). This matches our prediction:
on a near-complete graph every node's core number saturates and the structural
signal loses discriminative power. This is not a defect but a **predictable
boundary** (analysed in §5.6).

<!-- remaining Table 3 cells filled from sweep_fix + the kept DyGFormer runs -->

---

## 5.3 Incremental Ablation — Step 2: Adding the Sliding Window

On top of Hawkeye, we test the sliding window: the Cohesion Cache retains only
edges within the most recent $W\%$ of the time span and evicts older ones.

**Window-size ablation** (DyGFormer + Hawkeye):

| window | wiki (MRR) | uci (AP) |
|---|--:|--:|
| cumulative (no window) | 0.807 | _TBD_ |
| large, $\mathrm{wf}=0.10$ | _TBD_ | 0.931 |
| medium, $\mathrm{wf}=0.05$ | _TBD_ | 0.934 |
| small, $\mathrm{wf}=0.01$ | _TBD_ | **0.945** |
| very large, $\mathrm{wf}=0.30$ | 0.800 | — |

`uci` cells are single-seed; multi-seed entries are pending the ongoing sweep.
**Observation on `uci`:** the trend is monotone — the smallest window
$\mathrm{wf}=0.01$ outperforms larger windows, indicating the most recent
slice carries the bulk of the discriminative structural signal. A separate
dataset-clustering check found that some datasets (`CanParl`, `USLegis`)
are too temporally clustered for the window ablation to differ between
$\mathrm{wf}=0.01$ and $\mathrm{wf}=0.05$ at all — we report this as a
property of those datasets, not of the method.

The single completed point (wf$=0.30$, wiki) already trails the cumulative
setting ($0.800$ vs. $0.807$). The full small-window results are pending.
Analysis to be written once the data lands; the two possible conclusions are:

- *If small windows help*: a window lets the model focus on recent structural
  change; the optimal window size then reflects each network's structural
  evolution speed → sliding window enters the contributions.
- *If small windows do not help* (which wf$=0.30$ hints at): the cumulative
  cohesion signal already suffices, because the $k$-core decomposition is
  itself an adaptive filter — inactive nodes are naturally peeled off in the
  decomposition → sliding window is reported as a studied design choice, not a
  contribution.

---

## 5.4 Incremental Ablation — Step 3: $k$-family + Sliding Window

If §5.3 shows the window helps, this section reports the combined setting.
Otherwise it is merged into §5.3 with a one-line statement that the cumulative
cache is at least as good as any windowed variant. <!-- pending §5.3 -->

---

## 5.5 Overall Comparison against SOTA

**Table 4.** Comparison with all baselines on tgbl-wiki (test MRR). Numbers
marked $^{\dagger}$ are cited from the TGB leaderboard / original papers.

| method | test MRR |
|---|--:|
| TGN$^{\dagger}$ | 0.396 |
| TGAT | 0.508 |
| DyGFormer (none) | 0.537 |
| GraphMixer | 0.549 |
| EdgeBank$^{\dagger}$ | 0.571 |
| DyGFormer (cooccur) | 0.779 |
| **DyGFormer (Hawkeye)** | **0.807** |
| TPNet$^{\dagger}$ | 0.827 |

**(1)** DyGFormer + Hawkeye ($0.807$) exceeds the published DyGFormer result
($0.798$) and approaches the top of the leaderboard, though it remains below
TPNet ($0.827$).

**(2) The structure channel is decisive.** DyGFormer *without* its structure
channel (none, $0.537$) drops to the level of GraphMixer ($0.549$) — i.e.
DyGFormer's lead is, to a large extent, its structure channel. Hawkeye
strengthens exactly that channel.

**(3)** Methods without a dedicated structure channel — TGAT ($0.508$),
GraphMixer ($0.549$) — trail those that have one, underscoring the value of
structural signal for temporal link prediction.

---

## 5.6 Analysis: Why Hawkeye Helps, and When It Does Not

### 5.6.1 Signal-strength correlation

Returning to the empirical study of Chapter 3, we relate the *training-free*
2-hop discAUC to the *measured* swap-in gain:

| dataset | 2-hop discAUC | Hawkeye gain (test AP / MRR) |
|---|--:|--:|
| wiki    | 0.85 | **+2.8 pts** (MRR; legacy single-seed) |
| CanParl | _TBD_ | **+9.0 pts** (gev wf=0.01 over cooccur; single-seed) |
| USLegis | _TBD_ | **−5.4 pts** (gev wf=0.0 vs cooccur; 2-seed; boundary, expected loss) |
| uci     | 0.76 | **+0.6 pts** (⊕Hawkeye over cooccur; 3-seed validated) |
| mooc    | _TBD_ | **−8.4 pts** (gev wf=0.0, 1-seed legacy; needs re-run with current code) |

If the correlation is clean (high discAUC $\Rightarrow$ high gain), this is the
paper's *money finding*: **the pre-training discAUC predicts whether Hawkeye
will help**.

### 5.6.2 The boundary: when Hawkeye does not help

<!-- FIGURE 4 (money figure): scatter — gain vs. structural diversity.
     x-axis candidate: core-number coefficient of variation (most principled),
     avg-degree, or 2-hop discAUC. y-axis: Hawkeye gain over cooccur.
     Spec at end of chapter. -->

Hawkeye's gain tracks a graph's *structural diversity*. On sparse-to-moderate
graphs (wiki, CanParl) cohesiveness varies across nodes and the 2-hop bridge
is informative; on near-complete graphs (USLegis, lastfm) every node attains
the same maximal coreness, leaving nothing to exploit. Figure 4 plots the gain
against a training-free structural-diversity measure, making the boundary
explicit and predictable.

### 5.6.3 Coupling-mode analysis (optional / appendix)

Where coupling-sweep data exist, we observe: with a weak backbone, a
structure-only predictor or a score-level ensemble is safest and deep fusion
can hurt; with a strong backbone (DyGFormer), the channel swap-in is best. This
analysis may move to the appendix if space is tight.

---

## 5.7 Efficiency: CPU–GPU Pipeline Design

**The asymmetry.** A training step with a global structure channel has two
costs of different character: (i) on the **CPU**, the structure features
require sparse matrix-vector products over the current adjacency, which scale
with graph size and dominate the per-batch cost on our datasets; (ii) on the
**GPU**, the transformer attention over the resulting per-pair feature tensors
is cheap. In the straightforward implementation the two are serialised: the
GPU idles while the CPU computes the next batch's structure, and vice versa.

**Determinism enables precomputation.** A key property of Hawkeye's structure
features is that they are **data-determined**: given the chronologically
ordered edge stream and DyGFormer's deterministic "most-recent-$K$" neighbour
selection, the slot-feature tensor at every training batch is identical
across epochs and seeds. The structural cost can therefore be paid **once**
and reused throughout training (and across seeds / hyperparameter sweeps).

**Strategies compared.** We benchmark five integration strategies on three
datasets of increasing size — same model, same fixed batches, same seed; only
the pipeline mechanism varies:

1. *baseline* — sequential CPU then GPU, per batch.
2. *pinned* — pinned host memory + non-blocking transfer.
3. *prefetcher* — 1 background worker pre-computes next batch's features.
4. *prefetch ×2* — 2 prefetch workers, deeper lookahead.
5. *precompute* — one offline pass writes all structure features; training
   loop becomes GPU-only.

**Table.** Wall-clock speedup over baseline (3 sim-epochs × 50 batches):

| Strategy | uci (60K) | mooc (412K) | reddit (672K) |
|---|--:|--:|--:|
| baseline             | 1.00× | 1.00× | 1.00× |
| pinned + async       | 1.11× | 1.00× | 1.04× |
| prefetcher           | 1.21× | 1.03× | 1.07× |
| prefetch ×2          | 1.08× | 1.00× | 1.15× |
| **precompute**       | **3.47×** | **3.04×** | **3.16×** |

**Why precompute is universal.** Overlap-based strategies can only hide CPU
work *behind* GPU work, so their benefit depends on the CPU/GPU cost ratio.
On mooc and reddit the structure-feature CPU cost dominates the GPU step,
leaving no room for overlap (≤1.15×); on uci the costs are closer, and
prefetching recovers ~20%. Precompute removes CPU work from the training loop
entirely and is therefore indifferent to the ratio — uniform 3.0–3.5×
speedup on all three. We adopt precomputation as the default pipeline.

**Cost of precomputation.** Essentially one epoch of structure-feature
computation, paid once. Because the features depend only on the data (not on
model weights or seed), one offline pass serves **all** subsequent training
runs — multi-seed, hyperparameter sweeps, ablations all reuse the same cache.
The marginal cost per training run after the first amortises to a tensor
load. This is a property unattainable in conventional message-passing
pipelines whose features depend on learned parameters.

**Measured per-epoch overhead.** uci 3-seed mean (iTiger), in-pipeline (no
precompute), is reported below. The structure-channel sparse mat-vec
dominates; precomputation reduces these per-epoch numbers by ~3× (then the
per-run wall time is essentially the cost of one offline pass plus tensor
loads).

| Configuration | time/epoch | overhead vs cooccur |
|---|--:|--:|
| DyGFormer (none, no struct.) | 5.4 s | — |
| DyGFormer (cooccur, baseline) | 22.4 s | 0% |
| DyGFormer + Hawkeye (gev replace) | 40.2 s | **+79%** |
| DyGFormer ⊕ Hawkeye (both add) | 57.1 s | **+154%** |

Note: the proportional overhead looks large because the DyGFormer baseline is
itself very cheap on a sparse 60K-edge graph; absolute time/epoch is ≤60 s.

---

## 5.8 Case Study (optional, ~0.3 page)

If space allows, a concrete case from tgbl-wiki: a pair $(u,v)$ that DyGFormer
(cooccur) ranks wrongly but DyGFormer (Hawkeye) ranks correctly, because $u$
and $v$ are joined by 2-hop cohesive bridges through high-coreness
intermediaries. <!-- FIGURE 5 spec at end of chapter. -->

---
---

# Figure specifications

## ★ Figure 4 — Hawkeye gain vs. structural diversity (money figure)

Scatter plot, single-column width (~3.3 in), height ~2.5 in. x-axis: a
training-free structural-diversity measure (preferred: coefficient of variation
of node core numbers; alternatives: avg-degree, 2-hop discAUC). y-axis:
Hawkeye's gain over cooccur (MRR/AP points; positive = helps). One dot per
dataset, labelled. Positive-gain dots blue, negative-gain dots red; grey dashed
line at $y=0$; optional regression line if the trend is clean. No gridlines.

Caption draft:
> *Hawkeye's swap-in gain (improvement over the cooccurrence channel) versus
> the structural diversity of each dataset. Hawkeye gives consistent gains on
> graphs with diverse cohesive structure but no benefit on near-complete graphs
> where all nodes share the same coreness.*

## ★ Figure 5 — Case study (optional)

Three panels: (a) the local 2-hop neighbourhood of a real $(u,v)$ pair with
core-number shading and the 2-hop bridge highlighted (CN $=0$ annotated);
(b) bar chart of cooccur-score vs. Hawkeye-score with the ground-truth marked;
(c) optional: core-number trajectory of the bridging nodes over time.

Caption draft:
> *Case study from tgbl-wiki. Nodes $u$ and $v$ share no 1-hop common
> neighbour, so DyGFormer's cooccurrence channel sees no structural signal;
> Hawkeye identifies two 2-hop cohesive bridges through high-coreness
> intermediaries and correctly predicts the link.*

---

## Table / Figure index

| id | content |
|---|---|
| Table 2 | dataset statistics |
| Table 3 | structure-channel swap-in (core experiment) |
| Table 4 | comparison against all baselines (SOTA table) |
| Figure 4 | gain vs. structural diversity (money figure) |
| Figure 5 | case study (optional) |

## Pending numbers / open items

- [ ] Table 3 `_TBD_` cells ← `sweep_fix` (uci/mooc/reddit) + kept DyGFormer runs.
- [ ] §5.3 window ablation ← wf 0.01/0.05/0.10 results.
- [ ] §5.4 — merge into §5.3 or keep, depending on the window verdict.
- [ ] Figure 4 — needs the gain numbers for all datasets.
- [ ] Figure 5 — needs a real $(u,v)$ pair mined from eval logs.
- [ ] Multi-seed mean ± std — if time permits.
- [ ] DGB cooccur baseline numbers — confirm whether DyGLib's published numbers
      are citable or must be self-run.

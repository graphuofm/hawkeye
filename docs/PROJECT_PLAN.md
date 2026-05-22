# GraphEagleVision — Project Plan (CIKM 2026)

Condensed from the original planning docs (paper outline, framework design,
experiment plan, related-work survey, critical analysis). This file is the
single in-repo reference; the full prose docs live with the author.

## Deadlines
- Abstract: **2026-05-16**
- Full paper: **2026-05-23**

## Core thesis
Two orthogonal predictive signals in temporal graphs:
1. **Interaction dynamics** — sequences of pairwise events (what existing methods model).
2. **Structural cohesiveness dynamics** — temporal evolution of dense-substructure
   membership (`degree → k-core → k-truss → k-clique`: stronger constraint = denser
   = costlier = finer-grained signal). *Largely ignored — our focus.*

Key questions: (RQ1) which cohesiveness indicators carry predictive info?
(RQ2) does the *temporal change* of an indicator beat its *static value*?
(RQ3) is the structural signal orthogonal to GNN interaction embeddings?

## Method = `GraphEagleVision` (a framework, not just a method)
- **Incremental structural maintenance** (CPU): degree O(1); k-core O(local) per
  edge; triangle O(min deg); k-truss (periodic recompute for now). Auto-scale the
  indicator set by graph size (>10M edges → degree+core; 1–10M → +triangle;
  <1M → +truss).
- **Rolling statistics** per node × indicator: `current, ema(trend), variance,
  delta, max_change` → 5K-dim feature. (No raw history stored.)
- **Structural encoder**: 2-layer MLP + LayerNorm (default); GRU/identity for ablation.
- **Fusion** with a base model's interaction embedding: `gated` (default) /
  `concat` / `additive` / `struct_only` (no GNN).
- **Plug-and-play**: `integration/` wraps TGN, DyGFormer, TGAT, … unchanged.

## Datasets
- **TGB** (`data/tgb/`): tgbl-wiki (small, exhaustive negatives), tgbl-review,
  tgbl-coin (22M edges), tgbl-flight (67M edges). Metric: **MRR**. (Leaderboard
  "-v2" names map to the base names in py-tgb ≥ 0.9.)
- **DyGLib** (13 datasets): Wikipedia, Reddit, MOOC, LastFM, Enron, Social Evo.,
  UCI, Flights, Can.Parl., US Legis., UN Trade, UN Vote, Contact. Metric:
  **AP/AUC** with random/historical/inductive negatives.

## SOTA / baselines (tgbl-wiki MRR, ~2026-01 leaderboard)
TPNet 0.827 · Heuristic(LocalGlobal) 0.821 · HyperEvent 0.810 · DyGFormer 0.798 ·
NAT 0.749 · TNCN 0.718 · CAWN 0.711 · EdgeBank(tw) 0.571 · **TGN 0.396** · DyRep 0.050.
→ pure structural heuristics rank #2; GNNs are not the ceiling. Must beat
Heuristic(LocalGlobal) or explain why. Direct competitors to discuss & compare:
**CTGCN** (TKDE'20, k-core+discrete-snapshot GCN), **TTGCN** (ACML'24, k-truss).
Our differentiators vs them: continuous time, modelling the *temporal evolution*
of coreness (not static decomposition), large-scale TGB validation, plug-and-play.

## Experiment groups (≈10 tables, ≈8 figures)
1. **Empirical study** — per-indicator predictive power (Table 2); static vs
   dynamic vs both (Table 3); orthogonality to TGN embeddings (Fig 3).
2. **Fusion modes** — pure-GNN / pure-struct / late-fusion / deep-fusion (Table 4)
   + component ablation (Table 5).
3. **Main tables** — DyGLib 13 datasets (Table 6); TGB large-scale (Table 7).
4. **Cohesiveness-hierarchy figure** (the core figure) — x = degree→core→truss,
   left-y = MRR/AP, right-y = compute time → "core = best efficiency/quality" (Fig 5).
5. **Ablations** — indicators, rolling-stat set, encoder, fusion (Tables 8–9).
6. **Efficiency & scalability** — incremental vs full recompute; overhead %;
   memory; 67M-edge scaling (Table 10, Fig 6).
7. **Deep analysis / case study** — tgbl-coin coreness-surge cases (Fig 7);
   gate-value analysis by node coreness (Fig 8a); GNN-depth experiment (Fig 8b);
   inductive setting; structure-aware attention viz.

Priorities if time-constrained: P0 = fusion-mode comparison, TGB main table,
hierarchy figure. P1 = empirical study (RQ1/RQ2), DyGLib main table, ablations.

## Implementation status / next steps
- [x] Core framework `gev/` (graph, indicators, rolling stats, encoder, fusion, framework)
- [x] Unit tests (incremental k-core/triangle/truss correctness vs networkx; rolling stats; framework smoke) — 22 pass
- [x] TGB data loader (datasets kept inside `data/tgb/`); tgbl-wiki downloaded
- [x] `experiments/run_tgb_structonly.py` — runs end-to-end on tgbl-wiki (pipeline verified vs EdgeBank leaderboard MRR)
- [ ] **Pairwise structural-feature module** (see finding below) — 1-hop + 2-hop CN/AA/RA/Jaccard + common-neighbor coreness + temporal deltas
- [ ] Re-validate the core hypothesis on a small **non-bipartite** dataset (tgbl-enron / DyGLib UCI / Enron) with pairwise feats
- [ ] TGN integration + plug-and-play validation
- [ ] DyGLib loader + 13-dataset matrix; remaining TGB datasets (esp. tgbl-coin, tgbl-flight — non-bipartite, used for case study)
- [ ] Baselines: EdgeBank, Heuristic(LocalGlobal), CTGCN
- [ ] Analysis/plotting scripts; paper draft

## Finding #1 (2026-05-12): node-only feats ≈ popularity prior; bipartite issue
First minimal experiment (struct_only `degree+core`+MLP on tgbl-wiki) gave val
MRR ≈ 0.025 — about the same as a trivial "rank dst by its degree" baseline
(≈ 0.020). Reasons:
- Ranking 1000 candidate destinations for a source is dominated by *destination-side*
  features, so `[emb(node_u); emb(node_v)]` of node-level structural feats just
  learns a popularity prior. The methods that work (EdgeBank 0.59, heuristics 0.82,
  TGN, DyGFormer) all encode **pairwise / neighbor-identity** signal. → we must add
  pairwise structural features to the link predictor; the node-feature side is most
  valuable as a *complement* to a base model (TGN), giving global structure on top
  of local interaction.
- **tgbl-wiki / tgbl-review are bipartite** (users↔pages/products): 1-hop common
  neighbors are always 0, so naive CN / k-truss pairwise feats are degenerate →
  need **bipartite-aware (2-hop)** pairwise features. tgbl-coin (addr↔addr) and
  tgbl-flight (airport↔airport) are non-bipartite — better for the structural story
  & the case studies. (The original docs' "pure core+MLP on tgbl-wiki ≈ 0.40-0.65"
  estimate is too optimistic for this setting.)
- The eval pipeline is correct (reproduces EdgeBank(tw) ≈ 0.59 on tgbl-wiki) — the
  low MRR is a modeling gap, not a bug. The current `run_tgb_structonly.py` (snapshot-
  of-node-features approach) stands as the "node-feat-only" ablation; the full struct
  model needs the pairwise module.

## Naming note
The original docs used "StructEvo" / "SHIELD" inconsistently; this project
is uniformly **GraphEagleVision** (importable as `gev`). Rename before
submission if desired.

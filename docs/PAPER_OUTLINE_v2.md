# Paper outline v2 — integrating the empirical findings

> Supersedes the planning-stage outline. Driven by what the measurements actually
> show (see `docs/LOGIC_CHAIN.md`). Numbers below are preliminary (single seed,
> sub-sampled val) — finalise from `analysis/summarize.py`.

## Working title
*Beyond Interaction Patterns: Cohesive-Bridge Structure and its Evolution for
Temporal Link Prediction* (alt: "From k-cores to 2-hop cohesive bridges: …").

## Thesis (revised by the data)
Temporal link prediction is dominated by interaction-recency in existing models;
a complementary, overlooked, **incrementally maintainable structural signal** is
the **cohesive bridge** between a candidate pair — how many shared *2-hop*
connections they have, how cohesive (high k-core) those bridges are, and how that
bridge is *evolving* (windowed trend / recency, not the noisy per-step change) —
together with the node-level cohesiveness *level* and its hierarchy
(degree → k-core → k-truss). Maintained on CPU at streaming cost and plugged into
any temporal-graph model, it gives a large boost (TGN on tgbl-wiki: test MRR
0.38 → 0.53).

## Contributions
- **C1.** A systematic empirical study of the k-family cohesiveness hierarchy and
  its temporal evolution as a TLP signal — over 6 TGB benchmarks, with single-feature
  discriminability (AUC, Cohen's d, single-feature MRR), revealing a precise picture
  (what works: 2-hop cohesive bridge, windowed trend, node coreness level/hierarchy;
  what doesn't: 1-hop common neighbours, triangles, per-step delta of coreness).
- **C2.** The "windowed-granularity" finding: the *per-step* change of coreness is
  noise, but a *smoothed/windowed* view (ema / trend at scale d / recency) is ≥ the
  static value — i.e. the dynamic signal exists but only at the right timescale.
- **C3.** `GraphEagleVision`: a lightweight, incremental, plug-and-play structural
  module — node rolling stats (current/ema/std/delta/max_change/trend_<d>/recency)
  of {degree, k-core, k-truss} + pairwise cohesive-bridge features (cn2 + its
  coreness-weighting + its dynamics) — with several coupling modes to a base TG
  model; a study of *how to couple* (late fusion / score-ensemble / FiLM / aux /
  …) showing the coupling interacts with base-model quality (deep fusion helps a
  strong base, hurts a weak one; learnable score-ensemble is robust).
- **C4.** Large-scale validation (TGB; scales to 22M / 67M-edge graphs because the
  structural side is CPU/incremental) and a case study (tgbl-coin: where `core` >
  `degree` in discriminability).

## Structure
1. **Introduction.** Motivating example (a coreness surge / a forming cohesive
   community precedes a burst of new edges); gap (TG models model interaction
   dynamics; CTGCN/TTGCN use *static* k-core/k-truss in *discrete snapshots* on
   small data, don't model the temporal trajectory; WL-expressivity & over-smoothing
   limit GNNs from computing global structure); contributions.
2. **Preliminaries.** Continuous-time dynamic graph; TLP; MRR/AP eval; the
   cohesiveness hierarchy (degree, k-core, k-truss, with definitions and the
   k-truss ⊆ (k-1)-core relation); related TG-model recap.
3. **Empirical study (the core).**
   - 3.1 Setup: indicators, the per-node rolling statistics, the pairwise
     structural features (1-hop, **2-hop**, cohesion-weighted, dynamic), the
     measurement protocol (single-feature discAUC / MRR vs the TGB negatives).
   - 3.2 RQ1 — which indicators / which features carry signal? → 2-hop cohesive
     bridge ≫ node-level ≫ 1-hop CN ≈ triangles ≈ chance; the hierarchy
     (degree → core → truss) in the pure k-family signal; `core > degree` on tgbl-coin.
   - 3.3 RQ2 — static value vs temporal change? → per-step delta = noise; **windowed
     trend / ema / recency ≥ static**; window-size matters (`sweep_window.sh`).
   - 3.4 RQ3 — is the structural signal orthogonal to a TG model's? → low corr with
     TGN embeddings; the `aux` coupling (force TGN to predict the indicators) doesn't
     help → the GNN isn't / can't learn it; TGN+GEV ≫ TGN.
   - 3.5 Findings table → motivates the method.
4. **GraphEagleVision.** Incremental structural maintenance (degree O(1), k-core
   O(local), triangle, k-truss periodic; vectorised CSR pairwise for dense graphs);
   rolling statistics (incl. multi-timescale trend, recency); structural encoder
   (MLP); **the coupling layer** (late fusion concat/additive/gated/attn, FiLM,
   score-ensemble with learnable α, aux) — with the principle that coupling depth
   should match base-model quality; complexity analysis (negligible GPU mem, <X%
   time).
5. **Experiments.**
   - 5.1 Setup (TGB benchmarks, evaluation, baselines: EdgeBank, TGN, DyRep, TGAT,
     DyGFormer, NAT, TNCN, CAWN, TPNet, CTGCN, Heuristic(LocalGlobal)).
   - 5.2 Main — TGN / DyGFormer ± GraphEagleVision across datasets (the "enhance"
     table); struct-only as a standalone baseline.
   - 5.3 Cohesiveness-hierarchy figure (degree → core → truss: MRR & compute time;
     "core is the sweet spot").
   - 5.4 Coupling comparison (how to couple matters; vs base-model quality).
   - 5.5 Ablations — indicators / pairwise_mode (none / generic / cohesion / all) /
     stat_groups (static / +ema / +trend / +recency / dynamic / all) / encoder.
   - 5.6 Efficiency & scalability (incremental vs full recompute; 22M/67M-edge runs).
   - 5.7 Case study (tgbl-coin: coreness trajectories + new-edge events; gate values).
6. **Related work.** TLP methods (interaction dynamics / local neighbourhood);
   dense-subgraph decomposition on dynamic graphs (computation, not prediction);
   structural features in graph learning (CTGCN, TTGCN — *static, discrete, small*;
   "What Do TG Models Learn?"); ours = continuous time + temporal evolution +
   2-hop cohesive bridge + large-scale + plug-and-play.
7. **Conclusion & limitations.** 1-hop CN / triangles uninformative; near-complete
   graphs have no structural variation; deep fusion needs care with a weak base;
   truss is costly on huge graphs (use degree+core there).

## Key numbers (preliminary — re-derive from results/summary/)
- empirical discAUC: cn2/aa2 ≈ 0.77/0.78 (uci), 0.85/0.84 (wiki), 0.96/0.97 (subreddit);
  cn2_x_sum/cn2_x_delta_sum ≈ 0.72–0.77; core.current ≈ 0.62–0.75; core.trend_* ≈
  core.current (slightly above on uci/enron); core.delta ≈ 0.46–0.56 (chance);
  1-hop CN / triangles ≈ 0.50.
- struct-only @ tgbl-uci: pw=cohesion: degree 0.022 → core 0.030 → truss 0.097 (val);
  pw=all: ≈ 0.11–0.13 (val) regardless of indicator; pw=generic ≈ 0.137 (val).
- TGN @ tgbl-wiki test 0.384; TGN+GEV(gated, pw=all/cohesion) test ≈ 0.52–0.53.
- TGN @ tgbl-uci test ≈ 0.10; struct-only test ≈ 0.17 > all TGN+GEV couplings
  (0.09–0.16); score_ensemble best among couplings there.
- TGN @ tgbl-subreddit test 0.561 (TGN+GEV pending).

## Open items before submission
- DyGFormer (strong base) integration — the cleanest "enhance" evidence.
- Multiple seeds (val/test discrepancies suggest noise).
- struct-only on wiki/subreddit/coin/lastfm (run_matrix struct, running).
- big datasets (coin 22M / comment 44M / flight 67M) — single-pass training.
- DyGLib 13-dataset suite + baselines (EdgeBank / Heuristic / CTGCN).
- Figures: hierarchy curve, coupling-vs-base-quality, RQ2 window-ablation, case study.

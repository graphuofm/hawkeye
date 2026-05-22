# GraphEagleVision — the logic chain (kept up to date as results come in)

> Discipline: state the hypothesis up front, report what the measurements actually
> show (including the parts that contradict the hypothesis), run *all* the planned
> datasets, and only then decide the framing. Do not "draw the target after
> shooting the arrow" — no cherry-picking a dataset/feature that works.

## 0. The question
Can the traditional **k-family** cohesiveness metrics (degree → k-core → k-truss → …,
constraint tightening down the family) — and especially **their temporal evolution** —
be **used for / enhance temporal link prediction**? Two arms: (a) standalone
structural model; (b) plug into a base TG model (TGN, …). Plus: how does
performance/cost trade off along the hierarchy (is k-core the sweet spot)?

## Headline narrative (three-layer, as the data supports it)
**L1 — node-level structural position has baseline predictive power.** The static
*core number* alone is moderately discriminative for "which destination forms an
edge" (discAUC 0.62–0.77 across datasets; `core > degree` on tgbl-coin). The
cohesiveness *hierarchy* shows up in the pure k-family signal: struct-only with
`pairwise=cohesion` on tgbl-uci gives degree 0.022 → core 0.030 → **truss 0.097**
(val MRR) — tighter constraint = more signal.

**L2 — a windowed/smoothed view extracts the dynamic signal that the raw step-change
buries.** The per-update `delta` of coreness is noise (discAUC ≈ 0.46–0.56, ≈ chance);
but `ema` / `trend_<d>` (windowed change at scale d) / `recency` (how recently it
moved) are ≥ the static value (e.g. tgbl-uci `core.trend_*` 0.738 vs `core.current`
0.727; tgbl-subreddit `core.ema` 0.810, `core.trend_0.9999` 0.800 vs 0.749). Key
insight: *look at the right granularity* — smoothed ≫ per-step. The window size
matters on some datasets (subreddit needs a long window). [RQ2 trained-model ablation
running: `experiments/sweep_window.sh` → `results/window_matrix.jsonl`.]

**L3 — the pairwise "2-hop cohesive bridge" is where the real predictive power is.**
Node-level features (static or dynamic) are weak *rankers* (single-feature MRR ≈ 0.02–0.03
on tgbl-uci); the **2-hop neighbourhood overlap** `cn2`/`aa2` and its coreness-weighted
/ dynamic versions `cn2_x_sum`/`cn2_x_delta_sum` reach single-feature MRR 0.07 (uci) /
0.44 (wiki) / 0.56 (subreddit). 1-hop CN and triangles are dead everywhere. When
node-level trend + pairwise cohesion are combined and fused with a base TG model, you
get a *large* improvement: **TGN on tgbl-wiki test MRR 0.38 → 0.53** (TGN+GEV, gated;
`pairwise=cohesion` ≈ `pairwise=all`, i.e. the k-family-derived pairwise feats are
enough). On a *weak*-base dataset (tgbl-uci, TGN ≈ 0.04) deep fusion *hurts* — struct
alone (0.17 test) > all TGN+GEV couplings (0.09–0.16); `score_ensemble` (learnable α)
is the robust coupling there → "how you couple matters, and it interacts with base
quality".

Honest caveats woven in: 1-hop CN / triangles dead; per-step delta noise; node-level
dynamics ≈ (not >) static; near-complete graphs (tgbl-lastfm) have no structural
variation so everything's dead; on a weak base model naive deep fusion can hurt.

## 1. Original hypothesis (from the planning docs)
- The *temporal change* of cohesiveness indicators carries predictive info
  *beyond* their static values ("trend beats position").
- Predictive power varies across the hierarchy; **k-core = best efficiency/quality**.
- Structural signal is largely **orthogonal** to what TG models capture.

## 2. What the measurement (empirical study, no training) actually shows
*(`analysis/empirical_study.py`: per-feature discAUC = max(AUC, 1-AUC) and
single-feature MRR for ranking the true destination against the TGB negatives;
datasets so far: tgbl-uci/wiki/subreddit (sparse), tgbl-enron (dense); coin/lastfm running.)*

- **F-a. Node-only k-family features ≈ a popularity prior for MRR ranking.** A model
  that only sees `[emb(node_u); emb(node_v)]` of node feats ≈ "rank by destination
  popularity" (struct-only node-only ≈ 0.02-0.03 MRR on tgbl-uci, = trivial). →
  link prediction needs *pairwise* structural signal; node feats are valuable as a
  *complement* to a base model.
- **F-b. 1-hop common neighbours / triangles / 1-hop k-truss are DEAD on the
  benchmarks.** discAUC ≈ 0.5 on every dataset (sparse graphs: random pairs share
  no direct neighbour; bipartite: 1-hop CN ≡ 0; near-complete dense: no variation).
- **F-c. 2-hop neighbourhood overlap (`cn2`, `aa2`) is BY FAR the strongest simple
  structural signal**: discAUC 0.77 / 0.85 / 0.97 (uci/wiki/subreddit), single-feature
  MRR 0.07 / 0.44 / 0.56 — on wiki/subreddit, "rank by #2-hop-common-neighbours"
  alone is near-EdgeBank/competitive. This is structural (not interaction-recency).
- **F-d. The *per-update* delta of coreness is noise** (discAUC ≈ 0.46-0.64 ≈ chance) —
  **but a *windowed/smoothed* trend** (`ema − ema_d` at d ∈ {0.99, 0.999, 0.9999}) and
  **"recency of last coreness change"** are as discriminative as the *static* coreness
  value, sometimes more (tgbl-uci: `core.trend_*` AUC ≈ 0.76, MRR1 ≈ 0.034-0.040 vs
  `core.current` AUC 0.74, MRR1 0.009; `core.recency` AUC ≈ 0.69). → the "dynamics"
  signal is real but only at the right *timescale* (and as "how recently it moved"),
  not as the raw step-change.
- **F-e. The 2-hop *cohesion-weighted / dynamic* pairwise feats** (`cn2_x_sum`,
  `cn2_x_delta_sum`, `cn2_x_dpos`) are nearly as strong as the raw `cn2` count
  (discAUC ≈ 0.72-0.77) — i.e. the k-family enters via the **2-hop cohesive bridge**,
  not via 1-hop k-core or a node's own k-core.
- **F-f. Static centrality *level* (degree, core)** is moderately discriminative on
  the destination side (discAUC 0.6-0.88) but weak as a single-feature ranker;
  **node-level dynamics weak.**
- **F-g. Hierarchy: degree → core helps modestly** (struct-only val 0.114 → 0.125 with
  full pairwise on tgbl-uci; cohesion-only test 0.029 → 0.103); **triangle/truss not
  yet shown to help** (dead on the sparse sets — need dense ones).

## 3. What the method (`GraphEagleVision`) therefore is
Per-node rolling stats (current, ema, std, delta, max_change, **trend at multiple
timescales**, **recency**) of {degree, core, (triangle, truss on small graphs)} →
MLP encoder; **pairwise structural features** dominated by the **2-hop cohesive
bridge** (`cn2`, `cn2` weighted by common-neighbour coreness, its dynamics) + 1-hop
heuristics + node-pair cohesion; fed (with an optional interaction embedding from a
base TG model, via gated/concat/additive fusion) to a small link-predictor MLP.
Structural side is CPU/incremental; encoder/fusion/predictor trained with BCE.

## 4. The "is it useful for link prediction?" test (running)
- struct-only across datasets (`run_tgb.py`) — beats trivial baselines? approaches EdgeBank?
- TGN baseline vs **TGN + GraphEagleVision** (`run_tgb_tgn.py`) — does the structural
  signal *improve* the base model? (TGN is weak on these sets, so the cleaner test is
  a stronger base model — DyGFormer integration is TODO.)
- ablations: pairwise_mode {none / generic / cohesion / all}; stat_groups {static /
  dynamic / all}; indicators {degree / core / triangle / truss / combos}.

### 4b. *How* to couple GraphEagleVision with a GNN — "the coupling matters"
The structural side and a base TG model can be combined many ways; we compare:
- **endpoints**: `none` (pure TGN), `struct_only` (pure structural).
- **late fusion** (fuse z and h_struct before the link head): `concat`, `additive`,
  `gated` (g·z+(1-g)·h_struct), `attn` (cross-attention), and `score_ensemble`
  (α·score_TGN+(1-α)·score_GEV, learnable α — the key "just average the two models"
  baseline; late fusion must beat this to be worth it). [implemented]
- **modulation**: `film` (h_struct → γ,β; z ← (1+γ)·z+β). [implemented]
- **auxiliary task**: `aux` (extra head making TGN's z predict the indicator values,
  added to the loss — probes whether the GNN *can* learn structure if forced; if it
  helps, the GNN wasn't learning it → supports the orthogonality claim). [implemented]
- **early fusion** (h_struct concatenated into the GNN's node input, so message
  passing sees structure). [TODO — needs a TGN model change]
- **deep fusion**: structure-aware attention bias in the GNN; coreness-weighted
  neighbour sampling; coreness-conditioned GNN depth. [TODO — invasive; the doc's
  "mode D" — decide whether worth it after the cheaper couplings]
`run_matrix.sh tgn` sweeps {baseline, score_ensemble, concat, additive, gated, attn,
film, aux} × datasets, plus GEV ablations (pairwise_mode, indicators) under gated.

## 5. Framing decision — AFTER all the data is in
Candidates: (A) "the **2-hop cohesive bridge** and its evolution is an effective,
overlooked, complementary structural signal for TLP" (with the honest nuances:
1-hop dead, per-step delta noise, k-core ≈ sweet spot, dynamics matter at the right
timescale); (B) nuanced "dynamics matter in community-forming networks, level matters
in stable networks" (pending dense-graph results, esp. tgbl-coin); (C) a rigorous
measurement paper that overturns the naive "1-hop k-core dynamics" hypothesis and
identifies what actually works. Decide by the weight of evidence across datasets,
not a corner of it.

## Result files
- `results/empirical/<dataset>.json` — per-feature discAUC / |d| / single-feature MRR (with-trend version)
- `results/sweep_hierarchy.jsonl` — struct-only hierarchy sweep on tgbl-uci (older 25-dim code)
- `results/struct_matrix.jsonl` — struct-only sweep with the current code (`run_matrix.sh struct`)
- `results/tgn_matrix.jsonl`, `results/tgn_baseline.jsonl` — TGN vs TGN+GEV (`run_matrix.sh tgn`)
- `results/coupling_matrix.jsonl` — the "how to couple" sweep (struct_only / TGN / 7 couplings) on uci+enron
- `results/window_matrix.jsonl` — RQ2 window-ablation (struct-only, stat_groups ∈ {current, +ema, +trend_*, +recency, dynamic, all}) on uci/enron/subreddit
- `results/summary/*.csv` and `python analysis/summarize.py` — consolidated tables. See `results/README.md`.

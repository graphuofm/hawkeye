# GraphEagleVision

**Structural Cohesiveness Dynamics for Temporal Link Prediction** — CIKM 2026 submission.

> Working title of the paper: *Beyond Interaction Patterns: How Structural
> Cohesiveness Dynamics Drive Temporal Link Prediction*.

## Idea in one paragraph

Existing temporal-graph models learn *interaction dynamics* — who interacted with
whom and when. We argue a second, largely-ignored signal is at least as
predictive: **structural cohesiveness dynamics** — how a node's membership in
dense substructures evolves. We study a hierarchy of cohesiveness indicators
(`degree → k-core → k-truss`, with triangles/clustering as cheap proxies),
maintain them **incrementally** over the edge stream, summarise their **temporal
evolution** with rolling statistics, and feed that as a complementary node
feature. `GraphEagleVision` (the module) can run **standalone** (no GNN) or be
**plugged into any temporal-graph model** (TGN, DyGFormer, TGAT, …) with
negligible overhead.

## Repo layout

```
gev/                       core framework (importable as `import gev`)
  graph/dynamic_graph.py    insertion-only simple graph
  indicators/               degree / k-core / triangle / k-truss / clustering (incremental)
  stats/rolling.py          per-node rolling statistics (current/ema/var/delta/max_change)
  encoder/                  MLP (default) / GRU / identity structural encoder
  fusion/                   struct_only / concat / additive / gated fusion
  framework.py              GraphEagleVision main module + GEVConfig
  data/tgb_loader.py        TGB datasets (kept inside `data/tgb/`)
  utils.py
integration/                wrappers to plug GEV into base TG models (TGN, ...)
experiments/                runnable experiment scripts + configs
analysis/                   plotting / analysis scripts (figures, tables)
baselines/                  scripts to run CTGCN / EdgeBank / heuristics
tests/                      unit tests (incremental-correctness, rolling stats, framework)
data/  results/             datasets & outputs (gitignored)
docs/                       project plan & design notes
```

## Environment

Uses the existing conda env **`tgnn`** (Python 3.9, CUDA 12.1):

```bash
PY=python
$PY -m pip install -e .          # install `gev` in editable mode
$PY -m pytest tests/ -q          # 22 tests, should pass
```

It already provides `torch 2.5.1+cu121`, `py-tgb 2.2.0`, `torch-geometric`,
`networkx`; `seaborn` and `python-igraph` were added. See `requirements.txt`.

Hardware available locally: 1× Quadro RTX 6000 (24 GB), 40 CPUs, 93 GB RAM.

## Quick start

```bash
PY=python

# download a dataset into ./data/tgb/
$PY -m gev.data.tgb_loader tgbl-wiki

# standalone structural model on tgbl-wiki
$PY experiments/run_tgb_structonly.py --dataset tgbl-wiki --indicators degree,core --epochs 30
```

## Status

See `docs/PROJECT_PLAN.md` for the full experiment plan and timeline (CIKM 2026:
abstract 2026-05-16, full paper 2026-05-23). Core framework + unit tests are in;
next: minimal-viable experiment on `tgbl-wiki`, then TGN integration, then the
full dataset matrix.

#!/bin/bash
# Cohesiveness-hierarchy (k-family) sweep, struct_only.
#   - relax->tighten:  degree -> core -> triangle -> truss   (single indicator)
#   - combinations
#   - pairwise mode:   cohesion (only k-family-derived pairwise feats, the clean signal)
#                      generic  (only classic CN/AA heuristics, a baseline)
#                      all      (both)
#   - static vs dynamic node features (on core)
# Runs sequentially on one GPU; per-run JSON appended to results/sweep_hierarchy.jsonl.
set -u
PY=/home/jding/miniconda3/envs/tgnn/bin/python
cd "$(dirname "$0")/.."
mkdir -p results/logs
OUT=results/sweep_hierarchy.jsonl
EPOCHS=${EPOCHS:-25}
PATIENCE=${PATIENCE:-7}
SEED=${SEED:-0}
DATASETS=${DATASETS:-"tgbl-uci"}   # sparse; dense graphs (enron/lastfm/flight) need vectorised pairwise (TODO)
VAL_SUB=${VAL_SUB:-3000}

run () {
  local ds="$1" tag="$2"; shift 2
  local log="results/logs/${ds}_${tag}.log"
  echo "=== [$(date +%H:%M:%S)] $ds | $tag | $* ===" | tee -a results/sweep_hierarchy.progress
  $PY experiments/run_tgb.py --dataset "$ds" --epochs "$EPOCHS" --patience "$PATIENCE" \
      --seed "$SEED" --no_download --val_subsample "$VAL_SUB" --out "$OUT" "$@" > "$log" 2>&1
  grep -E "RESULT" "$log" | tee -a results/sweep_hierarchy.progress
}

for ds in $DATASETS; do
  # --- hierarchy x pairwise_mode (the core comparison) ---
  for ind in degree core triangle truss; do
    extra=""; [ "$ind" = "truss" ] && extra="--truss_recompute_every 32"
    run "$ds" "${ind}_cohesion" --indicators "$ind" --pairwise_mode cohesion $extra
    run "$ds" "${ind}_all"      --indicators "$ind" --pairwise_mode all      $extra
  done
  run "$ds" "generic_only"      --indicators core --pairwise_mode generic     # CN/AA heuristic baseline
  # --- combinations (full pairwise) ---
  run "$ds" "degree_core"       --indicators degree,core --pairwise_mode all
  run "$ds" "core_triangle"     --indicators core,triangle --pairwise_mode all
  run "$ds" "deg_core_tri"      --indicators degree,core,triangle --pairwise_mode all
  # --- static vs dynamic node features (on core, cohesion-only pairwise) ---
  run "$ds" "core_static"       --indicators core --pairwise_mode cohesion --stat_groups current
  run "$ds" "core_dynamic"      --indicators core --pairwise_mode cohesion --stat_groups ema,std,delta,max_change
  # --- no pairwise at all (node-only ablation) ---
  run "$ds" "core_nopairwise"   --indicators degree,core --no_pairwise
done
echo "=== sweep done $(date) ===" | tee -a results/sweep_hierarchy.progress

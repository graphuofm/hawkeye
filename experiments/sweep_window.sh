#!/bin/bash
# RQ2 evidence: does adding a *windowed/smoothed* view of the indicators (ema /
# trend_<d> / recency) on top of the *static* value improve the trained model?
# struct-only; we isolate the node-feature contribution two ways:
#   - --no_pairwise  : node features ONLY (purest isolation; absolute MRR tiny)
#   - --pairwise_mode cohesion : node feats + k-family pairwise (more realistic range)
# Results -> results/window_matrix.jsonl
set -u
PY=python
cd "$(dirname "$0")/.."
mkdir -p results/logs
EPOCHS="${EPOCHS:-20}"; PATIENCE="${PATIENCE:-7}"; SEED="${SEED:-0}"; VAL_SUB="${VAL_SUB:-3000}"
TREND="0.99,0.999,0.9999"
DATASETS="${DATASETS:-tgbl-uci tgbl-enron tgbl-subreddit}"
OUT=results/window_matrix.jsonl
log() { echo "=== [$(date +%H:%M:%S)] $* ===" | tee -a results/window.progress; }

run() { local ds="$1" tag="$2"; shift 2
  log "$ds | $tag | $*"
  $PY experiments/run_tgb.py --dataset "$ds" --epochs "$EPOCHS" --patience "$PATIENCE" --seed "$SEED" \
      --no_download --val_subsample "$VAL_SUB" --trend_decays "$TREND" --indicators degree,core --out "$OUT" "$@" \
      > "results/logs/window_${ds}_${tag}.log" 2>&1
  grep -E "RESULT" "results/logs/window_${ds}_${tag}.log" | tee -a results/window.progress
}

# stat-group configs (the variable). 'current' = static value.
declare -A SG=(
  [current]="current"
  [cur_ema]="current,ema"
  [cur_t99]="current,trend_0.99"
  [cur_t9999]="current,trend_0.9999"
  [cur_rec]="current,recency"
  [dyn_notrend]="ema,std,delta,max_change"
  [all]="all"
)

for ds in $DATASETS; do
  # purest isolation: node features only
  for k in current cur_ema cur_t99 cur_t9999 cur_rec dyn_notrend all; do
    run "$ds" "np_${k}" --no_pairwise --stat_groups "${SG[$k]}"
  done
  # with k-family pairwise (realistic, larger MRR range) -- a few anchors
  for k in current cur_ema cur_t9999 all; do
    run "$ds" "coh_${k}" --pairwise_mode cohesion --stat_groups "${SG[$k]}"
  done
done
log "window sweep done $(date)"

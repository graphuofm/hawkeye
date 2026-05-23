#!/bin/bash
# Sliding-window graph vs cumulative graph: drop edges older than `window_fraction`
# of the dataset time span. Struct-only, degree+core, pw=cohesion.
set -u
PY=/home/jding/miniconda3/envs/tgnn/bin/python
cd "$(dirname "$0")/.."
mkdir -p results/logs
EPOCHS=${EPOCHS:-15}; PATIENCE=${PATIENCE:-5}; SEED=${SEED:-0}; VAL_SUB=${VAL_SUB:-2000}
TREND="0.99,0.999,0.9999"
DATASETS=${DATASETS:-"tgbl-uci tgbl-wiki tgbl-subreddit tgbl-enron"}
OUT=results/windowgraph_matrix.jsonl
log(){ echo "=== [$(date +%H:%M:%S)] $* ===" | tee -a results/windowgraph.progress; }
R(){ local ds=$1 wf=$2; local tag="wf${wf}"
  log "$ds | $tag"
  $PY experiments/run_tgb.py --dataset $ds --epochs $EPOCHS --patience $PATIENCE --seed $SEED \
      --no_download --val_subsample $VAL_SUB --trend_decays $TREND --indicators degree,core \
      --pairwise_mode cohesion --window_fraction $wf --out $OUT \
      > results/logs/wg_${ds}_${tag}.log 2>&1
  grep RESULT results/logs/wg_${ds}_${tag}.log | tee -a results/windowgraph.progress; }

for ds in $DATASETS; do
  for wf in 0.0 0.1 0.3 0.5; do  # 0.0 = cumulative (baseline)
    R $ds $wf
  done
done
log "windowgraph sweep done $(date)"

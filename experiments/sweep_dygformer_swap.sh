#!/bin/bash
# DyGFormer structure-channel swap-in ablation.
# 4-way × 5 datasets = 20 runs. uci/enron saturate the cooccur trick;
# wiki/subreddit/review/mooc/lastfm are the discriminative datasets.
set -u
PY=/home/jding/miniconda3/envs/tgnn/bin/python
cd "$(dirname "$0")/.."
mkdir -p results/logs
EPOCHS=${EPOCHS:-15}; PATIENCE=${PATIENCE:-5}; SEED=${SEED:-0}; VAL_SUB=${VAL_SUB:-2000}
TREND="0.99,0.999,0.9999"
DATASETS=${DATASETS:-"tgbl-wiki tgbl-subreddit tgbl-uci tgbl-enron"}
OUT=results/dygformer_swap_matrix.jsonl
log(){ echo "=== [$(date +%H:%M:%S)] $* ===" | tee -a results/dygformer_swap.progress; }
R(){ local ds=$1 sch=$2; local tag="${ds}_${sch}"
  log "$ds | struct_channel=$sch"
  $PY experiments/run_tgb_dygformer_swap.py \
      --dataset $ds --struct_channel $sch \
      --struct_indicators degree,core --struct_pairwise_mode cohesion \
      --struct_trend_decays $TREND \
      --epochs $EPOCHS --patience $PATIENCE --seed $SEED \
      --val_subsample $VAL_SUB --no_download \
      --out $OUT \
      > results/logs/dygfswap_${tag}.log 2>&1
  grep RESULT results/logs/dygfswap_${tag}.log | tee -a results/dygformer_swap.progress; }

for ds in $DATASETS; do
  for sch in none cooccur gev both; do
    R $ds $sch
  done
done
log "dygformer_swap sweep done $(date)"

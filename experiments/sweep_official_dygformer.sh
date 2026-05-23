#!/bin/bash
# Official DyGFormer + GEV cohesion-aware structure-channel swap-in.
# 4-way ablation × DyGLib-supported datasets.
# This is the CORE swap-in experiment for the paper.
set -u
PY=/home/jding/miniconda3/envs/tgnn/bin/python
ROOT=/home/jding/CIKM2026frp
cd "$ROOT/sota/DyGLib_TGB"
mkdir -p "$ROOT/results/logs"

EPOCHS=${EPOCHS:-20}; PATIENCE=${PATIENCE:-5}; NUM_RUNS=${NUM_RUNS:-1}
DATASETS=${DATASETS:-"tgbl-wiki tgbl-review"}
PROG=$ROOT/results/official_dygformer.progress
log(){ echo "=== [$(date +%H:%M:%S)] $* ===" | tee -a $PROG; }

R(){ local ds=$1 sch=$2
  local tag="${ds}_${sch}"
  log "official DyGFormer | $ds | struct=$sch"
  $PY train_link_prediction.py \
      --dataset_name $ds --model_name DyGFormer --patch_size 1 --max_input_sequence_length 32 \
      --num_runs $NUM_RUNS --gpu 0 \
      --num_epochs $EPOCHS --patience $PATIENCE \
      --structure_channel $sch --gev_indicators degree,core \
      > $ROOT/results/logs/officialdgf_${tag}.log 2>&1
  echo "[DONE] $ds | $sch | tail of log:" | tee -a $PROG
  tail -5 $ROOT/results/logs/officialdgf_${tag}.log | tee -a $PROG; }

for ds in $DATASETS; do
  for sch in none cooccur gev both; do
    R $ds $sch
  done
done
log "official DyGFormer 4-way sweep DONE $(date)"

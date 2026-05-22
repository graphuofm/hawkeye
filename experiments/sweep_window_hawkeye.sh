#!/bin/bash
# Hawkeye sliding-window ablation: small/medium/large windows (1% / 5% / 10%)
# vs cumulative. Run on a fast non-degenerate dataset (uci) and the headline
# dataset (wiki). The cohesion cache prunes edges older than wf*(t_max-t_min).
set -u
PY=python
ROOT=.
PROG="$ROOT/results/window_hawkeye.progress"
MAXJOBS=${MAXJOBS:-10}

log(){ echo "=== [$(date +%m-%d_%H:%M:%S)] $* ===" | tee -a "$PROG"; }
gpu_jobs(){ nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . ; }
wait_slot(){ while [ "$(gpu_jobs)" -ge "$MAXJOBS" ]; do sleep 60; done; }

# R_tgb / R_dgb <dataset> <wf> — DyGFormer + GEV with sliding window fraction wf
R_dgb(){
  local ds=$1 wf=$2; local tag="wf${wf}"
  wait_slot
  log "DGB $ds | window $tag"
  ( cd "$ROOT/sota/DyGLib" && nohup $PY train_link_prediction.py \
      --dataset_name "$ds" --model_name DyGFormer --patch_size 1 --max_input_sequence_length 32 \
      --num_runs 1 --gpu 0 --num_epochs 15 --patience 4 \
      --structure_channel gev --gev_indicators degree,core,truss --slot_backend full \
      --gev_window_fraction "$wf" \
      > "$ROOT/results/logs/win_${ds}_${tag}.log" 2>&1 & )
  sleep 25
}
R_tgb(){
  local ds=$1 wf=$2; local tag="wf${wf}"
  wait_slot
  log "TGB $ds | window $tag"
  ( cd "$ROOT/sota/DyGLib_TGB" && nohup $PY train_link_prediction.py \
      --dataset_name "$ds" --model_name DyGFormer --patch_size 1 --max_input_sequence_length 32 \
      --num_runs 1 --gpu 0 --num_epochs 20 --patience 5 \
      --structure_channel gev --gev_indicators degree,core \
      --gev_window_fraction "$wf" \
      > "$ROOT/results/logs/win_tgb_${ds}_${tag}.log" 2>&1 & )
  sleep 25
}

# small / medium / large windows (cumulative wf=0 already covered by the main sweep)
for wf in 0.01 0.05 0.10; do
  R_dgb uci $wf
done
for wf in 0.01 0.05 0.10; do
  R_tgb tgbl-wiki $wf
done

log "window ablation all launched"
wait
log "window ablation DONE"

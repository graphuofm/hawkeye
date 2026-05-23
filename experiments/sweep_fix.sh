#!/bin/bash
# Consolidated re-sweep after the save-folder-collision fix.
# Every gev/both run now gets a unique save dir (struct+wf+backend in the name).
# Covers: uci gev-variants (window ablation), mooc 4-rung, reddit 2-rung,
# wiki window ablation. cooccur/none/both non-colliding runs are untouched.
set -u
PY=/home/jding/miniconda3/envs/tgnn/bin/python
ROOT=/home/jding/CIKM2026frp
PROG="$ROOT/results/sweep_fix.progress"
MAXJOBS=${MAXJOBS:-9}

log(){ echo "=== [$(date +%m-%d_%H:%M:%S)] $* ===" | tee -a "$PROG"; }
gpu_jobs(){ nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . ; }
wait_slot(){ while [ "$(gpu_jobs)" -ge "$MAXJOBS" ]; do sleep 60; done; }

# DGB (DyGLib, AP/AUC).  D <dataset> <tag> <extra-args...>
D(){
  local ds=$1 tag=$2; shift 2
  wait_slot
  log "DGB $ds | $tag"
  ( cd "$ROOT/sota/DyGLib" && nohup $PY train_link_prediction.py \
      --dataset_name "$ds" --model_name DyGFormer --patch_size 1 --max_input_sequence_length 32 \
      --num_runs 1 --gpu 0 --num_epochs 15 --patience 4 "$@" \
      > "$ROOT/results/logs/fx_${ds}_${tag}.log" 2>&1 & )
  sleep 25
}
# TGB (DyGLib_TGB, MRR).  T <dataset> <tag> <extra-args...>
T(){
  local ds=$1 tag=$2; shift 2
  wait_slot
  log "TGB $ds | $tag"
  ( cd "$ROOT/sota/DyGLib_TGB" && nohup $PY train_link_prediction.py \
      --dataset_name "$ds" --model_name DyGFormer --patch_size 1 --max_input_sequence_length 32 \
      --num_runs 1 --gpu 0 --num_epochs 20 --patience 5 "$@" \
      > "$ROOT/results/logs/fx_${ds}_${tag}.log" 2>&1 & )
  sleep 25
}

GEV="--structure_channel gev --gev_indicators degree,core,truss --slot_backend full"

# --- uci: gev at 5 window fractions (covers 4-rung gev rung + window ablation) ---
D uci gev_wf0.0  $GEV --gev_window_fraction 0.0
D uci gev_wf0.01 $GEV --gev_window_fraction 0.01
D uci gev_wf0.05 $GEV --gev_window_fraction 0.05
D uci gev_wf0.10 $GEV --gev_window_fraction 0.10
D uci gev_wf0.30 $GEV --gev_window_fraction 0.30

# --- mooc: full 4-rung ---
D mooc cooccur   --structure_channel cooccur
D mooc gev_wf0.0 $GEV --gev_window_fraction 0.0
D mooc gev_wf0.30 $GEV --gev_window_fraction 0.30
D mooc both      --structure_channel both --gev_indicators degree,core,truss --slot_backend full

# --- reddit: cooccur + gev ---
D reddit cooccur --structure_channel cooccur
D reddit gev_wf0.0 $GEV --gev_window_fraction 0.0

# --- wiki: window ablation (cumulative gev already done = fix_wiki_gev) ---
T tgbl-wiki gev_wf0.01 --structure_channel gev --gev_indicators degree,core --gev_window_fraction 0.01
T tgbl-wiki gev_wf0.05 --structure_channel gev --gev_indicators degree,core --gev_window_fraction 0.05
T tgbl-wiki gev_wf0.10 --structure_channel gev --gev_indicators degree,core --gev_window_fraction 0.10

log "sweep_fix all launched; waiting"
wait
log "sweep_fix DONE"

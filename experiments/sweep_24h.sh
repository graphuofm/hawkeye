#!/bin/bash
# 24h ablation-ladder sweep for the CIKM paper.
# Per dataset: cooccur (SOTA baseline) / gev (+k-family) / gev+window / both.
# GEV runs ONLY on non-degenerate graphs — on near-complete graphs the k-truss
# decomposition (O(m^1.5)) hangs. Degenerate-graph boundary evidence comes from
# the already-completed USLegis run.
set -u
PY=/home/jding/miniconda3/envs/tgnn/bin/python
cd "$(dirname "$0")/../sota/DyGLib"
ROOT=/home/jding/CIKM2026frp
mkdir -p "$ROOT/results/logs"
PROG="$ROOT/results/sweep24h.progress"
MAXJOBS=${MAXJOBS:-5}

log(){ echo "=== [$(date +%m-%d_%H:%M:%S)] $* ===" | tee -a "$PROG"; }

# count GPU compute processes (accurate — wrapper shells don't use the GPU)
gpu_jobs(){ nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . ; }

wait_slot(){
  while [ "$(gpu_jobs)" -ge "$MAXJOBS" ]; do
    sleep 60
  done
}

# R <dataset> <tag> <extra-args...>
R(){
  local ds=$1 tag=$2; shift 2
  wait_slot
  log "launch $ds | $tag"
  nohup $PY train_link_prediction.py \
    --dataset_name "$ds" --model_name DyGFormer --patch_size 1 --max_input_sequence_length 32 \
    --num_runs 1 --gpu 0 --num_epochs 15 --patience 4 "$@" \
    > "$ROOT/results/logs/s24_${ds}_${tag}.log" 2>&1 &
  sleep 25
}

# ---- non-degenerate datasets only (k-family is tractable here) ----
# uci (avg_deg 63), mooc (115): full 4-rung ablation
for ds in uci mooc; do
  R "$ds" cooccur --structure_channel cooccur
  R "$ds" gev     --structure_channel gev  --gev_indicators degree,core,truss --slot_backend full
  R "$ds" gevwin  --structure_channel gev  --gev_indicators degree,core,truss --slot_backend full --gev_window_fraction 0.3
  R "$ds" both    --structure_channel both --gev_indicators degree,core,truss --slot_backend full
done

# reddit (avg_deg 122, 672k edges): bigger non-degenerate, 2-rung
R reddit cooccur --structure_channel cooccur
R reddit gev     --structure_channel gev --gev_indicators degree,core,truss --slot_backend full

log "sweep_24h all launched; waiting for completion"
wait
log "sweep_24h DONE"

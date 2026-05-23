#!/bin/bash
#SBATCH --job-name=mooc_small_1h
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00
#SBATCH --output=/project/jding2/hawkeye/slurm_logs/%x_%j.out
#
# 1-HOUR HARD-CAP small experiment slotted between Phase 1 (USLegis) and
# Phase 2 (wiki). Tests whether current code (wf=0.01 + native + memoise)
# also flips mooc from the legacy −8.4 pts to something positive, the same
# way it just did on USLegis (legacy −5.4 → current code single-seed +4).
#
# Strategy: prioritised single-seed runs; if SLURM kills us at 60min, the
# most informative pair (cooccur, gev) must have already landed. Native
# Bash watchdog cuts at 55min so we always finish cleanly inside SLURM's
# 60min hard window.
set -u
source /project/jding2/mc/etc/profile.d/conda.sh
conda activate /project/jding2/hkenv
export PYTHONNOUSERSITE=1
ROOT=/project/jding2/hawkeye
cd "$ROOT/sota/DyGLib"
DS=mooc
OUT="$ROOT/results/${DS}_small_1h"
mkdir -p "$OUT"
JDIR="saved_results/DyGFormer/${DS}"
WF=0.01

START=$(date +%s)
DEADLINE=$((START + 55*60))   # bash-side cap at 55 min

run() {
  local ch=$1 seed=$2
  local now; now=$(date +%s)
  if [ "$now" -ge "$DEADLINE" ]; then
    echo "=== [$(date +%H:%M:%S)] DEADLINE REACHED, skipping ${ch}_s${seed} ==="
    return 0
  fi
  local tag="${ch}_s${seed}"
  echo "=== [$(date +%H:%M:%S)] START ${tag} ==="
  local GEV_ARGS=""
  if [ "$ch" = "gev" ] || [ "$ch" = "both" ]; then
    GEV_ARGS="--gev_indicators degree,core,truss --slot_backend full --gev_window_fraction $WF"
  fi
  python train_link_prediction.py --dataset_name "$DS" --model_name DyGFormer \
    --patch_size 1 --max_input_sequence_length 32 --num_runs 1 --gpu 0 \
    --num_epochs 12 --patience 3 --structure_channel "$ch" --seed_offset "$seed" \
    $GEV_ARGS \
    > "$OUT/log_${tag}.txt" 2>&1 \
    && {
      local j
      j=$(ls -t "$JDIR"/*.json 2>/dev/null | head -1)
      if [ -n "$j" ]; then
        cp "$j" "$OUT/${tag}.json"
        python -c "import json;d=json.load(open('$OUT/${tag}.json'));m=d['test metrics'];n=d['new node test metrics'];print('    DONE ${tag}  test AP=%s  new-node AP=%s'%(m.get('average_precision'),n.get('average_precision')))"
      else
        echo "    ${tag}  NO RESULT -- see log_${tag}.txt"
      fi
    } || echo "    ${tag}  TRAIN FAILED -- see log_${tag}.txt"
}

# Priority order: cooccur and gev first (the critical comparison),
# then none (cheap), then both if time still left.
run cooccur 0
run gev     0
run none    0
run both    0

echo "=== Wall time: $(( ($(date +%s) - START) / 60 )) min ==="
echo "=== END [$(date +%H:%M:%S)] ==="

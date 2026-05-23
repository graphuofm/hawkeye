#!/bin/bash
#SBATCH --job-name=wiki_4ch_3seed
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --output=/project/jding2/hawkeye/slurm_logs/%x_%j.out
# tgbl-wiki main comparison: none / cooccur / Hawkeye(gev wf=0.01) / both
# x seed 0/1/2 = 12 runs strictly sequential. TGB MRR protocol.
set -u
source /project/jding2/mc/etc/profile.d/conda.sh
conda activate /project/jding2/hkenv
export PYTHONNOUSERSITE=1
ROOT=/project/jding2/hawkeye
cd "$ROOT/sota/DyGLib_TGB"
DS=tgbl-wiki
OUT="$ROOT/results/wiki_4ch_3seed"
mkdir -p "$OUT"
JDIR="saved_results/DyGFormer/${DS}"
WF=0.01

run() {
  local ch=$1 seed=$2
  local tag="${ch}_s${seed}"
  echo "=== [$(date +%H:%M:%S)] START ${tag} ==="
  local GEV_ARGS=""
  if [ "$ch" = "gev" ] || [ "$ch" = "both" ]; then
    GEV_ARGS="--gev_indicators degree,core,truss --slot_backend full --gev_window_fraction $WF"
  fi
  python train_link_prediction.py --dataset_name "$DS" --model_name DyGFormer \
    --patch_size 1 --max_input_sequence_length 32 --num_runs 1 --gpu 0 \
    --num_epochs 20 --patience 5 --structure_channel "$ch" --seed_offset "$seed" \
    $GEV_ARGS \
    > "$OUT/log_${tag}.txt" 2>&1
  local j
  j=$(ls -t "$JDIR"/*.json 2>/dev/null | head -1)
  if [ -n "$j" ]; then
    cp "$j" "$OUT/${tag}.json"
    python -c "import json;d=json.load(open('$OUT/${tag}.json'));m=d.get('test metrics',{});print('    DONE ${tag}  test MRR=%s'%(m.get('mrr','?')))"
  else
    echo "    ${tag}  NO RESULT -- see log_${tag}.txt"
  fi
}

for seed in 0 1 2; do
  for ch in none cooccur gev both; do
    run "$ch" "$seed"
  done
done
echo "=== ALL 12 RUNS DONE [$(date +%H:%M:%S)] ==="

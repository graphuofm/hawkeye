#!/bin/bash
#SBATCH --job-name=hawkeye_3cond
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/project/jding2/hawkeye/slurm_logs/%x_%j.out
# Hawkeye 3-condition x 3-window ablation on uci (temporal graph).
# ONE job, 9 runs strictly sequential. Each result copied out the moment
# it finishes. All windowed (no cumulative graph). C++ k-core/k-truss.
set -u
source /project/jding2/mc/etc/profile.d/conda.sh
conda activate /project/jding2/hkenv
export PYTHONNOUSERSITE=1
ROOT=/project/jding2/hawkeye
cd "$ROOT/sota/DyGLib"
DS=${DATASET:-uci}
OUT="$ROOT/results/hawkeye_3cond_${DS}"
mkdir -p "$OUT"
JDIR="saved_results/DyGFormer/${DS}"

echo "[native check]"
python -c "from gev import native; print(' native available:', native.available())"

run() {
  local cond=$1 ind=$2 wf=$3
  local tag="${cond}_wf${wf}"
  echo "=== [$(date +%H:%M:%S)] START ${tag}  (indicators=${ind}) ==="
  python train_link_prediction.py --dataset_name "$DS" --model_name DyGFormer \
    --patch_size 1 --max_input_sequence_length 32 --num_runs 1 --gpu 0 \
    --num_epochs 15 --patience 4 --structure_channel gev \
    --gev_indicators "$ind" --slot_backend full --gev_window_fraction "$wf" \
    > "$OUT/log_${tag}.txt" 2>&1
  local j
  j=$(ls -t "$JDIR"/*.json 2>/dev/null | head -1)
  if [ -n "$j" ]; then
    cp "$j" "$OUT/${tag}.json"
    python -c "import json;d=json.load(open('$OUT/${tag}.json'));m=d['test metrics'];n=d['new node test metrics'];print('    DONE ${tag}  test AP=%s  new-node AP=%s'%(m.get('average_precision',m.get('mrr')),n.get('average_precision',n.get('mrr'))))"
  else
    echo "    [$(date +%H:%M:%S)] ${tag}  NO RESULT -- see log_${tag}.txt"
  fi
}

for wf in 0.01 0.05 0.10; do
  run window     degree            "$wf"
  run winstruct  degree,core,truss "$wf"
  run purestruct core,truss        "$wf"
done
echo "=== ALL 9 RUNS DONE [$(date +%H:%M:%S)] ==="

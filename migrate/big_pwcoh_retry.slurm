#!/bin/bash
#SBATCH --job-name=big_pwcoh_retry
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:40:00
#SBATCH --output=/project/jding2/hawkeye/slurm_logs/%x_%j.out
# Retry tgbl-coin and tgbl-flight WITH cohesion, but with smaller
# subsamples and a lower max_2hop cap so the pairwise computation fits
# inside the wall-clock budget on these 22M/67M-edge graphs.
set -u
source /project/jding2/mc/etc/profile.d/conda.sh
conda activate /project/jding2/hkenv
export PYTHONNOUSERSITE=1
cd /project/jding2/hawkeye
PY=/project/jding2/hkenv/bin/python
OUT=/project/jding2/hawkeye/results/big_singlepass
mkdir -p "$OUT" "$OUT/logs"

START=$(date +%s)
DEADLINE=$((START + 95*60))

try() {
  local DS=$1 TAG=$2 ; shift 2
  local now; now=$(date +%s); [ "$now" -ge "$DEADLINE" ] && { echo "DEADLINE skip $DS $TAG"; return; }
  echo "=== [$(date +%H:%M:%S)] START ${DS} ${TAG} ==="
  timeout 3000 "$PY" experiments/run_tgb.py --dataset "$DS" \
    --indicators degree,core --pairwise_mode cohesion --single_pass --no_download \
    --val_subsample 500 --test_subsample 1000 \
    --pairwise_max_2hop 2000 \
    --seed 0 --batch_size 20000 \
    --out "$OUT/${DS}_${TAG}.jsonl" \
    > "$OUT/logs/${DS}_${TAG}.log" 2>&1 \
    && echo "    DONE ${DS} ${TAG}  $(grep RESULT $OUT/logs/${DS}_${TAG}.log | tail -1)" \
    || echo "    FAIL ${DS} ${TAG}"
}

try tgbl-coin   sp_pwcoh_small
try tgbl-flight sp_pwcoh_small
echo "=== big pwcoh retry END [$(date +%H:%M:%S)] wall=$((($(date +%s)-START)/60))min ==="

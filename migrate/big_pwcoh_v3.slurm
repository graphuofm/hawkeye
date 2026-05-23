#!/bin/bash
#SBATCH --job-name=big_pwcoh_v3
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=3:00:00
#SBATCH --output=/project/jding2/hawkeye/slurm_logs/%x_%j.out
#
# Third attempt at tgbl-coin and tgbl-flight WITH cohesion under the
# professor's single-pass protocol. Aggressively tuned to fit:
#   * smaller pairwise_max_2hop (500 vs default 20000)
#   * tiny val/test subsample (200 / 500 vs default 3000 / 5000)
#   * large batch_size (50000) to amortise per-batch overhead
#   * --pairwise_backend sparse  (force sparse mat-vec; the auto choice
#     sometimes picks 'loop' on huge graphs and that's why earlier runs
#     stalled)
# 3 h SLURM time budget, with 2.5 h inner timeout so the second dataset
# still has a chance even if the first hits the inner cap.
set -u
source /project/jding2/mc/etc/profile.d/conda.sh
conda activate /project/jding2/hkenv
export PYTHONNOUSERSITE=1
cd /project/jding2/hawkeye
PY=/project/jding2/hkenv/bin/python
OUT=/project/jding2/hawkeye/results/big_singlepass
mkdir -p "$OUT" "$OUT/logs"

START=$(date +%s)

run() {
  local DS=$1 TAG=$2
  local now; now=$(date +%s)
  local left=$((START + 165*60 - now))
  if [ "$left" -le 600 ]; then echo "skip $DS — only ${left}s left"; return; fi
  echo "=== [$(date +%H:%M:%S)] START ${DS} ${TAG}  (budget=${left}s) ==="
  timeout $((left - 60)) "$PY" experiments/run_tgb.py --dataset "$DS" \
    --indicators degree,core --pairwise_mode cohesion --single_pass --no_download \
    --pairwise_backend sparse \
    --pairwise_max_2hop 500 \
    --val_subsample 200 --test_subsample 500 \
    --seed 0 --batch_size 50000 \
    --out "$OUT/${DS}_${TAG}.jsonl" \
    > "$OUT/logs/${DS}_${TAG}.log" 2>&1 \
    && echo "    DONE ${DS} ${TAG}  $(grep RESULT $OUT/logs/${DS}_${TAG}.log | tail -1)" \
    || echo "    FAIL ${DS} ${TAG}  (see logs/${DS}_${TAG}.log tail)"
}

run tgbl-coin   sp_pwcoh_v3
run tgbl-flight sp_pwcoh_v3
echo "=== END [$(date +%H:%M:%S)] wall=$((($(date +%s)-START)/60))min ==="

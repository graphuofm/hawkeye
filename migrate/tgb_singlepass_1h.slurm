#!/bin/bash
#SBATCH --job-name=tgb_sp
#SBATCH --partition=bigTiger
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --output=/project/jding2/hawkeye/slurm_logs/%x_%j.out
#
# ONE-HOUR HARD-CAP single-pass run on ONE TGB dataset.
# Submit with env vars:
#   DS=tgbl-review     (required)
#   TAG=sp_pwcoh       (required)
#   INDICATORS=degree,core
#   PAIRWISE=cohesion   (or "none")
#   BATCH_SIZE=5000
# Example:
#   sbatch --job-name=tgb_sp_review_pwcoh \
#          --export=DS=tgbl-review,TAG=sp_pwcoh,INDICATORS=degree\,core,PAIRWISE=cohesion,BATCH_SIZE=5000 \
#          /project/jding2/hawkeye/migrate/tgb_singlepass_1h.slurm
set -u
source /project/jding2/mc/etc/profile.d/conda.sh
conda activate /project/jding2/hkenv
export PYTHONNOUSERSITE=1
cd /project/jding2/hawkeye
PY=/project/jding2/hkenv/bin/python

DS=${DS:?DS env var required}
TAG=${TAG:-sp_default}
INDICATORS=${INDICATORS:-degree,core}
PAIRWISE=${PAIRWISE:-cohesion}
BATCH_SIZE=${BATCH_SIZE:-5000}

OUT=results/big_singlepass
mkdir -p "$OUT" "$OUT/logs"

# Pairwise flag selector
PW_FLAG="--pairwise_mode $PAIRWISE"
if [ "$PAIRWISE" = "none" ]; then
  PW_FLAG="--no_pairwise"
fi

START=$(date +%s)
DEADLINE=$((START + 50*60))    # 50min soft deadline, SLURM kills at 60min anyway

echo "=== [$(date +%H:%M:%S)] START ${DS} ${TAG} (indicators=${INDICATORS} pairwise=${PAIRWISE}) ==="
# Use --no_download because data is pre-staged on /project/.../data/tgb/datasets/
timeout 2700 "$PY" experiments/run_tgb.py --dataset "$DS" \
  --indicators "$INDICATORS" $PW_FLAG --single_pass --no_download \
  --val_subsample 3000 --test_subsample 5000 --seed 0 \
  --batch_size "$BATCH_SIZE" \
  --out "$OUT/${DS}_${TAG}.jsonl" \
  > "$OUT/logs/${DS}_${TAG}.log" 2>&1 \
  && echo "    DONE ${DS} ${TAG}  $(grep RESULT $OUT/logs/${DS}_${TAG}.log | tail -1)" \
  || echo "    FAILED ${DS} ${TAG} -- see logs/${DS}_${TAG}.log"

echo "=== END [$(date +%H:%M:%S)] wall=$((($(date +%s)-START)/60))min ==="

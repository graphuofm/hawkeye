#!/bin/bash
#SBATCH --job-name=subreddit_3seed_sp
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1:40:00
#SBATCH --output=/project/jding2/hawkeye/slurm_logs/%x_%j.out
# Validate the tgbl-subreddit +19.1 headline with 3-seed runs of both
# baseline (degree only, no pairwise) and with-cohesion (degree+core, pairwise=cohesion).
# 6 runs total, ~12 min each = ~75 min, fits in 1h40m budget.
set -u
source /project/jding2/mc/etc/profile.d/conda.sh
conda activate /project/jding2/hkenv
export PYTHONNOUSERSITE=1
cd /project/jding2/hawkeye
PY=/project/jding2/hkenv/bin/python
OUT=/project/jding2/hawkeye/results/subreddit_3seed_sp
mkdir -p "$OUT" "$OUT/logs"

START=$(date +%s)
DEADLINE=$((START + 95*60))

run() {
  local TAG=$1 IND=$2 PW=$3 SEED=$4
  local now; now=$(date +%s)
  if [ "$now" -ge "$DEADLINE" ]; then echo "DEADLINE, skip $TAG s$SEED"; return; fi
  echo "=== [$(date +%H:%M:%S)] START ${TAG}_s${SEED} ==="
  local PWFLAG="--pairwise_mode $PW"
  [ "$PW" = "none" ] && PWFLAG="--no_pairwise"
  timeout 1500 "$PY" experiments/run_tgb.py --dataset tgbl-subreddit \
    --indicators "$IND" $PWFLAG --single_pass --no_download \
    --val_subsample 3000 --test_subsample 5000 --seed $SEED \
    --batch_size 10000 \
    --out "$OUT/${TAG}_s${SEED}.jsonl" \
    > "$OUT/logs/${TAG}_s${SEED}.log" 2>&1 \
    && echo "    DONE ${TAG}_s${SEED}  $(grep RESULT $OUT/logs/${TAG}_s${SEED}.log | tail -1)" \
    || echo "    FAIL ${TAG}_s${SEED}"
}

for SEED in 0 1 2; do
  run sp_deg_only degree            none      $SEED
  run sp_pwcoh    "degree,core"     cohesion  $SEED
done
echo "=== subreddit 3seed END [$(date +%H:%M:%S)] wall=$((($(date +%s)-START)/60))min ==="

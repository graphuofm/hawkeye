#!/bin/bash
#SBATCH --job-name=last_fill
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:45:00
#SBATCH --output=/project/jding2/hawkeye/slurm_logs/%x_%j.out
#
# Last fill before deadline (3h budget):
#   1. mooc both_s2 (DGB DyGFormer, ~30 min) — completes the only
#      remaining 2-seed cell in Table 3 (mooc oplus Hawkeye)
#   2. tgbl-review sp_pwcoh seeds 1 and 2 (single-pass, ~37 min each)
#      — completes the only single-seed cell in tab:scalability
# Total expected wall: 30 + 37 + 37 = ~104 min, well within 2:45.
set -u
source /project/jding2/mc/etc/profile.d/conda.sh
conda activate /project/jding2/hkenv
export PYTHONNOUSERSITE=1
PY=/project/jding2/hkenv/bin/python
HK=/project/jding2/hawkeye

# ---------- 1. mooc both_s2 ----------
cd $HK/sota/DyGLib
OUT_M=$HK/results/mooc_4ch_3seed
echo "=== [$(date +%H:%M:%S)] START mooc both_s2 ==="
$PY train_link_prediction.py --dataset_name mooc --model_name DyGFormer \
  --patch_size 1 --max_input_sequence_length 32 --num_runs 1 --gpu 0 \
  --num_epochs 15 --patience 4 --structure_channel both --seed_offset 2 \
  --gev_indicators degree,core,truss --slot_backend full --gev_window_fraction 0.01 \
  > $OUT_M/log_both_s2.txt 2>&1 \
  && {
    J=$(ls -t saved_results/DyGFormer/mooc/*.json | head -1)
    cp "$J" $OUT_M/both_s2.json
    $PY -c "import json;d=json.load(open('$OUT_M/both_s2.json'));m=d['test metrics'];n=d['new node test metrics'];print('    DONE mooc both_s2  test AP=%s  nn=%s'%(m.get('average_precision'),n.get('average_precision')))"
  } || echo "    FAIL mooc both_s2"

# ---------- 2. tgbl-review pwcoh s1 + s2 ----------
cd $HK
OUT_R=$HK/results/big_singlepass
mkdir -p $OUT_R/logs
for S in 1 2; do
  echo "=== [$(date +%H:%M:%S)] START tgbl-review sp_pwcoh s$S ==="
  timeout 3000 $PY experiments/run_tgb.py --dataset tgbl-review \
    --indicators degree,core --pairwise_mode cohesion --single_pass --no_download \
    --val_subsample 3000 --test_subsample 5000 --seed $S --batch_size 5000 \
    --out $OUT_R/tgbl-review_sp_pwcoh_s${S}.jsonl \
    > $OUT_R/logs/tgbl-review_sp_pwcoh_s${S}.log 2>&1 \
    && echo "    DONE s$S  $(grep RESULT $OUT_R/logs/tgbl-review_sp_pwcoh_s${S}.log | tail -1)" \
    || echo "    FAIL s$S"
done

echo "=== END [$(date +%H:%M:%S)] ==="

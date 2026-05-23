#!/bin/bash
#SBATCH --job-name=coin_pwcoh_v3
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=2:30:00
#SBATCH --output=/project/jding2/hawkeye/slurm_logs/%x_%j.out
set -u
source /project/jding2/mc/etc/profile.d/conda.sh
conda activate /project/jding2/hkenv
export PYTHONNOUSERSITE=1
cd /project/jding2/hawkeye
PY=/project/jding2/hkenv/bin/python
OUT=/project/jding2/hawkeye/results/big_singlepass
mkdir -p $OUT/logs
echo "=== [$(date +%H:%M:%S)] START tgbl-coin sp_pwcoh_v3 ==="
timeout 8400 $PY experiments/run_tgb.py --dataset tgbl-coin \
  --indicators degree,core --pairwise_mode cohesion --single_pass --no_download \
  --pairwise_backend sparse --pairwise_max_2hop 500 \
  --val_subsample 200 --test_subsample 500 \
  --seed 0 --batch_size 50000 \
  --out $OUT/tgbl-coin_sp_pwcoh_v3.jsonl \
  > $OUT/logs/tgbl-coin_sp_pwcoh_v3.log 2>&1 \
  && echo "    DONE  $(grep RESULT $OUT/logs/tgbl-coin_sp_pwcoh_v3.log | tail -1)" \
  || echo "    FAIL  (see logs/tgbl-coin_sp_pwcoh_v3.log tail)"

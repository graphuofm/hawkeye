#!/bin/bash
#SBATCH --job-name=big_singlepass
#SBATCH --partition=bigTiger
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=/project/jding2/hawkeye/slurm_logs/%x_%j.out
#
# Professor's "single-pass online" sweep on the BIG TGB datasets that the
# existing 12-run sweeps cannot afford:
#   tgbl-mooc      (~411K edges)
#   tgbl-review    (large recommendation)
#   tgbl-comment   (~44M  edges, very large)
#   tgbl-flight    (~67M  edges, near-complete -> degree-only)
#
# Adds new scalability evidence: cohesion-aware features measured at scale.
# Output goes to results/big_singlepass/ as JSONL.
set -u
source /project/jding2/mc/etc/profile.d/conda.sh
conda activate /project/jding2/hkenv
export PYTHONNOUSERSITE=1
cd /project/jding2/hawkeye
PY=/project/jding2/hkenv/bin/python
OUT=results/big_singlepass
mkdir -p "$OUT" "$OUT/logs"

START=$(date +%s)
DEADLINE=$((START + 9*60*60))   # 9h soft cap (SLURM allows 10h)

run() {
  local DS=$1 TAG=$2; shift 2
  local now; now=$(date +%s)
  if [ "$now" -ge "$DEADLINE" ]; then
    echo "=== [$(date +%H:%M:%S)] DEADLINE, skipping ${DS} ${TAG} ==="
    return 0
  fi
  echo "=== [$(date +%H:%M:%S)] START ${DS} ${TAG} ==="
  timeout 7200 "$PY" experiments/run_tgb.py --dataset "$DS" \
      --val_subsample 3000 --test_subsample 5000 --seed 0 \
      --out "$OUT/${DS}_${TAG}.jsonl" "$@" \
      > "$OUT/logs/${DS}_${TAG}.log" 2>&1 \
      && echo "    DONE ${DS} ${TAG} ($(grep RESULT "$OUT/logs/${DS}_${TAG}.log" | tail -1))" \
      || echo "    FAILED ${DS} ${TAG}  -- see logs/${DS}_${TAG}.log"
}

# --- tgbl-mooc (411K bipartite, full single-pass with k-family + 2-hop bridge)
run tgbl-mooc    sp_pwall      --indicators degree,core --pairwise_mode all      --single_pass --batch_size 5000

# --- tgbl-review (recommendation, mid-sized, single-pass + cohesion pairwise)
run tgbl-review  sp_pwcoh      --indicators degree,core --pairwise_mode cohesion --single_pass --batch_size 5000

# --- tgbl-comment (44M edges, degree+core only, no pairwise — scalability)
run tgbl-comment sp_deg_core   --indicators degree,core --no_pairwise            --single_pass --batch_size 10000

# --- tgbl-flight (67M near-complete, degree-only — the cheapest possible)
run tgbl-flight  sp_deg_only   --indicators degree      --no_pairwise            --single_pass --batch_size 10000

echo "=== END [$(date +%H:%M:%S)] wall=$((($(date +%s)-START)/60))min ==="

#!/bin/bash
# Multi-seed sweep: re-run the full 32-config sweep for seed1 and seed2
# (seed0 was the original submit_all + submit_rest batch). 32 x 2 = 64 jobs.
# The train script reads --seed_offset (added to run index) so each seed lands
# in its own ..._seed{N}.json without colliding with seed0.
S=/project/jding2/hawkeye/migrate/itiger_run.slurm
mkdir -p /project/jding2/hawkeye/slurm_logs
# sub <name> <runner> <dataset> <channel> <wf> <epochs> <seed>
sub(){ sbatch --job-name="$1" \
  --export=ALL,RUNNER=$2,DATASET=$3,CHANNEL=$4,WF=${5:-0.0},EPOCHS=$6,SEED=$7 "$S"; }

for SEED in 1 2; do
  # --- TGB wiki 4-way ---
  for ch in none cooccur gev both; do
    sub hk_wiki_${ch}_s${SEED} tgb tgbl-wiki $ch 0.0 20 $SEED
  done
  # --- DGB uci/mooc/reddit/CanParl 4-way ---
  for ds in uci mooc reddit CanParl; do
    for ch in none cooccur gev both; do
      sub hk_${ds}_${ch}_s${SEED} dgb $ds $ch 0.0 15 $SEED
    done
  done
  # --- boundary datasets: cooccur vs gev ---
  for ds in USLegis UNvote; do
    for ch in cooccur gev; do
      sub hk_${ds}_${ch}_s${SEED} dgb $ds $ch 0.0 15 $SEED
    done
  done
  # --- window ablation: wiki + uci gev at wf 0.01/0.05/0.10/0.30 ---
  for wf in 0.01 0.05 0.10 0.30; do
    sub hk_wikiwf${wf}_s${SEED} tgb tgbl-wiki gev $wf 20 $SEED
    sub hk_uciwf${wf}_s${SEED}  dgb uci       gev $wf 15 $SEED
  done
done
echo "multi-seed jobs submitted (seed1 + seed2, 64 total)"

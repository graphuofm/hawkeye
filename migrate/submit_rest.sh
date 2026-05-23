#!/bin/bash
# Submit the REMAINING Hawkeye sweep to iTiger SLURM (the 4 validation jobs
# wiki/uci x cooccur/gev are already running, so they are skipped here).
S=/project/jding2/hawkeye/migrate/itiger_run.slurm
mkdir -p /project/jding2/hawkeye/slurm_logs
sub(){ sbatch --job-name="$1" --export=ALL,RUNNER=$2,DATASET=$3,CHANNEL=$4,WF=${5:-0.0},EPOCHS=$6 "$S"; }

# --- TGB wiki: remaining channels ---
sub hk_wiki_none tgb tgbl-wiki none 0.0 20
sub hk_wiki_both tgb tgbl-wiki both 0.0 20

# --- DGB uci: remaining channels ---
sub hk_uci_none dgb uci none 0.0 15
sub hk_uci_both dgb uci both 0.0 15

# --- DGB mooc / reddit / CanParl: full 4-way ---
for ds in mooc reddit CanParl; do
  for ch in none cooccur gev both; do sub hk_${ds}_$ch dgb $ds $ch 0.0 15; done
done

# --- boundary (degenerate) datasets: cooccur vs gev only ---
for ds in USLegis UNvote; do
  for ch in cooccur gev; do sub hk_${ds}_$ch dgb $ds $ch 0.0 15; done
done

# --- window ablation: wiki + uci gev at wf 0.01/0.05/0.10/0.30 ---
for wf in 0.01 0.05 0.10 0.30; do
  sub hk_wikiwf$wf tgb tgbl-wiki gev $wf 20
  sub hk_uciwf$wf  dgb uci       gev $wf 15
done
echo "remaining jobs submitted"

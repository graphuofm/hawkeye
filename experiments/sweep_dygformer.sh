#!/bin/bash
# DyGFormer-lite baseline + DyGFormer-lite + GraphEagleVision (different couplings).
set -u
PY=python
cd "$(dirname "$0")/.."
mkdir -p results/logs
EPOCHS=${EPOCHS:-15}; PATIENCE=${PATIENCE:-5}; SEED=${SEED:-0}; VAL_SUB=${VAL_SUB:-2000}
TREND="0.99,0.999,0.9999"
DATASETS=${DATASETS:-"tgbl-uci tgbl-enron tgbl-wiki tgbl-subreddit"}
OUT=results/dygformer_matrix.jsonl
log(){ echo "=== [$(date +%H:%M:%S)] $* ===" | tee -a results/dygformer.progress; }
R(){ local ds=$1 tag=$2; shift 2
  log "$ds | $tag | $*"
  $PY experiments/run_tgb_dygformer.py --dataset $ds --epochs $EPOCHS --patience $PATIENCE --seed $SEED \
      --no_download --val_subsample $VAL_SUB --trend_decays $TREND --out $OUT "$@" \
      > results/logs/dygf_${ds}_${tag}.log 2>&1
  grep RESULT results/logs/dygf_${ds}_${tag}.log | tee -a results/dygformer.progress; }

for ds in $DATASETS; do
  R $ds "baseline"    --gev_indicators ""
  R $ds "gev_gated"   --gev_indicators degree,core --fusion gated --pairwise_mode all
  R $ds "scoreens"    --gev_indicators degree,core --coupling score_ensemble
  R $ds "gev_attn"    --gev_indicators degree,core --fusion attn --pairwise_mode all
done
log "dygformer sweep done $(date)"

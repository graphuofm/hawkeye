#!/bin/bash
# "How to couple GraphEagleVision with a GNN matters" — sweep the coupling modes.
# TGN baseline + struct_only + 6 late/modulation couplings + aux, on a couple datasets.
# Results -> results/coupling_matrix.jsonl ; logs -> results/logs/couple_*.log
set -u
PY=python
cd "$(dirname "$0")/.."
mkdir -p results/logs
EPOCHS="${EPOCHS:-15}"; PATIENCE="${PATIENCE:-5}"; SEED="${SEED:-0}"; VAL_SUB="${VAL_SUB:-2000}"
TREND="${TREND:-0.99,0.999,0.9999}"
DATASETS="${DATASETS:-tgbl-uci tgbl-enron}"
OUT_TGN=results/coupling_matrix.jsonl
OUT_STR=results/coupling_matrix.jsonl
log() { echo "=== [$(date +%H:%M:%S)] $* ===" | tee -a results/coupling.progress; }

tgn() { local ds="$1" tag="$2"; shift 2
  log "tgn $ds | $tag | $*"
  $PY experiments/run_tgb_tgn.py --dataset "$ds" --epochs "$EPOCHS" --patience "$PATIENCE" --seed "$SEED" \
      --no_download --val_subsample "$VAL_SUB" --trend_decays "$TREND" --out "$OUT_TGN" "$@" \
      > "results/logs/couple_${ds}_${tag}.log" 2>&1
  grep -E "RESULT" "results/logs/couple_${ds}_${tag}.log" | tee -a results/coupling.progress
}
struct() { local ds="$1" tag="$2"; shift 2
  log "struct $ds | $tag | $*"
  $PY experiments/run_tgb.py --dataset "$ds" --epochs "$EPOCHS" --patience "$PATIENCE" --seed "$SEED" \
      --no_download --val_subsample "$VAL_SUB" --trend_decays "$TREND" --out "$OUT_STR" "$@" \
      > "results/logs/couple_${ds}_${tag}.log" 2>&1
  grep -E "RESULT" "results/logs/couple_${ds}_${tag}.log" | tee -a results/coupling.progress
}

for ds in $DATASETS; do
  struct "$ds" "structonly"       --indicators degree,core --pairwise_mode all       # pure structural endpoint
  tgn    "$ds" "tgn_baseline"     --gev_indicators ""                                 # pure TGN endpoint
  tgn    "$ds" "couple_scoreens"  --gev_indicators degree,core --coupling score_ensemble
  tgn    "$ds" "couple_concat"    --gev_indicators degree,core --fusion concat
  tgn    "$ds" "couple_additive"  --gev_indicators degree,core --fusion additive
  tgn    "$ds" "couple_gated"     --gev_indicators degree,core --fusion gated
  tgn    "$ds" "couple_attn"      --gev_indicators degree,core --fusion attn
  tgn    "$ds" "couple_film"      --gev_indicators degree,core --fusion film
  tgn    "$ds" "couple_aux"       --gev_indicators degree,core --coupling aux --fusion gated
done
log "coupling sweep done $(date)"

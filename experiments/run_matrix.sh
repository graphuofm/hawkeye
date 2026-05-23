#!/bin/bash
# Master experiment driver. Runs sequentially on one GPU (the bottleneck is CPU:
# structure maintenance + pairwise). Append-only JSON results under results/.
#
#   bash experiments/run_matrix.sh empirical   # empirical study (measurement, no training)
#   bash experiments/run_matrix.sh struct      # struct-only hierarchy + ablations
#   bash experiments/run_matrix.sh tgn         # TGN baseline + TGN+GEV (k-family x GNN coupling)
#   bash experiments/run_matrix.sh all
#
# Env overrides: EPOCHS PATIENCE SEED VAL_SUB DATASETS DENSE_DATASETS
set -u
PY=/home/jding/miniconda3/envs/tgnn/bin/python
cd "$(dirname "$0")/.."
mkdir -p results/logs results/empirical
WHICH="${1:-all}"
EPOCHS="${EPOCHS:-20}"; PATIENCE="${PATIENCE:-6}"; SEED="${SEED:-0}"; VAL_SUB="${VAL_SUB:-3000}"
TREND="${TREND:-0.99,0.999,0.9999}"   # multi-timescale trend decays (windowed-change features)
# datasets that the re-stream + (loop/sparse) pipeline handles in reasonable time
DATASETS="${DATASETS:-tgbl-uci tgbl-wiki tgbl-subreddit tgbl-enron tgbl-lastfm}"   # add tgbl-review for the full set (slow)
log() { echo "=== [$(date +%H:%M:%S)] $* ===" | tee -a results/matrix.progress; }

# ----------------------------------------------------------------------------
do_empirical() {
  for ds in $DATASETS; do
    log "empirical $ds"
    $PY analysis/empirical_study.py --dataset "$ds" --indicators degree,core,triangle \
        --no_download --val_subsample "$VAL_SUB" --pairwise_mode all --trend_decays "$TREND" \
        --out "results/empirical/${ds}.json" > "results/logs/empirical_${ds}.log" 2>&1
    grep -E "by indicator|static vs dynamic|saved" "results/logs/empirical_${ds}.log" | tee -a results/matrix.progress
  done
}

# ----------------------------------------------------------------------------
struct_run() {  # $1=dataset $2=tag ; rest=extra args
  local ds="$1" tag="$2"; shift 2
  log "struct $ds | $tag"
  $PY experiments/run_tgb.py --dataset "$ds" --epochs "$EPOCHS" --patience "$PATIENCE" --seed "$SEED" \
      --no_download --val_subsample "$VAL_SUB" --trend_decays "$TREND" --out results/struct_matrix.jsonl "$@" \
      > "results/logs/struct_${ds}_${tag}.log" 2>&1
  grep -E "RESULT" "results/logs/struct_${ds}_${tag}.log" | tee -a results/matrix.progress
}
do_struct() {
  for ds in $DATASETS; do
    for ind in degree core triangle; do
      struct_run "$ds" "${ind}_cohesion" --indicators "$ind" --pairwise_mode cohesion
      struct_run "$ds" "${ind}_all"      --indicators "$ind" --pairwise_mode all
    done
    struct_run "$ds" "truss_cohesion"  --indicators truss --pairwise_mode cohesion --truss_recompute_every 32
    struct_run "$ds" "generic_only"    --indicators core  --pairwise_mode generic
    struct_run "$ds" "deg_core_tri"    --indicators degree,core,triangle --pairwise_mode all
    struct_run "$ds" "core_static"     --indicators core  --pairwise_mode cohesion --stat_groups current
    struct_run "$ds" "core_dynamic"    --indicators core  --pairwise_mode cohesion --stat_groups ema,std,delta,max_change
    struct_run "$ds" "core_nopairwise" --indicators degree,core --no_pairwise
  done
}

# ----------------------------------------------------------------------------
tgn_run() {  # $1=dataset $2=tag ; rest=extra args
  local ds="$1" tag="$2"; shift 2
  log "tgn $ds | $tag"
  $PY experiments/run_tgb_tgn.py --dataset "$ds" --epochs "$EPOCHS" --patience "$PATIENCE" --seed "$SEED" \
      --no_download --val_subsample "$VAL_SUB" --trend_decays "$TREND" --out results/tgn_matrix.jsonl "$@" \
      > "results/logs/tgn_${ds}_${tag}.log" 2>&1
  grep -E "RESULT" "results/logs/tgn_${ds}_${tag}.log" | tee -a results/matrix.progress
}
do_tgn() {
  for ds in $DATASETS; do
    # baselines
    tgn_run "$ds" "baseline"          --gev_indicators ""
    # coupling comparison (the "how to couple matters" experiment), GEV = degree,core, pairwise=all
    tgn_run "$ds" "couple_score_ens"  --gev_indicators degree,core --coupling score_ensemble
    tgn_run "$ds" "couple_concat"     --gev_indicators degree,core --fusion concat
    tgn_run "$ds" "couple_additive"   --gev_indicators degree,core --fusion additive
    tgn_run "$ds" "couple_gated"      --gev_indicators degree,core --fusion gated
    tgn_run "$ds" "couple_attn"       --gev_indicators degree,core --fusion attn
    tgn_run "$ds" "couple_film"       --gev_indicators degree,core --fusion film
    tgn_run "$ds" "couple_aux"        --gev_indicators degree,core --coupling aux --fusion gated
    # GEV ablations under the best (gated) coupling
    tgn_run "$ds" "gated_cohesion"    --gev_indicators degree,core --fusion gated --pairwise_mode cohesion
    tgn_run "$ds" "gated_nopairwise"  --gev_indicators degree,core --fusion gated --no_pairwise
    tgn_run "$ds" "gated_coreonly"    --gev_indicators core        --fusion gated --pairwise_mode all
  done
}

case "$WHICH" in
  empirical) do_empirical ;;
  struct)    do_struct ;;
  tgn)       do_tgn ;;
  all)       do_empirical; do_struct; do_tgn ;;
  *) echo "usage: $0 {empirical|struct|tgn|all}"; exit 1 ;;
esac
log "matrix '$WHICH' done"

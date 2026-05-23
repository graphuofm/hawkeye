#!/bin/bash
# OVERNIGHT comprehensive sweep — aiming for CIKM-level coverage.
#
# Three pieces:
#   A) Multi-seed (2 extra seeds) on headline configs for the 4 covered datasets
#   B) Full ablation on tgbl-lastfm (complete graph, expected honest negative)
#   C) Single-config struct-only on tgbl-review (4.8M edges, big)
#
# Runs sequentially. ~10-12 h wall on a Quadro RTX 6000.
set -u
PY=/home/jding/miniconda3/envs/tgnn/bin/python
cd "$(dirname "$0")/.."
mkdir -p results/logs

EPOCHS=${EPOCHS:-15}; PATIENCE=${PATIENCE:-5}; VAL_SUB=${VAL_SUB:-2000}
TREND="0.99,0.999,0.9999"
PROG=results/overnight.progress
log(){ echo "=== [$(date +%H:%M:%S)] $* ===" | tee -a $PROG; }

# ------------------------------------------------------------------ #
# A) Multi-seed headline (seeds 1, 2) on wiki/subreddit/enron/uci
# ------------------------------------------------------------------ #
TGN_OUT=results/tgn_multiseed.jsonl
DGF_OUT=results/dygformer_multiseed.jsonl
STRUCT_OUT=results/struct_multiseed.jsonl

run_tgn(){ local ds=$1 seed=$2 tag=$3 coupling=$4 fusion=$5
  log "TGN+GEV $ds seed=$seed | $tag"
  $PY experiments/run_tgb_tgn.py --dataset $ds --epochs $EPOCHS --patience $PATIENCE --seed $seed \
      --no_download --val_subsample $VAL_SUB --trend_decays $TREND \
      --gev_indicators degree,core --pairwise_mode cohesion \
      --coupling $coupling --fusion $fusion --out $TGN_OUT \
      > results/logs/over_tgn_${ds}_${tag}_s${seed}.log 2>&1
  grep RESULT results/logs/over_tgn_${ds}_${tag}_s${seed}.log | tee -a $PROG; }

run_dgf(){ local ds=$1 seed=$2 tag=$3 coupling=$4 fusion=$5
  log "DyGF+GEV $ds seed=$seed | $tag"
  $PY experiments/run_tgb_dygformer.py --dataset $ds --epochs $EPOCHS --patience $PATIENCE --seed $seed \
      --no_download --val_subsample $VAL_SUB --trend_decays $TREND \
      --gev_indicators degree,core --pairwise_mode cohesion \
      --coupling $coupling --fusion $fusion --out $DGF_OUT \
      > results/logs/over_dgf_${ds}_${tag}_s${seed}.log 2>&1
  grep RESULT results/logs/over_dgf_${ds}_${tag}_s${seed}.log | tee -a $PROG; }

run_struct(){ local ds=$1 seed=$2 pwm=$3
  log "STRUCT-only $ds seed=$seed | pw=$pwm"
  $PY experiments/run_tgb.py --dataset $ds --epochs $EPOCHS --patience $PATIENCE --seed $seed \
      --no_download --val_subsample $VAL_SUB --trend_decays $TREND \
      --indicators degree,core --pairwise_mode $pwm --out $STRUCT_OUT \
      > results/logs/over_struct_${ds}_pw${pwm}_s${seed}.log 2>&1
  grep RESULT results/logs/over_struct_${ds}_pw${pwm}_s${seed}.log | tee -a $PROG; }

# A1: TGN+GEV gated cohesion (single best coupling), seeds {1,2}
for ds in tgbl-uci tgbl-wiki tgbl-enron; do
  for seed in 1 2; do
    run_tgn $ds $seed "gated_cohesion" fusion gated
  done
done

# A2: DyGF+GEV gated cohesion (the headline 22x lift config), seeds {1,2}
for ds in tgbl-uci tgbl-wiki tgbl-enron; do
  for seed in 1 2; do
    run_dgf $ds $seed "gated_cohesion" fusion gated
  done
done

# ------------------------------------------------------------------ #
# B) tgbl-lastfm full ablation (complete-graph honest negative result)
# ------------------------------------------------------------------ #
LASTFM_OUT=results/lastfm_matrix.jsonl
log "lastfm — START full ablation"
for pwm in none generic cohesion all; do
  $PY experiments/run_tgb.py --dataset tgbl-lastfm --epochs $EPOCHS --patience $PATIENCE --seed 0 \
      --no_download --val_subsample $VAL_SUB --trend_decays $TREND \
      --indicators degree,core --pairwise_mode $pwm --out $LASTFM_OUT \
      > results/logs/over_lastfm_pw${pwm}.log 2>&1
  grep RESULT results/logs/over_lastfm_pw${pwm}.log | tee -a $PROG
done
# TGN+GEV gated on lastfm too
$PY experiments/run_tgb_tgn.py --dataset tgbl-lastfm --epochs $EPOCHS --patience $PATIENCE --seed 0 \
    --no_download --val_subsample $VAL_SUB --trend_decays $TREND \
    --gev_indicators degree,core --pairwise_mode cohesion --coupling fusion --fusion gated \
    --out results/tgn_multiseed.jsonl \
    > results/logs/over_tgn_lastfm_gated.log 2>&1
grep RESULT results/logs/over_tgn_lastfm_gated.log | tee -a $PROG

# ------------------------------------------------------------------ #
# C) tgbl-review — single struct-only run (big dataset, coverage-only)
# ------------------------------------------------------------------ #
REVIEW_OUT=results/review_matrix.jsonl
log "review — struct-only cohesion (big)"
$PY experiments/run_tgb.py --dataset tgbl-review --epochs 10 --patience 4 --seed 0 \
    --no_download --val_subsample 1000 --trend_decays $TREND \
    --indicators degree,core --pairwise_mode cohesion --single_pass --out $REVIEW_OUT \
    > results/logs/over_review_struct.log 2>&1
grep RESULT results/logs/over_review_struct.log | tee -a $PROG

log "overnight sweep DONE $(date)"

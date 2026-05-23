#!/bin/bash
# More TGN-vs-TGN+GEV runs: the missing datasets (subreddit/enron/lastfm) with the
# key couplings, and multi-seed on uci/wiki.
set -u
PY=/home/jding/miniconda3/envs/tgnn/bin/python
cd "$(dirname "$0")/.."
mkdir -p results/logs
TREND="0.99,0.999,0.9999"; OUT=results/tgn_matrix.jsonl
log(){ echo "=== [$(date +%H:%M:%S)] $* ===" | tee -a results/tgnrest.progress; }
T(){ local ds=$1 tag=$2; shift 2; log "tgn $ds | $tag | $*"
  $PY experiments/run_tgb_tgn.py --dataset $ds --epochs 20 --patience 6 --no_download --val_subsample 3000 --trend_decays $TREND \
      --out $OUT "$@" > results/logs/tgnrest_${ds}_${tag}.log 2>&1
  grep RESULT results/logs/tgnrest_${ds}_${tag}.log | tee -a results/tgnrest.progress; }

# missing datasets: TGN baseline + gated(pw=all) + score_ensemble
for ds in tgbl-enron tgbl-lastfm tgbl-subreddit; do
  T $ds "tgn_baseline" --seed 0 --gev_indicators ""
  T $ds "gev_gated"    --seed 0 --gev_indicators degree,core --fusion gated --pairwise_mode all
  T $ds "scoreens"     --seed 0 --gev_indicators degree,core --coupling score_ensemble
done
# multi-seed on uci + wiki
for sd in 1 2; do
  T tgbl-uci  "s${sd}_tgn"      --seed $sd --gev_indicators ""
  T tgbl-uci  "s${sd}_gevgated" --seed $sd --gev_indicators degree,core --fusion gated --pairwise_mode all
  T tgbl-uci  "s${sd}_scoreens" --seed $sd --gev_indicators degree,core --coupling score_ensemble
  T tgbl-wiki "s${sd}_tgn"      --seed $sd --gev_indicators ""
  T tgbl-wiki "s${sd}_gevgated" --seed $sd --gev_indicators degree,core --fusion gated --pairwise_mode all
  T tgbl-wiki "s${sd}_gevcoh"   --seed $sd --gev_indicators degree,core --fusion gated --pairwise_mode cohesion
done
log "sweep_tgn_rest done $(date)"

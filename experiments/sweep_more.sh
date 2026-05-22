#!/bin/bash
# Extra runs to fill gaps while the main batches run: multi-seed key struct-only
# configs (pressure-test noise) + flight degree-only single-pass (scalability).
set -u
PY=python
cd "$(dirname "$0")/.."
mkdir -p results/logs
TREND="0.99,0.999,0.9999"; OUT=results/more_matrix.jsonl
log(){ echo "=== [$(date +%H:%M:%S)] $* ===" | tee -a results/more.progress; }
S(){ local ds=$1 tag=$2; shift 2; log "struct $ds | $tag | $*"
  $PY experiments/run_tgb.py --dataset $ds --no_download --val_subsample 3000 --trend_decays $TREND \
      --out $OUT "$@" > results/logs/more_${ds}_${tag}.log 2>&1
  grep RESULT results/logs/more_${ds}_${tag}.log | tee -a results/more.progress; }

# multi-seed key struct-only configs on tgbl-uci (degree,core indicators, 20 ep)
for sd in 1 2; do
  S tgbl-uci "s${sd}_pwall"  --seed $sd --indicators degree,core --epochs 20 --patience 6 --pairwise_mode all
  S tgbl-uci "s${sd}_pwcoh"  --seed $sd --indicators degree,core --epochs 20 --patience 6 --pairwise_mode cohesion
  S tgbl-uci "s${sd}_pwgen"  --seed $sd --indicators degree,core --epochs 20 --patience 6 --pairwise_mode generic
  S tgbl-uci "s${sd}_pwnone" --seed $sd --indicators degree,core --epochs 20 --patience 6 --no_pairwise
  S tgbl-uci "s${sd}_truss_coh" --seed $sd --indicators truss --epochs 20 --patience 6 --pairwise_mode cohesion --truss_recompute_every 32
done
# tgbl-flight degree-only single-pass (O(1) maintenance -> tractable on 67M edges; scalability point)
S tgbl-flight "deg_only_sp" --seed 0 --indicators degree --no_pairwise --single_pass --batch_size 5000 --val_subsample 3000 --test_subsample 5000
log "sweep_more done $(date)"

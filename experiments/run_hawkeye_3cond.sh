#!/bin/bash
# Hawkeye — 3-condition x 3-window sequential ablation on a temporal graph.
# One run at a time; each result is copied out the moment it finishes.
#   C1 window        : sliding window only      (degree)
#   C2 window+struct : window + k-core/k-truss  (degree,core,truss)
#   C3 pure-struct   : pure structure in window (core,truss)
# windows: 1% / 5% / 10% of the time span.  No cumulative graph.
set -u
PY=/home/jding/miniconda3/envs/tgnn/bin/python
ROOT=/home/jding/CIKM2026frp
cd "$ROOT/sota/DyGLib"
OUT="$ROOT/results/hawkeye_3cond"
mkdir -p "$OUT"
JDIR="saved_results/DyGFormer/uci"
DS=uci

run() {
  local cond=$1 ind=$2 wf=$3
  local tag="${cond}_wf${wf}"
  echo "=== [$(date +%H:%M:%S)] START ${tag}  (indicators=${ind}) ==="
  $PY train_link_prediction.py --dataset_name "$DS" --model_name DyGFormer \
    --patch_size 1 --max_input_sequence_length 32 --num_runs 1 --gpu 0 \
    --num_epochs 15 --patience 4 --structure_channel gev \
    --gev_indicators "$ind" --slot_backend full --gev_window_fraction "$wf" \
    > "$OUT/log_${tag}.txt" 2>&1
  local j
  j=$(ls -t "$JDIR"/*.json 2>/dev/null | head -1)
  if [ -n "$j" ]; then
    cp "$j" "$OUT/${tag}.json"
    local r
    r=$($PY -c "import json;d=json.load(open('$OUT/${tag}.json'));m=d['test metrics'];n=d['new node test metrics'];print('test AP=%s  new-node AP=%s'%(m.get('average_precision',m.get('mrr')),n.get('average_precision',n.get('mrr'))))" 2>/dev/null)
    echo "    [$(date +%H:%M:%S)] DONE  ${tag}  ${r}"
  else
    echo "    [$(date +%H:%M:%S)] ${tag}  NO RESULT -- see log_${tag}.txt"
  fi
}

for wf in 0.01 0.05 0.10; do
  run window     degree            "$wf"
  run winstruct  degree,core,truss "$wf"
  run purestruct core,truss        "$wf"
done
echo "=== ALL 9 RUNS DONE [$(date +%H:%M:%S)] ==="

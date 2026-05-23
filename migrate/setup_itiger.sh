#!/bin/bash
# Hawkeye — environment setup on the iTiger HPC cluster.
# Run this ONCE on the iTiger LOGIN node (it has internet) after rsync-ing the
# code over. It recreates the conda env and extracts the DGB data.
#
#   bash migrate/setup_itiger.sh
#
set -e
cd "$(dirname "$0")/.."           # project root on iTiger (e.g. ~/hawkeye)
ROOT=$(pwd)
echo "[setup] project root: $ROOT"

# --- 1. conda env -----------------------------------------------------------
# NOTE: the anaconda module name varies by cluster. Check with `module avail`.
# Common names: anaconda3 / miniconda3 / Anaconda3. Adjust the next line.
module load anaconda3 2>/dev/null || module load miniconda3 2>/dev/null || true

conda create -y -n hawkeye python=3.9
# shellcheck disable=SC1091
source activate hawkeye 2>/dev/null || conda activate hawkeye

# torch 2.5.1 + CUDA 12.1 (works on Ada / H100)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
# PyG core
pip install torch_geometric==2.6.1
# PyG companion wheels — matched to torch 2.5.1+cu121 (NO compilation)
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
# rest
pip install py-tgb==2.2.0 numpy==2.0.2 scipy==1.13.1 pandas==2.3.3 \
    scikit-learn==1.6.1 tqdm==4.67.1 matplotlib==3.9.4 networkx==3.2.1

echo "[setup] conda env 'hawkeye' ready."

# --- 2. DGB data ------------------------------------------------------------
# The DGB processed datasets (ml_*.csv / *.npy) go under sota/DyGLib/processed_data/
if [ -f "$ROOT/TG_network_datasets.zip" ]; then
  echo "[setup] extracting DGB datasets ..."
  mkdir -p /tmp/dgb_x && unzip -q -o "$ROOT/TG_network_datasets.zip" -d /tmp/dgb_x
  mkdir -p "$ROOT/sota/DyGLib/processed_data"
  for d in /tmp/dgb_x/TG_network_datasets/*/; do
    name=$(basename "$d")
    cp -r "$d" "$ROOT/sota/DyGLib/processed_data/$name"
  done
  echo "[setup] DGB data installed."
else
  echo "[setup] WARNING: TG_network_datasets.zip not found in project root."
  echo "        scp it over, or skip DGB runs."
fi

# --- 3. fix the GEV root path ----------------------------------------------
# DyGFormer.py and train scripts hardcode the project root. Patch to $ROOT.
grep -rl "/home/jding/CIKM2026frp" "$ROOT/sota" 2>/dev/null | while read -r f; do
  sed -i "s#/home/jding/CIKM2026frp#$ROOT#g" "$f"
done
echo "[setup] patched project-root paths to $ROOT."

echo "[setup] DONE. TGB datasets download automatically on first run."
echo "Next: edit migrate/slurm_hawkeye.sh for iTiger's partition, then sbatch."

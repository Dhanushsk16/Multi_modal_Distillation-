#!/usr/bin/env bash
# One-time, OFFLINE: the server has no internet, so everything comes from
# offline.tar (extracted to $DATA/cache/offline).
#
#   1. installs Miniconda (Python 3.11) into ~/miniconda3 if it is not there
#   2. creates the conda env "peptide" as an offline clone of base
#   3. pip-installs torch 2.6.0+cu124, transformers, ... from the local wheels
#
#   bash ~/code/shape_pilot/scripts/setup_env.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

OFF=$DATA/cache/offline
INSTALLER=$(ls "$OFF"/Miniconda3-py311_*-Linux-x86_64.sh 2>/dev/null | head -1 || true)
test -d "$OFF/wheels" || { echo "missing $OFF/wheels -- extract offline.tar into $DATA/cache"; exit 1; }

# --- 1. Miniconda ------------------------------------------------------------
if [ ! -x "$HOME/miniconda3/bin/conda" ]; then
  test -n "$INSTALLER" || { echo "no Miniconda installer in $OFF"; exit 1; }
  echo "== installing Miniconda into ~/miniconda3"
  bash "$INSTALLER" -b -p "$HOME/miniconda3"
  "$HOME/miniconda3/bin/conda" init bash
fi
source "$HOME/miniconda3/etc/profile.d/conda.sh"
# Newer conda asks to accept channel Terms of Service; nothing is downloaded
# here, but the check still runs and would stop a non-interactive script.
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
# The offline clone needs the packages the installer unpacked into
# ~/miniconda3/pkgs, so keep that directory visible alongside the data-drive cache.
export CONDA_PKGS_DIRS="$CONDA_PKGS_DIRS,$HOME/miniconda3/pkgs"

# --- 2. env ------------------------------------------------------------------
if conda env list | grep -qE "^peptide\s"; then
  echo "== env 'peptide' already exists"
else
  echo "== creating env 'peptide' (offline clone of base)"
  conda create -y -n peptide --clone base --offline
fi
conda activate peptide

# --- 3. packages from local wheels -------------------------------------------
echo "== installing packages from $OFF/wheels"
pip install --no-index --find-links "$OFF/wheels" \
    "torch==2.6.0+cu124" "transformers>=4.46,<5" safetensors numpy pandas pyarrow

python - <<'EOF'
import torch, transformers
print("torch", torch.__version__, "| transformers", transformers.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), "| bf16:", torch.cuda.is_bf16_supported())
EOF
echo
echo "done. Open a new shell (or: source ~/.bashrc), then: conda activate peptide"

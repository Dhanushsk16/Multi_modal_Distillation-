#!/usr/bin/env bash
# One-time, OFFLINE (the server has no internet). Wheels come from
# $DATA/cache/offline2/, extracted from offline2.tar.
#
#   1. adds RDKit, scikit-learn, SciPy to the existing "peptide" env
#   2. creates the "pepland" env (offline clone of base, Python 3.11) with
#      CPU PyTorch 2.2.2, DGL, MLflow and RDKit -- PepLand cannot share an env
#      with PyTorch 2.6
#   3. checks both, including loading the PepLand weights
#
#   bash ~/code/peptidefusion/scripts/setup_envs.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"

W=$DATA/cache/offline2
test -d "$W/pepland" || { echo "missing $W -- extract offline2.tar into $DATA/cache"; exit 1; }
source "$HOME/miniconda3/etc/profile.d/conda.sh"
export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
export CONDA_PKGS_DIRS="$CONDA_PKGS_DIRS,$HOME/miniconda3/pkgs"

echo "== 1. peptide env: + rdkit scikit-learn scipy"
conda activate peptide
pip install -q --no-index --find-links "$W/peptide_extra" rdkit scikit-learn scipy
python -c "import rdkit, sklearn, scipy, torch; print('   peptide ok | rdkit', rdkit.__version__, '| sklearn', sklearn.__version__, '| torch', torch.__version__)"
conda deactivate

echo "== 2. pepland env"
if conda env list | grep -qE "^pepland\s"; then
  echo "   already exists"
else
  conda create -y -q -n pepland --clone base --offline > /dev/null
fi
conda activate pepland
pip install -q --no-index --find-links "$W/pepland" \
    torch==2.2.2 dgl rdkit "numpy<2" pandas scikit-learn scipy pyyaml mlflow-skinny "pydantic<3"

echo "== 3. PepLand weights load and embed"
python - <<'EOF'
import os, sys
sys.path.insert(0, os.environ["CODE"] + "/external")
import torch
from pepland.model.core import PepLandFeatureExtractor
e = PepLandFeatureExtractor(model_path=os.environ["CKPT_DIR"] + "/pepland_model",
                            pooling="avg", padding_mode="exclude"); e.eval()
with torch.no_grad():
    x = e(["N[C@@H](C)C(=O)N[C@@H](Cc1ccccc1)C(=O)O"])
print("   pepland ok | torch", torch.__version__, "| embedding", tuple(x.shape),
      "| finite", bool(torch.isfinite(x).all()))
EOF
echo "done"

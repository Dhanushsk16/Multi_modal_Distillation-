# Sourced by every script here. Keeps all runtime writes on the data drive and
# CPU use under the lab's 50% limit.
export DATA=${BTP_DATA:-/mnt/wwn-0x5000c500e64c5cc0/dhanushsk/BTPData}
export BTP_DATA=$DATA
export CKPT_DIR=$DATA/checkpoints
export RESULTS_DIR=$DATA/results
export LOG_DIR=$DATA/logs

# Caches off home (40 GB cap). trust_remote_code models copy their code into
# HF_HOME/modules, so this matters even with no downloads.
export HF_HOME=$DATA/cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=$DATA/cache/torch
export PIP_CACHE_DIR=$DATA/cache/pip
export CONDA_PKGS_DIRS=$DATA/cache/conda_pkgs
export XDG_CACHE_HOME=$DATA/cache/xdg

# CPU etiquette: at most half the machine.
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export RAYON_NUM_THREADS=8          # HF fast tokenizer thread pool
export TOKENIZERS_PARALLELISM=true

export CUDA_VISIBLE_DEVICES=0
export CODE=${CODE:-$HOME/code/shape_pilot}
export PYTHONPATH=$CODE/distill${PYTHONPATH:+:$PYTHONPATH}

mkdir -p "$CKPT_DIR" "$RESULTS_DIR" "$LOG_DIR" "$HF_HOME" "$TORCH_HOME" \
         "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$XDG_CACHE_HOME"

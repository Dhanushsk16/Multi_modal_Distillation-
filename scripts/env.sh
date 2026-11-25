# Sourced by every script. All runtime writes go to the data drive; CPU use stays
# under the lab's 50% limit.
export DATA=${BTP_DATA:-/mnt/wwn-0x5000c500e64c5cc0/dhanushsk/BTPData}
export BTP_DATA=$DATA
export CKPT_DIR=$DATA/checkpoints
export RESULTS_DIR=$DATA/results
export LOG_DIR=$DATA/logs

export HF_HOME=$DATA/cache/huggingface
export TORCH_HOME=$DATA/cache/torch
export PIP_CACHE_DIR=$DATA/cache/pip
export CONDA_PKGS_DIRS=$DATA/cache/conda_pkgs
export XDG_CACHE_HOME=$DATA/cache/xdg
export MPLCONFIGDIR=$DATA/cache/matplotlib
export DGLBACKEND=pytorch
export DGLDEFAULTDIR=$DATA/cache/dgl        # DGL writes ~/.dgl otherwise

# CPU threads for every step. Lab rule: at most half the machine (check: nproc).
#   THREADS=16 bash run_single.sh
export THREADS=${THREADS:-8}
export OMP_NUM_THREADS=$THREADS
export MKL_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=$THREADS
export RAYON_NUM_THREADS=$THREADS
export CUDA_VISIBLE_DEVICES=0

export CODE=${CODE:-$HOME/code/peptidefusion}
mkdir -p "$CKPT_DIR" "$RESULTS_DIR" "$LOG_DIR" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" \
         "$CONDA_PKGS_DIRS" "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$DGLDEFAULTDIR"

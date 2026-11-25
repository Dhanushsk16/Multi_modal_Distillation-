#!/usr/bin/env bash
# Pooling pilot: does average pooling throw away information? Mean / max / attention (/ GRU) pooling over the
# token or node vectors of the FROZEN encoders, same head and same splits, trained with labels on the benchmark
# molecules only. Decision rule: a learned pooler that gains clearly (more than ~0.03 on most benchmarks) is
# worth re-caching the corpus with; otherwise it is not. See src/finetune/pool_probe.py for the details.
#
#   pclm     PeptideCLM-2 token vectors (extracted on the fly) on the 4 main benchmarks and PepLand's 5 tasks.
#            GPU, light (a few GB); the GPU must be free (checked). poolers: mean max meanmax attn.
#   nodes    PepLand atom and fragment node vectors for the same molecules (pepland env, CPU, 8 threads). Same cost
#            as the earlier PepLand feature runs (30-60 min); skipped if the files exist.
#   pepland  PepLand pooling probe on those nodes: avg max attn gru (GPU, light).
#   all      pclm, nodes, pepland
#
#   tmux new -s pool
#   bash ~/code/peptidefusion/scripts/run_pool_probe.sh pclm        # first: this is the PeptideCLM-2 question
#   bash ~/code/peptidefusion/scripts/run_pool_probe.sh nodes       # CPU; can run while something else uses the GPU
#   bash ~/code/peptidefusion/scripts/run_pool_probe.sh pepland
#   SEEDS=1 bash ...     a quick look (default 3 seeds);   FORCE=1 skips the GPU check
# Finished benchmark x pooler pairs are kept (raw.csv), so re-running continues where it stopped.
# Logs: $LOG_DIR/pool_*.log. Results: results/pool_pclm_main, pool_pclm_pl, pool_pepland_main, pool_pepland_pl
# (raw.csv, attn_examples.csv = token weights of a few test molecules for the attention pooler).
set -uo pipefail
source "$(dirname "$0")/env.sh"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

STAGE=${1:-all}
LOG=$LOG_DIR/pool_${STAGE}_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "== $(date) | stage $STAGE | log: $LOG"
S=$CODE/src
MAIN4="CellPPD THPep PepMSND PAMPA"
PL="PL_cCPP PL_cSol PL_ncCPP PL_cBinding PL_ncBinding"
SEEDS=${SEEDS:-3}
export OMP_NUM_THREADS=8

gpu_free() {   # 0 = free (or FORCE=1), 1 = someone else is on the GPU
  local busy
  busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v dorado || true)
  if [ -n "$busy" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "GPU in use by:"; for pid in $(echo "$busy" | cut -d, -f1); do ps -o user=,etime=,args= -p "$pid"; done
    echo "not starting (FORCE=1 to override)"; return 1
  fi
}

pclm() {
  gpu_free || return 2
  conda activate peptide
  echo; echo "== pclm pooling, main benchmarks ($(date +%H:%M))"
  python "$S/finetune/pool_probe.py" --model pclm --bench $MAIN4 --seeds "$SEEDS" --out pool_pclm_main || return 1
  echo; echo "== pclm pooling, PepLand's five tasks ($(date +%H:%M))"
  python "$S/finetune/pool_probe.py" --model pclm --bench $PL --seeds "$SEEDS" --out pool_pclm_pl
}

nodes() {
  conda activate pepland
  echo; echo "== PepLand node vectors, main benchmarks ($(date +%H:%M))"
  python "$S/featurize_pepland_nodes.py" --bench $MAIN4 --threads 8 || return 1
  echo; echo "== PepLand node vectors, PepLand's five tasks ($(date +%H:%M))"
  python "$S/featurize_pepland_nodes.py" --feat pepland_suite --bench $PL --threads 8
}

pepland() {
  gpu_free || return 2
  conda activate peptide
  echo; echo "== PepLand pooling, main benchmarks ($(date +%H:%M))"
  python "$S/finetune/pool_probe.py" --model pepland --bench $MAIN4 --seeds "$SEEDS" --out pool_pepland_main || return 1
  echo; echo "== PepLand pooling, PepLand's five tasks ($(date +%H:%M))"
  python "$S/finetune/pool_probe.py" --model pepland --feat pepland_suite --bench $PL --seeds "$SEEDS" --out pool_pepland_pl
}

case "$STAGE" in
  pclm)    pclm ;;
  nodes)   nodes ;;
  pepland) pepland ;;
  all)     pclm; nodes; pepland ;;
  *)       echo "usage: run_pool_probe.sh pclm|nodes|pepland|all"; exit 1 ;;
esac
echo; echo "== done $(date +%H:%M)"

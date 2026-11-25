#!/usr/bin/env bash
# Single-modality check: featurise every benchmark molecule four ways, then score
# each representation (and their naive concatenation) with the same learners.
#
#   tmux new -s single
#   bash ~/code/peptidefusion/scripts/run_single.sh
#   THREADS=16 bash ~/code/peptidefusion/scripts/run_single.sh   # more CPU (<= half of nproc)
#
# Order and rough time on the lab server (8 CPU threads, A6000):
#   ecfp     CPU   ~1 min
#   mapc     CPU   ~1 h      (AmpHGT's large peptides cost ~1 s each)
#   pclm     GPU   minutes   (PeptideCLM-2 mlm-large, bf16)
#   pepland  CPU   ~1-2 h    (own conda env)
#   eval     CPU   ~1-2 h
#
# Safe to re-run: finished feature files are skipped. Stops at 06:45
# (DEADLINE=HH:MM); re-run the next night to continue. FORCE=1 skips the GPU check.
set -uo pipefail
source "$(dirname "$0")/env.sh"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

LOG=$LOG_DIR/single_modality_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "== $(date) | log: $LOG | CPU threads: $THREADS of $(nproc --all)"
if [ "$THREADS" -gt $(( $(nproc --all) / 2 )) ]; then echo "THREADS=$THREADS is more than half of $(nproc --all) -- lab rule, not starting"; exit 2; fi

# --- data present? ---------------------------------------------------------------
for need in "$DATA/data/benchmarks/amp_test.csv" "$DATA/data/benchmarks/PAMPA_clusters.csv" \
            "$CKPT_DIR/peptideclm-2-mlm-large/model.safetensors" "$CKPT_DIR/pepland_model/MLmodel"; do
  test -e "$need" || { echo "missing $need -- extract single_modality_data.tar into $DATA"; exit 1; }
done

# --- GPU check (dorado_basecall_server is a permanent idle service: ignored) ------
busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v dorado || true)
if [ -n "$busy" ] && [ "${FORCE:-0}" != 1 ]; then
  echo "GPU in use by:"
  for pid in $(echo "$busy" | cut -d, -f1); do ps -o user=,etime=,args= -p "$pid"; done
  echo "not starting (FORCE=1 to override)"; exit 2
fi

# --- time budget -----------------------------------------------------------------
DEADLINE=${DEADLINE:-06:45}
now=$(date +%s); stop=$(date -d "$DEADLINE" +%s)
[ "$stop" -le "$now" ] && stop=$(date -d "tomorrow $DEADLINE" +%s)
left() { echo $(( stop - $(date +%s) )); }
step() {   # name env command...
  local name=$1 env=$2; shift 2
  local t=$(left)
  if [ "$t" -le 60 ]; then echo "== $name: past $DEADLINE, stopping; re-run to continue"; exit 0; fi
  echo; echo "== $name  ($(date +%H:%M), env $env)"
  conda activate "$env"
  timeout --signal=INT --kill-after=60 "$t" "$@"
  local rc=$?
  conda deactivate
  if [ $rc = 124 ] || [ $rc = 130 ]; then echo "== $name stopped at $DEADLINE; re-run to continue"; exit 0; fi
  [ $rc = 0 ] || { echo "== $name FAILED (exit $rc)"; exit $rc; }
}

step ecfp    peptide python "$CODE/src/featurize.py" --rep ecfp --workers "$THREADS"
step mapc    peptide python "$CODE/src/featurize.py" --rep mapc --workers "$THREADS"
step pclm    peptide python "$CODE/src/featurize.py" --rep pclm
step pepland pepland python "$CODE/src/featurize_pepland.py" --threads "$THREADS"
step eval    peptide python "$CODE/src/eval_single.py"

echo; echo "== finished $(date) | results in $RESULTS_DIR/single_modality"

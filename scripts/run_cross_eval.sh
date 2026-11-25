#!/usr/bin/env bash
# Cross-evaluation of the two peptide encoders on each other's benchmarks, frozen features, same learners:
#   A. PepLand and PeptideCLM-2 on PeptideCLM-2's suite (CellPPD, THPep, PepMSND, PAMPA, AmpHGT)
#      uses the features that already exist (features/single_modality)      -> results/cross_main
#   B. the same two on PepLand's suite (c-CPP, c-Sol, nc-CPP, c-binding, nc-binding)
#      needs their features first (ECFP, PeptideCLM-2, PepLand)             -> results/cross_pepland
#
#   tmux new -s cross
#   bash ~/code/peptidefusion/scripts/run_cross_eval.sh
#   LEARNERS="linear" bash ~/code/peptidefusion/scripts/run_cross_eval.sh     # quick look, no trees / MLP
#
# Rough time (unmeasured): ECFP 1 min, PeptideCLM-2 on the GPU a few minutes, PepLand 30-60 min on 8 CPU
# threads (the slow part), then the evaluations 30-60 min. Steps whose output exists are skipped, so it is
# safe to stop and re-run. Never more than 8 CPU threads at a time. Refuses if someone else is on the GPU
# (FORCE=1 overrides). Log: $LOG_DIR/cross_eval_*.log
set -uo pipefail
source "$(dirname "$0")/env.sh"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

LOG=$LOG_DIR/cross_eval_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "== $(date) | log: $LOG"

SUITE="PL_cCPP PL_cSol PL_ncCPP PL_cBinding PL_ncBinding"
LEARNERS=${LEARNERS:-"linear gbt mlp"}
S=$CODE/src

busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v dorado || true)
if [ -n "$busy" ] && [ "${FORCE:-0}" != 1 ]; then
  echo "GPU in use by:"
  for pid in $(echo "$busy" | cut -d, -f1); do ps -o user=,etime=,args= -p "$pid"; done
  echo "not starting (FORCE=1 to override)"; exit 2
fi
for need in "$DATA/features/single_modality/ecfp.npz" "$DATA/features/single_modality/pclm.npz" \
            "$DATA/features/single_modality/pepland.npz" "$DATA/data/benchmarks/pepland/c-CPP.txt"; do
  test -e "$need" || { echo "missing $need"; exit 1; }
done

step() {   # name env command...
  local name=$1 env=$2; shift 2
  echo; echo "== $name  ($(date +%H:%M), env $env)"
  conda activate "$env"
  "$@"
  local rc=$?
  conda deactivate
  [ $rc = 0 ] || { echo "== $name FAILED (exit $rc)"; exit $rc; }
}

step "features: ECFP, PepLand suite"  peptide python "$S/featurize.py" --rep ecfp --bench $SUITE --feat pepland_suite --workers 8
step "features: PeptideCLM-2, PepLand suite" peptide python "$S/featurize.py" --rep pclm --bench $SUITE --feat pepland_suite
step "features: PepLand, PepLand suite (the slow one)" pepland python "$S/featurize_pepland.py" --bench $SUITE --feat pepland_suite --threads 8

step "A: PeptideCLM-2 suite, ECFP / PeptideCLM-2 / PepLand" peptide \
     python "$S/eval_single.py" --reps ecfp pclm pepland --learners $LEARNERS --out cross_main
step "B: PepLand suite, ECFP / PeptideCLM-2 / PepLand" peptide \
     python "$S/eval_single.py" --feat pepland_suite --bench $SUITE --reps ecfp pclm pepland \
     --learners $LEARNERS --out cross_pepland

echo; echo "== done $(date +%H:%M): results in $RESULTS_DIR/cross_main and $RESULTS_DIR/cross_pepland"

#!/usr/bin/env bash
# The PeptideCLM-2 authors' PepMSND post-training (train_pepmsnd_kan_paperstyle.py: KAN branch on the 140 descriptors + KAN fusion, backbone
# frozen for 3 epochs then trained, lr 1e-5 backbone / 1e-3 head, <= 120 epochs, patience 20 on the best-threshold MCC), with our TEACHER
# (default v3_mtr = the four-input one) as the backbone. src/teacher/pepmsnd_ft_kan_v3.py has the details; this script only runs it.
#
#   bash ~/code/peptidefusion/scripts/run_teacher_ft_kan.sh                      # v3_mtr, seed 103, 10 folds, BOTH selection rules (test, inner)
#   NAME=v3_base bash ...                                                         # another teacher
#   CONTROLS="frozen scratch" bash ...                                            # also: teacher never trained / random weights, same recipe
#   SELECTS=test FOLDS="1 2" bash ...                                             # a quick look
# Settings (environment): NAME=v3_mtr  SEEDS="103"  FOLDS="1 ... 10"  SELECTS="test inner"  CONTROLS=""  NEED_MB=4000 (free GPU memory to wait for)
#   SELECTS: test = the authors' protocol (the TEST fold picks the epoch: optimistic, comparable with the published 0.729 / 0.741 / 0.749);
#            inner = 10% of the training rows pick it (no test leak: the honest number).
# Needs `lightning` in the active env only because the authors' script file is imported for its classes (installed already for the repro).
# Results: results/pepmsnd_ft_kan_<select>/<label>/seed<S>/fold<k>/preds_fold<k>.csv; summaries are printed at the end and kept in the log.
set -uo pipefail
source "$(dirname "$0")/env.sh"
NAME=${NAME:-v3_mtr}
SEEDS=${SEEDS:-"103"}
FOLDS=${FOLDS:-"1 2 3 4 5 6 7 8 9 10"}
SELECTS=${SELECTS:-"test inner"}
CONTROLS=${CONTROLS:-""}
NEED_MB=${NEED_MB:-4000}
LOG=$LOG_DIR/teacher_ft_kan_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
say() { echo "$(date +%H:%M:%S) $*"; }
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate peptide
S=$CODE/src/teacher/pepmsnd_ft_kan_v3.py
say "== teacher $NAME | selections: $SELECTS | controls: ${CONTROLS:-none} | seeds: $SEEDS | folds: $FOLDS | log $LOG"
python -c "import lightning" 2> /dev/null || { say "!! 'import lightning' fails in this env: the authors' script file is imported for its classes and needs it"; exit 3; }
[ -f "$CKPT_DIR/teacher/$NAME/best.pt" ] || [ -f "$CKPT_DIR/teacher/$NAME/last.pt" ] || { say "!! no teacher checkpoint in $CKPT_DIR/teacher/$NAME"; exit 3; }

wait_gpu() {                                          # only the free memory counts
  local free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2> /dev/null | head -1 | tr -d ' ')
    [ -n "${free:-}" ] && [ "$free" -ge "$1" ] && break
    say "waiting for the GPU: ${free:-?} MB free, $1 MB needed (checking every 60 s)"; sleep 60
  done
}

say "step 0: does every molecule of the authors' folds have a vector in every view of the teacher?"
python "$S" --name "$NAME" --check --folds $FOLDS || { say "!! coverage check failed"; exit 4; }

for sel in $SELECTS; do
  for ctl in main $CONTROLS; do
    case $ctl in
      main) extra="";;
      frozen) extra="--freeze-epochs 100000";;
      scratch) extra="--init scratch";;
      *) say "!! unknown control $ctl (frozen | scratch)"; continue;;
    esac
    wait_gpu "$NEED_MB"
    say "=== $NAME | selection $sel | $ctl ==="
    # shellcheck disable=SC2086
    python "$S" --name "$NAME" --select "$sel" --seeds $SEEDS --folds $FOLDS $extra --no-summary || say "!! $sel / $ctl ended with status $?"
  done
done

say "=== SUMMARY ==="
for sel in $SELECTS; do
  say "--- selection $sel: results/pepmsnd_ft_kan_$sel ---"
  [ -d "$RESULTS_DIR/pepmsnd_ft_kan_$sel" ] && python "$CODE/src/repro/pepmsnd_summary.py" --root "$RESULTS_DIR/pepmsnd_ft_kan_$sel"
done
say "== done."

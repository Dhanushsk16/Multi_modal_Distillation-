#!/usr/bin/env bash
# Evaluate ONE student with everything: the best weights (EPOCH=0) or a snapshot written by train_student.py --save-every (EPOCH=2 -> epoch_02.pt).
# A NEW file: it only calls the other scripts, it does not touch run_student.sh (which may be running a training), so it is safe to use any time.
#
#   RUN=st6f EPOCH=2 bash ~/code/peptidefusion/scripts/run_student_eval.sh export    # CPU, ~1-2 min: the benchmark embeddings of the snapshot (name st6f_e02, st6f_e02_pre)
#   RUN=st6f EPOCH=2 bash ~/code/peptidefusion/scripts/run_student_eval.sh harness   # CPU: frozen output, learners linear kernel gbt mlp (2 seeds for gbt / mlp), 10 benchmarks
#   RUN=st6f EPOCH=2 bash ~/code/peptidefusion/scripts/run_student_eval.sh ft        # GPU (~3 GB): fine-tuned student + MLP head (student lr 1e-5, head lr 1e-3) AND the frozen-student MLP-head control
#   RUN=st6f EPOCH=2 bash ~/code/peptidefusion/scripts/run_student_eval.sh kan       # GPU: PepMSND with the authors' KAN recipe: authors' protocol, no-leak protocol, frozen control
#
# Results: results/harness_<NAME>_all/ (python src/student/show_bench.py <NAME> --learner ...), results/finetune_student_<NAME>/ (summary.csv, raw.csv),
#          results/pepmsnd_ft_kan_student_{test,inner}/<NAME>[_frozen]/ (summary.csv). NAME = RUN, or RUN_eNN for a snapshot.
# Settings (environment): LEARNERS="linear kernel gbt mlp"  SEEDS=2  BOOT=1000  WORKERS_H=4  HEXTRA="--reps P L H" (the harness's other inputs are not rerun)
#   FT_SEEDS=1  INNER=5  MAX_MODELS=5 (models per test fold)   KAN_SEEDS="103"   Extra arguments of ft / kan are passed on, e.g.
#   RUN=st6f EPOCH=2 bash .../run_student_eval.sh ft --bench CellPPD --epochs 3 --inner 2        (a smoke test)
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
STAGE=${1:?usage: RUN=st6f EPOCH=2 bash run_student_eval.sh export|harness|ft|kan}
shift
RUN=${RUN:?set RUN=<student run>, e.g. RUN=st6f}
EPOCH=${EPOCH:-0}
NAME=$RUN
[ "$EPOCH" -gt 0 ] && NAME=$(printf '%s_e%02d' "$RUN" "$EPOCH")
LOG=$LOG_DIR/student_eval_${STAGE}_${NAME}_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
say() { echo "$(date +%H:%M:%S) $*"; }
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate peptide
S=$CODE/src/student
say "== stage $STAGE | student $RUN | epoch $EPOCH (name $NAME) | log $LOG"

wait_gpu() {                                          # only the free memory counts
  local free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2> /dev/null | head -1 | tr -d ' ')
    [ -n "${free:-}" ] && [ "$free" -ge "$1" ] && break
    say "waiting for the GPU: ${free:-?} MB free, $1 MB needed (checking every 60 s)"; sleep 60
  done
}

rc=0
case "$STAGE" in
  export)
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python "$S/export_student.py" --name "$RUN" --epoch "$EPOCH" --workers 4
    rc=$?
    ;;
  harness)
    HEXTRA=${HEXTRA:-"--reps P L H"}
    OUT=${OUT:-harness_${NAME}_all} LEARNERS="${LEARNERS:-linear kernel gbt mlp}" SEEDS="${SEEDS:-2}" BOOT="${BOOT:-1000}" WORKERS_H="${WORKERS_H:-4}" HEXTRA="$HEXTRA" \
      bash "$HERE/run_student.sh" harness "$NAME" "${NAME}_pre"
    rc=$?
    ;;
  ft)
    wait_gpu "${NEED_MB:-4000}"
    OMP_NUM_THREADS=4 python "$S/finetune_student.py" --run "$RUN" --epoch "$EPOCH" --seeds "${FT_SEEDS:-1}" --inner "${INNER:-5}" --max-models "${MAX_MODELS:-5}" --workers 4 "$@"
    rc=$?
    ;;
  kan)
    wait_gpu "${NEED_MB:-4000}"
    export OMP_NUM_THREADS=4
    python "$S/pepmsnd_kan_student.py" --run "$RUN" --epoch "$EPOCH" --check || exit 1
    for sel in test inner; do
      python "$S/pepmsnd_kan_student.py" --run "$RUN" --epoch "$EPOCH" --select "$sel" --seeds ${KAN_SEEDS:-103} "$@" || rc=1
    done
    python "$S/pepmsnd_kan_student.py" --run "$RUN" --epoch "$EPOCH" --select inner --freeze-epochs 100000 --seeds ${KAN_SEEDS:-103} "$@" || rc=1
    ;;
  *) echo "unknown stage $STAGE"; rc=2;;
esac
say "== $STAGE finished with exit status $rc"
exit $rc

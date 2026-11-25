#!/usr/bin/env bash
# THPep with the JOURNAL PeptideCLM-2 paper's protocol (J. Chem. Inf. Model. 2026, 66, 8110): random 5-fold cross-validation over ALL 609
# molecules, three repetitions (benchmarks THPep_cv0 / _cv1 / _cv2, src/benchmarks.py), scored for the frozen v3_base teacher with every
# learner (linear, gbt, mlp, kernel) next to the single inputs (ECFP, PeptideCLM-2, PepLand, the three joined, PCA), and for the FINE-TUNED
# teacher (finetune_v3.py, one run per repetition). The last step prints the mean +- sd over the three repetitions next to the paper's numbers.
#
#   bash ~/code/peptidefusion/scripts/run_thpep_cv.sh                  # about 30-45 minutes, almost all of it gbt
#   TEACHER=v3_base THREADS=4 FINETUNE=1 bash ~/code/peptidefusion/scripts/run_thpep_cv.sh
# Settings (environment): TEACHER=v3_base  LEARNERS="linear gbt mlp kernel"  THREADS=4 (CPU threads, in 2 workers: small on purpose, so it can run
#   next to the first teacher training, before the benchmarking queue of run_variants_v3.sh starts)  FINETUNE=1 (GPU, < 1 GB)  SKIP_DONE=1
# Results: results/thpep_cv_<learner> (harness), results/thpep_cv_ft (fine-tuning); the summary is printed and kept in logs/thpep_cv_*.log
set -uo pipefail
source "$(dirname "$0")/env.sh"
SELF=$(cd "$(dirname "$0")" && pwd)
R=$SELF/run_teacher_v3.sh
TEACHER=${TEACHER:-v3_base}
LEARNERS=${LEARNERS:-"linear gbt mlp kernel"}
THREADS=${THREADS:-4}
WORKERS=${WORKERS:-$((THREADS / 2))}
FINETUNE=${FINETUNE:-1}
SKIP_DONE=${SKIP_DONE:-1}
PREFIX=thpep_cv
BENCH="THPep_cv0 THPep_cv1 THPep_cv2"
LOG=$LOG_DIR/thpep_cv_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
say() { echo "$(date +%H:%M:%S) $*"; }
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate peptide
say "== THPep, journal protocol (random 5-fold over all 609 molecules x 3 repetitions) | teacher $TEACHER | learners $LEARNERS | threads $THREADS ($WORKERS workers) | log $LOG"

for L in $LEARNERS; do
  if [ "$SKIP_DONE" = 1 ] && [ -f "$RESULTS_DIR/${PREFIX}_${L}/headline.csv" ]; then say "${PREFIX}_${L}: already done, skipped"; continue; fi
  say "harness: $L"
  env THREADS=$THREADS BENCH="$BENCH" TEACHERS="$TEACHER" LEARNERS=$L SEEDS=3 BOOT=0 WORKERS=$WORKERS OUT=${PREFIX}_${L} \
    EXTRA="--reps P L H P+L+H PCA --no-e-pass" bash "$R" harness || say "!! harness $L FAILED"
done

if [ "$FINETUNE" = 1 ]; then
  say "fine-tuning $TEACHER on the three repetitions (one run each)"
  # shellcheck disable=SC2086
  bash "$R" finetune "$TEACHER" --bench $BENCH --seeds 1 --out ${PREFIX}_ft --resume || say "!! fine-tuning FAILED"
fi

say "== SUMMARY"
python "$CODE/src/teacher/thpep_cv_summary.py" --prefix "$PREFIX" --teacher "$TEACHER" --learners $LEARNERS

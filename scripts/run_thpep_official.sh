#!/usr/bin/env bash
# THPep on the splits the PeptideCLM-2 authors ACTUALLY used (J. Chem. Inf. Model. 2026, 66, 8110): three splits (seeds 101 / 202 / 303), each with 122 test
# molecules, recovered from the result files of their repository (src/build_thpep_official_splits.py; benchmarks THPep_o101 / _o202 / _o303 in
# src/benchmarks.py). Two parts, then one summary (mean +- sd over the three splits):
#   1. FROZEN embeddings of the teacher v3_base and the student st6f (epoch 2) next to the single inputs P, L, H, read by every sklearn learner (linear,
#      kernel, gbt, mlp). CPU only, about 10-20 minutes.
#   2. FINE-TUNING (FINETUNE=1, the default): the teacher and the student trained end to end with the labels (`ft`) and the paired control with the weights
#      fixed and the same head (`frozen`). ONE model per run: fitted on the authors' 389 train rows, stopped early on their 98 validation rows (the paper's
#      protocol); the test molecules are never used for any choice. GPU, < 4 GB, waits for free memory; a few minutes per model.
#
#   bash ~/code/peptidefusion/scripts/run_thpep_official.sh
#   FINETUNE=0 bash ~/code/peptidefusion/scripts/run_thpep_official.sh          # part 1 only
#   FT_ONLY=1  bash ~/code/peptidefusion/scripts/run_thpep_official.sh          # part 2 only (after part 1 is done)
# Settings (environment): TEACHERS="v3_base st6f_e02"  LEARNERS="linear kernel gbt mlp"  SEEDS=5 (learner seeds for gbt / mlp)  THREADS=4  SKIP_DONE=1
#   FINETUNE=1  FT_SEEDS=3 (fine-tuning seeds per split; the paper has one)  FT_TEACHER=v3_base  FT_STUDENT=st6f  FT_EPOCH=2  NEED_MB=4000 (free GPU memory to wait for)
# Results: results/thpep_official_all (harness: raw.csv, per_bench.csv, ...), results/thpep_official_ft_teacher and results/thpep_official_ft_student
#   (raw.csv, summary.csv, preds_*.csv); the summary is also kept in logs/thpep_official_*.log
set -uo pipefail
source "$(dirname "$0")/env.sh"
SELF=$(cd "$(dirname "$0")" && pwd)
R=$SELF/run_teacher_v3.sh
TEACHERS=${TEACHERS:-"v3_base st6f_e02"}
LEARNERS=${LEARNERS:-"linear kernel gbt mlp"}
SEEDS=${SEEDS:-5}
THREADS=${THREADS:-4}
WORKERS=${WORKERS:-$((THREADS / 2))}
SKIP_DONE=${SKIP_DONE:-1}
FINETUNE=${FINETUNE:-1}
FT_ONLY=${FT_ONLY:-0}
FT_SEEDS=${FT_SEEDS:-3}
FT_TEACHER=${FT_TEACHER:-v3_base}
FT_STUDENT=${FT_STUDENT:-st6f}
FT_EPOCH=${FT_EPOCH:-2}
NEED_MB=${NEED_MB:-4000}
OUT=thpep_official_all
BENCH="THPep_o101 THPep_o202 THPep_o303"
LOG=$LOG_DIR/thpep_official_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
say() { echo "$(date +%H:%M:%S) $*"; }
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate peptide
say "== THPep, the authors' three splits | representations $TEACHERS | learners $LEARNERS | learner seeds $SEEDS | threads $THREADS ($WORKERS workers) | log $LOG"

python "$CODE/src/teacher/check_thpep_official.py" || { say "!! the molecule check failed: nothing was run"; exit 1; }

wait_gpu() {                                          # only the free memory counts
  local free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2> /dev/null | head -1 | tr -d ' ')
    [ -n "${free:-}" ] && [ "$free" -ge "$1" ] && break
    say "waiting for the GPU: ${free:-?} MB free, $1 MB needed (checking every 60 s)"; sleep 60
  done
}

if [ "$FT_ONLY" = 1 ]; then
  say "FT_ONLY=1: the frozen-embedding part is skipped"
elif [ "$SKIP_DONE" = 1 ] && [ -f "$RESULTS_DIR/$OUT/headline.csv" ]; then
  say "$OUT: already done, skipped (SKIP_DONE=0 reruns)"
else
  say "part 1: harness (frozen embeddings, sklearn learners)"
  env THREADS=$THREADS BENCH="$BENCH" TEACHERS="$TEACHERS" LEARNERS="$LEARNERS" SEEDS=$SEEDS BOOT=0 WORKERS=$WORKERS OUT=$OUT     EXTRA="--reps P L H --no-e-pass" bash "$R" harness || say "!! harness FAILED"
fi

if [ "$FINETUNE" = 1 ]; then
  say "part 2a: fine-tuning the teacher $FT_TEACHER (ft + frozen head), $FT_SEEDS seed(s) per split"
  wait_gpu "$NEED_MB"
  # shellcheck disable=SC2086
  bash "$R" finetune "$FT_TEACHER" --bench $BENCH --modes ft frozen --seeds "$FT_SEEDS" --out thpep_official_ft_teacher --resume || say "!! teacher fine-tuning FAILED"
  say "part 2b: fine-tuning the student $FT_STUDENT epoch $FT_EPOCH (ft + frozen head), $FT_SEEDS seed(s) per split"
  # shellcheck disable=SC2086
  RUN=$FT_STUDENT EPOCH=$FT_EPOCH FT_SEEDS=$FT_SEEDS NEED_MB=$NEED_MB bash "$SELF/run_student_eval.sh" ft --bench $BENCH --modes ft frozen --out thpep_official_ft_student --resume || say "!! student fine-tuning FAILED"
fi

say "== SUMMARY"
python "$CODE/src/teacher/thpep_official_summary.py" --out $OUT

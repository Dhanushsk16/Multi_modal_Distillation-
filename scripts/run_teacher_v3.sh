#!/usr/bin/env bash
# Teacher v3: evaluation harness first, then the multi-view autoencoder.
#
#   bash ~/code/peptidefusion/scripts/run_teacher_v3.sh prep                 # CPU: view statistics + the PCA baseline, both over ALL training rows of the pool
#                                                                             (SHARDS=6 PCA_ROWS=100000 = a quick sample version)
#   TEACHERS="t10" bash ~/code/peptidefusion/scripts/run_teacher_v3.sh harness            # CPU, hours: step 0, scores the baselines and teacher t10
#   BENCH="CellPPD THPep" SEEDS=2 TEACHERS="t10" bash .../run_teacher_v3.sh harness      # a quick look at two benchmarks
#   FORCE=1 bash ~/code/peptidefusion/scripts/run_teacher_v3.sh train abl_base --first 5 --epochs 60 --patience 8   # one run
#   FORCE=1 bash ~/code/peptidefusion/scripts/run_teacher_v3.sh seeds         # FIRST: the base configuration 3x (seeds 1-3) = training noise
#   FORCE=1 bash ~/code/peptidefusion/scripts/run_teacher_v3.sh abl           # then the ablations on the 10% subset, one run each (seed 1)
#   FORCE=1 bash ~/code/peptidefusion/scripts/run_teacher_v3.sh train v3 --epochs 200 --patience 10 --resume   # the full-pool run
#   EXTEND=1 FORCE=1 ... train v3 --epochs 200 --patience 10 --resume          # a second night: continue a finished run up to more epochs
#   HARNESS=1 ... train NAME ...    scores the finished teacher with the harness afterwards (CPU)
#   bash ~/code/peptidefusion/scripts/run_teacher_v3.sh adapt v3_base --suite both       # frozen teacher + adapter vectors + MLP, per benchmark (GPU, minutes)
#   VARIATIONS of the design (4th input PeptideCLM-2 MTR: train NAME --extra-input pclm_mtr; attention pooling: train NAME --fuse attn --n-queries 4):
#     bash ~/code/peptidefusion/scripts/run_variants_v3.sh       trains both on the whole pool one after the other and runs every learner on each (see its header)
#   LEARNERS=mlp BOOT=1000 SEEDS=3 TEACHERS="v3_base ..." OUT=final_mlp EXTRA="--reps P L H P+L+H PCA --no-e-pass" ... harness    # the MLP learner (CPU)
#   bash ~/code/peptidefusion/scripts/run_teacher_v3.sh finetune v3_base --seeds 3 --resume                                # fine-tune the teacher, PeptideCLM-2's five benchmarks (GPU, a few GB)
#   ... finetune v3_base --suite pepland --seeds 3 --resume                                                                # the same on PepLand's five tasks
#
# harness: --reps default P L H L+H P+L P+L+H PCA E P+L+H+E plus the TEACHERS; options via env: BENCH, SEEDS, LEARNERS, OUT, BOOT, WORKERS (benchmarks in parallel processes, sharing the 8-thread cap).
#   after seeds / abl:  TEACHERS="abl_base_s1 abl_base_s2 abl_base_s3 abl_noh_s1 abl_vd0_s1 ..." OUT=harness_abl ... harness
#   (teachers named <base>_s1, _s2, ... also get a seed-spread table)
# train / abl need the GPU (shared: FORCE=1 skips the check that nobody else is on it); both refuse 07:00-20:00 without FORCE=1 and
# keep view E in RAM (--e-on cpu) so about 17 GB stay off the GPU. Logs: $LOG_DIR/teacher_v3_*.log
set -uo pipefail
source "$(dirname "$0")/env.sh"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate peptide

STAGE=${1:?usage: run_teacher_v3.sh prep|harness|train NAME [args]|abl}
shift
rc=0                # the exit status of the stage (train / harness / finetune / adapt): a driver script can see a failure
LOG=$LOG_DIR/teacher_v3_${STAGE}_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "== $(date) | stage $STAGE | log: $LOG"
S=$CODE/src/teacher

night_gpu() {      # 0 = ok to start a GPU job
  local hour busy
  hour=$(date +%H)
  if [ "${FORCE:-0}" != 1 ] && (( 10#$hour >= 7 && 10#$hour < 20 )); then
    echo "it is $(date +%H:%M): long jobs run at night (20:00-07:00). FORCE=1 to start anyway"; return 1
  fi
  busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v dorado || true)
  if [ -n "$busy" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "GPU in use by:"; for pid in $(echo "$busy" | cut -d, -f1); do ps -o user=,etime=,args= -p "$pid"; done
    echo "not starting (FORCE=1 to override)"; return 1
  fi
}

harness_for() {    # NAME(s) -> CPU harness
  # shellcheck disable=SC2086
  python "$S/eval_harness.py" --teachers $1 --out "${OUT:-harness_${1// /_}}" ${BENCH:+--bench $BENCH} ${SEEDS:+--seeds $SEEDS} \
    ${LEARNERS:+--learners $LEARNERS} ${BOOT:+--boot $BOOT} ${WORKERS:+--workers $WORKERS} ${EXTRA:-}
}

train_one() {      # NAME [args]
  local name=$1; shift
  if [ -f "$CKPT_DIR/teacher/$name/latent_stats.npz" ] && [ "${EXTEND:-0}" != 1 ]; then
    echo "== $name is already trained ($CKPT_DIR/teacher/$name): skipped (EXTEND=1 with --resume and a higher --epochs continues it; or use another name)"; return 0
  fi
  night_gpu || return 2
  python "$S/train_teacher_v3.py" --name "$name" "$@" || { echo "== training $name failed"; return 1; }
  python "$S/train_teacher_v3.py" --name "$name" --embed-bench-from pepland_suite || return 1
  [ "${HARNESS:-0}" = 1 ] && harness_for "$name"
  return 0
}

case "$STAGE" in
  prep)
    python "$S/views_v3.py" --stats --shards "${SHARDS:-0}" || exit 1          # SHARDS=0: all training rows of the pool
    python "$S/pca_baseline.py" --fit --rows "${PCA_ROWS:-0}"                  # PCA_ROWS=0: all training rows (exact)
    ;;
  harness)
    harness_for "${TEACHERS:-t10}"
    rc=$?
    ;;
  train)
    NAME=${1:?usage: run_teacher_v3.sh train NAME [args]}; shift
    train_one "$NAME" "$@"
    rc=$?
    ;;
  seeds)    # FIRST: the base configuration three times (seeds 1-3) on the 10% subset = the teacher-training noise
    C="--first 5 --epochs ${ABL_EPOCHS:-60} --patience 8 --resume"
    for k in 1 2 3; do train_one abl_base_s$k $C --seed $k; done
    ;;
  abl)      # the ablations, one at a time, on the same subset, seed 1 (paired with abl_base_s1); 1 run each
    C="--first 5 --epochs ${ABL_EPOCHS:-60} --patience 8 --resume --seed 1"
    train_one abl_base_s1 $C
    train_one abl_noh_s1 $C --no-h
    train_one abl_vd0_s1 $C --view-dropout 0
    train_one abl_vd3_s1 $C --view-dropout 0.3
    train_one abl_wh2_s1 $C --w-h 2
    train_one abl_noe_s1 $C --no-e
    train_one abl_bal19_s1 $C --balance 0.19
    ;;
  finetune) # fine-tune / head-probe a trained version 3 teacher on the benchmarks (src/teacher/finetune_v3.py; its docstring has the options)
    NAME=${1:?usage: run_teacher_v3.sh finetune NAME [args]}; shift
    python "$S/finetune_v3.py" --name "$NAME" "$@"
    rc=$?
    ;;
  adapt)    # frozen teacher + per-view adapter vectors learned together with an MLP, per benchmark (src/teacher/adapter_mlp_v3.py; GPU, < 1 GB)
    NAME=${1:?usage: run_teacher_v3.sh adapt NAME [args]}; shift
    python "$S/adapter_mlp_v3.py" --name "$NAME" "$@"
    rc=$?
    ;;
  *) echo "usage: run_teacher_v3.sh prep|harness|train NAME [args]|seeds|abl|finetune NAME [args]|adapt NAME [args]"; exit 1 ;;
esac
echo "== done $(date +%H:%M) (exit status $rc)"
exit $rc

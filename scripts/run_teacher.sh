#!/usr/bin/env bash
# Train the CDI-style teacher on the cached views, embed the benchmark molecules, then run the gate:
# teacher vs ECFP alone on the benchmarks, same learners and splits as the single-modality check.
#
#   bash ~/code/peptidefusion/scripts/run_teacher.sh t1                         # all cached molecules, 40 epochs
#   bash ~/code/peptidefusion/scripts/run_teacher.sh t1 --epochs 5 --first 3    # quick look
#   bash ~/code/peptidefusion/scripts/run_teacher.sh t1 --resume                # continue a stopped run
#   GATE=0 bash ~/code/peptidefusion/scripts/run_teacher.sh t1                  # train only, no gate
#   GATE_FT=1 bash ~/code/peptidefusion/scripts/run_teacher.sh t1               # + the fine-tuning gate (teacher/finetune.py)
#   For a teacher that is already trained, run the fine-tuning gate directly:
#     python ~/code/peptidefusion/src/teacher/finetune.py --name t8
#
# Uses the finished shards plus the finished rows of PepLand shards that are still being cached
# (--strict: finished shards only). GPU for about 10-30 min, then 8 CPU threads for the gate (30-60 min);
# the gate waits if the caching is still running (it uses the allowed CPU). Refuses if someone else is on
# the GPU (FORCE=1 overrides). Log: $LOG_DIR/teacher_<name>_*.log; results: results/teacher_<name>/.
set -uo pipefail
source "$(dirname "$0")/env.sh"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

NAME=${1:?usage: run_teacher.sh NAME [options of train_teacher.py, e.g. --epochs 5 --first 3 --resume]}
shift
LOG=$LOG_DIR/teacher_${NAME}_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "== $(date) | teacher '$NAME' | log: $LOG"

# --- GPU check (dorado_basecall_server is a permanent idle service: ignored) ------------
busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v dorado || true)
if [ -n "$busy" ] && [ "${FORCE:-0}" != 1 ]; then
  echo "GPU in use by:"
  for pid in $(echo "$busy" | cut -d, -f1); do ps -o user=,etime=,args= -p "$pid"; done
  echo "not starting (FORCE=1 to override)"; exit 2
fi

# --- train ----------------------------------------------------------------------------------
conda activate peptide
python "$CODE/src/teacher/train_teacher.py" --name "$NAME" "$@"
rc=$?
[ $rc = 0 ] || { echo "== training failed (exit $rc)"; exit $rc; }

# --- gate -----------------------------------------------------------------------------------
# Two halves: (1) frozen embedding + simple learners (eval_single.py, CPU); (2) GATE_FT=1: the teacher's own networks fine-tuned
# on the five benchmarks at a low learning rate (teacher/finetune.py, GPU; FT_ARGS passes options, e.g. FT_ARGS="--seeds 3").
ft_gate() {
  [ "${GATE_FT:-0}" = 1 ] || return 0
  echo "== fine-tuning gate: teacher/finetune.py --name $NAME ${FT_ARGS:-} ($(date +%H:%M))"
  python "$CODE/src/teacher/finetune.py" --name "$NAME" --resume ${FT_ARGS:-}
}
if [ "${GATE:-1}" != 1 ]; then echo "== GATE=0: the frozen-embedding gate is skipped"; ft_gate; exit 0; fi
if pgrep -u "$(id -un)" -f "corpus/cache_views.py" > /dev/null; then
  echo "== the caching is still running and uses the allowed CPU: gate not started. Later:"
  echo "   python $CODE/src/eval_single.py --reps ecfp teacher_$NAME --out teacher_$NAME"
  ft_gate
  exit 0
fi
# The gate's representations (default: the teacher against ECFP alone). For teacher v2 (no PepLand, RDKit descriptors) use
#   GATE_REPS="ecfp pclm rdkit ecfp+rdkit ecfp+pclm+rdkit teacher_t3 teacher_$NAME ecfp+teacher_$NAME" \
#   GATE_LEARNERS="linear gbt mlp" GATE_OUT=teacher_${NAME}_gate bash run_teacher.sh $NAME ...
GATE_REPS=${GATE_REPS:-"ecfp teacher_$NAME"}
GATE_OUT=${GATE_OUT:-teacher_$NAME}
echo "== gate: $GATE_REPS ($(date +%H:%M))"
# shellcheck disable=SC2086
python "$CODE/src/eval_single.py" --reps $GATE_REPS ${GATE_LEARNERS:+--learners $GATE_LEARNERS} --out "$GATE_OUT"
echo "== done $(date +%H:%M): results in $RESULTS_DIR/$GATE_OUT/"
ft_gate

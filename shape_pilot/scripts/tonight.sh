#!/usr/bin/env bash
# Tonight's two jobs, in order:
#
#   1. Fair MLM comparison (~20-40 min): mlm-large, mlm-small and the four 10M
#      shapes, each scored on the same 20,000 held-out molecules with the same
#      masks, split by molecule type.
#   2. LR recheck (~1 h): the two tied shapes, s256x15 and s320x10, retrained
#      from the same random init at LR 5e-4 instead of 1e-3, then scored the
#      same way as step 1.
#
#   conda activate peptide
#   tmux new -s pilot
#   bash ~/code/shape_pilot/scripts/tonight.sh
#
# Safe to re-run: finished evaluations and runs are skipped.
set -uo pipefail
source "$(dirname "$0")/env.sh"

LOG=$LOG_DIR/tonight_$(date +%Y%m%d).log
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG") 2>&1          # everything below also goes to the log
echo "== $(date) | log: $LOG"

# --- GPU check (dorado_basecall_server is always there and is ignored) --------
busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v dorado || true)
if [ -n "$busy" ] && [ "${FORCE:-0}" != 1 ]; then
  echo "GPU in use by:"; echo "$busy"
  for pid in $(echo "$busy" | cut -d, -f1); do ps -o user=,etime=,args= -p "$pid"; done
  echo "not starting (FORCE=1 to override)"; exit 2
fi

EV=$RESULTS_DIR/mlm_eval
evaluate() {   # name model_dir [weights]
  if [ -f "$EV/$1.json" ]; then echo "== eval $1: done already"; return; fi
  echo "== eval $1  ($(date +%H:%M))"
  python "$CODE/distill/eval_mlm.py" --name "$1" --model "$2" ${3:+--weights "$3"} ${EVAL_ARGS:-} 2>&1 \
    | grep -vE "Warning|warnings.warn|Loading weights|deprecated"
}

# --- 1. fair comparison --------------------------------------------------------
for need in "$CKPT_DIR/peptideclm-2-mlm-large/model.safetensors" \
            "$CKPT_DIR/peptideclm-2-mlm-small/model.safetensors"; do
  test -e "$need" || { echo "missing $need -- extract tonight_data.tar into $DATA"; exit 1; }
done
evaluate mlm-large "$CKPT_DIR/peptideclm-2-mlm-large"
evaluate mlm-small "$CKPT_DIR/peptideclm-2-mlm-small"
for s in s256x15 s320x10 s384x7 s512x4; do
  evaluate "$s" "$CKPT_DIR/students/student-$s" "$CKPT_DIR/shape_pilot/$s/student_final.pt"
done
python "$CODE/distill/summarize_mlm_eval.py"

# --- 2. LR recheck -------------------------------------------------------------
for s in s256x15 s320x10; do
  TAG=lr5e-4 LR=5e-4 FORCE=1 bash "$CODE/scripts/run_shape.sh" "$s"
  f="$CKPT_DIR/shape_pilot/${s}_lr5e-4/student_final.pt"
  if [ -f "$f" ]; then evaluate "${s}_lr5e-4" "$CKPT_DIR/students/student-$s" "$f"
  else echo "== ${s}_lr5e-4 did not finish (time limit?) -- re-run tonight.sh to resume"; fi
done

echo; echo "================ RESULTS ================"
python "$CODE/distill/summarize_mlm_eval.py"
echo
python "$CODE/distill/summarize_pilot.py"
echo "== finished $(date)"

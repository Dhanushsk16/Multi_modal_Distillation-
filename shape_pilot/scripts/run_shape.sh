#!/usr/bin/env bash
# Train ONE shape of the 10M student: random init, MLM-only, 1 epoch of the 2M
# corpus. Same seed, data and validation split for every shape.
#
#   conda activate peptide
#   tmux new -s pilot
#   bash ~/code/shape_pilot/scripts/run_shape.sh s256x15
#
# Shapes: s256x15  s320x10  s384x7  s512x4
# TAG=lr5e-4 LR=5e-4 ...  -> a separate run (own output dir and log) from the
#                           same random init, e.g. to recheck the learning rate
#
# Safety:
#   - refuses to start if another process is using the GPU (FORCE=1 overrides)
#   - stops itself at 06:45 (DEADLINE=HH:MM to change); the run checkpoints every
#     1000 steps, so re-running the same command the next night resumes it
#   - never runs longer than 23 h
# Everything is written to the data drive; the log is $LOG_DIR/shape_pilot/<shape>.log
set -euo pipefail
source "$(dirname "$0")/env.sh"

name=${1:?usage: run_shape.sh <s256x15|s320x10|s384x7|s512x4>}
case $name in
  s256x15) d=256 h=4 f=512  L=15 ;;
  s320x10) d=320 h=5 f=640  L=10 ;;
  s384x7)  d=384 h=6 f=768  L=7  ;;
  s512x4)  d=512 h=8 f=1024 L=4  ;;
  *) echo "unknown shape $name"; exit 1 ;;
esac

CORPUS=$DATA/data/pretrain_subset_2M/pretrain_subset_2M.parquet
SMALL=$CKPT_DIR/peptideclm-2-mlm-small
MODEL=$CKPT_DIR/students/student-$name
run=$name${TAG:+_$TAG}
OUT=$CKPT_DIR/shape_pilot/$run
LOG=$LOG_DIR/shape_pilot/$run.log
LR=${LR:-1e-3}; WARMUP=${WARMUP:-1000}; MAX_TOKENS=${MAX_TOKENS:-32768}
SEED=${SEED:-0}; VAL=${VAL:-20000}; LIMIT=${LIMIT:-}
DEADLINE=${DEADLINE:-06:45}
mkdir -p "$(dirname "$LOG")" "$OUT" "$CKPT_DIR/students"

for need in "$CORPUS" "$SMALL/model.safetensors" "$DATA/data/benchmarks/CellPPD_test.csv"; do
  test -e "$need" || { echo "missing $need -- extract shape_pilot_data.tar into $DATA"; exit 1; }
done
python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 'no CUDA -- conda activate peptide?')"

# --- is the GPU free? -------------------------------------------------------
others=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader || true)
echo "== nvidia-smi compute processes:"; echo "${others:-   (none)}"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
if [ -n "$others" ] && [ "${FORCE:-0}" != 1 ]; then
  echo "another process is on the GPU -- not starting (FORCE=1 to override)"; exit 2
fi

# --- time budget: stop at DEADLINE, never more than 23 h ---------------------
now=$(date +%s); stop=$(date -d "$DEADLINE" +%s)
[ "$stop" -le "$now" ] && stop=$(date -d "tomorrow $DEADLINE" +%s)
budget=$(( stop - now )); [ $budget -gt 82800 ] && budget=82800
echo "== $run: lr=${LR:-1e-3} d=$d heads=$h ffn=$f blocks=$L | $(date) | stops by $(date -d @$((now+budget)))"

if [ ! -f "$MODEL/model.safetensors" ]; then
  python "$CODE/distill/init_student.py" --embed-dim $d --heads $h --ffn $f \
      --blocks $L --src "$SMALL" --out "$MODEL" --seed "$SEED" 2>&1 \
      | grep -vE "Warning|Loading weights" | tee -a "$LOG"
fi

if [ -f "$OUT/final_val.json" ]; then
  echo "== $run already finished" | tee -a "$LOG"
else
  set +e
  timeout --signal=INT --kill-after=120 $budget \
    python "$CODE/distill/train_kd.py" \
      --subset "$CORPUS" --teacher none --student "$MODEL" --out "$OUT" \
      --epochs 1 --lr "$LR" --warmup "$WARMUP" --max-tokens "$MAX_TOKENS" \
      --amp bf16 --seed "$SEED" --val-molecules "$VAL" ${LIMIT:+--limit $LIMIT} \
      --log-every 200 --eval-every 2000 --save-every 1000 2>&1 \
    | grep --line-buffered -vE "Warning|W[0-9]{4} |Loading weights" | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  set -e
  if [ "$rc" = 124 ] || [ "$rc" = 130 ]; then
    echo "== stopped at the time limit; re-run the same command to resume" | tee -a "$LOG"
    exit 0
  fi
  [ "$rc" = 0 ] || { echo "== training failed (exit $rc), see $LOG"; exit "$rc"; }
fi

[ -n "${TAG:-}" ] || python "$CODE/distill/bench_latency.py" --model "$name" --path "$MODEL" \
    --device cuda --dtype bf16 --out-dir "$RESULTS_DIR/shape_pilot/latency" 2>&1 \
  | grep -vE "Warning|Loading weights" | tee -a "$LOG"
python "$CODE/distill/summarize_pilot.py"

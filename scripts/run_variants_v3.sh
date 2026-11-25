#!/usr/bin/env bash
# Teacher v3 VARIATIONS: the whole GPU programme in one go, then the benchmarking, with no manual step in between.
#
#   bash ~/code/peptidefusion/scripts/run_variants_v3.sh                  # everything below
#   VARIANTS="attn" bash ~/code/peptidefusion/scripts/run_variants_v3.sh  # only the attention variation
#   DRY=1 bash ~/code/peptidefusion/scripts/run_variants_v3.sh            # print what would run, run nothing
#
# What runs (GPU jobs first, one training at a time; the CPU benchmarking follows each training at once):
#   0. checks: the PeptideCLM-2 MTR view is cached for ALL pool shards (else it stops with the fix); the MTR benchmark files are made if missing
#      and checked (same molecule order as the other views). GPU memory is reported.
#   1. adapter experiment on the FROZEN v3_base (src/teacher/adapter_mlp_v3.py): per-view adapter vectors + MLP, learned per benchmark, with the
#      plain-MLP control, at adapter lr 1e-2 (and the sensitivity runs 1e-3, 3e-2). It needs < 1 GB of GPU memory and minutes, so it runs in
#      the background WHILE the first training runs (ADAPT_WHEN=start; ADAPT_WHEN=after: once the trainings are finished; ADAPT=0: skip).
#   2. train v3_mtr on the whole pool: FOUR inputs (PepLand, PeptideCLM-2, ECFP, PeptideCLM-2 MTR), FIVE decoded views (the four + ESM-2),
#      concatenation (2048-d into the trunk). ~28 GB of GPU memory at the peak.
#   3. train v3_attn on the whole pool: the three v3 inputs, attention pooling with 4 learned query vectors (8 heads) instead of concatenation.
#      ~21 GB. (Both use the v3_base recipe: --epochs 100 --patience 10 --resume.)
#   4. CPU benchmarking of each new teacher, started the moment its training ends, in a queue (one harness job at a time, so the CPU cap of
#      8 threads holds): linear (also the raw ESM-2 inputs and the standard-chain pass), gbt, mlp, kernel -> results/<name>_<learner>.
#      v3_base is scored in the SAME run (same items and splits); the mtr run also scores MTR alone (M) and all four inputs joined (P+L+H+M).
#   The third "version" is v3_base itself, benchmarked in the new way of step 1 (results/adapter_v3_base[_pepland], adapter_lr*).
# Earlier harness times for one teacher: linear ~25 min, gbt ~1.7 h, mlp ~30 min, kernel ~30 min.
#
# Settings (environment): VARIANTS="mtr attn" EPOCHS=100 PATIENCE=10 LEARNERS="linear gbt mlp kernel" BOOT=1000 SEEDS=3 BASE=v3_base
#   ADAPT=1 ADAPT_WHEN=start ADAPT_LRS="1e-2 1e-3 3e-2"   OVERLAP_THREADS=6 (harness threads while a training runs; 8 when none does)
#   TRAIN_EXTRA="" (e.g. "--data-on cpu": the pool in RAM, ~40 GB RAM). If a training dies with CUDA out of memory and >= 45 GB of RAM are free,
#   it is retried once from its last epoch with the pool in RAM.   SKIP_DONE=1 (finished harness / adapter folders are not run again)
#   FINETUNE=0 (1: also finetune_v3.py on the new teachers, 3 seeds, both suites)
# If it stops: run the same line again. Trainings resume, finished work is skipped.
set -uo pipefail
source "$(dirname "$0")/env.sh"
SELF=$(cd "$(dirname "$0")" && pwd)
R=$SELF/run_teacher_v3.sh
VARIANTS=${VARIANTS:-"mtr attn"}
EPOCHS=${EPOCHS:-100}
PATIENCE=${PATIENCE:-10}
LEARNERS=${LEARNERS:-"linear gbt mlp kernel"}
BOOT=${BOOT:-1000}
SEEDS=${SEEDS:-3}
BASE=${BASE:-v3_base}
ADAPT=${ADAPT:-1}
ADAPT_WHEN=${ADAPT_WHEN:-start}
ADAPT_LRS=${ADAPT_LRS:-"1e-2 1e-3 3e-2"}
OVERLAP_THREADS=${OVERLAP_THREADS:-6}
TRAIN_EXTRA=${TRAIN_EXTRA:-}
SKIP_DONE=${SKIP_DONE:-1}
FINETUNE=${FINETUNE:-0}
DRY=${DRY:-0}
LOG=$LOG_DIR/variants_v3_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
say() { echo "$(date +%H:%M:%S) $*"; }
run() { if [ "$DRY" = 1 ]; then echo "   [dry] $*"; else "$@"; fi; }
if [ "$DRY" != 1 ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate peptide
fi
say "== run_variants_v3 | variations: $VARIANTS | adapters: $ADAPT ($ADAPT_WHEN, lr $ADAPT_LRS) | learners: $LEARNERS | log: $LOG"

setvar() {      # sets NAME, TRAIN_ARGS, REPS for a variation
  case "$1" in
    mtr)  NAME=v3_mtr;  TRAIN_ARGS="--extra-input pclm_mtr"; REPS="P L H M P+L+H P+L+H+M PCA" ;;
    attn) NAME=v3_attn; TRAIN_ARGS="--fuse attn --n-queries 4"; REPS="P L H P+L+H PCA" ;;
    *) say "unknown variation '$1' (mtr or attn)"; exit 2 ;;
  esac
}

# ------------------------------------------------------------------------------------------------ 0. checks
gpu_report() {
  local free
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2> /dev/null | head -1 | tr -d ' ')
  say "GPU memory free: ${free:-?} MB (v3_mtr needs ~28000 MB at its peak, v3_attn ~21000 MB, the adapter job < 1000 MB)"
  if [ -n "${free:-}" ] && [ "$free" -lt 29000 ] && [ -z "$TRAIN_EXTRA" ]; then
    say "WARNING: less than 29 GB free. The mtr training may run out of memory: it is retried once with the pool in RAM if that happens."
  fi
}

preflight_mtr() {   # the MTR view must exist for the whole pool, or the mtr variation would train on part of the pool
  local line have want f
  line=$(python "$CODE/src/corpus/cache_views.py" --status 2>/dev/null | sed -nE "s/.*pclm_mtr +([0-9]+) \/ ([0-9]+) shards finished.*/\1 \2/p" | head -1)
  if [ -z "$line" ]; then say "PREFLIGHT: cannot read the pclm_mtr line of cache_views.py --status: stopping"; exit 3; fi
  read -r have want <<< "$line"
  if [ "$have" -lt "$want" ]; then
    say "PREFLIGHT: the pclm_mtr cache has $have of $want shards: finish it first (bash $SELF/run_cache.sh, VIEWS=pclm_mtr), or VARIANTS=attn to run only the attention variation"
    exit 3
  fi
  say "preflight: the pclm_mtr pool cache is complete ($have of $want shards)"
  # the benchmark files (a few minutes on the GPU each, only if missing)
  [ -f "$DATA/features/single_modality/pclm_mtr.npz" ] || { say "making features/single_modality/pclm_mtr.npz"; python "$CODE/src/featurize.py" --rep pclm_mtr || exit 3; }
  [ -f "$DATA/features/pepland_suite/pclm_mtr.npz" ] || { say "making features/pepland_suite/pclm_mtr.npz"; \
    python "$CODE/src/featurize.py" --rep pclm_mtr --bench PL_cCPP PL_cSol PL_ncCPP PL_cBinding PL_ncBinding --feat pepland_suite || exit 3; }
  python - "$DATA" <<'EOF' || { say "PREFLIGHT: the MTR benchmark files do not match the other views: stopping"; exit 3; }
import sys
import numpy as np
for d in ("single_modality", "pepland_suite"):
    a = np.load("%s/features/%s/pclm.npz" % (sys.argv[1], d), allow_pickle=True)
    b = np.load("%s/features/%s/pclm_mtr.npz" % (sys.argv[1], d), allow_pickle=True)
    same = np.array_equal(a["smiles"], b["smiles"])
    print("preflight: %-15s pclm_mtr.npz %d molecules, same order as pclm.npz: %s, usable %.1f%%, %d-d" % (d, len(b["smiles"]), same, 100 * b["ok"].mean(), b["X"].shape[1]))
    if not same or b["X"].shape[1] != 1024:
        sys.exit(1)
# the pool shards themselves: a few of them must be finite, 1024-d and mostly usable (a NaN would stop the training in its first epoch)
import glob
import os
files = sorted(glob.glob("%s/features/corpus/pclm_mtr/shard_*.npz" % sys.argv[1]))
for f in (files[0], files[len(files) // 2], files[-1]):
    z = np.load(f)
    X, ok = z["X"], z["ok"]
    fin = bool(np.isfinite(X[ok].astype(np.float32)).all())
    print("preflight: %s %s, %d rows, usable %.1f%%, finite %s, mean |x| %.3f" % (os.path.basename(f), X.dtype, len(X), 100 * ok.mean(), fin, float(np.abs(X[ok].astype(np.float32)).mean())))
    if X.shape[1] != 1024 or not fin or ok.mean() < 0.9 or "lo" not in z.files:
        sys.exit(1)
EOF
}

# ------------------------------------------------------------------------------------------------ jobs
adapt_chain() {      # the third version: v3_base with learnable per-view vectors + MLP (GPU, tiny); lr 1e-2 with the control, then the sensitivity runs
  local lr first=1
  for lr in $ADAPT_LRS; do
    if [ "$first" = 1 ]; then
      first=0
      if [ "$SKIP_DONE" = 1 ] && [ -f "$RESULTS_DIR/adapter_$BASE/raw.csv" ] && [ -f "$RESULTS_DIR/adapter_${BASE}_pepland/raw.csv" ]; then
        say "adapter run (lr $lr): already done, skipped"; continue
      fi
      say "adapter experiment on $BASE, adapter lr $lr (with the plain-MLP control)"
      run bash "$R" adapt "$BASE" --suite both --adapter-lr "$lr" || say "!! adapter run (lr $lr) FAILED"
    else
      if [ "$SKIP_DONE" = 1 ] && [ -f "$RESULTS_DIR/adapter_lr${lr}/raw.csv" ] && [ -f "$RESULTS_DIR/adapter_lr${lr}_pepland/raw.csv" ]; then
        say "adapter sensitivity run (lr $lr): already done, skipped"; continue
      fi
      say "adapter sensitivity run on $BASE, adapter lr $lr"
      run bash "$R" adapt "$BASE" --suite both --modes adapt --adapter-lr "$lr" --out "adapter_lr${lr}" || say "!! adapter run (lr $lr) FAILED"
    fi
  done
  say "adapter experiments finished (results/adapter_${BASE}*, results/adapter_lr*)"
}

train_variant() {    # NAME TRAIN_ARGS: the v3_base recipe; one retry with the pool in RAM after a CUDA out-of-memory
  local rc free_gb last
  # shellcheck disable=SC2086
  run env FORCE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True bash "$R" train "$NAME" --epochs "$EPOCHS" --patience "$PATIENCE" --resume $TRAIN_ARGS $TRAIN_EXTRA
  rc=$?
  [ $rc = 0 ] && return 0
  if [ "$DRY" != 1 ] && [ -z "$TRAIN_EXTRA" ]; then
    last=$(ls -t "$LOG_DIR"/teacher_v3_train_*.log 2> /dev/null | head -1)
    free_gb=$(free -g | awk '/^Mem:/{print $7}')
    if [ -n "$last" ] && grep -qa "out of memory" "$last" && [ "${free_gb:-0}" -ge 45 ]; then
      say "CUDA out of memory in $NAME: retrying once from the last epoch with the pool in RAM (--data-on cpu; ${free_gb} GB RAM free)"
      # shellcheck disable=SC2086
      run env FORCE=1 bash "$R" train "$NAME" --epochs "$EPOCHS" --patience "$PATIENCE" --resume $TRAIN_ARGS --data-on cpu
      return $?
    fi
  fi
  return $rc
}

harness_variant() {  # NAME REPS THREADS : every learner, one harness run each (THREADS: 6 while another training runs, else 8: the CPU cap)
  local name=$1 reps=$2 thr=$3 L out extra wrk
  wrk=$((thr / 2))
  for L in $LEARNERS; do
    out=${name}_${L}
    if [ "$SKIP_DONE" = 1 ] && [ -f "$RESULTS_DIR/$out/headline.csv" ]; then say "harness $out: already done, skipped"; continue; fi
    if [ "$L" = linear ]; then extra="--reps $reps E P+L+H+E"      # linear is fast: also the raw ESM-2 inputs and the standard-chain pass
    else extra="--reps $reps --no-e-pass"; fi
    say "harness $out ($L, threads $thr, workers $wrk)"
    run env THREADS=$thr TEACHERS="$BASE $name" LEARNERS=$L SEEDS=$SEEDS BOOT=$BOOT WORKERS=$wrk OUT=$out EXTRA="$extra" bash "$R" harness \
      || say "!! harness $out FAILED: run this script again to redo it"
  done
  say "harness for $name finished (results/${name}_*)"
}

ensure_embeddings() {   # NAME: the teacher's benchmark embeddings must exist in both feature folders (a crash after the latent statistics would leave them out)
  local name=$1 d
  [ -f "$CKPT_DIR/teacher/$name/latent_stats.npz" ] || return 1
  for d in single_modality pepland_suite; do
    if [ ! -f "$DATA/features/$d/teacher_$name.npz" ]; then
      say "making the benchmark embeddings of $name for $d"
      python "$CODE/src/teacher/train_teacher_v3.py" --name "$name" --embed-bench-from "$d" || return 1
    fi
  done
  [ -f "$DATA/features/single_modality/teacher_$name.npz" ] && [ -f "$DATA/features/pepland_suite/teacher_$name.npz" ]
}

HARNESS_PID=""
enqueue_harness() {  # NAME REPS THREADS: starts in the background when the previous harness job has finished (one at a time: the 8-thread CPU cap)
  local name=$1 reps=$2 thr=$3 prev=$HARNESS_PID
  (
    while [ -n "$prev" ] && kill -0 "$prev" 2> /dev/null; do sleep "${QUEUE_POLL:-30}"; done
    harness_variant "$name" "$reps" "$thr"
  ) &
  HARNESS_PID=$!
}

# ------------------------------------------------------------------------------------------------ run
FAILED=()
ADAPT_PID=""
if [ "$DRY" != 1 ]; then
  gpu_report
  for v in $VARIANTS; do [ "$v" = mtr ] && preflight_mtr; done
fi

if [ "$ADAPT" = 1 ] && [ "$ADAPT_WHEN" = start ]; then
  say "== the adapter experiments start now, in the background (GPU < 1 GB), while the first training runs"
  adapt_chain &
  ADAPT_PID=$!
fi

read -r -a VLIST <<< "$VARIANTS"
for i in "${!VLIST[@]}"; do
  v=${VLIST[$i]}
  setvar "$v"
  say "== training $NAME ($v): $TRAIN_ARGS $TRAIN_EXTRA"
  if ! train_variant; then
    say "!! training $NAME FAILED (see its log in $LOG_DIR): going on with the next variation; run this script again later to continue $NAME from its last epoch"
    FAILED+=("training $NAME")
    continue
  fi
  if [ "$DRY" != 1 ] && ! ensure_embeddings "$NAME"; then
    say "!! $NAME: the benchmark embeddings are missing and could not be made: no benchmarking for it"
    FAILED+=("embeddings $NAME")
    continue
  fi
  say "== $NAME is trained: its benchmarking is queued on the CPU now; the GPU goes on"
  # another training follows this one: the benchmarking shares the CPU with it (6 threads); the last one has the CPU for itself (8)
  if [ $((i + 1)) -lt "${#VLIST[@]}" ]; then enqueue_harness "$NAME" "$REPS" "$OVERLAP_THREADS"; else enqueue_harness "$NAME" "$REPS" "${THREADS:-8}"; fi
done

if [ "$ADAPT" = 1 ] && [ "$ADAPT_WHEN" = after ]; then
  say "== the trainings are finished: the adapter experiments run now"
  adapt_chain
fi

if [ "$FINETUNE" = 1 ]; then
  for v in "${VLIST[@]}"; do
    setvar "$v"
    run bash "$R" finetune "$NAME" --seeds 3 --resume || FAILED+=("finetune $NAME")
    run bash "$R" finetune "$NAME" --suite pepland --seeds 3 --resume || FAILED+=("finetune pepland $NAME")
  done
fi

say "== all GPU jobs are done; waiting for the CPU benchmarking and the adapter runs to finish"
# NOT a bare "wait": it would also wait for the tee process of the logging redirect above, which never ends
[ -n "$ADAPT_PID" ] && wait "$ADAPT_PID"
[ -n "$HARNESS_PID" ] && wait "$HARNESS_PID"

# ------------------------------------------------------------------------------------------------ the checklist
say "== CHECKLIST of what this script should have produced (OK / MISSING)"
MISSING=0
chk() {   # LABEL PATH
  if [ "$DRY" = 1 ] || [ -e "$2" ]; then say "   OK       $1"; else say "   MISSING  $1   ($2)"; MISSING=$((MISSING + 1)); fi
}
for v in "${VLIST[@]}"; do
  setvar "$v"
  chk "$NAME trained (latent statistics)" "$CKPT_DIR/teacher/$NAME/latent_stats.npz"
  chk "$NAME benchmark embeddings" "$DATA/features/pepland_suite/teacher_$NAME.npz"
  for L in $LEARNERS; do chk "results/${NAME}_${L}" "$RESULTS_DIR/${NAME}_${L}/headline.csv"; done
done
if [ "$ADAPT" = 1 ]; then
  first=1
  for lr in $ADAPT_LRS; do
    if [ "$first" = 1 ]; then f=adapter_$BASE; first=0; else f=adapter_lr$lr; fi
    chk "results/$f (adapter lr $lr, PeptideCLM-2 suite)" "$RESULTS_DIR/$f/raw.csv"
    chk "results/${f}_pepland (adapter lr $lr, PepLand suite)" "$RESULTS_DIR/${f}_pepland/raw.csv"
  done
fi
if [ "${#FAILED[@]}" -gt 0 ] || [ "$MISSING" -gt 0 ]; then
  say "== FINISHED WITH PROBLEMS: ${FAILED[*]:-none recorded} | $MISSING expected result(s) missing. Run this same line again: finished work is skipped, trainings resume."
  exit 1
fi
say "== EVERYTHING IS DONE. Results in $RESULTS_DIR: v3_mtr_{linear,gbt,mlp,kernel}, v3_attn_{linear,gbt,mlp,kernel}, adapter_${BASE}, adapter_${BASE}_pepland, adapter_lr*"

#!/usr/bin/env bash
# Benchmarks every student AUTOMATICALLY as soon as its training and export are finished, while the ladder keeps training the next one.
# A separate CPU job (no GPU); start it once, in its own tmux window, and leave it:
#
#   tmux new -s bench
#   bash ~/code/peptidefusion/scripts/bench_watch.sh
#
# It looks every 2 minutes. A student is ready when its export (the benchmark embeddings that run_student.sh writes after the training) is newer than
# its best.pt. Then it runs the harness for <name> and <name>_pre (and the teacher v3_base next to them) into $RESULTS_DIR/harness_<name>/ and writes a
# DONE file there. A student that already has DONE is skipped, so it is safe to stop and start it again. A harness that fails writes FAILED (delete
# that file to retry). It ends by itself when the ladder is no longer running and nothing is left; Ctrl-C ends it any time.
# This is a NEW file: it does not touch run_student.sh, so it can be started while the ladder is running.
#
# Settings (environment): NAMES="st3 stc st1 st7 st2 st5 st6 stca st4"   LEARNERS=linear (e.g. LEARNERS="linear gbt mlp": slower)   BOOT=1000   WORKERS_H=4
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
NAMES=${NAMES:-"st3 stc st1 st7 st2 st5 st6 stca st4 st3f stcf st6f"}
LEARNERS=${LEARNERS:-linear}
BOOT=${BOOT:-1000}
WORKERS_H=${WORKERS_H:-4}
LOG=$LOG_DIR/student_bench_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
say() { echo "$(date +%H:%M:%S) $*"; }
say "== benchmark watcher | students: $NAMES | learners: $LEARNERS | bootstrap $BOOT | log $LOG"

ready() {                                   # the export of $1 is finished, and it is newer than the best weights it was made from
  local best=$CKPT_DIR/student_distill/$1/best.pt
  local last=$DATA/features/pepland_suite/teacher_${1}_pre.npz         # the last file export_student.py writes
  [ -f "$DATA/features/pepland_suite/ecfp.npz" ] || last=$DATA/features/single_modality/teacher_${1}_pre.npz
  [ -f "$best" ] && [ -f "$last" ] && [ "$last" -nt "$best" ]
}

while true; do
  todo=0; did=0
  for n in $NAMES; do
    d=$RESULTS_DIR/harness_$n
    [ -f "$d/DONE" ] && continue
    [ -f "$d/FAILED" ] && continue
    todo=1
    ready "$n" || continue
    say "=== benchmarking $n ($n and ${n}_pre against the teacher)"
    mkdir -p "$d"
    if OUT=harness_$n LEARNERS="$LEARNERS" BOOT="$BOOT" WORKERS_H="$WORKERS_H" bash "$HERE/run_student.sh" harness "$n" "${n}_pre"; then
      touch "$d/DONE"; say "=== $n benchmarked: $d"
    else
      touch "$d/FAILED"; say "!! the harness failed for $n (see the student_harness log; delete $d/FAILED to retry)"
    fi
    did=1
  done
  [ "$todo" = 0 ] && { say "== all students benchmarked"; break; }
  if [ "$did" = 0 ]; then
    pgrep -f "run_student.sh ladder" > /dev/null || { say "== the ladder is not running and nothing is ready: stopping (start me again after the next run)"; break; }
    sleep 120
  fi
done

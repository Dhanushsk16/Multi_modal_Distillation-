#!/usr/bin/env bash
# Wait until the GPU is free, then run a command (for example the teacher training).
#
#   bash ~/code/peptidefusion/scripts/wait_gpu_run.sh COMMAND [ARGS...]
#
# It starts COMMAND only when ALL of these hold:
#   1. the RDKit cache is complete (no cache_views.py running and every shard of the views in NEED_CACHE is there);
#   2. the GPU has been idle for IDLE seconds in a row. Idle means: no compute process other than dorado (the permanent
#      idle service), utilisation <= UTIL_MAX % in 4 samples 1 s apart, and at least MIN_FREE_MB MB of GPU memory free.
#      It is checked every POLL seconds; one busy poll restarts the count.
# There is NO night window and NO deadline (the GPU is shared and other lab users run jobs at any hour): it starts at any hour
# and COMMAND runs until it ends by itself. If it dies (crash, out of memory), run the same line again: --resume continues.
#
# Settings (environment): POLL=30 IDLE=60 UTIL_MAX=0 MIN_FREE_MB=30000 NEED_CACHE=rdkit (empty: do not wait for a cache)
#   CONFIRM_EVERY=10 (once the GPU looks idle, look again every 10 s until it has stayed idle for IDLE s; one busy look restarts the count)
#   NOTE_EVERY=600 (nothing is printed between the checks except one "still waiting" line every 10 minutes and the state changes)
#   IGNORE_PROCS=1: other users' processes do not count as busy (idle = utilisation <= UTIL_MAX in the samples AND MIN_FREE_MB free)
#   SHARED=1: the GPU is shared with a user who is on it 24/7, so do not wait for it to be idle: other processes and the utilisation
#   are ignored and only MIN_FREE_MB of free memory is required (the cache check still applies). The command
#   itself must not refuse a busy GPU either: run_teacher.sh needs FORCE=1 as well.
# Log: $LOG_DIR/wait_gpu_*.log. Run it inside tmux; Ctrl-C while waiting cancels it (nothing has been started yet).
#
#   tmux new -s wait5
#   GATE=0 bash ~/code/peptidefusion/scripts/wait_gpu_run.sh bash ~/code/peptidefusion/scripts/run_teacher.sh t5 ... --resume
set -uo pipefail
source "$(dirname "$0")/env.sh"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

[ "${1:-}" = "--" ] && shift
[ $# -gt 0 ] || { echo "usage: wait_gpu_run.sh COMMAND [ARGS...]   (see the header of this file)"; exit 1; }
POLL=${POLL:-30}
IDLE=${IDLE:-60}
UTIL_MAX=${UTIL_MAX:-0}
MIN_FREE_MB=${MIN_FREE_MB:-30000}
NEED_CACHE=${NEED_CACHE-rdkit}
LOG=$LOG_DIR/wait_gpu_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
say() { echo "$(date +%H:%M:%S) $*"; }
if [ "${SHARED:-0}" = 1 ]; then
  say "== wait_gpu_run | SHARED=1: ONLY the free GPU memory counts (>= ${MIN_FREE_MB} MB, held ${IDLE}s, checked every ${POLL}s); utilisation and other users' processes are ignored | log: $LOG"
else
  say "== wait_gpu_run | poll ${POLL}s, idle needed ${IDLE}s, util <= ${UTIL_MAX}%, free >= ${MIN_FREE_MB} MB | log: $LOG"
fi
say "== command: $*"
nvidia-smi -L > /dev/null 2>&1 || { say "nvidia-smi does not work here: stopping"; exit 1; }
conda activate peptide

LAST=""; LAST_T=0
note() {                          # the first message of a kind, then one line every NOTE_EVERY seconds (default 10 minutes) = "still running"
  local t kind; t=$(date +%s); kind=${1%%:*}
  if [ "$kind" != "$LAST" ] || [ $(( t - LAST_T )) -ge "${NOTE_EVERY:-600}" ]; then say "$1"; LAST="$kind"; LAST_T=$t; fi
}

cache_missing() {                 # prints why and returns 0 while the cache is not complete
  local v line have want
  if pgrep -u "$(id -un)" -f "corpus/cache_views.py" > /dev/null; then echo "cache_views.py is still running"; return 0; fi
  for v in $NEED_CACHE; do
    line=$(python "$CODE/src/corpus/cache_views.py" --status 2>/dev/null | sed -nE "s/.*$v +([0-9]+) \/ ([0-9]+) shards finished.*/\1 \2/p" | head -1)
    [ -n "$line" ] || { echo "cannot read the $v line of cache_views.py --status"; return 0; }
    read -r have want <<< "$line"
    [ "$have" -ge "$want" ] || { echo "$v cache has $have of $want shards (run run_rdkit.sh cache again if nothing is running)"; return 0; }
  done
  return 1
}

gpu_busy() {                      # prints why and returns 0 when the GPU is in use; returns 1 when it is idle
  local apps pid u free i
  if [ "${SHARED:-0}" = 1 ]; then # the GPU is shared with someone who is on it all the time: other processes and utilisation are ignored, only the free memory counts
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2> /dev/null | head -1 | tr -d ' ')
    case "$free" in ''|*[!0-9]*) echo "cannot read the free memory"; return 0 ;; esac
    [ "$free" -lt "$MIN_FREE_MB" ] && { echo "only ${free} MB of GPU memory free (SHARED=1: the other processes are ignored)"; return 0; }
    return 1
  fi
  apps=""
  [ "${IGNORE_PROCS:-0}" = 1 ] || apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2> /dev/null | grep -v dorado || true)
  if [ -n "$apps" ]; then
    for pid in $(echo "$apps" | cut -d, -f1); do
      echo "process $pid: $(ps -o user=,args= -p "$pid" 2> /dev/null | cut -c1-90)"
    done
    return 0
  fi
  for i in 1 2 3 4; do
    u=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i 0 2> /dev/null | head -1 | tr -d ' ')
    case "$u" in ''|*[!0-9]*) echo "cannot read the utilisation"; return 0 ;; esac
    [ "$u" -gt "$UTIL_MAX" ] && { echo "utilisation ${u}%"; return 0; }
    sleep 1
  done
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2> /dev/null | head -1 | tr -d ' ')
  case "$free" in ''|*[!0-9]*) echo "cannot read the free memory"; return 0 ;; esac
  [ "$free" -lt "$MIN_FREE_MB" ] && { echo "only ${free} MB of GPU memory free"; return 0; }
  return 1
}

# 1. the cache
while why=$(cache_missing); do
  note "waiting for the cache: $why"
  sleep "${CACHE_POLL:-60}"
done
say "the cache is complete"

# 2. an idle GPU (any hour)
idle_since=0
while true; do
  why=$(gpu_busy); rc=$?
  if [ $rc = 0 ]; then
    [ "$idle_since" != 0 ] && say "GPU busy again ($why): the idle count restarts"
    idle_since=0; note "GPU busy: $why"
  else
    [ "$idle_since" = 0 ] && { idle_since=$(date +%s); say "GPU idle: it has to stay idle for ${IDLE}s"; }
    held=$(( $(date +%s) - idle_since ))
    [ "$held" -ge "$IDLE" ] && break
  fi
  if [ "$idle_since" != 0 ]; then sleep "${CONFIRM_EVERY:-10}"; else sleep "$POLL"; fi      # idle seen: look again every CONFIRM_EVERY s until it has held IDLE s
done
say "GPU idle for $held s: starting"

# 3. run: no deadline, no time limit
"$@"
rc=$?
say "== command finished with exit code $rc"
[ $rc = 0 ] || say "the command failed: run the same line again to continue (the command must contain --resume)"
exit $rc

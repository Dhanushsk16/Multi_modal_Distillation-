#!/usr/bin/env bash
# A personal queue for the shared GPU: wait until no OTHER user has a job on it,
# make sure it stays free for a few minutes (so a gap between someone's sweep
# jobs doesn't count), then start the shape run.
#
#   bash ~/code/shape_pilot/scripts/wait_and_run.sh s256x15
#
# Ignores the always-on dorado_basecall_server and your own processes.
# Lab rule: long jobs run at night. It only starts inside the night window
# EARLIEST_START..LATEST_START (default 20:00..04:00); queued in the daytime, it
# waits for the evening. It gives up if the GPU is still busy at LATEST_START.
# run_shape.sh itself stops at 06:45 and resumes on the next run.
set -uo pipefail

shape=${1:?usage: wait_and_run.sh <shape>}
CHECK_EVERY=${CHECK_EVERY:-120}     # seconds between checks
QUIET_CHECKS=${QUIET_CHECKS:-3}     # consecutive free checks needed (~6 min)
EARLIEST_START=${EARLIEST_START:-20:00}
LATEST_START=${LATEST_START:-04:00}
me=$(whoami)

others() {
  # one line per GPU process that belongs to someone else
  nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null |
  while IFS=, read -r pid name; do
    pid=${pid// /}
    case $name in *dorado*) continue ;; esac
    owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
    [ "$owner" = "$me" ] && continue
    echo "${owner:-?} pid=$pid${name}"
  done
}

# minutes since midnight, for the night-window test
mins() { local h=${1%%:*} m=${1##*:}; echo $((10#$h * 60 + 10#$m)); }
in_window() {
  local now e l
  now=$(mins "$(date +%H:%M)"); e=$(mins "$EARLIEST_START"); l=$(mins "$LATEST_START")
  if [ "$e" -le "$l" ]; then [ "$now" -ge "$e" ] && [ "$now" -lt "$l" ]
  else [ "$now" -ge "$e" ] || [ "$now" -lt "$l" ]; fi    # window crosses midnight
}

if ! in_window; then
  echo "$(date +%H:%M) outside the night window ($EARLIEST_START-$LATEST_START);" \
       "waiting until $EARLIEST_START"
  while ! in_window; do sleep 60; done
fi

deadline=$(date -d "$LATEST_START" +%s)
[ "$deadline" -le "$(date +%s)" ] && deadline=$(date -d "tomorrow $LATEST_START" +%s)
echo "queued $shape at $(date +%H:%M); will start once the GPU is free for" \
     "$QUIET_CHECKS checks in a row, or give up at $LATEST_START"

quiet=0
while :; do
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "$(date +%H:%M) GPU never freed up before $LATEST_START -- not starting"; exit 1
  fi
  busy=$(others)
  if [ -z "$busy" ]; then
    quiet=$((quiet + 1))
    echo "$(date +%H:%M) free ($quiet/$QUIET_CHECKS)"
    [ $quiet -ge $QUIET_CHECKS ] && break
  else
    quiet=0
    echo "$(date +%H:%M) busy: $(echo "$busy" | tr '\n' ' ')"
  fi
  sleep "$CHECK_EVERY"
done

echo "$(date +%H:%M) GPU free -- starting $shape"
# FORCE=1 only skips run_shape's own check, which would trip on dorado.
FORCE=1 exec bash "$(dirname "$0")/run_shape.sh" "$shape"

#!/usr/bin/env bash
# Cache the PeptideCLM-2 (GPU) and PepLand (CPU) embeddings of the pool, both at once.
#
#   tmux new -s cache
#   bash ~/code/peptidefusion/scripts/run_cache.sh
#   PL_PROCS=14 PL_THREADS=2 bash ~/code/peptidefusion/scripts/run_cache.sh     # other split of the CPU
#   VIEWS=pepland PL_PROCS=24 bash ~/code/peptidefusion/scripts/run_cache.sh    # PepLand only (CPU; no GPU needed or checked)
#   VIEWS=pclm bash ~/code/peptidefusion/scripts/run_cache.sh                   # PeptideCLM-2 only (GPU)
#   VIEWS=pclm_mtr bash ~/code/peptidefusion/scripts/run_cache.sh               # PeptideCLM-2 MTR weights only (GPU), features/corpus/pclm_mtr
#   (VIEWS defaults to "pclm pepland"; ESM is cached separately: cache_views.py --view esm --workers 12 --esm-caps)
#
# PepLand runs as PL_PROCS processes of PL_THREADS threads (default 27 x 1; lab limit: the processes
# plus the GPU job use at most 32 of the 64 cores). Process i takes the pool shards with
# shard % PL_PROCS == i. Everything is resumable: finished shards are skipped, PepLand keeps a
# partial file inside the shard it is working on, so Ctrl+C, the deadline (DEADLINE=HH:MM, default
# 06:45) or a crash cost minutes. Run the same command again to continue.
# Progress any time: python ~/code/peptidefusion/src/corpus/cache_views.py --status (env peptide)
# Logs: $LOG_DIR/cache_pclm_*.log and cache_pepland_p<i>_*.log. FORCE=1 skips the GPU and daytime checks.
set -uo pipefail
source "$(dirname "$0")/env.sh"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

CV=$CODE/src/corpus/cache_views.py
PL_PROCS=${PL_PROCS:-27}
PL_THREADS=${PL_THREADS:-1}
VIEWS=${VIEWS:-"pclm pepland"}
want() { case " $VIEWS " in *" $1 "*) return 0 ;; esac; return 1; }
STAMP=$(date +%Y%m%d_%H%M)
echo "== $(date) | views: $VIEWS | PepLand: $PL_PROCS processes x $PL_THREADS thread(s) | PeptideCLM-2: GPU"

# --- lab rules ----------------------------------------------------------------------
if [ $(( PL_PROCS * PL_THREADS + 1 )) -gt $(( $(nproc --all) / 2 )) ]; then
  echo "$PL_PROCS x $PL_THREADS threads + the GPU job is more than half of $(nproc --all) cores -- not starting"; exit 2
fi
hour=$(date +%H)
if [ "${FORCE:-0}" != 1 ] && (( 10#$hour >= 7 && 10#$hour < 20 )); then
  echo "it is $(date +%H:%M): long jobs run at night (20:00-07:00). FORCE=1 to start anyway"; exit 2
fi
busy=""
{ want pclm || want pclm_mtr; } && busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v dorado || true)
if [ -n "$busy" ] && [ "${FORCE:-0}" != 1 ]; then
  echo "GPU in use by:"
  for pid in $(echo "$busy" | cut -d, -f1); do ps -o user=,etime=,args= -p "$pid"; done
  echo "not starting (FORCE=1 to override)"; exit 2
fi
avail=$(free -g | awk '/^Mem:/ {print $7}')
if want pepland && [ "$avail" -lt $(( PL_PROCS * 3 )) ]; then
  echo "only $avail GB of RAM available for $PL_PROCS PepLand processes (about 3 GB each): lower PL_PROCS"; exit 2
fi

# --- time budget --------------------------------------------------------------------
DEADLINE=${DEADLINE:-06:45}
now=$(date +%s); stop=$(date -d "$DEADLINE" +%s)
[ "$stop" -le "$now" ] && stop=$(date -d "tomorrow $DEADLINE" +%s)
t=$(( stop - now ))
echo "stopping at $DEADLINE ($(( t / 60 )) min from now)"

# --- the pool as plain text (the pepland env cannot read parquet) ----------------------
conda activate peptide
[ -f "$DATA/data/corpus/pool_smiles/index.json" ] || python "$CV" --export || exit 1

# --- start everything in the background -------------------------------------------------
trap 'echo; echo "== stopping: the processes save their partial shards and exit"; kill $(jobs -p) 2>/dev/null; wait; python "$CV" --status; exit 130' INT TERM

conda activate pepland
for (( i = 0; i < PL_PROCS; i++ )); do
  want pepland || break
  OMP_NUM_THREADS=$PL_THREADS MKL_NUM_THREADS=$PL_THREADS OPENBLAS_NUM_THREADS=$PL_THREADS \
  timeout --signal=INT --kill-after=120 "$t" \
    python "$CV" --view pepland --threads "$PL_THREADS" --part "$i" --parts "$PL_PROCS" \
    > "$LOG_DIR/cache_pepland_p${i}_$STAMP.log" 2>&1 &
done

conda activate peptide
want pclm && timeout --signal=INT --kill-after=120 "$t" python "$CV" --view pclm > "$LOG_DIR/cache_pclm_$STAMP.log" 2>&1 &
want pclm_mtr && timeout --signal=INT --kill-after=120 "$t" python "$CV" --view pclm_mtr > "$LOG_DIR/cache_pclm_mtr_$STAMP.log" 2>&1 &

sleep 150
echo "== $(date +%H:%M): $(jobs -rp | wc -l) processes still running"
for f in "$LOG_DIR/cache_pclm_$STAMP.log" "$LOG_DIR/cache_pclm_mtr_$STAMP.log" "$LOG_DIR/cache_pepland_p0_$STAMP.log"; do
  [ -f "$f" ] && { echo "--- $f"; tail -n 5 "$f"; }
done

# --- watch ----------------------------------------------------------------------------
while [ -n "$(jobs -rp)" ]; do
  sleep 300
  echo; echo "== $(date +%H:%M)"
  python "$CV" --status
done
echo; echo "== all processes ended $(date +%H:%M)"
python "$CV" --status
echo "(a view that is not at 27 / 27 shards: run this script again, it continues where it stopped)"

#!/usr/bin/env bash
# RDKit descriptors (src/rdkit_desc.py, 206 values per molecule): the fourth view of teacher v2.
# CPU only, env peptide. Three stages:
#
#   bench   the benchmark molecules: features/single_modality/rdkit.npz (CellPPD, THPep, PepMSND, PAMPA, AmpHGT) and
#           features/pepland_suite/rdkit.npz (PepLand's five tasks; the long c-Sol sequences make this one the slow
#           one). Minutes to about 30 min. Any time of day (BENCH_WORKERS, default 8).
#   time    times 400 pool molecules with the real worker count and prints the whole-pool estimate. A minute or two.
#   cache   every pool shard -> features/corpus/rdkit/shard_NNN.npz (+ names.json). Resumable (a finished shard is
#           skipped); stops at DEADLINE (default 06:45); night only (20:00-07:00) unless FORCE=1.
#           WORKERS (default 24; the lab limit is 32 of the 64 cores in total).
#   status  progress of every cached view.
#
#   tmux new -s rdkit
#   bash ~/code/peptidefusion/scripts/run_rdkit.sh bench
#   bash ~/code/peptidefusion/scripts/run_rdkit.sh time
#   bash ~/code/peptidefusion/scripts/run_rdkit.sh cache        # at night; run again to continue after a stop
# Logs: $LOG_DIR/rdkit_<stage>_*.log
set -uo pipefail
source "$(dirname "$0")/env.sh"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

STAGE=${1:-status}
LOG=$LOG_DIR/rdkit_${STAGE}_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "== $(date) | stage $STAGE | log: $LOG"
S=$CODE/src
WORKERS=${WORKERS:-24}
BENCH_WORKERS=${BENCH_WORKERS:-8}
MAIN5="CellPPD THPep PepMSND PAMPA AmpHGT"
PL="PL_cCPP PL_cSol PL_ncCPP PL_cBinding PL_ncBinding"
conda activate peptide

too_many() {
  if [ "$1" -gt 32 ] || [ "$1" -gt $(( $(nproc --all) / 2 )) ]; then
    echo "$1 workers is more than half of the $(nproc --all) cores: not starting"; return 0
  fi
  return 1
}

case "$STAGE" in
  bench)
    too_many "$BENCH_WORKERS" && exit 2
    echo "== main benchmarks ($(date +%H:%M))"
    python "$S/featurize.py" --rep rdkit --bench $MAIN5 --workers "$BENCH_WORKERS" || exit 1
    echo; echo "== PepLand's five tasks ($(date +%H:%M))"
    python "$S/featurize.py" --rep rdkit --feat pepland_suite --bench $PL --workers "$BENCH_WORKERS"
    ;;
  time)
    too_many "$WORKERS" && exit 2
    python "$S/corpus/cache_views.py" --view rdkit --workers "$WORKERS" --time 400
    ;;
  cache)
    too_many "$WORKERS" && exit 2
    hour=$(date +%H)
    if [ "${FORCE:-0}" != 1 ] && (( 10#$hour >= 7 && 10#$hour < 20 )); then
      echo "it is $(date +%H:%M): long jobs run at night (20:00-07:00). FORCE=1 to start anyway"; exit 2
    fi
    DEADLINE=${DEADLINE:-06:45}
    now=$(date +%s); stop=$(date -d "$DEADLINE" +%s)
    [ "$stop" -le "$now" ] && stop=$(date -d "tomorrow $DEADLINE" +%s)
    t=$(( stop - now ))
    echo "== rdkit cache: $WORKERS workers, stopping at $DEADLINE ($(( t / 60 )) min from now)"
    [ -f "$DATA/data/corpus/pool_smiles/index.json" ] || python "$S/corpus/cache_views.py" --export || exit 1
    timeout --signal=INT --kill-after=120 "$t" python "$S/corpus/cache_views.py" --view rdkit --workers "$WORKERS"
    rc=$?
    python "$S/corpus/cache_views.py" --status
    [ $rc = 0 ] || echo "== stopped (exit $rc); run the same command again to continue"
    exit $rc
    ;;
  status)
    python "$S/corpus/cache_views.py" --status
    ;;
  *) echo "usage: run_rdkit.sh bench|time|cache|status"; exit 1 ;;
esac
echo; echo "== done $(date +%H:%M)"

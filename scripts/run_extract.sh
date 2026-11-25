#!/usr/bin/env bash
# Corpus extraction: filter all three raw sources down to the peptides we keep.
#
#   tmux new -s extract
#   THREADS=32 bash ~/code/peptidefusion/scripts/run_extract.sh
#
# CPU only -- the GPU is never touched, so this can run alongside someone's job.
# Cost at THREADS=32 (the per-source figures are measured, from the 2026-11-02 run):
#   prefilter check   ~10 min   benchmarks + a 400k PubChem sample
#   pepland_stage1    ~40 min   7.9M rows, all peptides
#   pclm_pubchem      ~50 min   108M rows, ~82% dropped by the pre-filter first
#   pclm_esmatlas     ~6 h      9.5M rows, and its long peptides are the slow ones
# Cheapest first, so a short night still finishes something.
#
# Safe to re-run: finished shards and finished sources are skipped. Stops at
# 06:45 (DEADLINE=HH:MM) and resumes at the last complete shard the next night.
#
# The PubChem step refuses to start unless the pre-filter provably drops no real
# peptide, because one it drops is one we can never get back. SKIP_GATE=1
# overrides, after you have read results/corpus_inventory/prefilter_check.json.
set -uo pipefail
source "$(dirname "$0")/env.sh"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

LOG=$LOG_DIR/extract_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "== $(date) | log: $LOG | CPU threads: $THREADS of $(nproc --all)"
if [ "$THREADS" -gt $(( $(nproc --all) / 2 )) ]; then
  echo "THREADS=$THREADS is more than half of $(nproc --all) -- lab rule, not starting"; exit 2
fi

# --- sources present? ------------------------------------------------------------
for need in "$DATA/data/sources/pepland/pep_atlas_uniparc_smiles.zip" \
            "$DATA/data/sources/pclm_esmatlas/part_00000.parquet" \
            "$DATA/data/sources/pclm_pubchem/part_00000.parquet"; do
  test -e "$need" || { echo "missing $need -- extract corpus_sources.tar into $DATA/data"; exit 1; }
done
for d in pclm_esmatlas pclm_pubchem; do
  n=$(ls "$DATA/data/sources/$d"/part_*.parquet | wc -l)
  [ "$n" = 1362 ] || { echo "$d has $n parts, expected 1362 -- transfer incomplete"; exit 1; }
done
echo "sources ok | free on data drive: $(df -h "$DATA" | awk 'NR==2{print $4}')"

# --- time budget -----------------------------------------------------------------
DEADLINE=${DEADLINE:-06:45}
now=$(date +%s); stop=$(date -d "$DEADLINE" +%s)
[ "$stop" -le "$now" ] && stop=$(date -d "tomorrow $DEADLINE" +%s)
left() { echo $(( stop - $(date +%s) )); }
step() {   # name command...
  local name=$1; shift
  local t=$(left)
  if [ "$t" -le 300 ]; then echo "== $name: past $DEADLINE, stopping; re-run to continue"; exit 0; fi
  echo; echo "== $name  ($(date +%H:%M))"
  timeout --signal=INT --kill-after=120 "$t" "$@"
  local rc=$?
  if [ $rc = 124 ] || [ $rc = 130 ]; then echo "== $name stopped at $DEADLINE; re-run to continue"; exit 0; fi
  [ $rc = 0 ] || { echo "== $name FAILED (exit $rc)"; exit $rc; }
}

conda activate peptide
CHECK=$RESULTS_DIR/corpus_inventory/prefilter_check.json

[ -f "$CHECK" ] || step prefilter_check python "$CODE/src/corpus/prefilter_check.py" \
  --rows "${CHECK_ROWS:-400000}" --workers "$THREADS"

step pepland_stage1 python "$CODE/src/corpus/extract.py" --source pepland_stage1 --workers "$THREADS"

# --- gate: the pre-filter must not have dropped a single real peptide ------------
echo
python - "$CHECK" <<'PY'
import json, sys
g = json.load(open(sys.argv[1]))["gate"]
print("== pre-filter (>=%d carbonyls): pubchem recall %s, benchmark peptides dropped %d,"
      " rejects %s of the file" % (g["threshold"], g["pubchem_recall"],
                                   g["benchmark_peptides_dropped"], g["rejects_frac_of_file"]))
sys.exit(0 if g["pubchem_recall"] >= 1.0 and g["benchmark_peptides_dropped"] == 0 else 3)
PY
if [ $? != 0 ] && [ "${SKIP_GATE:-0}" != 1 ]; then
  echo "the pre-filter drops real peptides -- PubChem is our only non-canonical source,"
  echo "so not starting. Read $CHECK, lower MIN_CARBONYLS in src/corpus/extract.py,"
  echo "delete that json and re-run. (SKIP_GATE=1 to accept the loss)"; exit 3
fi

step pclm_pubchem  python "$CODE/src/corpus/extract.py" --source pclm_pubchem  --workers "$THREADS"
step pclm_esmatlas python "$CODE/src/corpus/extract.py" --source pclm_esmatlas --workers "$THREADS"

echo; echo "== finished $(date)"
for s in pepland_stage1 pclm_pubchem pclm_esmatlas; do
  echo "--- $s"; cat "$DATA/data/corpus/sources/$s.json"
done

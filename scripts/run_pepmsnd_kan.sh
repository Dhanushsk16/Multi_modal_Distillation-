#!/usr/bin/env bash
# PepMSND with KAN networks (src/teacher/pepmsnd_kan_v3.py): the benchmark's own recipe (KAN on 140 RDKit descriptors + species / environment, joined
# with an embedding branch by learnable weights) for the frozen v3_base teacher and every baseline input, 10-fold cluster CV, 3 seeds.
#
#   bash ~/code/peptidefusion/scripts/run_pepmsnd_kan.sh                       # our own KAN layer: nothing to install
#   KAN_IMPL=pykan bash ~/code/peptidefusion/scripts/run_pepmsnd_kan.sh        # the official pykan package (install the wheel first, see below)
#   EXTRA="--reps teacher_v3_base --seeds 1" bash ~/code/peptidefusion/scripts/run_pepmsnd_kan.sh      # a quick look
# Settings (environment): TEACHER=v3_base  KAN_IMPL=ours|pykan  SEEDS=3  OUT=pepmsnd_kan_<impl>  EXTRA="" (more options of the python script)
#
# The server has no internet, so `pip install pykan` cannot work there. pykan is ONE pure-python wheel (78 KB); download it elsewhere, copy it over and
# install it without dependencies (the environment already has torch, numpy, pandas, matplotlib, sympy, tqdm, pyyaml):
#   download:       python -m pip download pykan --no-deps -d kan_wheels
#   server:         conda activate peptide && pip install --no-deps ~/pykan-0.2.8-py3-none-any.whl && python -c "import kan; print(kan.__file__)"
# If the import names a missing module, that module is the only other thing to download the same way (pip download NAME --no-deps ...).
set -uo pipefail
source "$(dirname "$0")/env.sh"
TEACHER=${TEACHER:-v3_base}
KAN_IMPL=${KAN_IMPL:-ours}
SEEDS=${SEEDS:-3}
OUT=${OUT:-pepmsnd_kan_$KAN_IMPL}
EXTRA=${EXTRA:-}
LOG=$LOG_DIR/pepmsnd_kan_${KAN_IMPL}_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate peptide
echo "== $(date) | PepMSND + KAN | teacher $TEACHER | KAN: $KAN_IMPL | seeds $SEEDS | results/$OUT | log $LOG"
if [ "$KAN_IMPL" = pykan ]; then
  python -c "import kan" 2> /dev/null || { echo "pykan is not installed in this environment: see the header of this script (one wheel to upload), or run with KAN_IMPL=ours"; exit 3; }
fi
# shellcheck disable=SC2086
python "$CODE/src/teacher/pepmsnd_kan_v3.py" --teacher "$TEACHER" --kan-impl "$KAN_IMPL" --seeds "$SEEDS" --out "$OUT" $EXTRA
rc=$?
echo "== done $(date +%H:%M) (exit status $rc)"
exit $rc

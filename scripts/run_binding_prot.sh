#!/usr/bin/env bash
# PepLand's two binding tasks (c-binding, nc-binding) AS INTENDED: the peptide AND its protein partner.
#   1. the protein partners (1,227 distinct sequences, column seq1 of the two files) are embedded with ESM-2 3B (GPU, a few minutes + the
#      11 GB model load) -> features/pepland_suite/protein_esm.npz   (src/featurize_esm.py --proteins)
#   2. every learner (linear, gbt, mlp, kernel) is run on  [representation | protein vector]  for the frozen v3_base teacher and for every
#      baseline input (PepLand, PeptideCLM-2, ECFP, the three joined, PCA), plus the protein vector ALONE (the control: what does the peptide add?),
#      on four benchmarks (benchmarks.py):  PL_cBindingProt / PL_ncBindingProt   the same rows and folds as the peptide-only tasks
#                                           PL_cBindingProtUnseen / PL_ncBindingProtUnseen   folds by PROTEIN: no test protein in training
#   3. the fine-tuned teacher on the same four benchmarks (head on [teacher latent | protein vector], GPU, < 1 GB)
#   4. the summary: Spearman next to the published numbers and next to the SAME teacher without the protein (the earlier peptide-only run)
#
#   bash ~/code/peptidefusion/scripts/run_binding_prot.sh                       # everything; about 1-2 hours (estimate), mostly gbt
#   LEARNERS="linear kernel" FINETUNE=0 bash ~/code/peptidefusion/scripts/run_binding_prot.sh     # a quick look
#   FORCE_EMB=1 bash ...                                                         # embed the proteins again
# Settings (environment): TEACHER=v3_base  LEARNERS="linear gbt mlp kernel"  THREADS=8 (CPU threads, in WORKERS=4 processes, one per
#   benchmark)  FINETUNE=1  SKIP_DONE=1  NEED_MB=14000 (free GPU memory needed for the embedding step; it waits, no clock rule)
# Results: results/binding_prot_<learner> (harness), results/binding_prot_ft (fine-tuning); the summary is kept in logs/binding_prot_*.log
set -uo pipefail
source "$(dirname "$0")/env.sh"
SELF=$(cd "$(dirname "$0")" && pwd)
R=$SELF/run_teacher_v3.sh
TEACHER=${TEACHER:-v3_base}
LEARNERS=${LEARNERS:-"linear gbt mlp kernel"}
WORKERS=${WORKERS:-4}
FINETUNE=${FINETUNE:-1}
SKIP_DONE=${SKIP_DONE:-1}
FORCE_EMB=${FORCE_EMB:-0}
NEED_MB=${NEED_MB:-14000}
PREFIX=binding_prot
BENCH="PL_cBindingProt PL_ncBindingProt PL_cBindingProtUnseen PL_ncBindingProtUnseen"
PROT_FILE=$DATA/features/pepland_suite/protein_esm.npz
LOG=$LOG_DIR/binding_prot_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
say() { echo "$(date +%H:%M:%S) $*"; }
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate peptide
say "== binding tasks with the protein partner | teacher $TEACHER | learners $LEARNERS | threads $THREADS ($WORKERS workers) | log $LOG"

# ---------------------------------------------------------------------------------------- 1. the protein vectors
if [ "$FORCE_EMB" = 1 ] || [ ! -f "$PROT_FILE" ]; then
  if [ ! -d "$CKPT_DIR/esm2_t36_3B_UR50D" ]; then
    say "!! no ESM-2 3B weights in $CKPT_DIR/esm2_t36_3B_UR50D (the pool's ESM view was made with them: where did they go?)"; exit 3
  fi
  while true; do                                    # only the free memory counts; nothing here looks at the clock
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2> /dev/null | head -1 | tr -d ' ')
    [ -n "${free:-}" ] && [ "$free" -ge "$NEED_MB" ] && break
    say "waiting for the GPU: ${free:-?} MB free, $NEED_MB MB needed (checking every 60 s)"; sleep 60
  done
  say "step 1: ESM-2 3B embeddings of the protein partners (GPU memory free: $free MB)"
  python "$CODE/src/featurize_esm.py" --proteins
  rc=$?
  if [ "$rc" != 0 ] || [ ! -f "$PROT_FILE" ]; then say "!! the protein embedding failed (exit status $rc): nothing else can run"; exit 4; fi
else
  say "step 1: $PROT_FILE exists, skipped (FORCE_EMB=1 makes it again)"
fi

# ---------------------------------------------------------------------------------------- 2. the learners
for L in $LEARNERS; do
  if [ "$SKIP_DONE" = 1 ] && [ -f "$RESULTS_DIR/${PREFIX}_${L}/headline.csv" ]; then say "${PREFIX}_${L}: already done, skipped"; continue; fi
  say "step 2: harness, learner $L"
  env THREADS=$THREADS BENCH="$BENCH" TEACHERS="$TEACHER" LEARNERS=$L SEEDS=3 BOOT=0 WORKERS=$WORKERS OUT=${PREFIX}_${L} \
    EXTRA="--reps PROT P L H P+L+H PCA --no-e-pass" bash "$R" harness || say "!! harness $L FAILED"
done

# ---------------------------------------------------------------------------------------- 3. fine-tuning
if [ "$FINETUNE" = 1 ]; then
  say "step 3: fine-tuning $TEACHER with the protein vector (one run per benchmark)"
  # shellcheck disable=SC2086
  bash "$R" finetune "$TEACHER" --suite pepland --bench $BENCH --seeds 1 --out ${PREFIX}_ft --resume || say "!! fine-tuning FAILED"
fi

# ---------------------------------------------------------------------------------------- 4. the summary
say "== SUMMARY"
python "$CODE/src/teacher/binding_prot_summary.py" --prefix "$PREFIX" --teacher "$TEACHER" --learners $LEARNERS
say "== done."

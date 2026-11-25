#!/usr/bin/env bash
# Student tokenizer: learn atom-level BPE tokenizers (vocab 1024 / 2048 / 4096) on a sample of the pool and compare them
# with PeptideCLM-2's released tokenizer (A) and the same tokenizer with its digit defect fixed (B). CPU only (no GPU, so no
# GPU check), 8 threads, a few GB of RAM, roughly 10-30 minutes. See src/student/train_tokenizer.py.
#
#   tmux new -s tok
#   bash ~/code/peptidefusion/scripts/run_student_tokenizer.sh
#   VOCAB="2048" bash ...                      one vocabulary size only
#   bash ... --n-train 100000 --max-token-len 32    extra arguments go to train_tokenizer.py
#   REPORT=1 bash ...                          no learning: report A, B and the tokenizers already saved
# Output: $CKPT_DIR/student_tokenizer/atombpe_v*/ (the tokenizers), $RESULTS_DIR/student_tokenizer/ (summary.csv, by_band.csv,
# token_use.csv, examples.txt, report.json). Log: $LOG_DIR/student_tokenizer_*.log
set -uo pipefail
source "$(dirname "$0")/env.sh"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate peptide

LOG=$LOG_DIR/student_tokenizer_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "== $(date) | log: $LOG"
echo "threads: $THREADS (rayon $RAYON_NUM_THREADS) | cores: $(nproc) | load: $(cut -d' ' -f1-3 /proc/loadavg) | memory: $(free -g | awk '/Mem:/ {print $7" GB available of "$2" GB"}')"

POOL=$DATA/data/corpus/pool.parquet
[ -f "$POOL" ] || { echo "no pool at $POOL"; exit 1; }

ARGS=()
if [ "${REPORT:-0}" = 1 ]; then ARGS+=(--report-only); else ARGS+=(--vocab ${VOCAB:-1024 2048 4096}); fi
python "$CODE/src/student/train_tokenizer.py" "${ARGS[@]}" "$@"
rc=$?
echo; echo "== done $(date +%H:%M) (exit $rc)"
exit $rc

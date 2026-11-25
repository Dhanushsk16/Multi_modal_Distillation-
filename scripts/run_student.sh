#!/usr/bin/env bash
# The SMILES-only student, distilled from the frozen teacher v3_base (src/student/). Stages:
#
#   bash ~/code/peptidefusion/scripts/run_student.sh status                  # what exists on the server (read-only, a few seconds)
#   bash ~/code/peptidefusion/scripts/run_student.sh norm                    # CPU, ~20-30 min: neutralise + canonicalise the whole pool -> data/corpus/pool_norm.parquet
#   bash ~/code/peptidefusion/scripts/run_student.sh tok                     # CPU, some GB of RAM: learn the student tokenizer (vocab 1024, stratified sample, normalised)
#   FIRST=5 bash ~/code/peptidefusion/scripts/run_student.sh embed           # GPU, minutes: the teacher's latent of the first 5 shards (the 10% subset); FIRST=0 = every shard
#   FIRST=5 bash ~/code/peptidefusion/scripts/run_student.sh data            # CPU: token ids of the shards the teacher cache has
#   bash ~/code/peptidefusion/scripts/run_student.sh train st1 --targets latent --first 5 --epochs 60 --patience 8    # one run (any train_student.py flags)
#   bash ~/code/peptidefusion/scripts/run_student.sh ladder                  # the five-step ladder + the MLM-only control on the 10% subset, one after the other
#   STEPS="st1 st2" bash ~/code/peptidefusion/scripts/run_student.sh ladder  # only some of them: st1 st2 st3 st4 st5 stc st6 st7 stca
#   bash ~/code/peptidefusion/scripts/run_student.sh export st1              # benchmark embeddings of a trained student (the files the harness reads)
#   LEARNERS=linear BOOT=1000 bash ~/code/peptidefusion/scripts/run_student.sh harness st1 st1_pre st2   # CPU: v3_base + these through the harness
#
# The ladder:
#   st1  teacher latent only, cosine + smooth L1                              st4  NO latent: views P L H E only (+ InfoNCE + soft similarity matrix)
#   st2  st1 + InfoNCE + soft similarity matrix                               st5  st3 + masked-token loss on 20% of the steps
#   st3  st2 + the four view heads, EMA loss balancing, 10% head dropping     stc  the MLM-ONLY control: same architecture and data, masked-token loss only
#   st6  st3 + RANDOM-SMILES AUGMENTATION (read in a random spelling with p 0.5, targets stay the canonical string's) + spelling-consistency loss
#   st7  st5 + the same augmentation = everything on                          stca the MLM-only control WITH the augmentation (the fair control for st6 / st7)
#   st3f / stcf / st6f = st3 / stc / st6 on the FULL pool (FIRST=0 for embed, data and ladder; about 28 min per epoch for st3f at the shared-GPU speed of 2026-11-22;
#   st6f about 1.6x that: random spellings are longer and the consistency loss is an extra forward pass)
#   SAVE_EVERY=2 (default): the EMA weights are also kept as epoch_NN.pt after every 2nd epoch (SAVE_EVERY=0 = off); benchmark one with
#   python src/student/export_student.py --name st6f --epoch 4   (written as st6f_e04, the export of best.pt stays)
#   (the validation line of every run reports `spell` = cosine between the outputs for the canonical and a random spelling: the baseline for the augmentation)
# Settings (environment): TEACHER=v3_base  TOK=<tokenizer folder>  VOCAB=1024  NTRAIN=500000  RANDFRAC=0.5 (tok)  KRAND=8 (data)  PRANDOM=0.5 (st6 st7 stca)
#   FIRST=5  EPOCHS=60  PATIENCE=8  SEED=1  NEED_MB=15000  DATAON=cpu (the teacher's target vectors stay in RAM, not on the GPU: -6 GB for st3, identical results)
#   RETRIES=30 (a run that dies of "out of memory" waits 5 min and continues from its last epoch; started again, an unfinished run continues too)
#   EXTRA="..." (added to every ladder run, e.g. EXTRA="--ladder" or "--batch 1024")  STEPS="st1 st2 ..."  WORKERS=8
# GPU stages wait until NEED_MB of VRAM are free (only the free memory counts; no night rule, no deadline). Logs: $LOG_DIR/student_<stage>_*.log
set -uo pipefail
source "$(dirname "$0")/env.sh"
STAGE=${1:?usage: run_student.sh status|norm|tok|embed|data|train NAME [args]|ladder|export NAME|harness NAME...}
shift
LOG=$LOG_DIR/student_${STAGE}_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
say() { echo "$(date +%H:%M:%S) $*"; }
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate peptide
S=$CODE/src/student
TEACHER=${TEACHER:-v3_base}
VOCAB=${VOCAB:-1024}
TOKDIR=$CKPT_DIR/student_tokenizer_norm
TOK=${TOK:-$TOKDIR/atombpe_v$VOCAB}
FIRST=${FIRST:-5}
NEED_MB=${NEED_MB:-15000}                             # st3 measured 18.7 GB with the targets on the GPU; with them in RAM (DATAON=cpu) about 12 GB (estimate)
rc=0
say "== stage $STAGE | log $LOG"

wait_gpu() {                                          # only the free memory counts
  local free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2> /dev/null | head -1 | tr -d ' ')
    [ -n "${free:-}" ] && [ "$free" -ge "$1" ] && break
    say "waiting for the GPU: ${free:-?} MB free, $1 MB needed (checking every 60 s)"; sleep 60
  done
}

need_inputs() {                                       # the files train / ladder need
  [ -d "$TOK" ] || { say "!! no tokenizer at $TOK (stage tok)"; return 1; }
  [ -d "$DATA/features/corpus/student_$(basename "$TOK")" ] || { say "!! no token ids (stage data)"; return 1; }
  [ -d "$DATA/features/corpus/teacher_$TEACHER" ] || { say "!! no teacher latents in features/corpus/teacher_$TEACHER (stage embed)"; return 1; }
}

train_one() {                                         # NAME [train_student.py args]
  # A run is FINISHED only when train_student.py exited cleanly (file COMPLETE; best.pt alone is written after the first epoch). A run that was
  # interrupted or crashed continues from its last finished epoch (last.pt) when this is started again. The GPU is shared: if the run dies with
  # "out of memory" (another user took the memory), it waits 5 minutes and continues from its last finished epoch, up to RETRIES times.
  local name=$1; shift
  local rd=$CKPT_DIR/student_distill/$name resume="" try=0
  if [ -f "$rd/COMPLETE" ] && [ "${REDO:-0}" != 1 ] && [[ " $* " != *" --resume "* ]]; then
    say "== $name already trained ($rd): skipped (REDO=1 trains it again from scratch)"; return 0
  fi
  [ "${REDO:-0}" = 1 ] && rm -f "$rd/COMPLETE"
  [ "${REDO:-0}" != 1 ] && [ -f "$rd/last.pt" ] && { resume="--resume"; say "== $name was started before: it continues from its last finished epoch"; }
  export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
  while true; do
    wait_gpu "$NEED_MB"
    say "=== train $name | $* $resume"
    if python "$S/train_student.py" --name "$name" --tok "$TOK" --teacher "$TEACHER" "$@" $resume; then
      touch "$rd/COMPLETE"; return 0
    fi
    try=$((try + 1)); sleep 5                          # (lets the log catch up before it is searched)
    if tail -n 80 "$LOG" | grep -q "out of memory" && [ "$try" -lt "${RETRIES:-30}" ]; then
      say "!! $name ran out of GPU memory (shared GPU): waiting 5 minutes, then it continues from its last finished epoch (try $try of ${RETRIES:-30})"
      sleep 300; resume="--resume"
    else
      say "!! training $name failed"; return 1
    fi
  done
}

case "$STAGE" in
  status)
    say "-- what the student needs, on this server"
    for f in "$DATA/data/corpus/pool.parquet" "$DATA/data/corpus/pool_norm.parquet" "$DATA/features/corpus/views_v3_stats.npz" \
             "$CKPT_DIR/teacher/$TEACHER/best.pt" "$CKPT_DIR/teacher/$TEACHER/latent_stats.npz"; do
      [ -e "$f" ] && echo "  yes  $f ($(du -h "$f" | cut -f1))" || echo "  NO   $f"
    done
    for d in pepland pclm ecfp esm pclm_mtr "teacher_$TEACHER"; do
      p=$DATA/features/corpus/$d
      [ -d "$p" ] && echo "  yes  features/corpus/$d: $(ls "$p" | grep -c '^shard_[0-9]*\.npz$') shards ($(du -sh "$p" | cut -f1))" || echo "  NO   features/corpus/$d"
    done
    for d in "$DATA"/features/corpus/student_*; do [ -d "$d" ] && echo "  yes  token ids $(basename "$d"): $(ls "$d" | grep -c '^shard_[0-9]*\.npz$') shards"; done
    for d in "$CKPT_DIR"/student_tokenizer "$CKPT_DIR"/student_tokenizer_norm; do
      [ -d "$d" ] && echo "  yes  $d: $(ls "$d" | tr '\n' ' ')" || echo "  NO   $d"
    done
    [ -d "$CKPT_DIR/student_distill" ] && echo "  trained students: $(ls "$CKPT_DIR/student_distill" | tr '\n' ' ')"
    echo "  free VRAM: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader -i 0 2> /dev/null) | RAM available: $(free -g | awk '/Mem:/ {print $7" GB of "$2" GB"}') | cores $(nproc --all), load $(cut -d' ' -f1-3 /proc/loadavg)"
    ;;
  norm)
    python "$S/normalise.py" --write-pool "$DATA/data/corpus/pool_norm.parquet" --workers "${WORKERS:-8}"
    rc=$?
    ;;
  tok)
    # one vocabulary, a stratified half-non-canonical sample of the pool, normalised first (the student's input form); learned tokenizers go to
    # student_tokenizer_norm so the earlier un-normalised ones (checkpoints/student_tokenizer) cannot be picked up by mistake
    # RANDFRAC of the learning sample is written as random spellings (random-SMILES augmentation, decided 2026-11-22): without them BPE merges only
    # canonical fragments and random spellings fall apart into near-letters (2.7x longer in the laptop test); the report and held-out set stay canonical
    python "$S/train_tokenizer.py" --vocab "$VOCAB" --sample stratified --nc-frac 0.5 --n-train "${NTRAIN:-500000}" --fixed-heldout \
      --random-frac "${RANDFRAC:-0.5}" --tok-dir "$TOKDIR" --out "$RESULTS_DIR/student_tokenizer_norm" --workers "${WORKERS:-8}" "$@"
    rc=$?
    ;;
  embed)
    wait_gpu "${NEED_MB_EMBED:-8000}"
    python "$CODE/src/teacher/train_teacher_v3.py" --name "$TEACHER" --embed-pool $([ "$FIRST" != 0 ] && echo "--first $FIRST")
    rc=$?
    ;;
  data)
    # KRAND random spellings per molecule next to the canonical one (RDKit random atom / branch order; KRAND=0 = none); ~4 ms per molecule per core
    python "$S/make_student_data.py" --tok "$TOK" --k-random "${KRAND:-8}" --workers "${WORKERS:-8}" $([ "$FIRST" != 0 ] && echo "--first $FIRST") "$@"
    rc=$?
    ;;
  train)
    NAME=${1:?usage: run_student.sh train NAME [args]}; shift
    need_inputs || exit 3
    train_one "$NAME" "$@"
    rc=$?
    ;;
  ladder)
    need_inputs || exit 3
    C="--first $FIRST --epochs ${EPOCHS:-60} --patience ${PATIENCE:-8} --seed ${SEED:-1} --data-on ${DATAON:-cpu} --save-every ${SAVE_EVERY:-2} ${EXTRA:-}"
    REL="--w-nce 1 --w-rel 1"
    ALL="--targets latent,P,L,H,E $REL --balance ema --teacher-drop 0.1"
    AUG="--p-random ${PRANDOM:-0.5} --w-consist 1 --consist-frac 0.25"
    for step in ${STEPS:-st1 st2 st3 st4 st5 stc st6 st7 stca}; do
      case $step in
        st1) A="--targets latent";;
        st2) A="--targets latent $REL";;
        st3) A="$ALL";;
        st4) A="--targets P,L,H,E $REL --balance ema --teacher-drop 0.1";;
        st5) A="$ALL --mlm-frac 0.2";;
        stc) A="--targets none --mlm-frac 1";;
        st6) A="$ALL $AUG";;                                  # st3 + random-SMILES augmentation + the spelling-consistency loss
        st7) A="$ALL --mlm-frac 0.2 $AUG";;                   # st5 + augmentation: everything on
        stca) A="--targets none --mlm-frac 1 --p-random ${PRANDOM:-0.5}";;   # the MLM-only control WITH the augmentation (fair control for st6 / st7)
        st3f) A="$ALL";;                                      # st3 on the FULL pool: run with FIRST=0 (all shards; embed and data must have been run with FIRST=0 too)
        st6f) A="$ALL $AUG";;                                 # st6 (st3 + random-SMILES augmentation + consistency loss) on the FULL pool: FIRST=0 as for st3f
        stcf) A="--targets none --mlm-frac 1";;               # the MLM-only control on the FULL pool (the matched control for st3f)
        *) say "!! unknown step $step"; continue;;
      esac
      # shellcheck disable=SC2086
      train_one "$step" $C $A || { rc=1; continue; }
      python "$S/export_student.py" --name "$step" --workers "${WORKERS:-8}" || rc=1
    done
    ;;
  export)
    NAME=${1:?usage: run_student.sh export NAME}
    python "$S/export_student.py" --name "$NAME" --workers "${WORKERS:-8}"
    rc=$?
    ;;
  harness)
    [ $# -ge 1 ] || { echo "usage: run_student.sh harness NAME [NAME_pre ...]"; exit 1; }
    # shellcheck disable=SC2086
    python "$CODE/src/teacher/eval_harness.py" --teachers "$TEACHER" "$@" --out "${OUT:-harness_student}" ${BENCH:+--bench $BENCH} ${SEEDS:+--seeds $SEEDS} \
      ${LEARNERS:+--learners $LEARNERS} ${BOOT:+--boot $BOOT} ${WORKERS_H:+--workers $WORKERS_H} ${HEXTRA:-}
    rc=$?
    ;;
  *) echo "usage: run_student.sh status|norm|tok|embed|data|train NAME [args]|ladder|export NAME|harness NAME..."; exit 1 ;;
esac
say "== done (exit status $rc)"
exit $rc

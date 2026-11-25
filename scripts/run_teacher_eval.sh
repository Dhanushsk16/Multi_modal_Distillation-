#!/usr/bin/env bash
# Benchmark tests of a finished TEACHER and nothing else (TEACHER=t8 by default). ECFP / RDKit / PeptideCLM-2 / PepLand baselines are
# NOT run here: their results are already stored under results/. One line per stage:
#
#   frozen       CPU     PeptideCLM-2's five benchmarks (CellPPD THPep AmpHGT PAMPA PepMSND): teacher embedding + linear / GBT / MLP,
#                        fold mean and pooled score                                            -> results/teacher_<T>_gate
#   pl-features  GPU     PepLand's five tasks need ESM features and the teacher's embedding first (minutes; the ESM model loads in ~2.5 min)
#   pl-frozen    CPU     PepLand's five tasks (PL_cCPP PL_cSol PL_ncCPP PL_cBinding PL_ncBinding): teacher + linear / GBT / MLP
#                                                                                              -> results/teacher_<T>_pl_gate
#   heads        small   PepLand's own head (frozen vector + its MLP) on the teacher, both suites -> results/head_main_teacher_<T>, head_pepland_teacher_<T>
#   finetune     GPU     the teacher's own networks, frozen vs fine-tuned at a low lr (teacher/finetune.py), both suites
#                                                                                              -> results/finetune_<T>, finetune_<T>_pepland
#   esm-rows     CPU     OPTIONAL, not part of `all`: raw ESM next to the teacher on the molecules both have (--same-rows), both suites
#   all          frozen, pl-features, pl-frozen, heads, finetune in this order, stopping at the first failure
#
#   tmux new -s eval
#   bash ~/code/peptidefusion/scripts/run_teacher_eval.sh frozen
#   TEACHER=t8 FT_ARGS="--seeds 3" bash ~/code/peptidefusion/scripts/run_teacher_eval.sh finetune
#   FT_ARGS="--modes frozen ft scratch ecfp" ...   # adds the two opt-in controls to the fine-tuning (not run by default)
#
# Rules kept: CPU stages use at most THREADS (8) threads; GPU stages refuse while another process computes on the GPU (FORCE=1 overrides);
# `finetune` refuses 07:00-20:00 (FORCE=1 overrides) and stops at DEADLINE (06:45): run the same line again, it continues (finished
# benchmark/variant pairs are skipped). Needs: features/single_modality/{pclm,pepland,esm,teacher_<T>}.npz and, for the PepLand stages,
# features/pepland_suite/{ecfp,pclm,pepland}.npz (inputs, not results; bash run_cross_eval.sh makes them; ecfp.npz only fixes the molecule
# order there). Log: $LOG_DIR/teacher_eval_<stage>_*.log
set -uo pipefail
source "$(dirname "$0")/env.sh"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

STAGE=${1:-}
T=${TEACHER:-t8}
LOG=$LOG_DIR/teacher_eval_${STAGE:-none}_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "== $(date) | stage ${STAGE:-none} | teacher $T | log: $LOG"
S=$CODE/src
F=$DATA/features/single_modality
P=$DATA/features/pepland_suite
PL="PL_cCPP PL_cSol PL_ncCPP PL_cBinding PL_ncBinding"
MAIN="CellPPD THPep PepMSND PAMPA AmpHGT"
LEARNERS=${LEARNERS:-"linear gbt mlp"}
conda activate peptide

need() { for f in "$@"; do [ -f "$f" ] || { echo "missing $f"; return 1; }; done; }

gpu_free() {   # 0 = free (or FORCE=1), 1 = someone else is on the GPU
  local busy
  busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v dorado || true)
  if [ -n "$busy" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "GPU in use by:"; for pid in $(echo "$busy" | cut -d, -f1); do ps -o user=,etime=,args= -p "$pid"; done
    echo "not starting (FORCE=1 to override)"; return 1
  fi
}

frozen() {
  need "$F/teacher_$T.npz" || return 1
  echo; echo "== frozen: teacher_$T on PeptideCLM-2's five benchmarks ($(date +%H:%M))"
  python "$S/eval_single.py" --reps teacher_$T --learners $LEARNERS --out teacher_${T}_gate
}

pl_features() {
  need "$P/ecfp.npz" "$P/pclm.npz" "$P/pepland.npz" || { echo "(bash run_cross_eval.sh makes the PepLand-suite features)"; return 1; }
  gpu_free || return 2
  if [ ! -f "$P/esm.npz" ]; then
    echo; echo "== ESM features of PepLand's suite ($(date +%H:%M)); same caps setting as the pool cache and the main benchmark file"
    python "$S/featurize_esm.py" --bench --feat pepland_suite --ignore-caps --workers 8 || return 1
  else
    echo "== $P/esm.npz exists: skipped"
  fi
  echo; echo "== teacher $T embeds PepLand's suite ($(date +%H:%M))"
  python "$S/teacher/train_teacher.py" --name "$T" --embed-bench-from pepland_suite
}

pl_frozen() {
  need "$P/teacher_$T.npz" || { echo "(run the pl-features stage first)"; return 1; }
  echo; echo "== pl-frozen: teacher_$T on PepLand's five tasks ($(date +%H:%M))"
  python "$S/eval_single.py" --feat pepland_suite --bench $PL --reps teacher_$T --learners $LEARNERS --out teacher_${T}_pl_gate
}

heads() {
  need "$F/teacher_$T.npz" || return 1
  echo; echo "== PepLand's head on teacher_$T, PeptideCLM-2's suite ($(date +%H:%M))"
  python "$S/finetune/pepland_head.py" --reps teacher_$T --bench $MAIN --seeds 3 --out head_main_teacher_$T || return 1
  need "$P/teacher_$T.npz" || { echo "(PepLand's suite skipped: run the pl-features stage first)"; return 1; }
  echo; echo "== PepLand's head on teacher_$T, PepLand's suite ($(date +%H:%M))"
  python "$S/finetune/pepland_head.py" --feat pepland_suite --reps teacher_$T --bench $PL --seeds 3 --out head_pepland_teacher_$T
}

finetune() {
  hour=$(date +%H)
  if [ "${FORCE:-0}" != 1 ] && (( 10#$hour >= 7 && 10#$hour < 20 )); then
    echo "it is $(date +%H:%M): long GPU jobs run at night (20:00-07:00). FORCE=1 to start anyway"; return 2
  fi
  gpu_free || return 2
  need "$F/pclm.npz" "$F/pepland.npz" "$F/esm.npz" || return 1
  DEADLINE=${DEADLINE:-06:45}
  now=$(date +%s); stop=$(date -d "$DEADLINE" +%s)
  [ "$stop" -le "$now" ] && stop=$(date -d "tomorrow $DEADLINE" +%s)
  t=$(( stop - now ))
  echo "== finetune: stopping at $DEADLINE ($(( t / 60 )) min from now); run the same line again to continue"
  echo; echo "== finetune: PeptideCLM-2's suite ($(date +%H:%M))"
  # shellcheck disable=SC2086
  timeout --signal=INT --kill-after=120 "$t" python "$S/teacher/finetune.py" --name "$T" --resume ${FT_ARGS:-}
  rc=$?; [ $rc = 0 ] || { echo "== stopped (exit $rc); run again to continue"; return $rc; }
  need "$P/pclm.npz" "$P/pepland.npz" "$P/esm.npz" "$P/teacher_$T.npz" || { echo "(PepLand's suite skipped: pl-features first)"; return 1; }
  now=$(date +%s); t=$(( stop - now ))
  [ "$t" -gt 600 ] || { echo "== no time left before $DEADLINE; run again"; return 0; }
  echo; echo "== finetune: PepLand's suite ($(date +%H:%M))"
  # shellcheck disable=SC2086
  timeout --signal=INT --kill-after=120 "$t" python "$S/teacher/finetune.py" --name "$T" --suite pepland --resume ${FT_ARGS:-}
  rc=$?; [ $rc = 0 ] || echo "== stopped (exit $rc); run again to continue"
  return $rc
}

esm_rows() {   # optional: raw ESM next to the teacher on the molecules both have
  need "$F/esm.npz" "$F/teacher_$T.npz" || return 1
  echo; echo "== esm-rows: PeptideCLM-2's suite ($(date +%H:%M))"
  python "$S/eval_single.py" --same-rows --reps esm teacher_$T --learners $LEARNERS --out teacher_${T}_esmrows || return 1
  need "$P/esm.npz" "$P/teacher_$T.npz" || { echo "(PepLand's suite skipped: pl-features first)"; return 1; }
  echo; echo "== esm-rows: PepLand's suite ($(date +%H:%M))"
  python "$S/eval_single.py" --feat pepland_suite --bench $PL --same-rows --reps esm teacher_$T --learners $LEARNERS --out teacher_${T}_pl_esmrows
}

case "$STAGE" in
  frozen) frozen ;;
  pl-features) pl_features ;;
  pl-frozen) pl_frozen ;;
  heads) heads ;;
  finetune) finetune ;;
  esm-rows) esm_rows ;;
  all) frozen && pl_features && pl_frozen && heads && finetune ;;
  *) echo "usage: run_teacher_eval.sh frozen|pl-features|pl-frozen|heads|finetune|esm-rows|all   (TEACHER=t8, see the header)"; exit 1 ;;
esac
rc=$?
echo; echo "== stage $STAGE finished, exit code $rc ($(date +%H:%M))"
exit $rc

#!/usr/bin/env bash
# The two peptide encoders on each other's benchmarks with each one's OWN downstream recipe
# (not frozen features + sklearn):
#   heads  PepLand's recipe: frozen encoder + its MLP head (lr 5e-5 / 1e-5, 100-200 epochs, batch 32)
#          on both suites; the same head also on PeptideCLM-2 and ECFP vectors for an equal-treatment table.
#          Minutes, any time. Needs the PepLand-suite features (run_cross_eval.sh makes them) for part B.
#   check  CellPPD only, same recipe as pclm below: a reproduction check (the authors report MCC 0.875) AND a
#          timing run. It prints how long PepLand's five tasks will take (they are ~42-45x the CellPPD training
#          work, counted by rows and by padded tokens). The GPU must be free (checked). Expect minutes on the
#          A6000 (unmeasured; the same job took 87 min on a Kaggle T4 in fp32).
#   teacher the finished teacher (TEACHER=t3) on PepLand's five tasks: embeds the PepLand-suite molecules with it,
#          then PepLand's head and the linear/GBT/MLP probes, next to ECFP / PeptideCLM-2 / PepLand. CPU, minutes,
#          any time. Results: results/head_pepland_teacher, results/cross_pepland_teacher.
#   complement  does the teacher add anything to ECFP? ecfp vs teacher vs ecfp+teacher on the 5 main benchmarks,
#          with ecfp+pclm, ecfp+pepland and the raw 3-view concat as controls (linear / GBT / MLP, 3 seeds).
#          CPU only, expect 30-60 min (the cluster-CV benchmarks PepMSND and PAMPA dominate). Results: results/complement_t3.
#   pclm   PeptideCLM-2's recipe (the authors' released code: LoRA r=16 + head, lr 3e-4, batch 16, <=10 epochs,
#          inner 5-fold ensemble) via src/finetune/finetune_smiles.py: first CellPPD (skipped if `check` already
#          ran), then PepLand's five tasks. GPU, hours (estimate 3-7 h; run `check` to measure): night only,
#          stops at DEADLINE.
#
#   tmux new -s proper
#   bash ~/code/peptidefusion/scripts/run_proper_eval.sh heads
#   bash ~/code/peptidefusion/scripts/run_proper_eval.sh check         # minutes: timing + reproduction check
#   bash ~/code/peptidefusion/scripts/run_proper_eval.sh pclm          # at night
#   INNER=3 bash ...pclm      # fewer inner models (faster, less faithful);  FORCE=1 skips the GPU / daytime checks
#   DEADLINE=06:45 (default). Finished folds are kept, so re-running continues where it stopped.
# Logs: $LOG_DIR/proper_*.log. Results: results/head_main, head_pepland, finetune_pclm2_large.
set -uo pipefail
source "$(dirname "$0")/env.sh"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

STAGE=${1:-all}
LOG=$LOG_DIR/proper_${STAGE}_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
echo "== $(date) | stage $STAGE | log: $LOG"
S=$CODE/src
PL="PL_cCPP PL_cSol PL_ncCPP PL_cBinding PL_ncBinding"
MAIN="CellPPD THPep PepMSND PAMPA AmpHGT"
INNER=${INNER:-5}
conda activate peptide

heads() {
  echo; echo "== heads A: PeptideCLM-2's suite, head on PepLand / PeptideCLM-2 / ECFP ($(date +%H:%M))"
  python "$S/finetune/pepland_head.py" --reps pepland pclm ecfp --bench $MAIN --seeds 3 --out head_main || return 1
  if [ -f "$DATA/features/pepland_suite/pepland.npz" ]; then
    echo; echo "== heads B: PepLand's suite ($(date +%H:%M))"
    python "$S/finetune/pepland_head.py" --feat pepland_suite --reps pepland pclm ecfp --bench $PL --seeds 3 --out head_pepland || return 1
  else
    echo "== heads B skipped: $DATA/features/pepland_suite/pepland.npz does not exist yet (bash run_cross_eval.sh makes it)"
  fi
}

gpu_free() {   # 0 = free (or FORCE=1), 1 = someone else is on the GPU
  local busy
  busy=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader | grep -v dorado || true)
  if [ -n "$busy" ] && [ "${FORCE:-0}" != 1 ]; then
    echo "GPU in use by:"; for pid in $(echo "$busy" | cut -d, -f1); do ps -o user=,etime=,args= -p "$pid"; done
    echo "not starting (FORCE=1 to override)"; return 1
  fi
}

check() {
  gpu_free || return 2
  echo; echo "== check: CellPPD, $INNER inner models ($(date +%H:%M))"
  python "$S/finetune/finetune_smiles.py" --name pclm2_large --bench CellPPD --seeds 101 --inner "$INNER"
  rc=$?; [ $rc = 0 ] || { echo "== stopped (exit $rc)"; return $rc; }
  # time of the last $INNER models trained on CellPPD (from models.csv, written before python exits)
  mins=$(INNER="$INNER" python - <<'EOF'
import os, pandas as pd
p = os.path.join(os.environ["RESULTS_DIR"], "finetune_pclm2_large", "models.csv")
m = pd.read_csv(p) if os.path.exists(p) else pd.DataFrame()
m = m[(m.benchmark == "CellPPD") & (m.seed == 101)] if len(m) else m
print("%.1f" % (m.seconds.tail(int(os.environ["INNER"])).sum() / 60) if len(m) else "")
EOF
)
  if [ -n "$mins" ]; then
    lo=$(awk -v m="$mins" 'BEGIN{printf "%.1f", m*42/60}'); hi=$(awk -v m="$mins" 'BEGIN{printf "%.1f", m*45/60}')
    echo; echo "== CellPPD: $INNER models took $mins min of training."
    echo "== PepLand's five tasks are ~42-45x that work, so expect about $lo-$hi h for the pclm stage (INNER=$INNER)."
    echo "== (INNER=3 roughly halves it; leaving out nc-CPP removes 56% of the rows but only 27% of the padded tokens.)"
  else
    echo "== no CellPPD timing found in $RESULTS_DIR/finetune_pclm2_large/models.csv"
  fi
}

teacher() {   # the finished teacher on PepLand's five tasks (CPU, a few minutes); TEACHER=t3 by default
  local T=${TEACHER:-t3}
  local F="$DATA/features/pepland_suite"
  [ -f "$F/ecfp.npz" ] && [ -f "$F/pclm.npz" ] && [ -f "$F/pepland.npz" ] \
    || { echo "missing PepLand-suite features in $F (bash run_cross_eval.sh makes them)"; return 1; }
  echo; echo "== teacher $T on PepLand's suite: embed ($(date +%H:%M))"
  CUDA_VISIBLE_DEVICES="" python "$S/teacher/train_teacher.py" --name "$T" --embed-bench-from pepland_suite || return 1
  echo; echo "== teacher $T: PepLand's head ($(date +%H:%M))"
  python "$S/finetune/pepland_head.py" --feat pepland_suite --reps teacher_$T pepland pclm ecfp --bench $PL \
      --seeds 3 --out head_pepland_teacher || return 1
  echo; echo "== teacher $T: linear / GBT / MLP probes ($(date +%H:%M))"
  python "$S/eval_single.py" --feat pepland_suite --bench $PL --reps ecfp pclm pepland teacher_$T \
      --learners linear gbt mlp --out cross_pepland_teacher
}

complement() {   # does the teacher add anything to ECFP? CPU only; TEACHER=t3 by default
  local T=${TEACHER:-t3}
  echo; echo "== complement: ecfp vs teacher_$T vs ecfp+teacher_$T, with ecfp+pclm / ecfp+pepland / concat as controls ($(date +%H:%M))"
  OMP_NUM_THREADS=8 python "$S/eval_single.py" --reps ecfp teacher_$T ecfp+teacher_$T ecfp+pclm ecfp+pepland concat \
      --learners linear gbt mlp --out complement_$T
}

pclm() {
  hour=$(date +%H)
  if [ "${FORCE:-0}" != 1 ] && (( 10#$hour >= 7 && 10#$hour < 20 )); then
    echo "it is $(date +%H:%M): long GPU jobs run at night (20:00-07:00). FORCE=1 to start anyway"; return 2
  fi
  gpu_free || return 2
  DEADLINE=${DEADLINE:-06:45}
  now=$(date +%s); stop=$(date -d "$DEADLINE" +%s)
  [ "$stop" -le "$now" ] && stop=$(date -d "tomorrow $DEADLINE" +%s)
  t=$(( stop - now ))
  echo "== pclm: stopping at $DEADLINE ($(( t / 60 )) min from now), inner models per fold: $INNER"
  echo; echo "== pclm 1: reproduction check, CellPPD ($(date +%H:%M))"
  timeout --signal=INT --kill-after=120 "$t" python "$S/finetune/finetune_smiles.py" --name pclm2_large \
      --bench CellPPD --seeds 101 --inner "$INNER"
  rc=$?; [ $rc = 0 ] || { echo "== stopped (exit $rc); run again to continue"; return $rc; }
  now=$(date +%s); t=$(( stop - now ))
  [ "$t" -gt 600 ] || { echo "== no time left before $DEADLINE; run again"; return 0; }
  echo; echo "== pclm 2: PepLand's five tasks ($(date +%H:%M))"
  timeout --signal=INT --kill-after=120 "$t" python "$S/finetune/finetune_smiles.py" --name pclm2_large \
      --bench PL_ncBinding PL_cCPP PL_cSol PL_cBinding PL_ncCPP --seeds 101 --inner "$INNER"
  rc=$?; [ $rc = 0 ] || echo "== stopped (exit $rc); run again to continue"
  return $rc
}

case "$STAGE" in
  heads) heads ;;
  check) check ;;
  teacher) teacher ;;
  complement) complement ;;
  pclm)  pclm ;;
  all)   heads; pclm ;;
  *)     echo "usage: run_proper_eval.sh heads|check|teacher|complement|pclm|all"; exit 1 ;;
esac
echo; echo "== done $(date +%H:%M)"

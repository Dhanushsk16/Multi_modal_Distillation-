#!/usr/bin/env bash
# TWO REPRODUCTIONS OF PUBLISHED RESULTS, as close to the authors' own code and statements as the sources allow.
#
#   binding   PepLand's c-binding / nc-binding (published Spearman 0.503 / 0.768): PepLand vectors + ESM-2 t12 protein vectors, random
#             train/valid/test splits, two-linear-layer head, batch 32, 200 epochs, lr 5e-5 (src/repro/pepland_binding.py; every
#             sourced fact and every choice is in its docstring). The repository ships no downstream code, so the protocol comes from the
#             paper + the authors' answers in GitHub issue #6 (ESM2_t12 protein features, random splits).
#   pepmsnd   PeptideCLM-2 on PepMSND (published MCC 0.729 MLM / 0.749 MTR / 0.741 hybrid), the authors' OWN scripts
#             (external/peptideclm2_authors, verbatim, commit in SOURCE.txt) on their 10 cluster folds, seed 103, run through
#             src/repro/run_authors_script.py (local model folders, rotary-buffer repair). Two variants (VARIANTS, both by default):
#               kan          train_pepmsnd_kan_paperstyle.py: backbone fine-tuned (frozen for the first 3 epochs) + KAN branch on the 140
#                            RDKit descriptors + KAN fusion; AdamW 1e-5 backbone / 1e-3 head, batch 32, <= 120 epochs, patience 20 on the
#                            best-threshold MCC OF THE TEST FOLD. Most likely the source of the paper's Table 1 numbers.
#               species_env  train_pepmsnd_species_env.py: backbone fine-tuned + one-hot species / environment only, no KAN; AdamW 1e-5, batch 32,
#                            <= 10 epochs, patience 3 on the test fold's loss. Most likely the paper's "without KAN" bar (Fig. 5B, about 0.6;
#                            the earlier Distillation rerun got 0.618).
#             Both need `lightning` (the kan variant has no fallback); without it species_env runs as a plain-PyTorch port (pepmsnd_port.py).
#
#   bash ~/code/peptidefusion/scripts/run_reproduce.sh                     # both: binding first (minutes), then PepMSND (hours)
#   WHAT=binding bash ~/code/peptidefusion/scripts/run_reproduce.sh        # only the binding tasks
#   WHAT=pepmsnd bash ~/code/peptidefusion/scripts/run_reproduce.sh        # only PepMSND
#   WHAT=pepmsnd MODELS=mtr-large SEEDS=103 FOLDS="1 2" bash ...           # a quick look: two folds of one model
# Settings (environment):
#   WHAT="binding pepmsnd"   BIND_SEEDS=10 (random splits per cell)   GRU=1 (also the trainable GRU pooling, 3 seeds; needs the PepLand node
#   vectors, built on the way in the pepland env)   ESM_T12_DIR=$CKPT_DIR/esm2_t12_35M_UR50D (weights from scripts/get_esm_laptop.py, see below)
#   MODELS="mlm-large mtr-large"  (+ hybrid-large when its folder exists)   SEEDS="103 203 303"   FOLDS="1 2 3 4 5 6 7 8 9 10"
#   VARIANTS="kan species_env"   IMPL=auto|authors|port   NEED_MB=20000 (free GPU memory to wait for; nothing looks at the clock)
#   OUT_PEPMSND=repro_pepmsnd (results go to results/<OUT_PEPMSND>_<variant>)
# Memory: the authors' recipe trains all 337M weights in fp32 at batch 32 and pads every batch to the longest molecule. If a job runs out
# of memory it is repeated with micro-batches of 16, 8, 4 (gradients accumulated: still one optimiser step per 32 molecules; the kan variant
# stops at 8, its BatchNorm layers see the micro-batch).
# The ESM-2 t12 weights (140 MB) are the one thing the server cannot download. On a machine with internet:
#   python scripts\get_esm_laptop.py --repo facebook/esm2_t12_35M_UR50D
# Results: results/repro_pepland_binding/raw.csv, results/repro_pepmsnd_<variant>/<model>/seed<S>/fold<k>/preds_fold<k>.csv; the summaries are
# printed at the end and kept in logs/reproduce_*.log.
set -uo pipefail
source "$(dirname "$0")/env.sh"
WHAT=${WHAT:-"binding pepmsnd"}
BIND_SEEDS=${BIND_SEEDS:-10}
GRU=${GRU:-1}
SEEDS=${SEEDS:-"103 203 303"}
FOLDS=${FOLDS:-"1 2 3 4 5 6 7 8 9 10"}
IMPL=${IMPL:-auto}
NEED_MB=${NEED_MB:-20000}
OUT_PEPMSND=${OUT_PEPMSND:-repro_pepmsnd}
VARIANTS=${VARIANTS:-"kan species_env"}
ESM_T12_DIR=${ESM_T12_DIR:-$CKPT_DIR/esm2_t12_35M_UR50D}
AUTH=$CODE/external/peptideclm2_authors/training/03_PepMSND_training_code
SUITE=$DATA/features/pepland_suite
LOG=$LOG_DIR/reproduce_$(date +%Y%m%d_%H%M).log
exec > >(tee -a "$LOG") 2>&1
say() { echo "$(date +%H:%M:%S) $*"; }
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate peptide
say "== reproductions: $WHAT | log $LOG"

wait_gpu() {                                          # only the free memory counts
  local free
  while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0 2> /dev/null | head -1 | tr -d ' ')
    [ -n "${free:-}" ] && [ "$free" -ge "$1" ] && break
    say "waiting for the GPU: ${free:-?} MB free, $1 MB needed (checking every 60 s)"; sleep 60
  done
}

# ------------------------------------------------------------------------------------------------ binding
stage_binding() {
  say "=== BINDING: PepLand c-binding / nc-binding, ESM2-t12 protein vectors, random splits ==="
  local P12=$SUITE/protein_esm_t12.npz P3B=$SUITE/protein_esm.npz NODES=$SUITE/pepland_nodes.npz gruarg="" pf
  if [ ! -f "$P12" ]; then
    if [ ! -f "$ESM_T12_DIR/config.json" ]; then
      say "!! no ESM-2 t12 weights in $ESM_T12_DIR : download them (see the header of this script)"; return 3
    fi
    say "step 1: ESM2-t12 vectors of the 1,227 protein partners (35M parameters, CPU, a few minutes)"
    CUDA_VISIBLE_DEVICES= python "$CODE/src/featurize_esm.py" --proteins --model "$ESM_T12_DIR" --protein-file "$P12" || { say "!! protein embedding failed"; return 4; }
  else
    say "step 1: $P12 exists, skipped"
  fi
  if [ "$GRU" = 1 ]; then
    if [ ! -f "$NODES" ]; then
      say "step 2: PepLand node vectors of the binding peptides (pepland env, CPU)"
      ( conda activate pepland && python "$CODE/src/featurize_pepland_nodes.py" --feat pepland_suite --bench PL_cBinding PL_ncBinding --threads "$THREADS" ) \
        || say "!! node vectors failed: the GRU pooling is skipped"
    else
      say "step 2: $NODES exists, skipped"
    fi
    [ -f "$NODES" ] && gruarg="--gru"
  fi
  pf="t12=$P12"
  [ -f "$P3B" ] && pf="$pf esm3b=$P3B"
  say "step 3: the benchmark runs (protein vectors: $pf; $BIND_SEEDS random splits per cell; GRU: ${gruarg:-no})"
  # shellcheck disable=SC2086
  python "$CODE/src/repro/pepland_binding.py" --protein-file $pf --seeds "$BIND_SEEDS" $gruarg --threads 4 --out repro_pepland_binding
}

# ------------------------------------------------------------------------------------------------ PepMSND
stage_pepmsnd() {
  local variant=$1 out script ladder
  out=${OUT_PEPMSND}_$variant
  case $variant in
    kan) script=train_pepmsnd_kan_paperstyle.py; ladder="32 16 8";;
    species_env) script=train_pepmsnd_species_env.py; ladder="32 16 8 4";;
    *) say "!! unknown variant $variant (kan | species_env)"; return 2;;
  esac
  say "=== PEPMSND ($variant): PeptideCLM-2 on the 10 cluster folds, the authors' $script ==="
  local impl=$IMPL mlm mtr hyb tag name dir seed fold save jl mb acc rc models="${MODELS:-}"
  if [ "$impl" = auto ]; then
    if python -c "import lightning" 2> /dev/null; then impl=authors; else impl=port; fi
  fi
  if [ "$variant" = kan ] && [ "$impl" != authors ]; then
    say "!! the kan variant needs the authors' script and therefore lightning (python -c 'import lightning' fails here): skipped; species_env can run as the port"; return 3
  fi
  read -r mlm mtr hyb <<< "$(python - <<EOF
import sys
sys.path.insert(0, "$CODE/src")
from paths import PCLM_DIR, PCLM_MTR_DIR, _hf_model_dir
import os
ok = lambda d: d if os.path.exists(os.path.join(d, "model.safetensors")) else "-"
print(ok(PCLM_DIR), ok(PCLM_MTR_DIR), ok(_hf_model_dir("peptideclm-2-hybrid-large", "PCLM_HYBRID_DIR")))
EOF
)"
  if [ -z "$models" ]; then
    models="mlm-large mtr-large"
    [ "$hyb" != "-" ] && models="$models hybrid-large"
  fi
  say "implementation: $impl (authors = their script verbatim, needs lightning; port = same recipe in plain PyTorch) | models: $models | seeds: $SEEDS | folds: $FOLDS"
  say "model folders: mlm-large $mlm | mtr-large $mtr | hybrid-large $hyb  ('-' = not on the server; the hybrid model is optional)"
  [ "$impl" = authors ] && [ ! -f "$AUTH/scripts/$script" ] && { say "!! $AUTH is missing: upload the code tar (external/peptideclm2_authors)"; return 3; }
  for seed in $SEEDS; do
    for tag in $models; do
      case $tag in mlm-large) dir=$mlm;; mtr-large) dir=$mtr;; hybrid-large) dir=$hyb;; *) say "!! unknown model $tag"; continue;; esac
      [ "$dir" = "-" ] && { say "!! no weights for $tag: skipped"; continue; }
      name="aaronfeller/peptideclm-2-$tag"
      for fold in $FOLDS; do
        save=$RESULTS_DIR/$out/$tag/seed$seed/fold$fold
        if [ -f "$save/preds_fold$fold.csv" ]; then say "$tag seed $seed fold $fold: done, skipped"; continue; fi
        mkdir -p "$LOG_DIR/$out" "$save"
        wait_gpu "$NEED_MB"
        for mb in $ladder; do
          acc=$((32 / mb))
          jl=$LOG_DIR/$out/${tag}_seed${seed}_fold${fold}_mb${mb}.log
          say "$tag seed $seed fold $fold | micro-batch $mb (x$acc) | log $jl"
          if [ "$impl" = authors ]; then
            ACCUM=$acc python "$CODE/src/repro/run_authors_script.py" "$AUTH/scripts/$script" \
              --model_name "$name" --fold "$fold" --data_dir "$DATA/data/benchmarks/PepMSND_data" --save_path "$save" --seed "$seed" --batch_size "$mb" > "$jl" 2>&1
          else
            python "$CODE/src/repro/pepmsnd_port.py" --model_name "$name" --fold "$fold" --data_dir "$DATA/data/benchmarks/PepMSND_data" \
              --save_path "$save" --seed "$seed" --batch_size 32 --micro_batch "$mb" > "$jl" 2>&1
          fi
          rc=$?
          [ -f "$save/preds_fold$fold.csv" ] && { say "   done: $(tail -n 1 "$jl" | cut -c1-150)"; break; }
          if grep -qi "out of memory" "$jl"; then say "   out of memory at micro-batch $mb: trying a smaller one"; continue; fi
          say "!! $tag seed $seed fold $fold FAILED (exit status $rc), see $jl"; tail -n 8 "$jl"; break
        done
        rm -rf "$save/checkpoints"                      # Lightning keeps a 1.3 GB copy of the whole model per fold
      done
    done
  done
}

for w in $WHAT; do
  case $w in
    binding) stage_binding || say "!! binding stage ended with status $?";;
    pepmsnd) for v in $VARIANTS; do stage_pepmsnd "$v" || say "!! pepmsnd stage $v ended with status $?"; done;;
    *) say "!! unknown stage $w (binding | pepmsnd)";;
  esac
done

say "=== SUMMARY ==="
if [ -f "$RESULTS_DIR/repro_pepland_binding/raw.csv" ]; then
  python - <<EOF
import pandas as pd
pd.set_option("display.width", 220)
r = pd.read_csv("$RESULTS_DIR/repro_pepland_binding/raw.csv")
print("PepLand binding: %d runs" % len(r))
for t, d in r.groupby("task"):
    g = d.groupby(["split", "protein_vec", "features", "head"])
    s = g[["SCC", "PCC", "MAE", "RMSE"]].mean().round(3); s["SCC sd"] = g.SCC.std().round(3); s["seeds"] = g.size()
    print("\n%s (published SCC %s, PCC %s)" % (t, "0.503" if t == "c-binding" else "0.768", "0.450" if t == "c-binding" else "0.873"))
    print(s.to_string())
EOF
fi
for v in $VARIANTS; do
  if [ -d "$RESULTS_DIR/${OUT_PEPMSND}_$v" ]; then say "--- PepMSND, variant $v ---"; python "$CODE/src/repro/pepmsnd_summary.py" --root "$RESULTS_DIR/${OUT_PEPMSND}_$v"; fi
done
say "== done."

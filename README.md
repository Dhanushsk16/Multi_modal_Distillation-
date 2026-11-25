# PeptideFusion

**Knowledge distillation of a fused multi-view peptide model into a small SMILES-only student.**
B.Tech major project, Department of Biochemical Engineering and Biotechnology, IIT Delhi.
Author: Dhanush S K. Supervisor: Professor Ishaan Gupta.

The project follows a midterm study that compressed PeptideCLM-2 (337 M parameters) without training; its code is in a separate repository, [Dhanushsk16/Peptideclm2-model-distillation-](https://github.com/Dhanushsk16/Peptideclm2-model-distillation-). This part takes the step that study named, distillation:

1. **Teacher.** Three frozen published models describe each peptide differently: PepLand (a graph network, 300-d), PeptideCLM-2 (a SMILES language model, 1024-d) and extended-connectivity fingerprint counts (2048-d). A small fusion network (24.99 M parameters at inference) maps them to one 1024-d embedding. ESM-2 (3B, 2560-d), which only reads the 20 natural amino acids, is used as an extra reconstruction target and never as an input.
2. **Student.** A 10.82 M-parameter transformer that reads **only the SMILES string** is trained to reproduce the teacher's embedding (and each input view). At inference nothing else is needed: no graph model, no fingerprint, no PeptideCLM-2, no ESM-2.

The student has 31 times fewer parameters than PeptideCLM-2 and 34 times fewer than the chain of models the teacher needs. Running time was also measured, on one laptop (see [Running time](#running-time)). To try the final models without any data, see the [quick check](#quick-check-no-gpu-no-benchmark-data).

## The main models

Many designs were trained along the way (all of them are listed in [`weights/MODELS.md`](weights/MODELS.md)). Two are the result of the project:

| Role | Run | Weights | What it is |
|---|---|---|---|
| **Main teacher** | `v3_base` | `weights/teacher/v3_base/best.pt` | The fusion network described above: inputs PepLand, PeptideCLM-2 and ECFP counts, ESM-2 as a reconstruction target only; 35.27 M parameters in training, 24.99 M at inference; trained on the whole 3.40 M-molecule pool. |
| **Main student** | `st6f` | `weights/student_distill/st6f/best.pt`, tokenizer `weights/student_tokenizer_norm/atombpe_v1024` | The 10.82 M-parameter SMILES-only student, distilled from `v3_base` on the whole pool (21.1 M parameters in the file, of which 10.82 M are used at inference). |

Every teacher and student number below comes from these two models. All other runs are earlier designs, ablations or controls: the teacher generations t1 to t11, the `abl_*` ablations, the student ladder st1 to stca and the size pilot.

## Results

Seven peptide benchmarks, embeddings read with small learners on frozen features (the student and the teacher are not fine-tuned). Primary metric per benchmark; "best published" is the strongest number in the source paper's results table.

| Benchmark | Metric | Best published | Teacher | Student | Student − published | Molecules above 30 residues |
|---|---|---|---|---|---|---|
| CellPPD | MCC | 0.892 | 0.920 | 0.913 | +0.021 | 4.1 % |
| THPep | MCC | 0.756 | 0.719 | 0.728 | −0.028 | 0.2 % |
| PepMSND | MCC | 0.749 | 0.543 | 0.520 | −0.229 | 23.3 % |
| AmpHGT | MCC | 0.884 | 0.904 | 0.842 | −0.042 | 68.8 % |
| PL_cCPP | AUROC | 0.885 | 0.954 | 0.959 | +0.074 | 8.0 % |
| PL_cSol | AUROC | 0.730 | 0.703 | 0.693 | −0.037 | 100 % |
| PL_ncCPP | Spearman | 0.628 | 0.757 | 0.736 | +0.108 | 0.0 % |

The student is above the best published score on 3 of 7 benchmarks (CellPPD, PL_cCPP, PL_ncCPP) and below it on 4 (THPep, PepMSND, AmpHGT, PL_cSol); the teacher is above on 4 of 7. That count takes the best of four learners per benchmark; with each learner on its own, with no choice made on the test scores, the student is above on 2 or 3 of 7. Please read the numbers with these caveats:

- The published scores come from fine-tuned models; these are frozen embeddings, so the comparison is indicative, not like for like.
- The best of four learners (linear, RBF kernel, gradient-boosted trees, MLP) is chosen per benchmark on the test scores. This is optimistic and is applied equally to teacher and student.
- Splits: CellPPD, PepMSND and AmpHGT use the authors' released splits. THPep uses the three splits of the PeptideCLM-2 authors (122 test molecules each), rebuilt here from their repository by `src/build_thpep_official_splits.py`; the table gives the mean over the three (standard deviation of the student 0.104), run with `scripts/run_thpep_official.sh`, results in `results/thpep_official_all/`. On a split made earlier here the student scored 0.861 and the teacher 0.840; that split is not used. The three PepLand tasks use splits made here, because the PepLand authors released none.
- The published PeptideCLM-2 MCC values are test-threshold scores (recomputed from the authors' released prediction files): for THPep the published 0.756 is the best-threshold MCC, and the same predictions at the usual decision threshold give 0.693.
- Each network was trained once. The student is below the published score on the three benchmarks with the most long peptides (PepMSND, AmpHGT, PL_cSol; the pool it was trained on holds peptides of at most 50 residues), but also on THPep, which has almost none, so the pattern is not clean. Whether length is a cause was not tested.
- The benchmarks were not held out from the design work. The teacher generation, the student recipe (st6 was chosen among eight by its average linear-probe score on these benchmarks) and the variants were compared on the benchmark test folds. Only the choice of checkpoint and the stopping of a run used a held-out slice of the training pool.
- The corpus was cleaned of molecules within ECFP4 Tanimoto 0.9 of any benchmark molecule, but the pretraining data of PeptideCLM-2 and PepLand may overlap with the benchmarks, and that overlap cannot be removed.

Result files: `runs/st6f_results/results/harness_st6f_e02_all/*.csv` (per benchmark, per learner), with the student's training history in `runs/st6f_results/checkpoints/student_distill/st6f/history.csv`. The trained weights are described in [`weights/README.md`](weights/README.md). `python scripts/headline_table.py` recomputes this table, and each learner on its own, from the stored result files; the sources of the published numbers are in [`REFERENCES.md`](REFERENCES.md).

## Running time

Time per molecule in milliseconds, from a SMILES string to the 1024-d embedding, on 512 benchmark molecules drawn at random (64 on the CPU); median of five timed passes. One laptop (NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB; 8 CPU threads; mains power), the trained weights, plain PyTorch (bfloat16 on the GPU, fp32 on the CPU; nothing compiled, exported or quantised).

| Setting | Student | PeptideCLM-2 | Teacher, views cached | Teacher, new molecule* | Student faster than PeptideCLM-2 |
|---|---|---|---|---|---|
| GPU, batch 1 | 11.7 | 73.3 | 2.31 | 76.6 | 6.3× |
| GPU, batch 16 | 2.37 | 12.2 | 0.124 | 13.4 | 5.2× |
| GPU, batch 64 | 1.80 | 18.0 | 0.034 | 19.1 | 10.0× |
| CPU, batch 1 | 9.19 | 297.4 | 4.52 | 302.8 | 32.4× |
| CPU, batch 16 | 4.59 | 443.4 | 0.447 | 444.7 | 96.7× |

\* Fingerprint + PeptideCLM-2 + fusion network. PepLand was not timed (it needs DGL, which is not installed on the laptop), so this is a lower bound. Student = RDKit normalisation + tokenising + forward pass; PeptideCLM-2 = tokenising + forward pass; "Teacher, views cached" = the fusion network on stored view vectors, i.e. what the teacher costs once its three views exist.

- The forward pass alone (no RDKit, no tokenising) is 7, 15 and 70 times faster than PeptideCLM-2 on the GPU at batch 1, 16 and 64. At batch 64 most of the student's time is RDKit normalisation (1.5 ms per molecule).
- PeptideCLM-2 is slower per molecule at batch 64 than at batch 16 because a few long molecules pad each batch (longest input 541 tokens). The student reads sequences about three times shorter (median 19 tokens against 60), which is part of its speed.
- One laptop and one run; the lab server's GPU is shared and was not timed. `runs/time_models/results.csv` was written by the first version of the script, before it also logged the GPU state; the measurement code is the same.
- Reproduce: `python src/student/time_models.py` (the model paths default to `src/paths.py`; `--student`, `--tok`, `--teacher`, `--pclm`, `--out` override them). Files: `runs/time_models/results.csv` and `summary.txt`.

## Method in brief

- **Training pool.** 3.40 M peptide molecules (2 to 50 residues; about 19 % with non-canonical residues) from PepLand's stage-1 peptides, ESMAtlas peptides and PubChem molecules of the PeptideCLM-2 corpus, with molecules within ECFP4 Tanimoto 0.9 of any benchmark molecule removed (`src/corpus/clean_corpus.py --thresh 0.9`; the script's default is 0.8).
- **Teacher** (`src/teacher/train_teacher_v3.py`). One encoder per view (to 512-d), concatenation (1536-d), a residual-MLP trunk to a linear 1024-d latent, and four shallow decoders that rebuild P, L, H and E from the latent. View dropout (p = 0.1) during training. 35.27 M parameters in training, 24.99 M at inference.
- **Student** (`src/student/model.py`, `src/student/train_student.py`). Atom-level byte-pair tokenizer (vocabulary 1024), 8 pre-norm transformer blocks of width 320 (5 heads, rotary positions, SwiGLU 768), attentive pooling with 4 learned queries, a linear layer to 1024-d. Losses: cosine plus smooth L1 on the latent and on four training-only view heads, in-batch InfoNCE, a relational (similarity-matrix) term and a consistency term between two random SMILES spellings of one molecule.
- **Evaluation** (`src/benchmarks.py`, `src/teacher/eval_harness.py`). Every model goes through the same four learners and the same split protocol per benchmark.

## Repository layout

```
src/
  paths.py            every path in one place (BTP_DATA overrides the data root)
  benchmarks.py       the benchmark loaders and the split protocol of each
  featurize*.py       frozen views: ECFP counts, PepLand, PeptideCLM-2, ESM-2
  eval_single.py, kernel_learners.py     learners and the single-view check
  build_thpep_official_splits.py         THPep splits of the PeptideCLM-2 authors, rebuilt from their repository
  benchmark_splits/   those split ids and SHA-256 hashes of the author files used
  corpus/             building the training pool: extraction, cleaning, leakage removal, view caches
  teacher/            the teachers and the evaluation harness: train_teacher.py trained the first two generations (t1 to t11), train_teacher_v3.py the final design (v3_*)
  student/            tokenizer, student model, distillation training, export, evaluation, timing (time_models.py)
  finetune/, repro/   fine-tuning recipes and reproductions of published pipelines
scripts/              launch scripts (lab server, conda envs "peptide" and "pepland"); the header of each script lists its stages; headline_table.py recomputes the table above
shape_pilot/          the student size pilot (four models of about ten million parameters), written on the server; RUN.md explains it
weights/              every trained model in the server's checkpoints/ layout: 21 teacher runs, the student ladder st1 to stca, the final student st6f, the size pilot, all tokenizers. MODELS.md lists and explains them; the .pt files themselves are not in git but in the Zenodo record (see weights/README.md)
results/, logs/       the result tables and logs of every run, in the server's layout (results/INDEX.md lists the folders)
data/                 data/README.md describes the corpus files (data/corpus/*.parquet: the cleaned corpus and the 3.40 M-molecule training pool); the corpus files are not distributed (see below)
runs/st6f_results/    a compact copy of the final student's results (CSV, logs, training history); the results of all other runs are in results/ and logs/
runs/time_models/     running-time results (results.csv, summary.txt)
external/             third-party code at pinned commits, with its licences: PepLand (incl. its 26 MB weights), mapchiral, the PeptideCLM-2 authors' PepMSND training scripts
```

## Quick check (no GPU, no benchmark data)

What can be run from a clone on a laptop (Python with torch, numpy, pandas, rdkit and tokenizers; a CPU is enough):

1. `python scripts/headline_table.py` recomputes the results table above from the stored result files.
2. `python <script> --help` works for every Python script that has a command line (52 of them, in `src/`, `scripts/` and `shape_pilot/`).
3. The two main models are in the Zenodo record ([10.5281/zenodo.24507677](https://doi.org/10.5281/zenodo.24507677)), file `PeptideFusion_main_models.tar` (226 MB). From a git clone, place them with
   ```
   tar -xf PeptideFusion_main_models.tar --strip-components=1 -C <path-to-clone>
   ```
   The archive entries start with `PeptideFusion/`, which `--strip-components=1` removes; the files land in `weights/teacher/v3_base/best.pt` and `weights/student_distill/st6f/best.pt` and can be compared with `weights/SHA256SUMS.txt`.
4. Load the student and embed a peptide with the snippet in [`weights/README.md`](weights/README.md) (a few seconds on a CPU). The teacher loads in the same way; its input is the three cached view vectors, not a SMILES string:
   ```python
   import sys, torch
   sys.path[:0] = ["src", "src/teacher"]
   import train_teacher_v3 as TV3
   ck = torch.load("weights/teacher/v3_base/best.pt", map_location="cpu", weights_only=True)
   teacher = TV3.TeacherV3.from_cfg(ck["model_cfg"], 0.0); teacher.load_state_dict(ck["model"]); teacher.eval()
   ```

What cannot be run without more: training, the feature caches (about 45 GB) and the benchmark evaluation. They need the lab server and a GPU, the benchmark datasets and the published input models (PeptideCLM-2, PepLand, ESM-2), none of which are in this repository (see "Not in this repository"). A script that needs them stops with an error that names a path under `/mnt/…` (the lab server's data drive) or says that no benchmark molecules were found: the data root is not set.

To run the scripts on your own data root, set `BTP_DATA=<root>` (see `src/paths.py`). The code reads models from `$BTP_DATA/checkpoints/` and the benchmark files from `$BTP_DATA/data/benchmarks/`. In this repository the models are in `weights/`, so `$BTP_DATA/checkpoints` has to point to it (for example `ln -s <path-to-clone>/weights $BTP_DATA/checkpoints`). One more detail: a student checkpoint stores its tokenizer folder as the lab server's absolute path, and `export_student.py`, `finetune_student.py` and `baseline_features.py` read it from there (`spelling_test.py` has a `--tok` option), so those three can only run where that path exists. The loading snippet in `weights/README.md` does not depend on it.

## Running it

The scripts were written for a shared lab server (one RTX A6000, a home directory that holds code only and a separate data drive). All runtime paths come from `src/paths.py` and `scripts/env.sh`; set `BTP_DATA=/your/data/root` to relocate them. Every script's header comment shows its stages and settings. The order of the pipeline:

1. Environments: `scripts/setup_envs.sh` (offline installation on the server). PepLand needs its own environment (PyTorch 2.2, DGL).
2. Corpus: `scripts/run_extract.sh` → `src/corpus/clean_corpus.py` → `src/corpus/build_pool.py` → `scripts/run_cache.sh` (frozen views of every pool molecule).
3. Teacher: `scripts/run_teacher_v3.sh` (`prep`, `harness`, `train`).
4. Student: `scripts/run_student_tokenizer.sh`, then `scripts/run_student.sh` (`norm`, `tok`, `embed`, `data`, `train`, `export`), and `scripts/run_student_eval.sh` for the benchmarks.

## Environment

Everything ran on one lab server (NVIDIA RTX A6000 48 GB, CUDA 12.4, driver 550) in two conda environments: `peptide` (PyTorch 2.6 and the packages below) and `pepland` (Python 3.11, CPU PyTorch 2.2.2, DGL, MLflow, RDKit, NumPy below 2, which PepLand needs); `scripts/setup_envs.sh` builds both offline. The third-party packages that the code imports are: torch, numpy, pandas, scipy, scikit-learn, rdkit, pyarrow, transformers, tokenizers, huggingface_hub, joblib and psutil, plus pykan for the PepMSND reproduction (`scripts/run_pepmsnd_kan.sh`) and the code in `external/`. Exact package versions were not recorded, and there is no automated test suite; `src/corpus/check_sources.py`, `src/corpus/prefilter_check.py` and the `--check` options of several scripts are consistency checks that ship with the code.

## Extracting the embeddings of the large models

Benchmark molecules, one file per representation (`features/<rep>.npz`, conda env `peptide`; the PeptideCLM-2 and ESM-2 weights must be in `$BTP_DATA/checkpoints/`):

```
python src/featurize.py --rep ecfp         # also: mapc, pclm (PeptideCLM-2 mlm-large, 1024-d), pclm_mtr, rdkit
python src/featurize_esm.py --bench        # ESM-2 3B (2560-d); canonical peptides only
python src/featurize_pepland.py            # PepLand (300-d); conda env "pepland" (PyTorch 2.2 + DGL); uses external/pepland
```

The 3.4 M-molecule training pool, shard by shard (resumable, about 45 GB for all views):

```
python src/corpus/cache_views.py --export                # text copy of the pool for the PepLand environment
bash scripts/run_cache.sh                                # PeptideCLM-2 (GPU) and PepLand (CPU) together
python src/corpus/cache_views.py --view esm              # ESM-2; --view rdkit / pclm_mtr for the other views
python src/corpus/cache_views.py --status                # progress
```

The teacher's and the student's own embeddings:

```
python src/teacher/train_teacher_v3.py --name v3_base --embed-pool        # the student's distillation targets (teacher latent of every pool molecule)
python src/student/export_student.py --name st6f                          # benchmark embeddings of a trained student
```

PeptideCLM-2 must be loaded through `featurize.load_pclm` (see Known pitfalls). ESM-2 weights: `scripts/get_esm_laptop.py` downloads them on a machine with internet.

## Not in this repository

- **Data.** The benchmark files and the pretraining corpus belong to their authors. `src/benchmarks.py` documents the expected layout under `data/benchmarks/`; the sources are the PeptideCLM-2 repository (https://github.com/AaronFeller/PeptideCLM-2) and the PepLand repository (https://github.com/zhangruochi/pepland); `src/corpus/download_hf.py` fetches the PeptideCLM-2 pretraining files from Hugging Face. `src/benchmark_splits/repo_data_sha256.txt` lists the hashes of the author files used.
- **Published models.** PeptideCLM-2 (Hugging Face `aaronfeller/peptideclm-2-mlm-large`), PepLand, ESM-2 3B (`facebook/esm2_t36_3B_UR50D`, `scripts/get_esm_laptop.py`). The PepLand code and weights, mapchiral and the PeptideCLM-2 authors' scripts are copied into `external/` (their licences are kept; see `external/README.md`).
- **The large weight files of `weights/`** (5.8 GB of `.pt` and `.safetensors` files; the final teacher is 141 MB and the final student 84 MB, and GitHub rejects files above 100 MB) are in the Zenodo record [10.5281/zenodo.24507677](https://doi.org/10.5281/zenodo.24507677), together with a zip of this repository and `SHA256SUMS.txt` for the archives. To restore the full folder, unzip `PeptideFusion_repo_files.zip` and extract every `PeptideFusion_weights__*.tar` (one archive per trained run) in the same place (the archive entries start with `PeptideFusion/`); `PeptideFusion_main_models.tar` holds only the final teacher and student. The folders in git hold their descriptions (`weights/MODELS.md`, `weights/README.md` with a loading example) and SHA-256 checksums to verify the files.
- **The corpus** (`data/corpus/`, 4 GB: the cleaned corpus and the 3.40 M-molecule training pool). It is built from data that belongs to its authors and is not distributed; `data/README.md` lists every file with its SHA-256, so a rebuild with the scripts of the pipeline above can be checked.
- **The pool embedding caches** (about 45 GB) and the PeptideCLM-2 and ESM-2 weights; the section above shows how to rebuild the caches.

## Known pitfalls with PeptideCLM-2

- The rotary buffers (non-persistent `theta` and `cache`) are not in the checkpoint and come out as garbage after loading, even with one model per process, so the embeddings are silently wrong. Always load through `featurize.load_pclm`, which rebuilds and verifies them.
- Its attention builds an fp32 padding mask, which `scaled_dot_product_attention` rejects under fp16/bf16 on CUDA. Cast the mask to the query dtype.

## Notes

- **Licence.** The code, trained models, results and documentation written for this project are released under the Creative Commons Attribution 4.0 International licence (CC BY 4.0; full text in [`LICENSE`](LICENSE)): please credit Dhanush S K when reusing them. Third-party code in `external/` keeps its own licences (PeptideCLM-2 repository: MIT; PepLand: Apache-2.0; mapchiral: MIT, as stated in its README), and the benchmark datasets belong to their authors.
- **Loading checkpoints.** 35 of the 39 `.pt` files in `weights/`, including the final teacher (`v3_base`) and student (`st6f`), load with `torch.load(..., weights_only=True)` (plain tensors and numbers, nothing is executed); the other four (the smoke test and the first-generation teachers t1 to t3) also store numpy scalars. The training, export and evaluation scripts load the checkpoints that the same code wrote with `weights_only=False`, because resuming a run needs the optimizer and scheduler state: load only checkpoints from a source you trust, such as the SHA-256-verified files of the Zenodo record.
- **Remote model code.** PeptideCLM-2 ships its own model code, so it is loaded with `trust_remote_code=True` from a local copy of the Hugging Face snapshot (`aaronfeller/peptideclm-2-mlm-large`, revision `1a20894f16fbf294f2219130f9960a89d143d800`); inspect that code before running it. `src/corpus/clean_corpus.py` also reads `data/corpus/bench_fps.pkl`, a pickle written by the same script.

## Acknowledgements

Professor Ishaan Gupta (supervisor); the authors of PeptideCLM-2 (Feller et al.), PepLand (Zhang et al.), ESM-2 (Lin et al.) and the Chemical Dice Integrator (Kumar et al.), whose released models, data and ideas this work builds on. The works used are listed in [`REFERENCES.md`](REFERENCES.md).

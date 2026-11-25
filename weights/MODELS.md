# Trained models

Every model that was trained in this project, in the layout of `$BTP_DATA/checkpoints/`. Optimizer states (`last.pt`, `latest.pt`) were left out because they are three times the size of the weights and only serve to resume training; a few early runs saved only a `last.pt`, which is kept for them. Parameter counts and settings are read from the checkpoints. All files are verified in `SHA256SUMS.txt`. The `.pt` files are not stored in git (see `README.md`).

## Teachers (`teacher/`)

| Run | What it is | Settings recorded in the checkpoint | Epoch | Parameters | Files |
|---|---|---|---|---|---|
| `t1` | First generation: the published wiring (a leave-one-out autoencoder per view and a super-embedding autoencoder). Learning rate 1e-3; stopped by hand after 9 of the 40 planned epochs because the loss spiked early in training. | sea_input=latents, epochs=40 | 9 | 28,694,192 | last.pt 344 MB (training checkpoint with optimizer state: the only weights of this run) |
| `t2` | First generation (same wiring as t1). | sea_input=latents, epochs=40 | 40 | 28,694,192 | last.pt 344 MB (training checkpoint with optimizer state: the only weights of this run) |
| `t3` | First generation (same wiring); the one used in the first comparison with the fingerprint and PeptideCLM-2. | sea_input=both, epochs=40 | 40 | 35,603,420 | last.pt 427 MB (training checkpoint with optimizer state: the only weights of this run) |
| `t5` | First generation, other views: ECFP counts, PeptideCLM-2 and RDKit descriptors (no PepLand). | views=ecfp+pclm+rdkit, sea_input=both, epochs=200, view_dropout=0.1 | 196 | 80,636,934 | best.pt 323 MB |
| `t8` | Second generation: ESM-2 as an optional input view; molecules ESM-2 cannot read get an imputed vector (fill imputed). First pool (2.64 M molecules). | views=pclm+pepland+esm, optional_views=esm, fill=imputed, sea_input=latents, epochs=200, view_dropout=0.1 | 110 | 73,414,832 | best.pt 294 MB |
| `t9` | Second generation: as t8, but a learned mask token replaces the missing ESM-2 vector (fill token). First pool. | views=pclm+pepland+esm, optional_views=esm, fill=token, sea_input=latents, epochs=200, view_dropout=0.1 | 120 | 73,417,392 | best.pt 294 MB |
| `t10` | Second generation: as t9, trained on the extended pool (3.40 M molecules, peptides of 31 to 50 residues added). | views=pclm+pepland+esm, optional_views=esm, fill=token, sea_input=latents, epochs=200, view_dropout=0.1 | 135 | 73,417,392 | best.pt 294 MB |
| `t11` | Second generation: as t10 with the ESM-2 vector also in the super-embedding input (sea_input both); stopped early, under-trained. | views=pclm+pepland+esm, optional_views=esm, fill=token, sea_input=both, epochs=200, view_dropout=0.1 | 10 | 89,330,140 | best.pt 357 MB |
| `v3_base` | FINAL TEACHER (third generation): inputs PepLand, PeptideCLM-2 and ECFP counts, ESM-2 as a reconstruction target only; trained on the whole 3.40 M pool. The student is distilled from this model. | epochs=100, view_dropout=0.1, w_h=1.0, balance=0.5 | 86 | 35,270,444 | best.pt 141 MB, latent_stats.npz 13 kB |
| `v3_mtr` | Variant of v3_base with the PeptideCLM-2 MTR embedding as a fourth input (extra_input pclm_mtr). | epochs=100, extra_input=pclm_mtr, fuse=concat, view_dropout=0.1, w_h=1.0, balance=0.5 | 83 | 42,097,452 | best.pt 169 MB, latent_stats.npz 13 kB |
| `v3_attn` | Variant of v3_base with attention pooling over the view encodings instead of concatenation (fuse attn). | epochs=100, fuse=attn, view_dropout=0.1, w_h=1.0, balance=0.5 | 86 | 37,374,252 | best.pt 150 MB, latent_stats.npz 13 kB |
| `abl_base_s1` | Ablation base design on a subset of 500,000 molecules (the first 5 of 51 pool shards, 14.7 % of the pool), seed 1. | first=5, epochs=60, view_dropout=0.1, w_h=1.0, balance=0.5, seed=1 | 60 | 35,270,444 | best.pt 141 MB, latent_stats.npz 13 kB |
| `abl_base_s2` | Ablation base design, seed 2: the spread between training runs. | first=5, epochs=60, view_dropout=0.1, w_h=1.0, balance=0.5, seed=2 | 57 | 35,270,444 | best.pt 141 MB, latent_stats.npz 13 kB |
| `abl_base_s3` | Ablation base design, seed 3: the spread between training runs. | first=5, epochs=60, view_dropout=0.1, w_h=1.0, balance=0.5, seed=3 | 59 | 35,270,444 | best.pt 141 MB, latent_stats.npz 13 kB |
| `abl_noe_s1` | Ablation: without the ESM-2 target. | first=5, epochs=60, no_e=True, view_dropout=0.1, w_h=1.0, balance=0.5, seed=1 | 60 | 31,596,844 | best.pt 127 MB, latent_stats.npz 13 kB |
| `abl_noh_s1` | Ablation: without the ECFP (H) view. | first=5, epochs=60, no_h=True, view_dropout=0.1, w_h=1.0, balance=0.5, seed=1 | 57 | 26,345,260 | best.pt 105 MB, latent_stats.npz 13 kB |
| `abl_vd0_s1` | Ablation: view dropout 0. | first=5, epochs=60, view_dropout=0.0, w_h=1.0, balance=0.5, seed=1 | 60 | 35,270,444 | best.pt 141 MB, latent_stats.npz 13 kB |
| `abl_vd3_s1` | Ablation: view dropout 0.3. | first=5, epochs=60, view_dropout=0.3, w_h=1.0, balance=0.5, seed=1 | 60 | 35,270,444 | best.pt 141 MB, latent_stats.npz 13 kB |
| `abl_wh2_s1` | Ablation: the ECFP (H) loss weighted 2. | first=5, epochs=60, view_dropout=0.1, w_h=2.0, balance=0.5, seed=1 | 60 | 35,270,444 | best.pt 141 MB, latent_stats.npz 13 kB |
| `abl_bal19_s1` | Ablation: batch composition balance 0.19 instead of half and half. | first=5, epochs=60, view_dropout=0.1, w_h=1.0, balance=0.19, seed=1 | 59 | 35,270,444 | best.pt 141 MB, latent_stats.npz 13 kB |
| `smoke` | A short test run of the training code. | views=pclm+pepland+esm, optional_views=esm, fill=imputed, sea_input=latents, first=2, epochs=1, view_dropout=0.0 | 1 | 32,890,544 | best.pt 132 MB |

Generations: 1 = t1 to t5 (published wiring), 2 = t8 to t11 (ESM-2 as an optional input), 3 = v3_* (ESM-2 as a target only).

## Students (`student_distill/`)

The ladder st1 to stca was trained under one budget: the first 5 of 51 pool shards (500,000 molecules), 20 epochs, seed 1, tokenizer `student_tokenizer_norm/atombpe_v1024`, teacher v3_base. st6f is the final student on the whole pool. All have the same architecture (8 blocks, width 320, 5 heads, SwiGLU 768).

| Run | What it is | Training molecules | Epoch of best.pt | Parameters in the file | Files |
|---|---|---|---|---|---|
| `st1` | Ladder: the teacher latent only (cosine + smooth L1). | 494,988 | 20 | 10,821,185 | best.pt 43 MB |
| `st2` | Ladder: st1 + in-batch contrastive loss + relational (similarity-matrix) loss. | 494,988 | 20 | 10,821,185 | best.pt 43 MB |
| `st3` | Ladder: st2 + the four view heads, balanced by running means, with head dropping. | 494,988 | 20 | 21,108,077 | best.pt 84 MB |
| `st4` | Ladder: no latent target, the four views only (+ contrastive and relational terms). | 494,988 | 5 | 21,108,077 | best.pt 84 MB |
| `st5` | Ladder: st3 + masked-token loss on 20 % of the steps. | 494,988 | 20 | 21,109,101 | best.pt 84 MB |
| `st6` | Ladder: st3 + random-SMILES augmentation (p 0.5) + spelling-consistency loss. | 494,988 | 14 | 21,108,077 | best.pt 84 MB |
| `st7` | Ladder: st5 + the same augmentation (everything on). No weights were produced; not benchmarked. | 494,988 |  |  | (no weights) |
| `stc` | Control without a teacher: masked-token loss only. | 494,988 | 20 | 10,822,209 | best.pt 43 MB |
| `stca` | Control without a teacher: masked-token loss only, with random spellings. | 494,988 | 20 | 10,822,209 | best.pt 43 MB |
| `st6f` | FINAL STUDENT: the st6 recipe on the whole 3.40 M pool. best.pt = EMA weights of the best epoch (epoch 2); epoch_02/04/06.pt are snapshots. | 3,366,362 | 2 | 21,108,077 | best.pt 84 MB, epoch_02.pt 84 MB, epoch_04.pt 84 MB, epoch_06.pt 84 MB |

The parameters in the file include the four training-only view heads; the student needs 10,821,184 of them at inference (the other 10,286,893 are the heads). The models without view heads (st1, st2 and the two controls) have about 10.8 M parameters in the file.

## Student shape pilot (`shape_pilot/`, `students/`)

Four shapes of a student of about ten million parameters, trained from random initialisation on the masked-token objective for one epoch of the 2 M-molecule corpus, same seed, data and validation split for all (code: `shape_pilot/`). A `_lr5e-4` run repeats a shape at learning rate 5e-4. `students/student-<shape>/` holds the same four shapes in Hugging Face format (`config.json`, `model.safetensors`, tokenizer files).

| Run | Shape | Final validation | Files |
|---|---|---|---|
| `s256x15` | width 256, 15 blocks, 4 heads | loss 0.3462 | student_final.pt 40 MB |
| `s256x15_lr5e-4` | width 256, 15 blocks, 4 heads | loss 0.3471 | student_final.pt 40 MB |
| `s320x10` | width 320, 10 blocks, 5 heads | loss 0.3477 | student_final.pt 42 MB |
| `s320x10_lr5e-4` | width 320, 10 blocks, 5 heads | loss 0.3444 | student_final.pt 42 MB |
| `s384x7` | width 384, 7 blocks, 6 heads | loss 0.3645 | student_final.pt 43 MB |
| `s512x4` | width 512, 4 blocks, 8 heads | loss 0.3816 | student_final.pt 44 MB |

## Tokenizers

Atom-level byte-pair-encoding tokenizers learned from the pool (`src/student/train_tokenizer.py`).

| Folder | Vocabulary | Learned from | What it is |
|---|---|---|---|
| `student_tokenizer/atombpe_v1024` | 1024 | 200,000 molecules | candidates compared at the start (vocabulary 1024, 2048, 4096) |
| `student_tokenizer/atombpe_v2048` | 2048 | 200,000 molecules | candidates compared at the start (vocabulary 1024, 2048, 4096) |
| `student_tokenizer/atombpe_v4096` | 4096 | 200,000 molecules | candidates compared at the start (vocabulary 1024, 2048, 4096) |
| `st_random_200k/atombpe_v2048` | 2048 | 200,000 molecules | tokenizer sampling experiment: random sample of 200,000 molecules |
| `st_strat_200k/atombpe_v2048` | 2048 | 200,000 molecules | tokenizer sampling experiment: stratified sample of 200,000 molecules |
| `st_strat_500k/atombpe_v2048` | 2048 | 500,000 molecules | tokenizer sampling experiment: stratified sample of 500,000 molecules |
| `student_tokenizer_norm/atombpe_v1024` | 1024 | 500,000 molecules | FINAL: learned from the normalised pool; used by every student above |

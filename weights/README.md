# weights/

Every network trained in this project (**see [`MODELS.md`](MODELS.md) for the full list**: 21 teacher runs, the student ladder, the size pilot and the tokenizers), in the same folder layout as `$BTP_DATA/checkpoints/` on the lab server (so the scripts find them if you copy or link this folder as `$BTP_DATA/checkpoints`; the published PeptideCLM-2, PepLand and ESM-2 weights live in the same place and are not repeated here).

| File | What it is | Size |
|---|---|---|
| `student_distill/st6f/best.pt` | The SMILES-only student (EMA weights of the best epoch, epoch 2 of the run). 21,108,077 parameters in the file, of which 10,821,184 are used at inference; the rest are the four training-only view heads. | 84 MB |
| `student_distill/st6f/{history.csv, meta.json, COMPLETE}` | Training history and metadata of the run. | small |
| `student_tokenizer_norm/atombpe_v1024/` | The student's tokenizer (atom-level BPE, vocabulary 1024). | small |
| `teacher/v3_base/best.pt` | The fusion teacher (35,270,444 parameters in training; 24,990,208 at inference once the decoders and the view-dropout vectors are dropped). | 141 MB |
| `teacher/v3_base/latent_stats.npz` | Per-dimension statistics with which the teacher's latent is standardised. | small |

The `.pt` files are **not stored in git** (the teacher is above GitHub's 100 MB file limit); they are in the Zenodo record [10.5281/zenodo.24507677](https://doi.org/10.5281/zenodo.24507677) (the main README explains how to restore them). `SHA256SUMS.txt` lists the checksums of all files, so a download can be verified (`sha256sum -c SHA256SUMS.txt`, or `certutil -hashfile <file> SHA256` on Windows).

## Using the student

From the repository root, with PyTorch and RDKit installed (CPU is enough):

```python
import sys, torch
sys.path[:0] = ["src/student", "src"]
import model as M, tokenizer as T, normalise as NM

ck = torch.load("weights/student_distill/st6f/best.pt", map_location="cpu", weights_only=True)
net = M.Student(**ck["cfg"]); net.load_state_dict(ck["model"]); net.eval()
tok = T.load_tokenizer("weights/student_tokenizer_norm/atombpe_v1024")

smiles, _ = NM.normalise_many(["NCCCC[C@H](N)C(=O)N[C@@H](Cc1ccccc1)C(=O)O"], 1)   # neutralise + canonical SMILES, as in training
ids = [tok.with_special(e, 128) for e in tok.encode_batch(smiles)]                   # at most 128 tokens
x = torch.tensor(ids)                                  # one molecule; for a batch pad with T.PAD and pass x != T.PAD as the mask
with torch.no_grad():
    z = net(x, x != T.PAD)["z"]                        # [1, 1024] embedding
```

`src/student/export_student.py` does the same for whole benchmark files. The embedding is meant to be read by a simple learner (linear, kernel, gradient-boosted trees, MLP); see `src/teacher/eval_harness.py`.

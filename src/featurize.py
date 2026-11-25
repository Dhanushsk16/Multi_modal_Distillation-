"""Featurise every benchmark SMILES with one representation; save to features/<rep>.npz.

    python src/featurize.py --rep ecfp     # count ECFP, radius 2, 2048 (CPU)
    python src/featurize.py --rep mapc     # MAPC, 1-bit MinHash, 2048 (CPU)
    python src/featurize.py --rep pclm     # PeptideCLM-2 mlm-large mean pool, 1024 (GPU)
    python src/featurize.py --rep pclm_mtr # PeptideCLM-2 mtr-large mean pool, 1024 (GPU); features/<feat>/pclm_mtr.npz
    python src/featurize.py --rep rdkit    # RDKit 2D descriptors, 206 (CPU; src/rdkit_desc.py)

PepLand runs in its own environment: src/featurize_pepland.py.

Output: features/<rep>.npz with
    smiles  (N,)    the keys, same order as X
    X       (N, d)
    ok      (N,)    False where featurisation failed (row of X is zeros)
    names   (d,)    rdkit only: the descriptor names, one per column
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from benchmarks import all_smiles  # noqa: E402
from paths import EXTERNAL, FEAT_DIR as FEAT, PCLM_DIR, PCLM_MTR_DIR  # noqa: E402


# --------------------------------------------------------------------- ECFP
def _ecfp(smi, n=2048):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    g = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n)
    return g.GetCountFingerprintAsNumPy(m).astype(np.float32)


# --------------------------------------------------------------------- MAPC
def _mapc(smi, n=2048):
    """Chiral MAP fingerprint as 1-bit MinHash.

    mapchiral returns 2048 MinHash values: arbitrary integers whose only meaning is
    position-wise equality. Keeping the lowest bit of each gives balanced binary
    features whose agreement tracks MinHash Jaccard (r = 0.985 measured on
    benchmark peptides). Folding the shingles instead saturates on peptides
    (~1750 of 2048 bits set), so it is not used."""
    from rdkit import Chem
    from mapchiral.mapchiral import get_fingerprint
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    h = get_fingerprint(m, max_radius=2, n_permutations=n)
    return (h & 1).astype(np.float32)


def _init_mapc():
    sys.path.insert(0, os.path.join(EXTERNAL, "mapchiral"))


# --------------------------------------------------------------------- RDKit descriptors
def _rdkit(smi):
    import rdkit_desc
    return rdkit_desc.describe(smi)


def cpu_rep(fn, smiles, workers, init=None):
    with Pool(workers, initializer=init) as p:
        out = p.map(fn, smiles, chunksize=64)
    d = next(len(x) for x in out if x is not None)
    ok = np.array([x is not None for x in out])
    X = np.stack([x if x is not None else np.zeros(d, np.float32) for x in out])
    return X, ok


# --------------------------------------------------------------- PeptideCLM-2
def _patch_sdpa():
    """Their attention passes an fp32 padding mask; CUDA SDPA rejects it against
    bf16 queries. Cast it to the query dtype (clamped so -1e9 stays finite)."""
    import torch
    import torch.nn.functional as F
    orig = F.scaled_dot_product_attention
    if getattr(orig, "_patched", False):
        return

    # Their code calls SDPA with keywords (query=, key=, value=), so the wrapper
    # must accept the same names.
    def safe(query, key, value, attn_mask=None, **kw):
        if attn_mask is not None and attn_mask.dtype not in (torch.bool, query.dtype):
            attn_mask = attn_mask.to(query.dtype).clamp_(min=torch.finfo(query.dtype).min)
        return orig(query, key, value, attn_mask=attn_mask, **kw)

    safe._patched = True
    F.scaled_dot_product_attention = safe


def fix_rotary(model):
    """Rebuild every rotary theta/cache from the config, then check them against an independent
    recomputation. The checkpoint does not hold these non-persistent buffers, so after
    from_pretrained they contain whatever was in memory. The model repairs only the obviously
    invalid ones (theta outside (0, 1], non-finite cache); a finite garbage cache goes through and
    the embeddings are silently wrong (seen on the laptop in 1 of 6 fresh processes: cosine 0.96
    against the correct ones; the server's 'two models in one process' pitfall is the same thing)."""
    import torch
    n = 0
    dev = next(model.parameters()).device
    for m in model.modules():
        if not (hasattr(m, "theta") and hasattr(m, "cache") and hasattr(m, "build_rope_cache")):
            continue
        m.theta = m._compute_theta(device=dev)
        m.build_rope_cache(m.max_seq_len)
        pos = torch.arange(m.max_seq_len, dtype=torch.float32)
        th = float(m.base) ** (-torch.arange(0, m.dim, 2, dtype=torch.float32)[: m.dim // 2] / float(m.dim))
        ang = torch.einsum("i,j->ij", pos, th)
        ref = torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)
        assert torch.allclose(m.theta.float().cpu(), th, rtol=1e-5, atol=0), "rotary theta is wrong"
        assert torch.allclose(m.cache.float().cpu(), ref, atol=1e-4), "rotary cache is wrong"
        n += 1
    assert n > 0, "no rotary modules found: has the model code changed?"
    return n


def load_pclm(model_dir):
    """-> (tokenizer, model, device) with the rotary buffers rebuilt and verified (fix_rotary).
    Load once per process and reuse it."""
    import torch
    from transformers import AutoModel, AutoTokenizer
    _patch_sdpa()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, trust_remote_code=True,
                                      use_safetensors=True).to(dev).eval()
    fix_rotary(model)
    return tok, model, dev


def pclm(smiles, model_dir, max_tokens=8192):
    """One-shot: load the model, embed, return. (The corpus cache uses load_pclm +
    pclm_encode so the model is loaded once for all shards.)"""
    tok, model, dev = load_pclm(model_dir)
    return pclm_encode(tok, model, dev, smiles, max_tokens)


def pclm_encode(tok, model, dev, smiles, max_tokens=8192):
    import torch
    lens = np.array([len(x) + 2 for x in tok(smiles, add_special_tokens=True)["input_ids"]])
    order = np.argsort(lens)
    X = np.zeros((len(smiles), model.config.embed_dim), np.float32)
    i = 0
    with torch.no_grad():
        while i < len(order):
            T = lens[order[i]]
            j = i
            while j < len(order) and lens[order[j]] * (j - i + 1) <= max_tokens:
                j += 1
            j = max(j, i + 1)
            idx = order[i:j]
            b = tok([smiles[k] for k in idx], return_tensors="pt", padding=True).to(dev)
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev == "cuda")):
                out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
            X[idx] = out.mean_pool.float().cpu().numpy()
            i = j
    ok = np.isfinite(X).all(1)
    return X, ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rep", choices=["ecfp", "mapc", "pclm", "pclm_mtr", "rdkit"], required=True)
    ap.add_argument("--workers", type=int, default=8, help="lab rule: at most half the CPU")
    ap.add_argument("--pclm-dir", default=None, help="model folder (default: mlm-large for pclm, mtr-large for pclm_mtr)")
    ap.add_argument("--bench", nargs="*", default=None, help="benchmarks to featurise (default: the main ones + AmpHGT)")
    ap.add_argument("--feat", default=None, help="features subfolder under features/ (default single_modality)")
    a = ap.parse_args()

    smiles = all_smiles(a.bench)
    feat = os.path.join(os.path.dirname(FEAT), a.feat) if a.feat else FEAT
    out = os.path.join(feat, a.rep + ".npz")
    if os.path.exists(out):
        print("%s: already done (%s)" % (a.rep, out)); return
    t0 = time.time()
    if a.rep == "ecfp":
        X, ok = cpu_rep(_ecfp, smiles, a.workers)
    elif a.rep == "mapc":
        X, ok = cpu_rep(_mapc, smiles, a.workers, init=_init_mapc)
    elif a.rep == "rdkit":
        X, ok = cpu_rep(_rdkit, smiles, a.workers)
        X = np.where(ok[:, None], X, 0).astype(np.float32)     # NaN stays only inside usable rows
    else:
        X, ok = pclm(smiles, a.pclm_dir or (PCLM_MTR_DIR if a.rep == "pclm_mtr" else PCLM_DIR))
    os.makedirs(feat, exist_ok=True)
    extra = {}
    if a.rep == "rdkit":
        import rdkit_desc
        extra["names"] = np.array(rdkit_desc.names())
    np.savez_compressed(out + ".tmp.npz", smiles=np.array(smiles), X=X, ok=ok, **extra)
    os.replace(out + ".tmp.npz", out)          # atomic: a killed run leaves no half file
    print("%s: %d molecules -> %s | failed %d | %.1f min"
          % (a.rep, len(smiles), X.shape, (~ok).sum(), (time.time() - t0) / 60))


if __name__ == "__main__":
    main()

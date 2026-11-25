"""PepLand NODE-level embeddings (atoms and fragments, 300-d each) for the benchmark SMILES, so that a pooling
can be trained on them (src/finetune/pool_probe.py: avg / max / attention / the released GRU pooling).
The ordinary featurize_pepland.py keeps only the average-pooled 300-d vector.

Runs in the pepland environment (torch + DGL), CPU:

    conda activate pepland
    python src/featurize_pepland_nodes.py --bench CellPPD THPep PepMSND PAMPA                  # -> features/single_modality/
    python src/featurize_pepland_nodes.py --feat pepland_suite --bench PL_cCPP PL_cSol PL_ncCPP PL_cBinding PL_ncBinding

Same extractor, same settings as featurize_pepland.py (padding_mode="exclude": only real nodes are returned).
Writes features/<feat>/pepland_nodes.npz: smiles, ok, a_flat [Natoms, 300] fp16 + a_off [N+1],
f_flat [Nfrags, 300] fp16 + f_off [N+1] (molecule i owns rows off[i]:off[i+1]). Molecules the fragmenter cannot
handle get no nodes and ok=False. Time: about the same as featurize_pepland.py (the GNN dominates); the file is a
few GB at most for these benchmarks (AmpHGT, 16.9k long molecules, is left out on purpose).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from benchmarks import all_smiles  # noqa: E402
from paths import EXTERNAL, FEAT_DIR, PEPLAND_MODEL  # noqa: E402

sys.path.insert(0, EXTERNAL)                                 # parent of the pepland checkout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--threads", type=int, default=8, help="lab rule: at most half the CPU")
    ap.add_argument("--bench", nargs="*", default=None)
    ap.add_argument("--feat", default=None, help="features subfolder under features/ (default single_modality)")
    ap.add_argument("--max-molecules", type=int, default=0, help="only the first N molecules (smoke test)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    feat = os.path.join(os.path.dirname(FEAT_DIR), a.feat) if a.feat else FEAT_DIR
    out = os.path.join(feat, "pepland_nodes.npz")
    if os.path.exists(out):
        print("pepland nodes: already done (%s)" % out); return

    from pepland.model.core import PepLandFeatureExtractor
    ext = PepLandFeatureExtractor(model_path=PEPLAND_MODEL, pooling="avg", padding_mode="exclude")
    ext.eval()

    smiles = all_smiles(a.bench)
    if a.max_molecules:
        smiles = smiles[:a.max_molecules]
    n = len(smiles)
    a_pieces, f_pieces = [None] * n, [None] * n
    ok = np.zeros(n, bool)

    def one_batch(idx):
        A, Fr, ac, fc = ext.extract_atom_fragment_embedding([smiles[k] for k in idx], return_counts=True)
        for r, k in enumerate(idx):
            na, nf = int(ac[r]), int(fc[r])
            a_k = A[r, :na].float().cpu().numpy()
            f_k = Fr[r, :nf].float().cpu().numpy()
            if na and nf and np.isfinite(a_k).all() and np.isfinite(f_k).all() and max(np.abs(a_k).max(), np.abs(f_k).max()) < 6e4:
                a_pieces[k], f_pieces[k] = a_k.astype(np.float16), f_k.astype(np.float16)
                ok[k] = True

    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, a.batch):
            idx = list(range(i, min(i + a.batch, n)))
            try:
                one_batch(idx)
            except Exception:
                for k in idx:                            # isolate the failing molecule
                    try:
                        one_batch([k])
                    except Exception:
                        pass
            if (i // a.batch) % 50 == 0:
                print("%6d/%d  %.1f min" % (i, n, (time.time() - t0) / 60), flush=True)

    empty = np.zeros((0, 300), np.float16)
    a_pieces = [p if p is not None else empty for p in a_pieces]
    f_pieces = [p if p is not None else empty for p in f_pieces]
    a_off = np.concatenate([[0], np.cumsum([len(p) for p in a_pieces])]).astype(np.int64)
    f_off = np.concatenate([[0], np.cumsum([len(p) for p in f_pieces])]).astype(np.int64)
    os.makedirs(feat, exist_ok=True)
    np.savez_compressed(out + ".tmp.npz", smiles=np.array(smiles), ok=ok, a_flat=np.concatenate(a_pieces), a_off=a_off,
                        f_flat=np.concatenate(f_pieces), f_off=f_off)
    os.replace(out + ".tmp.npz", out)
    print("pepland nodes: %d molecules, %d atoms, %d fragments -> %s | failed %d | %.1f min"
          % (n, a_off[-1], f_off[-1], out, (~ok).sum(), (time.time() - t0) / 60))


if __name__ == "__main__":
    main()

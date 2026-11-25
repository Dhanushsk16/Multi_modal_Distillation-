"""PepLand 300-d embeddings for every benchmark SMILES -> features/pepland.npz.

Runs in its own environment (torch 2.2 + DGL + mlflow), not the main one:

    conda activate pepland
    python src/featurize_pepland.py

Uses the authors' feature extractor: average readout over real atom and fragment
nodes (padding_mode="exclude"). Molecules the fragmenter cannot handle get a
zero row and ok=False, so the evaluator can report coverage instead of crashing.
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
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bench", nargs="*", default=None, help="benchmarks to featurise (default: the main ones + AmpHGT)")
    ap.add_argument("--feat", default=None, help="features subfolder under features/ (default single_modality)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    feat = os.path.join(os.path.dirname(FEAT_DIR), a.feat) if a.feat else FEAT_DIR
    out = os.path.join(feat, "pepland.npz")
    if os.path.exists(out):
        print("pepland: already done (%s)" % out); return

    from pepland.model.core import PepLandFeatureExtractor
    ext = PepLandFeatureExtractor(
        model_path=PEPLAND_MODEL,
        pooling="avg", padding_mode="exclude")
    ext.eval()

    smiles = all_smiles(a.bench)
    X = np.zeros((len(smiles), 300), np.float32)
    ok = np.zeros(len(smiles), bool)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(smiles), a.batch):
            chunk = smiles[i:i + a.batch]
            try:
                X[i:i + len(chunk)] = ext(chunk).float().cpu().numpy()
                ok[i:i + len(chunk)] = True
            except Exception:
                for j, s in enumerate(chunk):           # isolate the failing molecule
                    try:
                        X[i + j] = ext([s]).float().cpu().numpy()[0]
                        ok[i + j] = True
                    except Exception:
                        pass
            if (i // a.batch) % 50 == 0:
                print("%6d/%d  %.1f min" % (i, len(smiles), (time.time() - t0) / 60), flush=True)
    ok &= np.isfinite(X).all(1)
    os.makedirs(feat, exist_ok=True)
    np.savez_compressed(out + ".tmp.npz", smiles=np.array(smiles), X=X, ok=ok)
    os.replace(out + ".tmp.npz", out)
    print("pepland: %d molecules -> %s | failed %d | %.1f min"
          % (len(smiles), X.shape, (~ok).sum(), (time.time() - t0) / 60))


if __name__ == "__main__":
    main()

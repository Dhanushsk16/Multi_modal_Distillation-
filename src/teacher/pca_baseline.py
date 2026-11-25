"""The PCA baseline of the evaluation harness: standardised PepLand + PeptideCLM-2 + ECFP side by side (300 + 1024 + 2048 = 3372
numbers), reduced to 1024 dimensions by PCA. It is the simplest "fusion" there is, with the same output size as the teacher; a
learned teacher has to match it.

    python src/teacher/pca_baseline.py --fit                  # ALL training rows of the pool: the covariance of the 3372 numbers is
                                                              # accumulated shard by shard, then an exact eigendecomposition (CPU, minutes)
    python src/teacher/pca_baseline.py --fit --rows 100000    # a quick version: randomized PCA on the first 100,000 training rows

The teacher trains on the whole pool, so the baseline it is compared with is fitted on the whole pool too. Writes
features/corpus/pca_plh_1024.npz (needs the standardising statistics of `views_v3.py --stats`).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import views_v3 as V  # noqa: E402
from paths import CACHE_DIR  # noqa: E402

PCA_FILE = os.path.join(CACHE_DIR, "pca_plh_1024.npz")


def stack(arrays, st):
    """{view: raw [n, d]} -> standardised PepLand | PeptideCLM-2 | ECFP side by side, float32 [n, 3372]."""
    return np.concatenate([V.prep(v, arrays[v], st) for v in V.INPUTS], 1)


def _save(out, components, mean, explained, rows):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out + ".tmp.npz", components=components.astype(np.float32), mean=mean.astype(np.float32),
             explained=explained.astype(np.float32), rows=rows)
    os.replace(out + ".tmp.npz", out)


def fit_all(k=1024, out=PCA_FILE, stats=None):
    """Exact PCA of ALL training rows (P, L and H fine), without ever holding them together: sum of x, sum of x x^T per shard."""
    have = set.intersection(*[set(V.shard_ids(v)) for v in V.INPUTS])
    shards = sorted(have)
    assert shards, "no shard is cached in all of PepLand, PeptideCLM-2 and ECFP"
    st = stats or (V.load_stats() if os.path.exists(V.STATS_FILE) else None)
    assert st is not None, "run `views_v3.py --stats` first: the baseline standardises with its statistics"
    d = sum(V.DIM[v] for v in V.INPUTS)
    S, s1, n = np.zeros((d, d)), np.zeros(d), 0
    t0 = time.time()
    for i, s in enumerate(shards):
        arrs, _, _ = V.read_pool([s], views=V.INPUTS)
        X = stack(arrs, st)
        S += X.T @ X                                       # float32 product of one shard, added in float64
        s1 += X.sum(0, dtype=np.float64)
        n += len(X)
        if (i + 1) % 5 == 0 or i + 1 == len(shards):
            print("   covariance: shard %d of %d | %s rows | %.0f s" % (i + 1, len(shards), format(n, ","), time.time() - t0), flush=True)
    mean = s1 / n
    cov = (S - n * np.outer(mean, mean)) / (n - 1)
    print("eigendecomposition of the %d x %d covariance of %s rows" % (d, d, format(n, ",")), flush=True)
    w, U = np.linalg.eigh(cov)
    order = np.argsort(w)[::-1]
    w, U = np.clip(w[order], 0, None), U[:, order]
    comp = U[:, :k].T.copy()
    flip = np.sign(comp[np.arange(k), np.abs(comp).argmax(1)])            # fixed sign: the largest entry of a component is positive
    comp *= flip[:, None]
    explained = w[:k] / w.sum()
    _save(out, comp, mean, explained, n)
    print("PCA on ALL %s training rows x %d dims -> %d components | explained variance %.1f%% | wrote %s"
          % (format(n, ","), d, k, 100 * float(explained.sum()), out))


def fit(rows=100_000, k=1024, seed=0, out=PCA_FILE, stats=None):
    """The quick version: randomized PCA on the first `rows` training rows."""
    from sklearn.decomposition import PCA
    have = set.intersection(*[set(V.shard_ids(v)) for v in V.INPUTS])
    shards = sorted(have)
    assert shards, "no shard is cached in all of PepLand, PeptideCLM-2 and ECFP"
    # shards are shuffled, so the first ones are a random sample; read as many as `rows` needs
    arrs, _, r = V.read_pool(shards, views=V.INPUTS, max_rows=rows)
    arrs = {v: x[:rows] for v, x in arrs.items()}
    st = stats or (V.load_stats() if os.path.exists(V.STATS_FILE) else V.fit_stats(arrs))
    X = stack(arrs, st)
    print("PCA on %s rows x %d dims -> %d components" % (format(len(X), ","), X.shape[1], k), flush=True)
    p = PCA(n_components=k, svd_solver="randomized", random_state=seed).fit(X)
    _save(out, p.components_, p.mean_, p.explained_variance_ratio_, len(X))
    print("explained variance of the %d components: %.1f%% | wrote %s" % (k, 100 * float(p.explained_variance_ratio_.sum()), out))


def load(path=PCA_FILE):
    z = np.load(path)
    return {"components": z["components"], "mean": z["mean"]}


def transform(arrays, st, pca):
    """{view: raw [n, d]} (PepLand, PeptideCLM-2, ECFP) -> the 1024-d PCA embedding, float32."""
    return ((stack(arrays, st) - pca["mean"]) @ pca["components"].T).astype(np.float32)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--rows", type=int, default=0, help="0 = all training rows of the pool (default); N = quick randomized PCA on N rows")
    ap.add_argument("--k", type=int, default=1024)
    a = ap.parse_args()
    assert a.fit, "give --fit"
    fit_all(a.k) if a.rows == 0 else fit(a.rows, a.k)

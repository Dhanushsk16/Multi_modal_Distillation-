"""The four views of teacher v3 and their preprocessing, in numpy (used by the evaluation harness and the PCA baseline;
train_teacher_v3.py applies the same formulas to torch tensors).

    P  pepland  300    input    graph network
    L  pclm     1024   input    SMILES language model (mean pool)
    H  ecfp     2048   input    hand-crafted substructure counts
    E  esm      2560   TARGET ONLY (never an input), present for standard chains only

Preprocessing (decided 2026-11-17):
    every view: standardise each dimension with the mean and std of the TRAINING rows (E: of the rows that have it);
    H: log(1 + count) first, and the std is floored at 0.1 so rare patterns do not blow up;
    P, L, E: the std is floored at 1e-3 x the RMS std over dimensions (a guard against constant dimensions; it only matters for dimensions that are nearly constant);
    every standardised value is clipped to +-10 (as in teachers t1-t11).

    python src/teacher/views_v3.py --stats               # ALL training rows of the pool, shard by shard (mean and spread need no sample);
                                                         # writes features/corpus/views_v3_stats.npz and reports clipping over all rows
    python src/teacher/views_v3.py --stats --shards 6    # only the first 6 shards (a quick test)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import CACHE_DIR  # noqa: E402

VIEWS = ("pepland", "pclm", "ecfp", "esm")                   # P L H E
LETTER = {"P": "pepland", "L": "pclm", "H": "ecfp", "E": "esm", "M": "pclm_mtr"}      # M: PeptideCLM-2 MTR (only the optional extra input)
DIM = {"pepland": 300, "pclm": 1024, "ecfp": 2048, "esm": 2560, "pclm_mtr": 1024}
INPUTS = ("pepland", "pclm", "ecfp")
EXTRA_INPUTS = ("pclm_mtr",)                                 # optional fourth input of a teacher (train_teacher_v3.py --extra-input pclm_mtr)
TARGET_ONLY = "esm"
H_FLOOR = 0.1
REL_FLOOR = 1e-3
CLIP = 10.0
VAL_MOD = 100                                                # a pool row is validation when row % 100 == 0 (as in train_teacher.py)
STATS_FILE = os.path.join(CACHE_DIR, "views_v3_stats.npz")


def to_float(view, X):
    """Raw cached values -> float32 (H: log1p of the counts)."""
    x = np.asarray(X, np.float32)
    return np.log1p(x) if view == "ecfp" else x


def finish_stats(view, mu, sd):
    """Mean and spread (float64 [d]) of one view -> the floored float32 statistics used everywhere."""
    if view == "ecfp":
        sd = np.maximum(sd, H_FLOOR)
    else:
        sd = np.maximum(sd, REL_FLOOR * np.sqrt((sd ** 2).mean()) + 1e-12)
    return {"mu": np.asarray(mu).astype(np.float32), "sd": np.asarray(sd).astype(np.float32)}


class Running:
    """Mean and variance of each dimension over rows that arrive batch by batch (Chan's merge, float64): exact, and the rows never
    have to be in memory together."""

    def __init__(self):
        self.n, self.mu, self.m2 = 0, None, None

    def add(self, x):                                     # x: [m, d]
        x = np.asarray(x, np.float64)
        if len(x):
            mu = x.mean(0)
            self.merge(len(x), mu, ((x - mu) ** 2).sum(0))

    def merge(self, n_b, mu_b, m2_b):
        if self.n == 0:
            self.n, self.mu, self.m2 = int(n_b), np.asarray(mu_b, np.float64), np.asarray(m2_b, np.float64)
            return
        n = self.n + n_b
        d = np.asarray(mu_b, np.float64) - self.mu
        self.mu = self.mu + d * (n_b / n)
        self.m2 = self.m2 + np.asarray(m2_b, np.float64) + d * d * (self.n * n_b / n)
        self.n = int(n)

    def mean_sd(self):                                    # population sd, as numpy's std
        return self.mu, np.sqrt(self.m2 / self.n)


def fit_stats(arrays, present=None):
    """arrays {view: raw [N, d]}, present {view: bool [N]} (a view absent for a row is left out of its own statistics).
    -> {view: {"mu": float32 [d], "sd": float32 [d]}}. For rows that are all in memory; the pool uses fit_stats_stream."""
    st = {}
    for v, X in arrays.items():
        keep = None if present is None else present.get(v)
        x = to_float(v, X if keep is None else X[keep]).astype(np.float64)
        assert len(x) >= 1000, "only %d rows to standardise %s: too few" % (len(x), v)
        st[v] = finish_stats(v, x.mean(0), x.std(0))
    return st


def fit_stats_stream(shards, views=None, cache_dir=CACHE_DIR, log=print):
    """The statistics of ALL training rows of the given pool shards, one shard at a time (E: of the rows that have it).
    -> (stats, n_rows, n_rows_with_E)"""
    import time
    views = tuple(views or VIEWS)
    acc = {v: Running() for v in views}
    n_rows = n_e = 0
    t0 = time.time()
    for k, s in enumerate(shards):
        arrs, pres, rows = read_pool([s], views, cache_dir)
        for v in views:
            x = arrs[v] if pres[v].all() else arrs[v][pres[v]]
            acc[v].add(to_float(v, x))
        n_rows += len(rows)
        n_e += int(pres[TARGET_ONLY].sum()) if TARGET_ONLY in views else 0
        if (k + 1) % 5 == 0 or k + 1 == len(shards):
            log("   statistics: shard %d of %d | %s rows | %.0f s" % (k + 1, len(shards), format(n_rows, ","), time.time() - t0))
    for v in views:
        assert acc[v].n >= 1000, "only %d rows to standardise %s: too few" % (acc[v].n, v)
    return {v: finish_stats(v, *acc[v].mean_sd()) for v in views}, n_rows, n_e


def prep(view, X, st):
    """Raw rows -> standardised float32, clipped to +-CLIP."""
    z = (to_float(view, X) - st[view]["mu"]) / st[view]["sd"]
    return np.clip(z, -CLIP, CLIP, out=z)


def save_stats(st, path=STATS_FILE, **meta):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    flat = {"%s_%s" % (v, k): a for v, d in st.items() for k, a in d.items()}
    np.savez(path + ".tmp.npz", **flat, **{k: np.asarray(x) for k, x in meta.items()})
    os.replace(path + ".tmp.npz", path)


def load_stats(path=STATS_FILE):
    z = np.load(path, allow_pickle=True)
    return {v: {"mu": z[v + "_mu"], "sd": z[v + "_sd"]} for v in VIEWS if v + "_mu" in z.files}


def shard_ids(view, cache_dir=CACHE_DIR):
    d = os.path.join(cache_dir, view)
    names = os.listdir(d) if os.path.isdir(d) else []
    return sorted(int(f[6:9]) for f in names if f.startswith("shard_") and f.endswith(".npz") and ".tmp" not in f)


def read_pool(shards, views=VIEWS, cache_dir=CACHE_DIR, max_rows=0):
    """Training rows (row % VAL_MOD != 0) of the given pool shards that have every INPUT view fine.
    -> (arrays {view: raw [n, d]}, present {view: bool [n]} (E: its own ok flag), rows [n])"""
    arrs, pres, rows = {v: [] for v in views}, {v: [] for v in views}, []
    got = 0
    for s in shards:
        z = {v: np.load(os.path.join(cache_dir, v, "shard_%03d.npz" % s)) for v in views}
        n = min(len(z[v]["ok"]) for v in views)
        lo = int(z[views[0]]["lo"])
        r = np.arange(lo, lo + n)
        keep = r % VAL_MOD != 0
        for v in views:
            if v in INPUTS:
                keep &= z[v]["ok"][:n]
        for v in views:
            arrs[v].append(np.asarray(z[v]["X"][:n])[keep])
            pres[v].append(z[v]["ok"][:n][keep] if v == TARGET_ONLY else np.ones(int(keep.sum()), bool))
        rows.append(r[keep])
        got += int(keep.sum())
        if max_rows and got >= max_rows:
            break
    return ({v: np.concatenate(a) for v, a in arrs.items()}, {v: np.concatenate(p) for v, p in pres.items()}, np.concatenate(rows))


def main():
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", action="store_true", help="fit the statistics on the pool's training rows and write STATS_FILE")
    ap.add_argument("--shards", type=int, default=0, help="0 = every shard cached in all four views (the default); N = only the first N (a quick test)")
    ap.add_argument("--out", default=STATS_FILE)
    a = ap.parse_args()
    assert a.stats, "give --stats"
    per_view = {v: set(shard_ids(v)) for v in VIEWS}
    have = set.intersection(*per_view.values())
    if len({len(x) for x in per_view.values()}) > 1:
        print("WARNING: the views are cached for different numbers of shards %s; only the %d shards cached in all of them are used"
              % ({v: len(x) for v, x in per_view.items()}, len(have)))
    shards = sorted(have)[: a.shards] if a.shards else sorted(have)
    assert shards, "no shard is cached in all of %s under %s" % (list(VIEWS), CACHE_DIR)
    print("statistics of ALL training rows of %d shards (%d..%d)" % (len(shards), shards[0], shards[-1]), flush=True)
    st, n_rows, n_e = fit_stats_stream(shards)
    print("training rows: %s | E present for %.1f%%" % (format(n_rows, ","), 100 * n_e / max(n_rows, 1)), flush=True)
    save_stats(st, a.out, shards=np.array(shards), n_rows=n_rows)
    # a second pass over the same rows: how many standardised values are clipped, or large
    clip = {v: 0 for v in VIEWS}
    big = {v: 0 for v in VIEWS}
    cnt = {v: 0 for v in VIEWS}
    t0 = time.time()
    for k, s in enumerate(shards):
        arrs, pres, _ = read_pool([s])
        for v in VIEWS:
            x = arrs[v][pres[v]]
            z = np.abs((to_float(v, x) - st[v]["mu"]) / st[v]["sd"])
            clip[v] += int((z > CLIP).sum())
            big[v] += int((z > 5).sum())
            cnt[v] += z.size
        if (k + 1) % 5 == 0 or k + 1 == len(shards):
            print("   clipping check: shard %d of %d | %.0f s" % (k + 1, len(shards), time.time() - t0), flush=True)
    for v in VIEWS:
        print("%-8s dims %4d | sd: min %.4g median %.4g max %.4g | clipped at +-%g: %.3f%% of values | |z| > 5: %.3f%%" % (
            v, DIM[v], st[v]["sd"].min(), np.median(st[v]["sd"]), st[v]["sd"].max(), CLIP,
            100 * clip[v] / cnt[v], 100 * big[v] / cnt[v]))
        if v == "ecfp":
            print("         H: dimensions held at the 0.1 floor: %d of %d" % (int((st[v]["sd"] <= H_FLOOR + 1e-9).sum()), DIM[v]))
    print("wrote", a.out)


if __name__ == "__main__":
    main()

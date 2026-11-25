"""Merge the three extracted sources and remove benchmark leakage.

    python src/corpus/clean_corpus.py --workers 32            # THRESH=0.8 by default

1. concatenate pclm_pubchem, pepland_stage1, pclm_esmatlas (in that order, so a
   molecule found in several sources keeps the non-canonical-rich label) and drop
   exact duplicate SMILES across sources;
2. fingerprint every corpus molecule (ECFP4 bits, 2048) and find its most similar
   benchmark molecule (CellPPD, THPep, PepMSND, PAMPA, AmpHGT, every split, plus the five PepLand-paper
   sets PL_cCPP, PL_cSol, PL_cBinding, PL_ncCPP, PL_ncBinding; PEPLAND=0 disables);
3. drop corpus molecules whose Tanimoto to any benchmark molecule is >= THRESH
   (1.0 would be exact matches only). The benchmark SMILES are standardised the
   same way as the corpus first.

Writes $DATA/data/corpus/scored.parquet (EVERY row with bench_sim, bench_name), so a
re-run with another --thresh only re-cuts it (seconds); then clean.parquet (the kept rows), clean.json (counts) and
leak_report.csv (removed molecules per benchmark and threshold).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from extract import Workers  # noqa: E402
from paths import CORPUS_DIR, CORPUS_SRC  # noqa: E402

ORDER = ("pclm_pubchem", "pepland_stage1", "pclm_esmatlas")
THRESHOLDS = (0.6, 0.7, 0.8, 0.9, 1.0)
CHUNK = 200_000

_bench = None       # (fingerprints, names, generator), filled by init_worker in each worker
PEPLAND_TOO = os.environ.get("PEPLAND", "1") != "0"
BENCH_CACHE = os.path.join(CORPUS_DIR, "bench_fps.pkl")


def bench_fp(smiles):
    """ECFP4 bits of the standardised molecule, WITHOUT Chem.MolToSmiles: on at least
    one AmpHGT peptide the canonical-SMILES step never returns, and we only need the
    fingerprint to compare against."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from peptide_filters import _largest, _uncharge
    m = Chem.MolFromSequence(smiles[4:]) if smiles.startswith("seq:") else Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    try:
        m = _uncharge.uncharge(_largest.choose(m))
        Chem.SanitizeMol(m)
    except Exception:
        return None
    return rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048).GetFingerprint(m).ToBinary()


def build_bench_cache(workers):
    """Fingerprint every benchmark molecule once (watched workers); the scoring
    workers then just load the pickle."""
    import pickle
    import benchmarks
    seen = {}
    for name in benchmarks.selected():
        for s in benchmarks.load(name).smiles.dropna().astype(str):
            seen.setdefault(s, name)
    if PEPLAND_TOO:
        for s, name in benchmarks.pepland_items().items():
            seen.setdefault(s, name)
    pool = Workers(min(workers, 16), bench_fp)
    try:
        res = dict(pool.run(list(seen)))
    finally:
        pool.close()
    fps, names, blobs, failed = [], [], set(), []
    for s, name in seen.items():
        b = res[s]
        if b is None:
            failed.append((name, s))
        elif b not in blobs:
            blobs.add(b)
            fps.append(b)
            names.append(name)
    os.makedirs(CORPUS_DIR, exist_ok=True)
    with open(BENCH_CACHE + ".tmp", "wb") as fh:
        pickle.dump((fps, names), fh)
    os.replace(BENCH_CACHE + ".tmp", BENCH_CACHE)
    return len(fps), failed


def init_worker():
    global _bench
    import pickle
    from rdkit import DataStructs
    from rdkit.Chem import rdFingerprintGenerator
    with open(BENCH_CACHE, "rb") as fh:
        blobs, names = pickle.load(fh)
    fps = [DataStructs.ExplicitBitVect(b) for b in blobs]
    _bench = (fps, np.array(names), rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048))


def nearest_benchmark(smiles):
    """(max Tanimoto to any benchmark molecule, its benchmark name)."""
    from rdkit import Chem, DataStructs
    fps, names, gen = _bench
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return 0.0, ""
    sims = DataStructs.BulkTanimotoSimilarity(gen.GetFingerprint(m), fps)
    i = int(np.argmax(sims))
    return float(sims[i]), str(names[i])


def score_all(a, t0):
    """Merge the sources and score every molecule against the benchmarks."""
    parts = [pd.read_parquet(os.path.join(CORPUS_SRC, s + ".parquet")) for s in ORDER]
    df = pd.concat(parts, ignore_index=True)
    n_in = len(df)
    df = df.drop_duplicates(subset="smiles", ignore_index=True)
    n_cross = n_in - len(df)
    print("merged %s rows, %s cross-source duplicates dropped" %
          (format(n_in, ","), format(n_cross, ",")), flush=True)
    if a.limit:
        df = df.head(a.limit).copy()

    n_bench, failed = build_bench_cache(a.workers)
    print("benchmark molecules: %s fingerprinted, %d could not be (hang/parse) | %.1f min"
          % (format(n_bench, ","), len(failed), (time.time() - t0) / 60), flush=True)
    assert len(failed) < 0.01 * (n_bench + len(failed)), "too many benchmark molecules failed"

    pool = Workers(a.workers, nearest_benchmark, init_worker)
    sim = np.zeros(len(df), dtype=np.float32)
    who = np.empty(len(df), dtype=object)
    skipped = 0
    try:
        for lo in range(0, len(df), CHUNK):
            smi = df.smiles.iloc[lo:lo + CHUNK].tolist()
            res = {}
            for s, r in pool.run(smi):
                if r is None:
                    skipped += 1
                    r = (1.0, "unscored")       # unscorable: drop it, never keep blindly
                res[s] = r
            for k, s in enumerate(smi):
                sim[lo + k], who[lo + k] = res[s]
            # a worker that cannot even start (or a systematic error) shows up here as
            # "unscorable"; stop rather than silently drop the whole corpus
            assert skipped <= 20 + 0.0005 * (lo + len(smi)),                 "%d of %d molecules unscorable -- see the worker traceback above" % (skipped, lo + len(smi))
            print("scored %s / %s | %.1f min" % (format(min(lo + CHUNK, len(df)), ","),
                                                 format(len(df), ","), (time.time() - t0) / 60),
                  flush=True)
    finally:
        pool.close()
    df["bench_sim"], df["bench_name"] = sim, who
    meta = {"benchmark_fingerprints": n_bench, "benchmark_unfingerprinted": len(failed),
            "input_rows": n_in, "cross_source_duplicates": n_cross, "unscorable_dropped": skipped}
    return df, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--thresh", type=float, default=float(os.environ.get("THRESH", 0.8)))
    ap.add_argument("--limit", type=int, default=0, help="first N rows only (smoke test)")
    a = ap.parse_args()
    out_pq = os.path.join(CORPUS_DIR, "clean.parquet")
    scored_pq = os.path.join(CORPUS_DIR, "scored.parquet")     # every row, every score
    scored_js = os.path.join(CORPUS_DIR, "scored.json")
    t0 = time.time()
    os.makedirs(CORPUS_DIR, exist_ok=True)

    if os.path.exists(scored_pq) and os.path.exists(scored_js) and not a.limit:
        df = pd.read_parquet(scored_pq)
        with open(scored_js) as fh:
            meta = json.load(fh)
        print("re-cutting the saved scores: %s rows" % format(len(df), ","), flush=True)
    else:
        df, meta = score_all(a, t0)
        if not a.limit:
            df.to_parquet(scored_pq + ".tmp", index=False)
            os.replace(scored_pq + ".tmp", scored_pq)
            with open(scored_js, "w") as fh:
                json.dump(meta, fh, indent=1)

    rep = []
    for th in THRESHOLDS:
        hit = df[df.bench_sim >= th - 1e-6]
        for name, g in hit.groupby("bench_name"):
            rep.append((th, name, len(g), int((g.kind == "noncanonical").sum())))
    pd.DataFrame(rep, columns=["thresh", "benchmark", "removed", "of_which_noncanonical"])         .to_csv(os.path.join(CORPUS_DIR, "leak_report.csv"), index=False)

    keep = df[df.bench_sim < a.thresh - 1e-6].reset_index(drop=True)
    stats = dict(meta, scored=len(df), thresh=a.thresh,
                 removed_as_benchmark_neighbours=len(df) - len(keep), kept=len(keep))
    for src, g in keep.groupby("source"):
        stats["kept_" + src] = len(g)
    stats["kept_canonical"] = int((keep.kind == "canonical").sum())
    stats["kept_noncanonical"] = int((keep.kind == "noncanonical").sum())
    stats["noncanonical_share"] = round(stats["kept_noncanonical"] / max(len(keep), 1), 4)
    assert stats["scored"] - stats["removed_as_benchmark_neighbours"] == stats["kept"]

    keep.to_parquet(out_pq + ".tmp", index=False)
    os.replace(out_pq + ".tmp", out_pq)
    with open(os.path.join(CORPUS_DIR, "clean.json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    print(json.dumps(stats, indent=1))
    print("done in %.1f min" % ((time.time() - t0) / 60))


if __name__ == "__main__":
    main()

"""The ONE input normalisation of the student (decision of 2026-11-16): neutralise charges, then RDKit canonical SMILES.

    normalise("NC(N)=[NH2+]")  ->  "N=C(N)N"          # RDKit's Uncharger, then Chem.MolToSmiles(isomericSmiles=True, canonical=True)

Everything the student reads goes through this function: the tokenizer's learning sample, its checks, the distillation pool, the
benchmarks and any new benchmark. Not touched on purpose: salts / disconnected fragments (a ".[Cl-]" stays), stereo (kept), tautomers.
Charges that cannot be removed stay (quaternary ammonium, nitro groups, counter-ions). A SMILES RDKit cannot parse is returned
unchanged and counted. Canonical SMILES depend a little on the RDKit version: build the final student data in the same env.

    python src/student/normalise.py --bench                      # what it changes in the ten benchmarks
    python src/student/normalise.py --pool --max-molecules 500000 # ... and in (a prefix of) the pool
    python src/student/normalise.py --csv NEW.csv --col SMILES   # ... and in a new benchmark
    python src/student/normalise.py --write-pool OUT.parquet     # the whole pool: column smiles_norm, same row order (CPU, 8 workers)
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

_UN = None


def _init():
    global _UN
    from rdkit import RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    _UN = rdMolStandardize.Uncharger()


def _charged(m):
    return any(a.GetFormalCharge() != 0 for a in m.GetAtoms())


def normalise(smiles):
    """-> the normalised SMILES, or None when RDKit cannot parse the input."""
    from rdkit import Chem
    if _UN is None:
        _init()
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    try:
        m = _UN.uncharge(m)
    except Exception:
        pass
    return Chem.MolToSmiles(m, isomericSmiles=True, canonical=True)


def _one(s):
    r = normalise(s)
    return (s, False) if r is None else (r, True)


def _stats_one(s):
    from rdkit import Chem
    if _UN is None:
        _init()
    m = Chem.MolFromSmiles(s)
    if m is None:
        return (None, None, None)
    before = _charged(m)
    r = Chem.MolToSmiles(_UN.uncharge(m), isomericSmiles=True, canonical=True)
    m2 = Chem.MolFromSmiles(r)
    return (r, before, bool(m2 is not None and _charged(m2)))


def normalise_many(smiles, workers=8, pool=None):
    """-> (list of normalised SMILES, number that RDKit could not parse and were left as they are)."""
    smiles = list(smiles)
    if workers <= 1 or len(smiles) < 2000:
        out = [_one(s) for s in smiles]
    else:
        from multiprocessing import Pool
        own = pool is None
        pool = pool or Pool(workers, initializer=_init)
        try:
            out = pool.map(_one, smiles, chunksize=1000)
        finally:
            if own:
                pool.close()
                pool.join()
    return [o[0] for o in out], sum(1 for o in out if not o[1])


def _rand_k(arg):
    s, k = arg
    from rdkit import Chem
    if _UN is None:
        _init()
    m = Chem.MolFromSmiles(s)
    if m is None:
        return [s] * k, False
    return [Chem.MolToSmiles(m, isomericSmiles=True, canonical=False, doRandom=True) for _ in range(k)], True


def random_spellings(smiles, k, workers=8, pool=None):
    """RANDOM valid spellings (atom order and branch order drawn at random, stereo kept, the same molecule) of every normalised SMILES.
    -> (list of lists with k spellings per input, number that RDKit could not parse: their spellings are the string itself).
    Training-time augmentation only (decided 2026-11-22): the student still READS canonical strings at inference. RDKit's random writer cannot be
    seeded, so two calls differ: keep the files made from it (make_student_data.py) as the record."""
    smiles = list(smiles)
    args = [(s, k) for s in smiles]
    if workers <= 1 or len(smiles) < 2000:
        out = [_rand_k(x) for x in args]
    else:
        from multiprocessing import Pool
        own = pool is None
        pool = pool or Pool(workers, initializer=_init)
        try:
            out = pool.map(_rand_k, args, chunksize=500)
        finally:
            if own:
                pool.close()
                pool.join()
    return [o[0] for o in out], sum(1 for o in out if not o[1])


def randomise_fraction(smiles, frac, workers=8, seed=0):
    """A copy of `smiles` in which a random fraction `frac` of the entries is replaced by one random spelling (the tokenizer's learning sample).
    -> (new list, number replaced)."""
    import numpy as np
    smiles = list(smiles)
    n = int(round(frac * len(smiles)))
    if n <= 0:
        return smiles, 0
    pick = np.random.RandomState(seed).choice(len(smiles), size=n, replace=False)
    sp, _ = random_spellings([smiles[i] for i in pick], 1, workers)
    for i, s in zip(pick, sp):
        smiles[i] = s[0]
    return smiles, n


def change_stats(smiles, workers=8):
    """What normalisation does to a set. -> dict: n, unparsable, changed (any), charged_before, still_charged."""
    smiles = list(smiles)
    if workers <= 1 or len(smiles) < 2000:
        res = [_stats_one(s) for s in smiles]
    else:
        from multiprocessing import Pool
        with Pool(workers, initializer=_init) as p:
            res = p.map(_stats_one, smiles, chunksize=1000)
    ok = [(s, r) for s, r in zip(smiles, res) if r[0] is not None]
    return {"n": len(smiles), "unparsable": len(smiles) - len(ok), "changed": sum(s != r[0] for s, r in ok),
            "charged_before": sum(bool(r[1]) for _, r in ok), "still_charged": sum(bool(r[2]) for _, r in ok)}


def show(name, st):
    n = max(st["n"], 1)
    print("%-26s n %9s | changed %6.2f%% | charged before %6.2f%% -> after %5.2f%% | unparsable %d" % (
        name, format(st["n"], ","), 100 * st["changed"] / n, 100 * st["charged_before"] / n, 100 * st["still_charged"] / n, st["unparsable"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="the ten benchmarks")
    ap.add_argument("--pool", action="store_true", help="the pool (a prefix with --max-molecules)")
    ap.add_argument("--max-molecules", type=int, default=0)
    ap.add_argument("--csv", nargs="*", default=[])
    ap.add_argument("--col", default=None)
    ap.add_argument("--pool-file", default=None)
    ap.add_argument("--write-pool", default=None, help="write the normalised pool to this parquet (column smiles_norm, pool row order)")
    ap.add_argument("--workers", type=int, default=8, help="lab rule: at most half of the CPU")
    a = ap.parse_args()
    assert a.workers <= 32
    from paths import CORPUS_DIR
    pool_path = a.pool_file or os.path.join(CORPUS_DIR, "pool.parquet")
    if a.bench:
        import benchmarks as BM
        for b in ["CellPPD", "THPep", "AmpHGT", "PAMPA", "PepMSND", "PL_cCPP", "PL_cSol", "PL_ncCPP", "PL_cBinding", "PL_ncBinding"]:
            show(b, change_stats(sorted(set(BM.load(b).smiles.dropna().astype(str))), a.workers))
    for f in a.csv:
        import pandas as pd
        d = pd.read_csv(f, sep=None, engine="python")
        col = a.col or next(c for c in d.columns if str(c).strip().lower() in ("smiles", "canonical_smiles", "isomeric_smiles", "merge_smiles", "smi"))
        show(os.path.basename(f), change_stats(d[col].dropna().astype(str).drop_duplicates().tolist(), a.workers))
    if a.pool or a.write_pool:
        import pyarrow as pa
        import pyarrow.parquet as pq
        from multiprocessing import Pool
        tot = {"n": 0, "unparsable": 0, "changed": 0, "charged_before": 0, "still_charged": 0}
        writer = None
        with Pool(a.workers, initializer=_init) as p:
            for b in pq.ParquetFile(pool_path).iter_batches(batch_size=100_000, columns=["smiles"]):
                sm = b.column("smiles").to_pylist()
                if a.max_molecules and tot["n"] + len(sm) > a.max_molecules and not a.write_pool:
                    sm = sm[: a.max_molecules - tot["n"]]
                res = p.map(_stats_one, sm, chunksize=1000)
                for s, r in zip(sm, res):
                    tot["n"] += 1
                    if r[0] is None:
                        tot["unparsable"] += 1
                        continue
                    tot["changed"] += s != r[0]
                    tot["charged_before"] += bool(r[1])
                    tot["still_charged"] += bool(r[2])
                if a.write_pool:
                    if writer is None:
                        writer = pq.ParquetWriter(a.write_pool + ".tmp", pa.schema([("smiles_norm", pa.string())]))
                    writer.write_table(pa.table({"smiles_norm": [r[0] if r[0] is not None else s for s, r in zip(sm, res)]}))
                print("  pool: %s molecules (%d changed so far)" % (format(tot["n"], ","), tot["changed"]), flush=True)
                if a.max_molecules and tot["n"] >= a.max_molecules and not a.write_pool:
                    break
        if writer is not None:
            writer.close()
            os.replace(a.write_pool + ".tmp", a.write_pool)
            print("wrote", a.write_pool)
        show("POOL" + (" (first %d)" % a.max_molecules if a.max_molecules and not a.write_pool else ""), tot)


if __name__ == "__main__":
    main()

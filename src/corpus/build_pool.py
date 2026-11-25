"""Select the molecules to cache: the teacher / distillation pool.

    python src/corpus/build_pool.py        # writes $CORPUS_DIR/pool.parquet and pool.json

From clean.parquet: residues <= --cap (30, the PepLand range), every non-canonical
molecule, and --canonical canonical molecules spread evenly over source x residue bin
(a stratum smaller than its quota is taken whole and its unused share goes to the rest).
Rows are shuffled and numbered into shards of --shard rows, so any prefix of shards has
the full mix: views can be cached, and a teacher trained, shard by shard. Near-duplicate
removal and the final mix are cut later by subsetting the cache.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import CORPUS_DIR  # noqa: E402

EDGES = [1, 4, 9, 14, 19, 24, 30, 40, 50]
LABELS = ["2-4", "5-9", "10-14", "15-19", "20-24", "25-30", "31-40", "41-50"]


def water_fill(sizes, total):
    """Equal quota per stratum; a stratum smaller than its quota is exhausted and its
    unused share is shared out among the others."""
    sizes = np.asarray(sizes, dtype=np.int64)
    quota = np.zeros(len(sizes), dtype=np.int64)
    left = min(int(total), int(sizes.sum()))
    live = np.arange(len(sizes))
    while left > 0 and len(live):
        if left < len(live):
            quota[live[:left]] += 1
            break
        take = np.minimum(sizes[live] - quota[live], left // len(live))
        quota[live] += take
        left -= int(take.sum())
        live = live[quota[live] < sizes[live]]
    return quota


def extend(a):
    """Append the molecules ABOVE the pool's current residue cap (up to --cap) as NEW shards. The existing rows, their order and their
    shard numbers are not touched, so every cached shard (features/corpus/<view>/shard_NNN.npz) stays valid, and the teacher's
    validation rows (global row % 100 == 0) stay the same molecules. Afterwards: `cache_views.py --export` (the pool changed),
    then cache the new shards for every view. Run again with a larger --cap to add the next band (e.g. 51-100 residues)."""
    import shutil
    old = pd.read_parquet(a.out)
    n_old = len(old)
    assert old.shard.is_monotonic_increasing, "the pool is not ordered by shard"
    old_cap = int(old.residues.max())
    assert a.cap > old_cap, "the pool already holds molecules of %d residues; --cap %d adds nothing (use a larger --cap)" % (old_cap, a.cap)
    backup = a.out.replace(".parquet", "_v1.parquet")
    if not os.path.exists(backup):
        shutil.copy2(a.out, backup)
        print("backup of the old pool:", backup)
    cols = ["smiles", "residues", "kind", "cyclic", "source"]
    flt = [("residues", ">", old_cap), ("residues", "<=", a.cap)]
    try:
        df = pd.read_parquet(a.src, columns=cols, filters=flt)
    except Exception:                                      # an engine without filter support: read everything, then cut
        df = pd.read_parquet(a.src, columns=cols)
        df = df[(df.residues > old_cap) & (df.residues <= a.cap)]
    kinds = ["canonical"] + (["noncanonical"] if a.with_noncanonical else [])
    df = df[df.kind.isin(kinds)]
    df = df[~df.smiles.isin(set(old.smiles))]
    print("candidates with %d < residues <= %d (%s): %s" % (old_cap, a.cap, "+".join(kinds), format(len(df), ",")))
    assert len(df) > 0, "nothing to add"
    if a.take and a.take < len(df):
        df = df.sample(n=a.take, random_state=a.seed)
    df = df.sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    size = int(np.ceil(len(df) / a.new_shards)) if a.new_shards else a.shard
    first = int(old.shard.max()) + 1
    df["shard"] = (first + df.index // size).astype(np.int32)
    new = pd.concat([old, df.astype(old.dtypes[df.columns].to_dict())], ignore_index=True)
    assert new.smiles.is_unique, "duplicate SMILES in the extended pool"
    assert (new.smiles.values[:n_old] == old.smiles.values).all() and (new.shard.values[:n_old] == old.shard.values).all(), \
        "the old rows changed"
    new.to_parquet(a.out + ".tmp", index=False)
    os.replace(a.out + ".tmp", a.out)
    n_nc = int((new.kind == "noncanonical").sum())
    jpath = a.out.replace(".parquet", ".json")
    stats = json.load(open(jpath)) if os.path.exists(jpath) else {}
    stats.update({"rows": len(new), "noncanonical": n_nc, "canonical": len(new) - n_nc, "noncanonical_share": round(n_nc / len(new), 4),
                  "cap": a.cap, "shards": int(new.shard.max()) + 1,
                  "by_source_kind": {"%s/%s" % k: int(v) for k, v in new.groupby(["source", "kind"]).size().items()}})
    stats.setdefault("extensions", []).append({"residues_above": old_cap, "residues_up_to": a.cap, "kinds": kinds, "rows_added": len(df),
                                               "new_shards": "%d-%d" % (first, int(new.shard.max())), "rows_per_new_shard": size, "seed": a.seed})
    with open(jpath, "w") as fh:
        json.dump(stats, fh, indent=1)
    add = new.iloc[n_old:]
    print("added %s molecules in shards %d..%d (%s rows each, the last one fewer)" % (format(len(df), ","), first, int(new.shard.max()), format(size, ",")))
    print("  by source:", {k: int(v) for k, v in add.groupby("source").size().items()})
    edges = sorted({e for e in (old_cap, 35, 40, 45, 50, 60, 70, 80, 90, 100, a.cap) if old_cap <= e <= a.cap})
    print("  by residues:", {str(k): int(v) for k, v in add.groupby(pd.cut(add.residues, edges), observed=False).size().items()})
    print("pool now: %s molecules, %.1f%% non-canonical, %d shards" % (format(len(new), ","), 100 * n_nc / len(new), int(new.shard.max()) + 1))
    print("next: python src/corpus/cache_views.py --export   (the pool file changed), then cache the new shards")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=30, help="max residues")
    ap.add_argument("--canonical", type=int, default=2_000_000, help="canonical molecules to take")
    ap.add_argument("--shard", type=int, default=100_000, help="rows per shard")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--src", default=os.path.join(CORPUS_DIR, "clean.parquet"))
    ap.add_argument("--out", default=os.path.join(CORPUS_DIR, "pool.parquet"))
    ap.add_argument("--extend", action="store_true",
                    help="append every molecule above the pool's current cap and up to --cap (e.g. --extend --cap 50) as NEW shards; "
                         "existing rows and shards are untouched (see extend())")
    ap.add_argument("--new-shards", type=int, default=0,
                    help="with --extend: split the added molecules into this many shards (match the number of PepLand processes, e.g. 24); "
                         "default: --shard rows each")
    ap.add_argument("--with-noncanonical", action="store_true", help="with --extend: also add the non-canonical molecules of the band")
    ap.add_argument("--take", type=int, default=0, help="with --extend: add at most this many (0 = all)")
    a = ap.parse_args()
    if a.extend:
        assert os.path.exists(a.out), "no pool to extend: %s" % a.out
        return extend(a)
    if os.path.exists(a.out):
        print("already done:", a.out)
        return

    df = pd.read_parquet(a.src, columns=["smiles", "residues", "kind", "cyclic", "source"])
    df = df[df.residues <= a.cap]
    nc = df[df.kind == "noncanonical"]
    ca = df[df.kind == "canonical"].copy()
    ca["bin"] = pd.cut(ca.residues, EDGES, labels=LABELS)
    keys, groups = zip(*[(k, g) for k, g in ca.groupby(["source", "bin"], observed=True)])
    quota = water_fill([len(g) for g in groups], a.canonical)
    rng = np.random.default_rng(a.seed)
    picked = [g.iloc[rng.choice(len(g), size=int(q), replace=False)].drop(columns="bin")
              for g, q in zip(groups, quota)]
    print("canonical strata (source, residues): available -> taken")
    for k, g, q in zip(keys, groups, quota):
        print("  %-15s %-6s %10s -> %10s" % (k[0], k[1], format(len(g), ","), format(int(q), ",")))

    pool = pd.concat([nc] + picked, ignore_index=True)
    pool = pool.sample(frac=1.0, random_state=a.seed).reset_index(drop=True)
    pool["shard"] = (pool.index // a.shard).astype(np.int32)
    assert pool.smiles.is_unique and (pool.residues <= a.cap).all()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    pool.to_parquet(a.out + ".tmp", index=False)
    os.replace(a.out + ".tmp", a.out)
    n_nc = int((pool.kind == "noncanonical").sum())
    stats = {"rows": len(pool), "noncanonical": n_nc, "canonical": len(pool) - n_nc,
             "noncanonical_share": round(n_nc / len(pool), 4), "cap": a.cap, "shard_rows": a.shard,
             "shards": int(pool["shard"].max()) + 1, "seed": a.seed,
             "by_source_kind": {"%s/%s" % k: int(v) for k, v in pool.groupby(["source", "kind"]).size().items()}}
    with open(a.out.replace(".parquet", ".json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Rebuild the THPep splits the PeptideCLM-2 authors actually used (J. Chem. Inf. Model. 2026, 66, 8110) from their public repository.

The repository (github.com/AaronFeller/PeptideCLM-2, commit 6b9708d, 2026-07-23) ships data/THPep_main90_smiles_classes.csv (609 molecules, no split), but the
split of each of the three seeds (101, 202, 303) is released as data inside the result files of the authors' runs:
    figure_generation/results/runs_LoRA_highrank/thpep/CheMeleon/seed_<s>/chemeleon_work/fold_0/{train,val,test}.csv   (sample_id, smiles, target)
Every other model of the authors (XGBoost, ChemBERTa, PeptideCLM-2 MLM / MTR / Hybrid) was scored on the same test molecules (checked here).
Per seed: train 389, val 98, test 122 molecules (35 positives in each test set); the paper reports the mean over the three seeds.

    git clone https://github.com/AaronFeller/PeptideCLM-2.git <DIR>          (core.longpaths true on Windows)
    python -I src/build_thpep_official_splits.py <DIR>             ->  src/benchmark_splits/thpep_official_splits.csv  (id, class, s101, s202, s303 = train / val / test)

Nothing from the clone is executed; only its CSV files are read. The script stops without writing if any check fails.
"""
import itertools
import os
import sys

import pandas as pd

SEEDS = (101, 202, 303)
MODELS = ("peptideclm-2-mlm-large", "peptideclm-2-mtr-large", "peptideclm-2-hybrid-large")


def main(repo):
    full = pd.read_csv(os.path.join(repo, "data", "THPep_main90_smiles_classes.csv"))
    assert len(full) == 609 and full.id.is_unique, "THPep_main90 is not 609 unique ids"
    lab = full.set_index("id")["class"]
    base = os.path.join(repo, "figure_generation", "results", "runs_LoRA_highrank", "thpep")
    out = full[["id", "class"]].copy()
    tests = {}
    for seed in SEEDS:
        w = os.path.join(base, "CheMeleon", "seed_%d" % seed, "chemeleon_work", "fold_0")
        parts = {n: pd.read_csv(os.path.join(w, n + ".csv")) for n in ("train", "val", "test")}
        sizes = {n: len(p) for n, p in parts.items()}
        assert sizes == {"train": 389, "val": 98, "test": 122}, (seed, sizes)
        ids = {n: set(p.sample_id) for n, p in parts.items()}
        assert all(not (ids[a] & ids[b]) for a, b in itertools.combinations(ids, 2)), "overlap between train / val / test, seed %d" % seed
        assert set().union(*ids.values()) == set(full.id), "seed %d does not cover the 609 molecules exactly once" % seed
        for n, p in parts.items():
            assert (lab.loc[p.sample_id].to_numpy() == p.target.to_numpy()).all(), "label mismatch, seed %d %s" % (seed, n)
        assert int(parts["test"].target.sum()) == 35
        # the authors' other models were scored on the very same test molecules
        xs = pd.read_csv(os.path.join(base, "xgboost-morgan", "seed_%d" % seed, "predictions.csv"))
        assert set(xs.sample_id) == ids["test"], "xgboost test set differs, seed %d" % seed
        test_smiles = set(full.set_index("id").loc[sorted(ids["test"]), "smiles"])
        for m in MODELS:
            r = pd.read_csv(os.path.join(base, m, "seed_%d" % seed, "THPep_%s_results.csv" % m))
            assert len(r) == 122 and set(r.smiles) == test_smiles, "%s test set differs, seed %d" % (m, seed)
        col = {}
        for n in ("train", "val", "test"):
            col.update({i: n for i in ids[n]})
        out["s%d" % seed] = out.id.map(col)
        tests[seed] = ids["test"]
        print("seed %d: train %d / val %d / test %d (35 positives), disjoint, covers all 609, labels agree, same test set in 4 other models' files (XGBoost + PeptideCLM-2 MLM / MTR / Hybrid)" % (seed, *sizes.values()))
    print("test-set overlaps:", {"%d&%d" % (a, b): len(tests[a] & tests[b]) for a, b in itertools.combinations(SEEDS, 2)})
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_splits", "thpep_official_splits.csv")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    out.to_csv(dest, index=False, lineterminator="\n")
    print("written:", dest, "(%d rows)" % len(out))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])

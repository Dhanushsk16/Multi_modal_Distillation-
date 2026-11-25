"""Prove the carbonyl pre-filter is safe before trusting it on 108M PubChem rows.

    python src/corpus/prefilter_check.py --rows 400000 --workers 32

Two independent checks, because they fail in different ways:

  benchmarks  every molecule in our five labelled sets, classified. A real
              peptide the filter drops is a silent hole in the corpus, so the
              count must be 0 -- per benchmark, since an earlier amide-pattern
              filter looked fine overall while quietly dropping 18% of PAMPA.
  pubchem     a uniform sample of the source itself: recall over the peptides
              extract.py would keep (must be 1.0), and how much of the file the
              filter rejects, which is the only reason to have it.

Writes $DATA/results/corpus_inventory/prefilter_check.json. run_extract.sh reads
the "gate" block from it and refuses to start PubChem unless it is clean.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import benchmarks  # noqa: E402
from extract import MIN_CARBONYLS, SOURCES, carbonyls, peptidic  # noqa: E402
from peptide_filters import classify  # noqa: E402
from paths import RESULTS_DIR, SOURCES_DIR  # noqa: E402

OUT = os.path.join(RESULTS_DIR, "corpus_inventory")
MIN_RES = SOURCES["pclm_pubchem"]["min_res"]        # the strict case: 4


def sample_pubchem(rows, n_parts=24, seed=0):
    import pyarrow.parquet as pq
    parts = sorted(glob.glob(os.path.join(SOURCES_DIR, "pclm_pubchem", "part_*.parquet")))
    assert parts, "no PubChem parts in " + SOURCES_DIR
    pick = np.linspace(0, len(parts) - 1, min(n_parts, len(parts))).astype(int)
    s = pd.concat([pq.read_table(parts[int(i)], columns=["smiles"]).to_pandas().smiles
                   for i in pick], ignore_index=True).dropna()
    return s.sample(min(rows, len(s)), random_state=seed).tolist()


def kept_mask(res, min_res):
    """What extract.py would keep, duplicates aside."""
    return np.array([r["std"] is not None and r["kind"] != "not_peptide"
                     and r["residues"] >= min_res for r in res])


def check_benchmarks(workers, thresholds):
    out, worst = {}, 0
    for name in benchmarks.MAIN + benchmarks.STRESS:
        smiles = sorted(set(benchmarks.load(name).smiles.dropna().astype(str)))
        with Pool(workers) as pool:
            res = pool.map(classify, smiles, chunksize=50)
        keep = kept_mask(res, MIN_RES)
        row = {"molecules": len(smiles), "peptides_ge_%d_res" % MIN_RES: int(keep.sum())}
        for t in thresholds:
            miss = [s for s, k in zip(smiles, keep) if k and not peptidic(s, t)]
            row["dropped_at_%d" % t] = len(miss)
            row["frac_dropped_at_%d" % t] = round(len(miss) / max(1, int(keep.sum())), 4)
            if t == MIN_CARBONYLS[MIN_RES]:
                worst = max(worst, len(miss))
                row["examples"] = miss[:3]
        out[name] = row
        print("%-10s %s molecules, %s peptides, dropped at %d: %d"
              % (name, format(len(smiles), ","), format(int(keep.sum()), ","),
                 MIN_CARBONYLS[MIN_RES], row["dropped_at_%d" % MIN_CARBONYLS[MIN_RES]]),
              flush=True)
    return out, worst


def check_pubchem(rows, workers, thresholds):
    t0 = time.time()
    smiles = sample_pubchem(rows)
    with Pool(workers) as pool:
        res = pool.map(classify, smiles, chunksize=200)
    keep = kept_mask(res, MIN_RES)
    print("pubchem: %s rows, %s would be kept (%.1f min)"
          % (format(len(smiles), ","), format(int(keep.sum()), ","),
             (time.time() - t0) / 60), flush=True)
    out = {"rows": len(smiles), "min_res": MIN_RES, "kept_by_classify": int(keep.sum()),
           "carbonyls_median_of_kept": float(np.median(
               [carbonyls(s) for s, k in zip(smiles, keep) if k])) if keep.any() else None,
           "thresholds": {}}
    for t in thresholds:
        passes = np.array([peptidic(s, t) for s in smiles])
        missed = [s for s, p, k in zip(smiles, passes, keep) if k and not p]
        out["thresholds"][str(t)] = {
            "rejects_frac_of_file": round(1 - float(passes.mean()), 4),
            "recall_of_kept": round(float((passes & keep).sum() / max(1, keep.sum())), 5),
            "missed": len(missed),
            "missed_examples": missed[:3],
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=400_000)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--thresholds", type=int, nargs="*", default=[2, 3, 4])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    t = MIN_CARBONYLS[MIN_RES]
    assert t in a.thresholds, "the threshold in use (%d) must be among those checked" % t

    bench, dropped = check_benchmarks(a.workers, a.thresholds)
    pub = check_pubchem(a.rows, a.workers, a.thresholds)
    res = {"threshold_in_use": t, "benchmarks": bench, "pubchem": pub,
           "gate": {"threshold": t,
                    "benchmark_peptides_dropped": dropped,
                    "pubchem_recall": pub["thresholds"][str(t)]["recall_of_kept"],
                    "rejects_frac_of_file": pub["thresholds"][str(t)]["rejects_frac_of_file"]}}
    with open(os.path.join(OUT, "prefilter_check.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps(res["gate"], indent=1))


if __name__ == "__main__":
    main()

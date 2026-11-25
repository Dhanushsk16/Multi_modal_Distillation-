"""THPep with the journal PeptideCLM-2 paper's protocol: summary of the runs of scripts/run_thpep_cv.sh.

    python src/teacher/thpep_cv_summary.py --prefix thpep_cv --teacher v3_base

Protocol (J. Chem. Inf. Model. 2026, 66, 8110, Methods "Downstream Evaluation Protocols"): THPep had no shipped split, so "random 5-fold cross
validation", all models "trained in triplicate", "mean +- standard deviation across independent seeds". Here: ALL 609 THPep molecules, three
repetitions (benchmarks THPep_cv0 / _cv1 / _cv2 = three different random fold assignments, benchmarks.py); every molecule is tested once per
repetition; one MCC per repetition on the pooled out-of-fold predictions (learners with seeds: averaged over their seeds first); the table gives
the mean +- sd (n - 1) over the three repetitions. Reads results/<prefix>_<learner>/raw.csv (the harness) and results/<prefix>_ft/raw.csv
(finetune_v3.py, mode ft).
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks import THPEP_CV  # noqa: E402
from paths import RESULTS_DIR  # noqa: E402

NAMES = {"H": "ECFP counts", "L": "PeptideCLM-2 (frozen)", "P": "PepLand (frozen)", "P+L+H": "the three joined", "PCA": "PCA of the three, 1024-d"}
# the journal paper's own THPep numbers (Table 1, MCC, mean +- sd over its seeds); the prior published method is quoted without a sd
PUBLISHED = [("PeptideCLM-2 MLM, fine-tuned 337M", 0.756, 0.019), ("PeptideCLM-2 Hybrid, fine-tuned 337M", 0.747, 0.036),
             ("PeptideCLM-2 MTR, fine-tuned 337M", 0.698, 0.036), ("THPep's own method (prior published)", 0.710, None),
             ("RDKit99 descriptors (their baseline)", 0.673, 0.092), ("ChemBERTa-2 MTR (their baseline)", 0.618, 0.118),
             ("CheMeleon (their baseline)", 0.600, 0.062), ("Morgan fingerprint (their baseline)", 0.597, 0.040)]


def ms(x):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    return (float(x.mean()), float(x.std(ddof=1)) if len(x) > 1 else float("nan"), len(x))


def fmt(m, s, n):
    return "%.3f +- %.3f%s" % (m, s, "" if n == len(THPEP_CV) else "  (%d of %d repetitions)" % (n, len(THPEP_CV)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="thpep_cv")
    ap.add_argument("--teacher", default="v3_base")
    ap.add_argument("--learners", nargs="*", default=["linear", "gbt", "mlp", "kernel"])
    a = ap.parse_args()
    tname = "teacher_" + a.teacher
    scores = {}                                              # (input, learner) -> MCC per repetition
    for l in a.learners:
        f = os.path.join(RESULTS_DIR, "%s_%s" % (a.prefix, l), "raw.csv")
        if not os.path.exists(f):
            print("no %s: skipped" % f)
            continue
        r = pd.read_csv(f)
        r = r[(r.subset == "all") & (r.rowset == "all") & (r.learner == l) & r.benchmark.isin(THPEP_CV) & (r.rep != "REF")]
        per = r.groupby(["rep", "benchmark"]).MCC.mean().unstack()        # learner seeds averaged: one score per repetition
        for rep in per.index:
            scores[(rep, l)] = per.loc[rep].to_numpy()
    ft = os.path.join(RESULTS_DIR, "%s_ft" % a.prefix, "raw.csv")
    if os.path.exists(ft):
        r = pd.read_csv(ft)
        r = r[r["mode"] == "ft"]
        col = "pooled_MCC" if "pooled_MCC" in r else "MCC"
        scores[(tname, "fine-tuned")] = r.groupby("benchmark")[col].mean().reindex(THPEP_CV).to_numpy()
    if not scores:
        print("nothing to summarise")
        return
    inputs = [i for i in list(NAMES) + [tname] if any(k[0] == i for k in scores)]
    learners = [l for l in a.learners + ["fine-tuned"] if any(k[1] == l for k in scores)]
    print("THPep, random 5-fold cross-validation over ALL 609 molecules, 3 repetitions: MCC, mean +- sd over the repetitions\n")
    rows = {}
    for i in inputs:
        label = NAMES.get(i, "%s (frozen)" % a.teacher if i == tname else i)
        rows[label] = {l: fmt(*ms(scores[(i, l)])) if (i, l) in scores else "-" for l in learners}
    print(pd.DataFrame(rows).T[learners].to_string())
    print("\nbest learner per input (mean over the repetitions) and the mean over all learners:")
    for i in inputs:
        m = {l: ms(scores[(i, l)])[0] for l in learners if (i, l) in scores}
        frozen = [v for l, v in m.items() if l != "fine-tuned"]
        best = max(m, key=m.get)
        print("  %-28s best %-10s %.3f | mean of the frozen-probe learners %.3f" % (NAMES.get(i, "%s (frozen)" % a.teacher if i == tname else i), best, m[best],
                                                                           np.mean(frozen) if frozen else float("nan")))
    print("\nthe journal paper's own THPep numbers (same protocol, 3 seeds), for comparison:")
    for n, m, s in PUBLISHED:
        print("  %-40s %.3f%s" % (n, m, "" if s is None else " +- %.3f" % s))
    print("\nrepetition-to-repetition spread of the same model is the sd above; differences smaller than about 2 sd are not resolved with 3 repetitions.")


if __name__ == "__main__":
    main()

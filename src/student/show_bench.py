"""One table: the teacher and every benchmarked student side by side, benchmark by benchmark (read-only, CPU, a second).

    python src/student/show_bench.py                      # every student that bench_watch.sh has finished (results/harness_<name>/DONE)
    python src/student/show_bench.py st3 stc             # only these
    python src/student/show_bench.py --bench CellPPD THPep      # only some benchmarks
    python src/student/show_bench.py --inputs --pre             # also PepLand / PeptideCLM-2 / ECFP alone, and the 1280-d vector of each student
    python src/student/show_bench.py --learner gbt              # another learner (default linear; ALL = mean over the learners the harness ran)

Reads results/harness_<name>/per_bench.csv (written by eval_harness.py; the score of every representation on every benchmark, the benchmark's own
primary metric, mean over learner seeds) and writes the table to results/student_benchmark_table.csv. Frozen-probe numbers of OUR protocol: they are not
the published numbers of the papers and must be labelled so.
"""
from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from paths import RESULTS_DIR  # noqa: E402

METRIC = {"CellPPD": "MCC", "THPep": "MCC", "PepMSND": "MCC", "PAMPA": "R2", "AmpHGT": "MCC", "PL_cCPP": "AUROC", "PL_cSol": "AUROC",
          "PL_ncCPP": "Spearman", "PL_cBinding": "Spearman", "PL_ncBinding": "Spearman"}
# The BEST PUBLISHED number per benchmark = the strongest number in the source paper's results table, whichever row (checked 2026-11-24): PeptideCLM-2 (J. Chem.
# Inf. Model. 2026, 66, 8110, Table 1, fine-tuned 337M; MCC at the best test threshold) for THPep 0.756, PepMSND 0.749 (MTR, with descriptor-KAN fusion) and AmpHGT 0.884;
# CellPPD 0.892 is CheMeleon's row of the same table (PeptideCLM-2 itself 0.875); PepLand (Briefings in Bioinformatics 2025, Table 1, splits not stated): PL_cCPP 0.885 and
# PL_cSol 0.730 are the PepLand + ESM2 rows (PepLand alone 0.838 / 0.662), PL_ncCPP 0.628. PAMPA is the pooled R2 (about 0.58). The two PL_ binding tasks used the protein
# partner: not comparable with a peptide-only probe. (Until 2026-11-24 this table held the superseded preprint numbers.)
PUBLISHED = {"CellPPD": 0.892, "THPep": 0.756, "PepMSND": 0.749, "PAMPA": 0.58, "AmpHGT": 0.884, "PL_cCPP": 0.885, "PL_cSol": 0.730, "PL_ncCPP": 0.628,
             "PL_cBinding": 0.503, "PL_ncBinding": 0.768}
LADDER = ["st1", "st2", "st3", "st4", "st5", "stc", "st6", "st7", "stca"]
INPUTS = [("P", "PepLand"), ("L", "PeptideCLM-2"), ("H", "ECFP counts")]


def finished():
    names = [d[len("harness_"):] for d in sorted(os.listdir(RESULTS_DIR))
             if d.startswith("harness_") and os.path.exists(os.path.join(RESULTS_DIR, d, "DONE"))]
    return sorted(names, key=lambda n: (LADDER.index(n) if n in LADDER else 99, n))


def scores(name, learner, rowset):
    """-> DataFrame [rep x benchmark] of the primary metric from results/harness_<name>/per_bench.csv"""
    f = os.path.join(RESULTS_DIR, "harness_%s" % name, "per_bench.csv")
    if not os.path.exists(f):
        return None
    df = pd.read_csv(f)
    df = df[df.rowset == rowset]
    if learner != "ALL":
        df = df[df.learner == learner]
    if df.empty:
        return None
    return df.groupby(["rep", "benchmark"]).primary.mean().unstack("benchmark")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", help="students (default: every one with a DONE file)")
    ap.add_argument("--teacher", default="v3_base")
    ap.add_argument("--learner", default="linear")
    ap.add_argument("--rowset", default="all", choices=["all", "has_E"])
    ap.add_argument("--bench", nargs="*", default=None)
    ap.add_argument("--pre", action="store_true", help="also the 1280-d pooled vector (<name>_pre) of each student")
    ap.add_argument("--inputs", action="store_true", help="also PepLand, PeptideCLM-2 and ECFP alone")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "student_benchmark_table.csv"))
    a = ap.parse_args()
    names = a.names or finished()
    if not names:
        sys.exit("no benchmarked student yet (no results/harness_*/DONE): bench_watch.sh writes them")
    cols, first = {}, None
    for n in names:
        t = scores(n, a.learner, a.rowset)
        if t is None:
            print("no results for %s with learner %s (results/harness_%s/per_bench.csv)" % (n, a.learner, n))
            continue
        if first is None:
            first = t                                                 # the teacher and the three inputs are the same in every harness run
            if "teacher_" + a.teacher in t.index:
                cols["teacher " + a.teacher] = t.loc["teacher_" + a.teacher]
            if a.inputs:
                for key, label in INPUTS:
                    if key in t.index:
                        cols[label + " alone"] = t.loc[key]
        for suffix, label in (("", n), ("_pre", n + "_pre")):
            if suffix and not a.pre:
                continue
            if "teacher_" + n + suffix in t.index:
                cols[label] = t.loc["teacher_" + n + suffix]
    tab = pd.DataFrame(cols)
    order = [b for b in METRIC if b in tab.index] + [b for b in tab.index if b not in METRIC]
    tab = tab.loc[order]
    if a.bench:
        tab = tab.loc[[b for b in a.bench if b in tab.index]]
    main_rows = [b for b in tab.index if b in METRIC]
    avg_key = "AVERAGE (all %d)" % len(tab)
    summary = {avg_key: tab.mean()}
    if "PAMPA" in tab.index and len(tab) > 1:
        summary["AVERAGE without PAMPA"] = tab.drop(index="PAMPA").mean()
    if a.bench is None and main_rows:
        summary["AVERAGE of the ten main"] = tab.loc[main_rows].mean()
    seven = [b for b in main_rows if b not in ("PAMPA", "PL_cBinding", "PL_ncBinding")]
    if a.bench is None and len(seven) == 7:                           # the seven where the R2 scale and the protein-partner tasks do not interfere
        summary["AVERAGE of 7 (no PAMPA, no binding)"] = tab.loc[seven].mean()
    shown = pd.concat([tab, pd.DataFrame(summary).T])
    tcol = "teacher " + a.teacher
    if tcol in shown.columns:                                         # how far each student's average is from the teacher's
        shown.loc["student - teacher (AVERAGE of all)"] = pd.Series({n: shown.loc[avg_key, n] - shown.loc[avg_key, tcol] for n in names if n in shown.columns})
    pub = pd.Series({b: v for b, v in PUBLISHED.items() if b in tab.index})
    if len(pub):                                                      # how many benchmarks each column is above the published number
        shown.loc["ABOVE the published number (of %d)" % len(pub)] = pd.Series({c: float((tab.loc[pub.index, c] > pub).sum()) for c in tab.columns})
        shown.insert(0, "published", pub)
        subsets = {avg_key: list(pub.index), "AVERAGE without PAMPA": [b for b in pub.index if b != "PAMPA"],
                   "AVERAGE of the ten main": [b for b in pub.index if b in METRIC], "AVERAGE of 7 (no PAMPA, no binding)": seven}
        for k, members in subsets.items():                            # the published numbers averaged over the same benchmarks (a crude line: the metrics differ)
            if k in shown.index and members:
                shown.loc[k, "published"] = pub.loc[[m for m in members if m in pub.index]].mean()
    shown.insert(0, "metric", [METRIC.get(i, "") for i in shown.index])
    print("frozen probe, OUR protocol (learner: %s, rows: %s); primary metric per benchmark (higher is better)\n" % (a.learner, a.rowset))
    print(shown.round(3).fillna("").to_string())
    print("\n'published' = the papers' own numbers, NOT like-for-like: PeptideCLM-2's (CellPPD, THPep, PepMSND, PAMPA, AmpHGT) are FINE-TUNED 337M models (PepMSND with a descriptor-KAN"
          "\nfusion; PAMPA pooled R2); PepLand's (PL_*) have no split stated and our probe on them is easier; PL_cBinding / PL_ncBinding published use the PROTEIN PARTNER (ours: peptide only).")
    shown.to_csv(a.out)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()

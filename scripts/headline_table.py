#!/usr/bin/env python
"""Recompute the headline table of the README from the stored result files.

    python scripts/headline_table.py                  # best of the four learners per benchmark (as in the README), and every learner on its own
    python scripts/headline_table.py --results PATH   # another harness folder

Reads results/harness_st6f_e02_all/per_bench.csv (written by src/teacher/eval_harness.py): the primary metric of every representation on every
benchmark, for each of the four learners (linear, kernel, gbt, mlp). THPep is taken from results/thpep_official_all/thpep_official_summary.csv: the mean
over the three splits that the PeptideCLM-2 authors used (rebuilt by src/build_thpep_official_splits.py); the THPep score on a split made here, which is
in per_bench.csv, is printed for comparison and not used (--own-thpep uses it instead). "teacher_v3_base" is the main teacher, "teacher_st6f_e02" the embedding of the
main student (the weights of epoch 2, which are best.pt). Frozen embeddings, our protocol; the published numbers are those of the papers
(fine-tuned models; REFERENCES.md) and are not directly comparable.

The README table takes the best of the four learners per benchmark, chosen on the test scores: that is optimistic. The second table gives each
learner on its own, which involves no selection, so the effect of the choice can be read off.
"""
import argparse
import os

import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SEVEN = ["CellPPD", "THPep", "PepMSND", "AmpHGT", "PL_cCPP", "PL_cSol", "PL_ncCPP"]
METRIC = {"CellPPD": "MCC", "THPep": "MCC", "PepMSND": "MCC", "AmpHGT": "MCC", "PL_cCPP": "AUROC", "PL_cSol": "AUROC", "PL_ncCPP": "Spearman"}
# best published number = the strongest in the source paper's results table, whichever row (see REFERENCES.md)
PUBLISHED = {"CellPPD": 0.892, "THPep": 0.756, "PepMSND": 0.749, "AmpHGT": 0.884, "PL_cCPP": 0.885, "PL_cSol": 0.730, "PL_ncCPP": 0.628}
SOURCE = {"CellPPD": "CheMeleon row of the PeptideCLM-2 paper's Table 1 (PeptideCLM-2 itself 0.875)", "THPep": "PeptideCLM-2 MLM, Table 1", "PepMSND": "PeptideCLM-2 MTR with KAN, Table 1",
          "AmpHGT": "PeptideCLM-2 MLM, Table 1", "PL_cCPP": "PepLand + ESM-2, PepLand paper Table 1 (PepLand alone 0.838)", "PL_cSol": "PepLand + ESM-2, Table 1 (PepLand alone 0.662)", "PL_ncCPP": "PepLand, Table 1"}
LEARNERS = ["linear", "kernel", "gbt", "mlp"]

ap = argparse.ArgumentParser()
ap.add_argument("--results", default=os.path.join(ROOT, "results", "harness_st6f_e02_all"))
ap.add_argument("--own-thpep", action="store_true", help="use the THPep split made here instead of the authors' three splits")
a = ap.parse_args()
d = pd.read_csv(os.path.join(a.results, "per_bench.csv"))
d = d[d.rowset == "all"]
tab = {name: d[d.rep == rep].pivot(index="benchmark", columns="learner", values="primary")[LEARNERS] for name, rep in (("teacher", "teacher_v3_base"), ("student", "teacher_st6f_e02"))}

own = {who: tab[who].loc["THPep"].copy() for who in tab}
OFFICIAL = os.path.join(ROOT, "results", "thpep_official_all", "thpep_official_summary.csv")
note = "THPep: the split made here (--own-thpep)"
if os.path.exists(OFFICIAL) and not a.own_thpep:
    S = pd.read_csv(OFFICIAL)
    for who, rep in (("teacher", "teacher_v3_base"), ("student", "teacher_st6f_e02")):
        g = S[S.rep == rep].set_index("learner")["mean"]
        tab[who].loc["THPep"] = [float(g[l]) for l in LEARNERS]
    note = "THPep: mean over the authors' three splits (on the split made here the best learner scores: teacher %.3f, student %.3f)" % (own["teacher"].max(), own["student"].max())
print("Best learner per benchmark (as in the README; the choice is made on the test scores)\n")
print("%-9s %-9s %9s | %8s %-12s | %8s %-12s | source of the published number" % ("benchmark", "metric", "published", "teacher", "(learner)", "student", "(learner)"))
above = {"teacher": 0, "student": 0}
for b in SEVEN:
    cells = []
    for who in ("teacher", "student"):
        row = tab[who].loc[b]
        top = row.max()
        cells.append("%8.3f %-12s" % (top, "(%s)" % row.idxmax()))
        above[who] += round(float(top), 3) > PUBLISHED[b]
    print("%-9s %-9s %9.3f | %s | %s | %s" % (b, METRIC[b], PUBLISHED[b], cells[0], cells[1], SOURCE[b]))
print("\nabove the best published score: teacher %d of 7, student %d of 7" % (above["teacher"], above["student"]))
print(note + "\n")

print("Each learner on its own (no selection)\n")
print("%-9s %9s | %s" % ("benchmark", "published", " | ".join("%-17s" % ("student " + l) for l in LEARNERS)))
n = {l: 0 for l in LEARNERS}
for b in SEVEN:
    vals = [float(tab["student"].loc[b, l]) for l in LEARNERS]
    for l, v in zip(LEARNERS, vals):
        n[l] += round(v, 3) > PUBLISHED[b]
    print("%-9s %9.3f | %s" % (b, PUBLISHED[b], " | ".join("%17.3f" % v for v in vals)))
print("%-9s %9s | %s" % ("above", "of 7", " | ".join("%17d" % n[l] for l in LEARNERS)))

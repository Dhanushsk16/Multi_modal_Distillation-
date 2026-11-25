"""Summary of PepLand's two binding tasks WITH the protein partner (run_binding_prot.sh).

    python src/teacher/binding_prot_summary.py --prefix binding_prot --teacher v3_base --learners linear gbt mlp kernel

Reads results/<prefix>_<learner>/ (harness: one table per benchmark, rows = representation, columns = learner) and results/<prefix>_ft/
(fine-tuning), prints per benchmark the Spearman correlation (the published metric; pooled over the five folds) and Pearson for nc-binding,
next to the published numbers and next to the SAME teacher without the protein (the earlier peptide-only run, results/final_<learner>).
Benchmarks (benchmarks.py): PL_cBindingProt / PL_ncBindingProt (folds as the peptide-only tasks: rows with the same peptide share a fold)
and ...ProtUnseen (folds by protein: the protein of a test pair is never in the training data). PROT = the protein vector alone.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import RESULTS_DIR  # noqa: E402

PUB = {"c": {"Spearman": 0.503, "Pearson": 0.45, "note": "PepLand (ESM2 15B protein features); ESM2 alone 0.452"},
       "nc": {"Spearman": 0.768, "Pearson": 0.873, "note": "PepLand (ESM2 15B protein features)"}}
BENCH = {"c": ("PL_cBindingProt", "PL_cBindingProtUnseen", "PL_cBinding"), "nc": ("PL_ncBindingProt", "PL_ncBindingProtUnseen", "PL_ncBinding")}
NAME = {"PROT": "protein alone", "P": "PepLand", "L": "PeptideCLM-2", "H": "ECFP", "P+L+H": "joined three", "PCA": "PCA"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="binding_prot")
    ap.add_argument("--teacher", default="v3_base")
    ap.add_argument("--learners", nargs="*", default=["linear", "gbt", "mlp", "kernel"])
    a = ap.parse_args()
    T = "teacher_" + a.teacher
    raws = {}
    for L in a.learners:
        p = os.path.join(RESULTS_DIR, "%s_%s" % (a.prefix, L), "raw.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            raws[L] = d[(d.rowset == "all") & (d.subset == "all") & (d.rep != "REF")]
        else:
            print("(no %s)" % p)
    old = {}
    for L in a.learners:                                                   # the same teacher without the protein, earlier run
        p = os.path.join(RESULTS_DIR, "final_%s" % L, "raw.csv")
        if os.path.exists(p):
            d = pd.read_csv(p)
            old[L] = d[(d.rowset == "all") & (d.subset == "all") & (d.rep == T)]
    ft = None
    p = os.path.join(RESULTS_DIR, "%s_ft" % a.prefix, "raw.csv")
    if os.path.exists(p):
        ft = pd.read_csv(p)
    for kind in ("c", "nc"):
        for bench, label in ((BENCH[kind][0], "folds as the peptide-only task"), (BENCH[kind][1], "folds by PROTEIN (unseen proteins in test)")):
            print("\n" + "=" * 100)
            print("%s | %s | Spearman (pooled over the 5 folds; mean over seeds) | published: Spearman %.3f, Pearson %.3f (%s)"
                  % (bench, label, PUB[kind]["Spearman"], PUB[kind]["Pearson"], PUB[kind]["note"]))
            rows = {}
            for L, d in raws.items():
                x = d[d.benchmark == bench].groupby("rep").Spearman.mean()
                for rep, v in x.items():
                    rows.setdefault(rep, {})[L] = v
            if ft is not None and (ft.benchmark == bench).any():
                col = "pooled_Spearman" if "pooled_Spearman" in ft else "Spearman"
                rows.setdefault(T, {})["fine-tuned"] = float(ft[ft.benchmark == bench][col].mean())
            if not rows:
                print("   (nothing yet)")
                continue
            tab = pd.DataFrame(rows).T
            order = [r for r in ("PROT", "P", "L", "H", "P+L+H", "PCA", T) if r in tab.index]
            tab = tab.loc[order]
            tab.index = [NAME.get(i, "teacher " + a.teacher if i == T else i) for i in tab.index]
            tab["best"] = tab.max(axis=1)
            print(tab.round(3).to_string())
            pe = {L: float(old[L][old[L].benchmark == BENCH[kind][2]].Spearman.mean()) for L in old if (old[L].benchmark == BENCH[kind][2]).any()}
            if pe and bench == BENCH[kind][0]:
                print("   same teacher WITHOUT the protein (earlier peptide-only run, same folds):", {k: round(v, 3) for k, v in pe.items()})
            if bench in BENCH[kind][:2]:
                pr = {L: float(d[(d.benchmark == bench) & (d.rep == T)].Pearson.mean()) for L, d in raws.items() if ((d.benchmark == bench) & (d.rep == T)).any()}
                if pr:
                    print("   Pearson, teacher %s:" % a.teacher, {k: round(v, 3) for k, v in pr.items()})


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""THPep on the authors' three splits: mean +- sd of the test MCC over THPep_o101 / _o202 / _o303.

    python ~/code/peptidefusion/src/teacher/thpep_official_summary.py [--out thpep_official_all]

Part 1, FROZEN embeddings read by sklearn learners: results/<out>/raw.csv (eval_harness.py), every representation and learner. The learner seeds (gbt, mlp) are
averaged first, then the three splits.
Part 2, FINE-TUNING (when present): results/thpep_official_ft_teacher and results/thpep_official_ft_student (finetune_v3.py / finetune_student.py).
    ft      the model trained end to end with the labels (low learning rate for its own weights) + a new head
    frozen  the model's weights fixed, the SAME head trained on its fixed output (the paired control of ft)
  One model per run: fitted on the authors' 389 train rows and stopped early on their 98 validation rows; the 122 test molecules are never used for any choice.
  The fine-tuning seeds are averaged first, then the three splits.
MCC is taken at each learner's own fixed decision rule (probability 0.5; kernel SVM: decision value 0), never at a threshold picked on the test set.
For the paper's numbers see the printed note: its PeptideCLM-2 MLM 0.756 is the MCC at the BEST threshold on the test set (recomputed from the authors' own
prediction files: 0.742 / 0.778 / 0.747); at the usual logit threshold 0 the same files give 0.621 / 0.732 / 0.725 (mean 0.693).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from benchmarks import THPEP_OFF  # noqa: E402
from paths import RESULTS_DIR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="thpep_official_all")
a = ap.parse_args()
pd.set_option("display.width", 220)


def over_splits(v):
    return np.nanmean(v), (np.nanstd(v, ddof=1) if np.isfinite(v).sum() > 1 else np.nan)


# ------------------------------------------------------------------ part 1: frozen embeddings + sklearn learners
r = pd.read_csv(os.path.join(RESULTS_DIR, a.out, "raw.csv"))
r = r[(r.subset == "all") & (r.rowset == "all") & r.benchmark.isin(THPEP_OFF) & (r.rep != "REF")]
if r.empty:
    sys.exit("no THPep_o* rows in %s/raw.csv" % a.out)
per = r.groupby(["rep", "learner", "benchmark"]).MCC.mean().reset_index()          # mean over the learner seeds, one value per split
rows = []
for (rep, lr), g in per.groupby(["rep", "learner"]):
    v = g.set_index("benchmark").MCC.reindex(THPEP_OFF).to_numpy()
    m, s = over_splits(v)
    rows.append(dict(rep=rep, learner=lr, o101=v[0], o202=v[1], o303=v[2], mean=m, sd=s, n_splits=int(np.isfinite(v).sum())))
t = pd.DataFrame(rows)
print("=== PART 1. FROZEN embeddings + sklearn learners: test MCC on the authors' three splits (122 test molecules each) ===")
print(t.round(3).to_string(index=False))
print("\n--- best learner per representation (by the 3-split mean; chosen on the test scores, so optimistic) ---")
best = t.loc[t.groupby("rep")["mean"].idxmax()][["rep", "learner", "mean", "sd"]]
print(best.round(3).to_string(index=False))
t.to_csv(os.path.join(RESULTS_DIR, a.out, "thpep_official_summary.csv"), index=False)

# ------------------------------------------------------------------ part 2: fine-tuning
FT = (("thpep_official_ft_teacher", "teacher v3_base", "teacher_v3_base"), ("thpep_official_ft_student", "student st6f_e02", "teacher_st6f_e02"))
ftrows, found = [], False
for tag, label, rep in FT:
    p = os.path.join(RESULTS_DIR, tag, "raw.csv")
    if not os.path.isfile(p):
        print("\n(no fine-tuning results yet in results/%s)" % tag)
        continue
    f = pd.read_csv(p)
    f = f[f.benchmark.isin(THPEP_OFF)]
    if f.empty:
        continue
    found = True
    for mode, g in f.groupby("mode"):
        v = g.groupby("benchmark").MCC.mean().reindex(THPEP_OFF).to_numpy()
        m, s = over_splits(v)
        bl = best[best.rep == rep]
        ftrows.append(dict(model=label, mode=mode, o101=v[0], o202=v[1], o303=v[2], mean=m, sd=s, seeds_per_split=int(g.groupby("benchmark").seed.nunique().min()),
                           best_epoch=g.best_epoch.mean(), best_frozen_learner_mean=(float(bl["mean"].iloc[0]) if len(bl) else np.nan)))
if found:
    ft = pd.DataFrame(ftrows)
    ft["minus_best_frozen_learner"] = ft["mean"] - ft["best_frozen_learner_mean"]
    print("\n=== PART 2. FINE-TUNING: test MCC on the authors' three splits (mean over the fine-tuning seeds, then over the splits) ===")
    print("    ft = trained end to end with the labels | frozen = weights fixed, the same new head | last column: against the best sklearn learner of part 1 on the same embedding")
    print(ft.round(3).to_string(index=False))
    ft.to_csv(os.path.join(RESULTS_DIR, a.out, "thpep_official_finetune_summary.csv"), index=False)

print("\nPaper (PeptideCLM-2 MLM, fine-tuned, 3 seeds): 0.756 +- 0.019 at the best threshold on each test set; 0.693 at the usual threshold (recomputed from the authors' predictions).")
print("saved:", os.path.join(RESULTS_DIR, a.out, "thpep_official_summary.csv"))

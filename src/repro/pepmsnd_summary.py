"""Scores of the PepMSND reproduction (scripts/run_reproduce.sh) next to the published numbers.

    python src/repro/pepmsnd_summary.py [--root results/repro_pepmsnd]

Reads <root>/<model>/seed<S>/fold<k>/preds_fold<k>.csv (true_label, predicted_prob, predicted_label; 10 folds per seed), prints per
model and seed the MCC and AUROC in two ways (pooled = all 10 test folds together, 640 molecules; fold mean = the 10 per-fold values
averaged), then the mean and standard deviation over seeds, next to:
  * PeptideCLM-2 paper (J. Chem. Inf. Model. 2026, 66, 8110, Table 1; "mean +- sd across independent seeds"; the paper says three seeds,
    does not say whether its MCC is pooled or per fold): MTR 0.749 +- 0.007, hybrid 0.741 +- 0.016, MLM 0.729 +- 0.006;
    PepMSND's own published model: 0.726;
  * the earlier Distillation project's run of the authors' scripts on mlm-large (reference/prior_results/compression_summary_agg.csv):
    MCC 0.618 +- 0.003, AUROC 0.867 (3 seeds).
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from paths import RESULTS_DIR  # noqa: E402

PUB = {"mlm-large": "0.729 +- 0.006", "mtr-large": "0.749 +- 0.007", "hybrid-large": "0.741 +- 0.016"}


def best_thr_mcc(y, pr):
    """MCC at the threshold that is best on these very predictions (what the KAN script's val_mcc_best monitor reports): an oracle number."""
    y = np.asarray(y).astype(int)
    return max(matthews_corrcoef(y, (pr >= t).astype(int)) for t in np.unique(pr))


def score(parts):
    y = np.concatenate([p.true_label.to_numpy() for p in parts]).astype(int)
    pr = np.concatenate([p.predicted_prob.to_numpy() for p in parts])
    pooled = (matthews_corrcoef(y, (pr >= 0.5).astype(int)), roc_auc_score(y, pr) if len(set(y)) > 1 else np.nan)
    per = [(matthews_corrcoef(p.true_label.astype(int), p.predicted_label.astype(int)),
            roc_auc_score(p.true_label.astype(int), p.predicted_prob) if p.true_label.nunique() > 1 else np.nan) for p in parts]
    oracle = float(np.mean([best_thr_mcc(p.true_label, p.predicted_prob.to_numpy()) for p in parts]))
    return pooled[0], pooled[1], float(np.mean([m for m, _ in per])), float(np.nanmean([a for _, a in per])), oracle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(RESULTS_DIR, "repro_pepmsnd"))
    a = ap.parse_args()
    rows = []
    for mdir in sorted(glob.glob(os.path.join(a.root, "*"))):
        if not os.path.isdir(mdir):
            continue
        for sdir in sorted(glob.glob(os.path.join(mdir, "seed*"))):
            files = sorted(glob.glob(os.path.join(sdir, "fold*", "preds_fold*.csv")), key=lambda f: int(re.findall(r"fold(\d+)\.csv", f)[0]))
            if not files:
                continue
            parts = [pd.read_csv(f) for f in files]
            mcc, auc, mcc_f, auc_f, mcc_o = score(parts)
            mcc_t = np.nan                                  # pepmsnd_ft_kan_v3.py also saves label_thr = the label at the threshold found on the set that picked the epoch
            if all("label_thr" in p for p in parts):
                mcc_t = matthews_corrcoef(np.concatenate([p.true_label.to_numpy() for p in parts]).astype(int), np.concatenate([p.label_thr.to_numpy() for p in parts]).astype(int))
            rows.append({"model": os.path.basename(mdir), "seed": os.path.basename(sdir)[4:], "folds": len(files),
                         "MCC pooled": mcc, "MCC fold mean": mcc_f, "AUROC pooled": auc, "AUROC fold mean": auc_f, "MCC best-thr fold mean (oracle)": mcc_o,
                         "MCC pooled at saved thr": mcc_t})
    if not rows:
        print("no predictions under", a.root)
        return
    raw = pd.DataFrame(rows)
    print("=== per model and seed (a seed is complete with 10 folds) ===")
    print(raw.round(4).to_string(index=False))
    done = raw[raw.folds == 10]
    if len(done):
        cols = ["MCC pooled", "MCC fold mean", "MCC best-thr fold mean (oracle)", "AUROC pooled", "AUROC fold mean"] + (["MCC pooled at saved thr"] if done["MCC pooled at saved thr"].notna().any() else [])
        agg = done.groupby("model")[cols].agg(["mean", "std"]).round(4)
        agg.columns = ["%s %s" % c for c in agg.columns]
        agg.insert(0, "seeds", done.groupby("model").size())
        agg["published MCC"] = [PUB.get(m, "") for m in agg.index]
        print("\n=== mean and sd over complete seeds, next to the paper (Table 1 MCC) ===")
        print(agg.to_string())
    print("\nreferences: PepMSND's own model 0.726 MCC; earlier Distillation run of the authors' scripts on mlm-large 0.618 +- 0.003 MCC, "
          "AUROC 0.867.\nNote: the authors' script chooses the stopping epoch on the TEST fold, so these numbers are optimistic by construction; "
          "src/finetune/finetune_smiles.py (no test-fold selection) is the honest comparison.")
    print("MCC columns: pooled and fold mean use the saved 0.5 threshold; best-thr (oracle) takes the best threshold per fold on the test predictions "
          "(what the KAN script monitors as val_mcc_best): an upper bound, not a usable score. 'at saved thr' (teacher runs only) = the threshold found on the set "
          "that picked the epoch: the test fold for --select test (still optimistic), an inner split for --select inner (honest).")
    raw.to_csv(os.path.join(a.root, "summary_by_seed.csv"), index=False)


if __name__ == "__main__":
    main()

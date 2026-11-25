"""Single-modality check: is each representation informative on its own?

Every representation goes through the SAME learners and the SAME split protocol
per benchmark, so differences come from the features alone:

    linear   logistic / ridge regression on standardised features, C or alpha
             picked by 5-fold CV inside the training data
    gbt      sklearn HistGradientBoosting, fixed settings, 3 seeds

Representations: whichever of features/{ecfp,mapc,pclm,pepland}.npz exist, plus
"concat" (all available ones side by side) -- the naive fusion any learned
fusion has to beat. Teacher embeddings (features/teacher_<name>.npz, written by
teacher/train_teacher.py) are extra representations; they never enter "concat".

Metrics: MCC (+ AUROC) for classification, R2 (+ Spearman) for regression.
Fixed splits report test; cluster-CV benchmarks report the mean over folds AND (pooled_*) one score on the predictions of all
folds put together. The pooled R2 is what PeptideCLM-2's PAMPA script reports (it concatenates the six held-out clusters), and it
is not the fold mean: a cluster whose labels barely vary has a hugely negative per-fold R2.

    python src/eval_single.py                  # main benchmarks + AmpHGT
    python src/eval_single.py --bench CellPPD  # one benchmark
    # the teacher gate, written to its own folder (the single-modality results stay as they are):
    python src/eval_single.py --reps ecfp teacher_t1 --out teacher_t1
    # neural probe on the frozen vectors (a better fit for dense embeddings than trees):
    python src/eval_single.py --reps ecfp teacher_t3 --learners mlp --out teacher_t3_mlp
    # does the teacher add anything to ECFP? ("a+b" = both side by side; "concat" = all raw views)
    python src/eval_single.py --reps ecfp teacher_t3 ecfp+teacher_t3 ecfp+pclm ecfp+pepland concat \
        --learners linear gbt mlp --out complement_t3
    # teacher v2 (ecfp + pclm + rdkit, no PepLand): is the teacher better than the simple stack of its own inputs?
    python src/eval_single.py --reps ecfp rdkit ecfp+rdkit ecfp+pclm+rdkit teacher_t5 ecfp+teacher_t5         --learners linear gbt mlp --out teacher_t5_gate
    # PepLand's own benchmark suite (features made with featurize*.py --feat pepland_suite):
    python src/eval_single.py --feat pepland_suite --out pepland_suite --reps ecfp pclm pepland \
        --learners linear gbt mlp --bench PL_cCPP PL_cSol PL_ncCPP PL_cBinding PL_ncBinding
    # raw ESM only exists for canonical peptides (ok = False elsewhere), so alone it is scored on a SMALLER set than the
    # others. --same-rows scores every listed representation on the molecules ALL of them have (fair comparison):
    python src/eval_single.py --same-rows --reps ecfp pclm pepland esm teacher_t8 --learners linear gbt mlp --out esm_gate_t8
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LogisticRegressionCV, RidgeCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef, mean_absolute_error,
                             mean_squared_error, r2_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from benchmarks import TASK, load, selected  # noqa: E402
from kernel_learners import KernelRidgeCV, KernelSVMCV  # noqa: E402
from paths import FEAT_DIR as FEAT, RESULTS_DIR, SINGLE_OUT as OUT  # noqa: E402

N_JOBS = int(os.environ.get("OMP_NUM_THREADS", "8"))        # lab rule: <= half the CPU
REPS = ["ecfp", "mapc", "pclm", "pepland"]                  # the single views; "concat" stacks only these
EXTRA = ["rdkit"]                                           # RDKit descriptors (rdkit_desc.py): loaded, usable as
                                                            # "rdkit" or "ecfp+rdkit", but not part of "concat"
SEEDS = [0, 1, 2]
warnings.filterwarnings("ignore")


def load_reps():
    teachers = sorted(f[:-4] for f in os.listdir(FEAT) if f.startswith("teacher_") and f.endswith(".npz")) \
        if os.path.isdir(FEAT) else []
    reps = {}
    for r in REPS + EXTRA + teachers:
        p = os.path.join(FEAT, r + ".npz")
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            X = z["X"]
            if r == "rdkit":                          # raw descriptors -> signed log, NaN -> median (no labels used)
                import rdkit_desc
                X = rdkit_desc.preprocess_np(X, z["ok"])
            reps[r] = (dict(zip(z["smiles"].tolist(), range(len(z["smiles"])))), X, z["ok"])
    return reps


def parts(rep, reps):
    """The loaded representations a name stands for ('a+b' = a and b; 'concat' = every single view)."""
    return [r for r in reps if r in REPS] if rep == "concat" else rep.split("+")


def common_ok(df, all_reps, reps):
    """Rows of df that every representation in all_reps (and every part of an 'a+b') has: the same-rows set."""
    ok = np.ones(len(df), bool)
    for rep in all_reps:
        for r in parts(rep, reps):
            idx, _, good = reps[r]
            ok &= good[np.array([idx[s] for s in df.smiles])]
    return ok


def matrix(df, rep, reps):
    """Feature matrix for df rows; 'concat' stacks every loaded single view (not the teachers).
    Context columns (extra_*) are appended for every representation."""
    # "a+b" stacks the named representations side by side (e.g. ecfp+teacher_t3: does the teacher add to ECFP?)
    names = parts(rep, reps)
    blocks, ok = [], np.ones(len(df), bool)
    for r in names:
        idx, X, good = reps[r]
        ii = np.array([idx[s] for s in df.smiles])
        blocks.append(X[ii]); ok &= good[ii]
    extra = [c for c in df.columns if c.startswith("extra_")]
    if extra:
        blocks.append(df[extra].to_numpy(np.float32))
    return np.hstack(blocks), ok


def learners(task, seed):
    if task == "clf":
        return {
            "linear": make_pipeline(StandardScaler(), LogisticRegressionCV(
                Cs=np.logspace(-4, 1, 8), cv=5, max_iter=3000, scoring="neg_log_loss", n_jobs=N_JOBS)),
            "gbt": HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                                  max_leaf_nodes=31, l2_regularization=1.0,
                                                  random_state=seed),
            "mlp": make_pipeline(StandardScaler(), MLPClassifier(
                hidden_layer_sizes=(512,), alpha=1e-3, early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=10, max_iter=300, random_state=seed)),
            "kernel": KernelSVMCV(),                       # RBF kernel SVM, (gamma, C) tuned inside the training data (kernel_learners.py)
        }
    return {
        "linear": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-1, 5, 13))),
        "gbt": HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                             max_leaf_nodes=31, l2_regularization=1.0,
                                             random_state=seed),
        "mlp": TransformedTargetRegressor(
            regressor=make_pipeline(StandardScaler(), MLPRegressor(
                hidden_layer_sizes=(512,), alpha=1e-3, early_stopping=True, validation_fraction=0.1,
                n_iter_no_change=10, max_iter=300, random_state=seed)),
            transformer=StandardScaler()),
        "kernel": KernelRidgeCV(),                         # RBF kernel ridge (= a Gaussian process mean), (gamma, lambda) by exact leave-one-out
    }


def score(task, y, pred):
    if task == "clf":
        return {"MCC": matthews_corrcoef(y, pred >= 0.5), "AUROC": roc_auc_score(y, pred),
                "ACC": accuracy_score(y, pred >= 0.5), "F1": f1_score(y, pred >= 0.5)}
    return {"R2": r2_score(y, pred), "Spearman": spearmanr(y, pred).correlation,
            "Pearson": float(np.corrcoef(y, pred)[0, 1]), "MAE": mean_absolute_error(y, pred),
            "RMSE": float(np.sqrt(mean_squared_error(y, pred)))}


def fit_predict(model, task, Xtr, ytr, Xte):
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1] if task == "clf" else model.predict(Xte)


def run(bench, rep, reps, which=("linear", "gbt"), same=None):
    """same: list of representations; when given, only the molecules ALL of them have are used (see --same-rows)."""
    df = load(bench).dropna(subset=["smiles"]).reset_index(drop=True)
    df["smiles"] = df.smiles.astype(str)
    task = TASK[bench]
    X, ok = matrix(df, rep, reps)
    n_all = len(df)
    if same:
        ok = ok & common_ok(df, same, reps)
    df, X = df[ok].reset_index(drop=True), X[ok]
    if len(df) == 0:                # e.g. PAMPA with raw ESM: no molecule is readable by every representation
        print("   %s %s: no usable molecule (with --same-rows: none has every representation): skipped" % (bench, rep), flush=True)
        return []
    y = df.y.to_numpy()
    rows = []
    # Cluster CV iff the split column holds fold ids. Not "dtype != object": on
    # pandas >= 3 'train'/'test' load as a string dtype, which would turn a fixed
    # split into a 2-fold CV that also trains on the test set.
    cv = pd.api.types.is_integer_dtype(df.split)
    assert cv or set(df.split) == {"train", "test"}, "unexpected split values in " + bench
    folds = sorted(df.split.unique()) if cv else ["test"]
    if same:            # dropping molecules can empty a fold or leave one class: refuse that benchmark instead of crashing
        for f in folds:
            te = (df.split == f).to_numpy()
            if te.sum() < 10 or (~te).sum() < 10 or (task == "clf" and (len(set(df.y[te])) < 2 or len(set(df.y[~te])) < 2)):
                print("   %s %s: the same-rows set (%d of %d molecules) leaves a split too small or with one class: skipped"
                      % (bench, rep, len(df), n_all), flush=True)
                return []
    for lname in which:
        seeds = [0] if lname in ("linear", "kernel") else SEEDS
        for seed in seeds:
            per, ally, allp = [], [], []
            for f in folds:
                te = (df.split == f).to_numpy() if cv else (df.split == "test").to_numpy()
                tr = ~te
                m = learners(task, seed)[lname]
                p = fit_predict(m, task, X[tr], y[tr], X[te])
                ally.append(y[te]), allp.append(p)
                s = score(task, y[te], p)
                if bench == "AmpHGT":                    # length stress: <=30 residues
                    short = df.loc[te, "smiles"].map(n_residues).to_numpy() <= 30
                    s.update({k + "_le30": v for k, v in score(task, y[te][short], p[short]).items()})
                per.append(s)
            row = {"benchmark": bench, "rep": rep, "learner": lname, "seed": seed,
                   "n": len(df), "coverage": round(ok.mean(), 4),
                   **{k: float(np.mean([s[k] for s in per])) for k in per[0]}}
            if cv:                                       # one score on the pooled predictions of all folds
                row.update({"pooled_" + k: float(v) for k, v in score(task, np.concatenate(ally), np.concatenate(allp)).items()})
            rows.append(row)
    return rows


_RES = None


def n_residues(smi):
    global _RES
    from rdkit import Chem
    if _RES is None:
        _RES = Chem.MolFromSmarts("[NX3][CX4][CX3](=O)")
    m = Chem.MolFromSmiles(smi)
    return len(m.GetSubstructMatches(_RES)) if m is not None else 0


def main():
    global FEAT
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="*", default=None)
    ap.add_argument("--feat", default=None, help="features subfolder under features/ (default single_modality); "
                                                 "e.g. pepland_suite for the PL_* benchmarks")
    ap.add_argument("--reps", nargs="*", default=None)
    ap.add_argument("--learners", nargs="*", default=["linear", "gbt"], choices=["linear", "gbt", "mlp", "kernel"],
                    help="default linear gbt; mlp = one hidden layer (512), early stopping, 3 seeds; "
                         "kernel = RBF kernel SVM (classification) / kernel ridge (regression), tuned inside the training data, deterministic")
    ap.add_argument("--out", default=None, help="results folder name under results/ (default single_modality)")
    ap.add_argument("--same-rows", action="store_true",
                    help="score every representation in --reps on the molecules ALL of them have (needed to compare raw ESM, which "
                         "only exists for canonical peptides, with representations that cover everything)")
    a = ap.parse_args()
    if a.feat:
        FEAT = os.path.join(os.path.dirname(FEAT), a.feat)
    a.bench = a.bench or selected()
    reps = load_reps()
    assert reps, "no features in %s -- run featurize.py first" % FEAT
    for r in a.reps or []:
        for part in ([] if r == "concat" else r.split("+")):
            assert part in reps, "no features for '%s' (have: %s)" % (part, ", ".join(reps))
    out = os.path.join(RESULTS_DIR, a.out) if a.out else OUT
    singles = [r for r in reps if r in REPS]
    names = (a.reps or list(reps)) + (["concat"] if len(singles) > 1 and not a.reps else [])
    print("representations:", names)
    if any("esm" in r.split("+") for r in names) and not a.same_rows:
        print("NOTE: raw ESM exists only for canonical peptides: without --same-rows it is scored on fewer molecules than the "
              "others (see the coverage column); the numbers are not comparable")
    os.makedirs(out, exist_ok=True)

    rows = []
    for b in a.bench:
        for r in names:
            t0 = time.time()
            new = run(b, r, reps, tuple(a.learners), names if a.same_rows else None)
            if not new:
                continue
            rows += new
            last = rows[-1]
            key = "MCC" if TASK[b] == "clf" else "R2"
            print("%-8s %-10s done (%.1fs) | %s %s %.3f" % (b, r, time.time() - t0, last["learner"], key, last[key]), flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(out, "raw.csv"), index=False)

    metric_cols = [c for c in raw.columns if c in ("MCC", "AUROC", "ACC", "F1", "R2", "Spearman", "Pearson",
                                                   "MAE", "RMSE", "MCC_le30") or c.startswith("pooled_")]
    agg = raw.groupby(["benchmark", "rep", "learner"])[metric_cols + ["coverage"]].agg(["mean", "std"])
    agg.to_csv(os.path.join(out, "summary.csv"))

    print("\n=== headline: MCC (classification) / R2 (PAMPA), mean over seeds ===")
    raw["headline"] = np.where(raw.benchmark.map(TASK) == "clf", raw.get("MCC"), raw.get("R2"))
    tab = raw.groupby(["benchmark", "learner", "rep"]).headline.mean().unstack("rep").round(3)
    print(tab.to_string())
    for col, label in (("AUROC", "AUROC (classification)"), ("ACC", "accuracy (classification)"),
                       ("F1", "F1 (classification)"), ("Pearson", "Pearson r (regression)"),
                       ("Spearman", "Spearman (regression)"), ("RMSE", "RMSE (regression, lower is better)")):
        if col in raw and raw[col].notna().any():
            print("\n=== %s, mean over seeds ===" % label)
            print(raw.dropna(subset=[col]).groupby(["benchmark", "learner", "rep"])[col].mean()
                  .unstack("rep").round(3).to_string())
    for col, label in (("pooled_R2", "POOLED R2 over all folds (PAMPA: the authors' metric)"),
                       ("pooled_MCC", "POOLED MCC over all folds (cluster-CV benchmarks)")):
        if col in raw and raw[col].notna().any():
            print("\n=== %s, mean over seeds ===" % label)
            print(raw.dropna(subset=[col]).groupby(["benchmark", "learner", "rep"])[col].mean()
                  .unstack("rep").round(3).to_string())
    if "MCC_le30" in raw:
        print("\nAmpHGT test restricted to <=30 residues (MCC):")
        print(raw[raw.benchmark == "AmpHGT"].groupby(["learner", "rep"]).MCC_le30.mean()
              .unstack("rep").round(3).to_string())
    print("\nwrote", out)


if __name__ == "__main__":
    main()

"""Evaluation harness for the teacher (step 0 of teacher v3). CPU only, frozen embeddings only (no fine-tuning).

What it fixes in the earlier protocol (eval_single.py + "best of six variants"): nothing is selected on the test set, every embedding
goes through the SAME learners and splits, there are 5 learner seeds, every comparison carries a paired-bootstrap interval over the
test items, and standard and modified chains are scored separately.

    python src/teacher/eval_harness.py --teachers t10 --out harness_t10
    python src/teacher/eval_harness.py --teachers t10 t8 --bench CellPPD THPep --seeds 3 --learners linear gbt
    python src/teacher/eval_harness.py --reps P L H P+L+H PCA --select none
    python src/teacher/eval_harness.py --teachers t10 --workers 4          # 4 benchmarks at a time, 2 threads each (of the 8-thread cap)

Representations (--reps, joined by "+" they sit side by side):
    P  PepLand 300     L  PeptideCLM-2 1024     H  ECFP counts 2048 (log1p)     E  ESM-2 2560 (standard chains only)
    PCA  P+L+H standardised and cut to 1024 dimensions (needs `pca_baseline.py --fit`)
    PROT  the protein partner alone (ESM-2 3B vector; only the PL_cBindingProt-type benchmarks have it, benchmarks.py). On those benchmarks every
          other representation is [representation | protein vector], so PROT is the "no peptide" control
    any other name = features/<suite>/<name>.npz, e.g. teacher_t10 (--teachers t10 adds teacher_t10)
Default --reps: P L H L+H P+L P+L+H PCA E P+L+H+E, plus the teachers.

Learners (eval_single.py): linear (regularisation picked by CV INSIDE the training data), gbt (fixed settings), mlp (fixed settings,
early stopping on a split of the training data), kernel (RBF kernel SVM for classification / kernel ridge for regression, both tuned inside
the training data: kernel_learners.py). linear and kernel are deterministic (one run); gbt and mlp run --seeds times.

Protocol: the splits the benchmarks ship with (fixed train/test, or the fixed cluster folds); the score of a cluster-CV benchmark is
ONE score on the pooled out-of-fold predictions. Primary metric per benchmark: the published one (MCC, PAMPA pooled R2, AUROC for
the two canonical PepLand-suite classification sets, Spearman for the regressions).

Reference ("the best single input"): per benchmark, learner and outer fold, the one of P, L, H with the best score on a validation
split (20%, stratified) cut from that fold's TRAINING data, never from the test data. The differences to each single input are also
written (per_input.csv), so any other reference can be computed afterwards.

Headline: the average over benchmarks of (score - score of the comparison), per learner and over learners, with a paired bootstrap
interval over test items (each benchmark's test items resampled on their own, --boot draws, then averaged over benchmarks) and the
number of benchmarks won / lost / tied. Two comparisons, two row groups: versus REF (the best single input, frozen by the validation
split) and versus H (the ECFP counts alone: the number a reviewer checks first). Intervals: 95% for the average and for wins /
losses / ties, 99.5% for the per-benchmark "clearly worse" / "clearly better" counts (ten benchmarks at 95% give a false alarm in
about 40% of cases). The "spread" of a difference = this interval. Per benchmark: per_bench_ci.csv (diff, 95% and 99.5% bounds).
Because one benchmark can dominate an average of raw metrics (PAMPA's R2 spans 0.9 while the others span 0.05-0.3), every headline,
pass mark and seed spread is also given WITHOUT the benchmarks of --without (default PAMPA), next to average ranks (ranks_*.csv) and a
strict bar, the best single input ON THE TEST DATA (versus BEST, an optimistic reference).
Pass mark for a teacher (passmark.csv): average difference to REF >= 0, at least the PCA baseline, no benchmark clearly worse (99.5%).
Teachers named <base>_s1, <base>_s2, ... are repeats of one configuration: seed_spread.csv gives the spread of their headline number
(training noise, which the intervals above do not cover).

E: raw ESM exists only for standard chains. Representations that contain E are scored on the benchmark items that have E (a second
pass, "has_E", in which P+L+H and the teachers are scored on the same items); everything else on all items. The test items are also
split into standard (E present) and modified (E absent) for the by-type table.

Writes results/<out>/: raw.csv, summary.csv, headline.csv, per_bench.csv, per_bench_ci.csv, per_input.csv, by_type.csv, passmark.csv,
seed_spread.csv and prints the tables. The standard-chain rows of by_type.csv are where E can matter: read the "no E decoder" ablation there.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import eval_single as ES  # noqa: E402
import pca_baseline as PB  # noqa: E402
import views_v3 as V  # noqa: E402
from benchmarks import MAIN, PL_PROT, PL_SUITE, STRESS, TASK, THPEP_CV, THPEP_OFF, load  # noqa: E402
from paths import FEAT_DIR, RESULTS_DIR  # noqa: E402

warnings.filterwarnings("ignore")
PRIMARY = {"CellPPD": "MCC", "THPep": "MCC", "PepMSND": "MCC", "PAMPA": "R2", "AmpHGT": "MCC", "PL_cCPP": "AUROC", "PL_cSol": "AUROC",
           "PL_ncCPP": "Spearman", "PL_cBinding": "Spearman", "PL_ncBinding": "Spearman"}
PRIMARY.update({n: "MCC" for n in THPEP_CV})                      # THPep with the journal paper's random 5-fold protocol (benchmarks.py)
PRIMARY.update({n: "MCC" for n in THPEP_OFF})                     # THPep on the authors' three actual splits (benchmarks.py, THPEP_OFF)
PRIMARY.update({n: "Spearman" for n in PL_PROT})                  # the binding tasks with the protein partner (benchmarks.py, PL_*Prot)
BENCHES = MAIN + STRESS + PL_SUITE
SUITE = os.path.join(os.path.dirname(FEAT_DIR), "pepland_suite")
SINGLES = ("P", "L", "H")


def short(name):
    """Display name: teacher_abl_noh_s1 -> noh_s1, teacher_t10 -> t10."""
    n = str(name)
    for pre in ("teacher_abl_", "teacher_"):
        if n.startswith(pre):
            return n[len(pre):]
    return n


def show(df, chunk=6, index=True):
    """Print a wide table in column chunks that fit a narrow terminal (the CSV files keep everything)."""
    cols = list(df.columns)
    for i in range(0, len(cols), chunk):
        print(df[cols[i:i + chunk]].to_string(index=index))
        if i + chunk < len(cols):
            print()
DEFAULT_REPS = ["P", "L", "H", "L+H", "P+L", "P+L+H", "PCA", "E", "P+L+H+E"]
SELECT_METRIC = {"clf": "AUROC", "reg": "Spearman"}              # the metric the reference is chosen by (validation data only)


def suite_dir(bench):
    return SUITE if (bench in PL_SUITE or bench in PL_PROT) else FEAT_DIR


def tokens(rep):
    return rep.split("+")


def has_e(rep):
    return "E" in tokens(rep)


class Store:
    """The benchmark feature files of one suite folder, loaded once."""

    def __init__(self, folder):
        self.folder, self.files = folder, {}

    def get(self, name):
        if name not in self.files:
            p = os.path.join(self.folder, name + ".npz")
            if os.path.exists(p):
                z = np.load(p, allow_pickle=True)
                self.files[name] = (dict(zip(z["smiles"].tolist(), range(len(z["smiles"])))), z["X"], z["ok"])
            else:
                self.files[name] = None
        return self.files[name]


def block(store, smiles, token, st=None, pca=None):
    """-> (features [n, d] float32, ok [n]) for one token of a representation; None if its file is missing."""
    if token in V.LETTER:
        v = V.LETTER[token]
        g = store.get(v)
        if g is None:
            return None
        idx, X, ok = g
        ii = np.array([idx[s] for s in smiles])
        x = X[ii]
        return V.to_float(v, np.minimum(x, 255) if v == "ecfp" else x), ok[ii]
    if token == "PCA":
        arrs, oks = {}, np.ones(len(smiles), bool)
        for v in V.INPUTS:
            g = store.get(v)
            if g is None:
                return None
            idx, X, ok = g
            ii = np.array([idx[s] for s in smiles])
            arrs[v] = np.minimum(X[ii], 255) if v == "ecfp" else X[ii]
            oks &= ok[ii]
        return PB.transform(arrs, st, pca), oks
    g = store.get(token)
    if g is None:
        return None
    idx, X, ok = g
    ii = np.array([idx[s] for s in smiles])
    return np.asarray(X[ii], np.float32), ok[ii]


def matrix(store, df, rep, st, pca):
    """The feature matrix of a representation for the rows of df, plus its ok vector (None if a file is missing)."""
    xs, ok = [], np.ones(len(df), bool)
    if tokens(rep) == ["PROT"]:                       # the protein partner ALONE (PL_*Prot benchmarks): the control that shows what the peptide adds
        cols = [c for c in df.columns if c.startswith("extra_prot_")]
        return (df[cols].to_numpy(np.float32), ok) if cols else None
    for t in tokens(rep):
        b = block(store, df.smiles.tolist(), t, st, pca)
        if b is None:
            return None
        xs.append(b[0])
        ok &= b[1]
    extra = [c for c in df.columns if c.startswith("extra_")]            # PepMSND: context columns belong to every representation
    if extra:
        xs.append(df[extra].to_numpy(np.float32))
    return np.hstack(xs), ok


def safe_score(task, y, p):
    try:
        return ES.score(task, y, p)
    except Exception:
        return {}


def primary_metric(bench, y, p):
    """The benchmark's primary metric alone (the bootstrap calls this thousands of times)."""
    from scipy.stats import spearmanr
    from sklearn.metrics import matthews_corrcoef, r2_score, roc_auc_score
    m = PRIMARY[bench]
    try:
        if m == "MCC":
            return matthews_corrcoef(y, p >= 0.5)
        if m == "AUROC":
            return roc_auc_score(y, p)
        if m == "R2":
            return r2_score(y, p)
        return spearmanr(y, p).correlation
    except Exception:
        return np.nan


def folds_of(df):
    cv = pd.api.types.is_integer_dtype(df.split)
    assert cv or set(df.split) == {"train", "test"}, "unexpected split values"
    return cv, (sorted(df.split.unique()) if cv else ["test"])


def test_mask(df, cv, f):
    return (df.split == f).to_numpy() if cv else (df.split == "test").to_numpy()


def too_small(df, task, cv, folds):
    y = df.y.to_numpy()
    for f in folds:
        te = test_mask(df, cv, f)
        if te.sum() < 10 or (~te).sum() < 10 or (task == "clf" and (len(set(y[te])) < 2 or len(set(y[~te])) < 2)):
            return True
    return False


def oof_predict(X, df, task, learner, seed, cv, folds):
    """Predictions for every row that is in some test fold (NaN elsewhere)."""
    y = df.y.to_numpy()
    pred = np.full(len(df), np.nan)
    for f in folds:
        te = test_mask(df, cv, f)
        m = ES.learners(task, seed)[learner]
        pred[te] = ES.fit_predict(m, task, X[~te], y[~te], X[te])
    return pred


def choose_reference(Xs, df, task, learner, cv, folds):
    """Per outer fold, the single input (P, L or H) with the best score on a validation split of that fold's TRAINING data.
    -> {fold: token}. Uses a 20% stratified split, learner seed 0, the metric SELECT_METRIC."""
    y = df.y.to_numpy()
    choice = {}
    for f in folds:
        tr = np.flatnonzero(~test_mask(df, cv, f))
        rng = np.random.default_rng(0)
        if task == "clf":
            va = np.concatenate([rng.permutation(tr[y[tr] == c])[: max(1, int(0.2 * (y[tr] == c).sum()))] for c in np.unique(y[tr])])
        else:
            va = rng.permutation(tr)[: max(1, int(0.2 * len(tr)))]
        fit = np.setdiff1d(tr, va)
        best, best_s = None, -np.inf
        for t in SINGLES:
            m = ES.learners(task, 0)[learner]
            p = ES.fit_predict(m, task, Xs[t][fit], y[fit], Xs[t][va])
            s = safe_score(task, y[va], p).get(SELECT_METRIC[task], np.nan)
            if np.isfinite(s) and s > best_s:
                best, best_s = t, s
        choice[f] = best or SINGLES[0]
    return choice


def run_bench(bench, reps, learners, seeds, select, st, pca, e_pass=True):
    """-> (rows list, store dict for the bootstrap). One benchmark, both passes."""
    store = Store(suite_dir(bench))
    df = load(bench).dropna(subset=["smiles"]).reset_index(drop=True)
    df["smiles"] = df.smiles.astype(str)
    task = TASK[bench]
    y_all = df.y.to_numpy()
    cv, folds = folds_of(df)
    rows, keep = [], {}
    mats = {}
    for r in set(reps) | set(SINGLES):
        m = matrix(store, df, r, st, pca)
        if m is None:
            if r in reps:
                print("   %s: %s skipped (a feature file or the PCA file is missing)" % (bench, r), flush=True)
            continue
        mats[r] = m
    std_flag = None
    g = store.get("esm")
    if g is not None:
        idx, _, ok = g
        std_flag = ok[np.array([idx[s] for s in df.smiles])]
    passes = [("all", [r for r in reps if not has_e(r) and r in mats])]
    if e_pass and std_flag is not None:
        e_reps = [r for r in reps if has_e(r) and r in mats]
        extra = [r for r in reps if not has_e(r) and r in mats and (r == "P+L+H" or r.startswith("teacher_") or "teacher_" in r)]
        if e_reps:
            passes.append(("has_E", e_reps + [r for r in extra if r not in e_reps]))
    for rowset, rlist in passes:
        if not rlist:
            continue
        ok = np.ones(len(df), bool)
        for r in set(rlist) | set(SINGLES):
            if r in mats:
                ok &= mats[r][1]
        if rowset == "has_E":
            ok &= std_flag
        sub = df[ok].reset_index(drop=True)
        if len(sub) == 0 or too_small(sub, task, cv, folds if not cv else sorted(sub.split.unique())):
            print("   %s [%s]: too few usable items (%d of %d) or a one-class split: skipped" % (bench, rowset, len(sub), len(df)), flush=True)
            continue
        cv2, folds2 = folds_of(sub)
        y = sub.y.to_numpy()
        flag = None if std_flag is None else std_flag[ok]
        X = {r: mats[r][0][ok] for r in set(rlist) | set(SINGLES) if r in mats}
        preds = {}
        for lname in learners:
            for seed in ([0] if lname in ("linear", "kernel") else range(seeds)):
                for r in rlist:
                    t0 = time.time()
                    p = oof_predict(X[r], sub, task, lname, seed, cv2, folds2)
                    preds[(r, lname, seed)] = p
                    msk = ~np.isnan(p)
                    subsets = {"all": msk}
                    if flag is not None and rowset == "all":
                        subsets.update({"standard": msk & flag, "modified": msk & ~flag})
                    shown = np.nan
                    for sname, mk in subsets.items():
                        if mk.sum() < 10:
                            continue
                        sc = safe_score(task, y[mk], p[mk])
                        if sname == "all":
                            shown = sc.get(PRIMARY[bench], np.nan)
                        rows.append({"benchmark": bench, "rowset": rowset, "subset": sname, "rep": r, "learner": lname, "seed": seed,
                                     "n": int(len(sub)), "n_test": int(mk.sum()), **sc})
                    print("   %-12s %-6s %-12s %-6s seed %d | %s %.3f (%.0fs)" % (
                        bench, rowset, r, lname, seed, PRIMARY[bench], shown, time.time() - t0), flush=True)
                for sname in SINGLES:                         # the single inputs are always scored: they are the reference pool
                    if (sname, lname, seed) not in preds:
                        preds[(sname, lname, seed)] = oof_predict(X[sname], sub, task, lname, seed, cv2, folds2)
        # the reference per learner: chosen on validation data, predictions assembled fold by fold
        refs, choices = {}, {}
        for lname in learners:
            if select == "val":
                ch = choose_reference(X, sub, task, lname, cv2, folds2)
            else:
                ch = {f: "H" for f in folds2}
            choices[lname] = ch
            for seed in ([0] if lname in ("linear", "kernel") else range(seeds)):
                ref = np.full(len(sub), np.nan)
                for f in folds2:
                    te = test_mask(sub, cv2, f)
                    ref[te] = preds[(ch[f], lname, seed)][te]
                refs[(lname, seed)] = ref
                msk = ~np.isnan(ref)
                rows.append({"benchmark": bench, "rowset": rowset, "subset": "all", "rep": "REF", "learner": lname, "seed": seed,
                             "n": int(len(sub)), "n_test": int(msk.sum()), "ref_choice": ",".join("%s" % ch[f] for f in folds2),
                             **safe_score(task, y[msk], ref[msk])})
        keep[(bench, rowset)] = {"y": y, "preds": preds, "refs": refs, "task": task}
    return rows, keep


_REF_BOOT = {}


def boot_diffs(keep, rep, learner, n_boot, rowset="all", seed=0, against="REF"):
    """Paired bootstrap of (primary score of rep - primary score of what it is compared with) per benchmark, learner seed `seed`.
    against = "REF" (the reference single input, chosen on validation data) or one input token such as "H".
    The draws of a benchmark are the same for every representation (seeded by its name) and the scores of what it is compared with
    are computed once. -> {bench: array [n_boot]}; benchmarks resampled independently."""
    import zlib
    out = {}
    for (bench, rs), d in keep.items():
        if rs != rowset or (rep, learner, seed) not in d["preds"]:
            continue
        other = d["refs"].get((learner, seed)) if against == "REF" else d["preds"].get((against, learner, seed))
        if other is None:
            continue
        p, ref, y = d["preds"][(rep, learner, seed)], other, d["y"]
        m = np.flatnonzero(~np.isnan(p) & ~np.isnan(ref))
        rng = np.random.default_rng(zlib.crc32(("%s|%s|%d" % (bench, rs, n_boot)).encode()))
        draws = np.empty(n_boot)
        key = (bench, rs, learner, seed, n_boot, against)
        cached = key in _REF_BOOT
        refs = _REF_BOOT.setdefault(key, np.empty(n_boot))
        for b in range(n_boot):
            ii = rng.choice(m, len(m), replace=True)
            if not cached:
                refs[b] = primary_metric(bench, y[ii], ref[ii])
            draws[b] = primary_metric(bench, y[ii], p[ii]) - refs[b]
        out[bench] = draws
    return out


def _bench_job(args):
    bench, reps, learners, seeds, select, st, pca, e_pass = args
    t0 = time.time()
    r, k = run_bench(bench, reps, learners, seeds, select, st, pca, e_pass)
    return bench, r, k, time.time() - t0


def run_parallel(a, reps, st, pca):
    """Benchmarks are independent: run them in --workers separate processes. The thread allowance (OMP_NUM_THREADS, the lab cap of 8)
    is divided between the workers, so the machine is not asked for more than before; the scores are the same as with one process
    (every draw is seeded by the benchmark's name and the learner seed)."""
    import concurrent.futures as cf
    import multiprocessing as mp
    total = int(os.environ.get("OMP_NUM_THREADS", "8"))
    w = min(a.workers, len(a.bench))
    per = max(1, total // w)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(per)                                  # inherited by the spawned workers before they import numpy / sklearn
    print("workers: %d processes x %d threads = %d threads (cap %d)" % (w, per, w * per, total), flush=True)
    order = sorted(a.bench, key=lambda b: -len(load(b)))          # the big benchmarks first, so the small ones fill the gaps
    out = {}
    with cf.ProcessPoolExecutor(max_workers=w, mp_context=mp.get_context("spawn")) as ex:
        futs = [ex.submit(_bench_job, (b, reps, a.learners, a.seeds, a.select, st, pca, not a.no_e_pass)) for b in order]
        for f in cf.as_completed(futs):
            b, r, k, dt = f.result()
            out[b] = (r, k)
            print("== %s done (%.0f s)" % (b, dt), flush=True)
    rows, keep = [], {}
    for b in a.bench:                                             # the order of --bench, as in a single process
        rows += out[b][0]
        keep.update(out[b][1])
    return rows, keep


def pass_mark(pb, head, learners, out, without=()):
    """The relaxed pass mark for a teacher (2026-11-17), on all usable items, per learner and over the learners:
    (1) the average difference to the reference single input is >= 0, (2) the teacher is at least the PCA baseline (average
    difference >= 0), (3) no benchmark is clearly worse than the reference: its paired-bootstrap 99.5% interval of the difference
    does not lie entirely below 0 (99.5%, because ten benchmarks at 95% give one false alarm in about 40% of cases)."""
    pb = pb[pb.rowset == "all"]
    teachers = [r for r in pb.rep.unique() if str(r).startswith("teacher_")]
    if not teachers:
        return
    h_ref = head[(head.rowset == "all") & (head.versus == "REF")]
    h_h = head[(head.rowset == "all") & (head.versus == "H")]
    has_pca = "PCA" in set(pb.rep)

    def avg(rep, l):
        x = pb[(pb.rep == rep) & ((pb.learner == l) if l != "ALL" else True)]
        return x.groupby("benchmark").primary.mean()

    rows = []
    for t in teachers:
        for l in learners + ["ALL"]:
            r_ = h_ref[(h_ref.rep == t) & (h_ref.learner == l)]
            if r_.empty:
                continue
            ref_d = float(r_.mean_diff.iloc[0])
            ref_wo = float(r_.mean_diff_wo.iloc[0]) if "mean_diff_wo" in r_.columns else np.nan
            hd = h_h[(h_h.rep == t) & (h_h.learner == l)]
            h_d = float(hd.mean_diff.iloc[0]) if len(hd) else np.nan
            dpca = (avg(t, l) - avg("PCA", l)) if has_pca else None
            pca_d = float(dpca.mean()) if has_pca else np.nan
            pca_wo = float(dpca.drop(labels=[b_ for b_ in without if b_ in dpca.index]).mean()) if has_pca else np.nan
            worse = worse_wo = np.nan
            if "worse995" in h_ref.columns:
                sel = h_ref[(h_ref.rep == t) & (h_ref.learner.isin(learners if l == "ALL" else [l]))]
                worse = float(sel.worse995.sum()) if sel.worse995.notna().any() else np.nan
                if "worse995_wo" in sel.columns and sel.worse995_wo.notna().any():
                    worse_wo = float(sel.worse995_wo.sum())
            ok = [ref_d >= 0] + ([pca_d >= 0] if np.isfinite(pca_d) else []) + ([worse == 0] if np.isfinite(worse) else [])
            ok_wo = [ref_wo >= 0] + ([pca_wo >= 0] if np.isfinite(pca_wo) else []) + ([worse_wo == 0] if np.isfinite(worse_wo) else [])
            rows.append({"teacher": t, "learner": l, "mean_diff_vs_REF": ref_d, "mean_diff_vs_PCA": pca_d, "mean_diff_vs_H": h_d,
                         "clearly_worse_benchmarks_99.5": worse, "PASS": "yes" if all(ok) else "no",
                         "vs_REF_wo": ref_wo, "vs_PCA_wo": pca_wo, "worse_wo": worse_wo, "PASS_wo": "yes" if all(ok_wo) else "no",
                         "checked": "REF" + (", PCA" if np.isfinite(pca_d) else "") + (", 99.5% intervals" if np.isfinite(worse) else "")})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "passmark.csv"), index=False)
    print("\n=== PASS MARK (teachers, all usable items) ===")
    print("(pass = average difference to the reference single input >= 0, at least the PCA baseline, no benchmark clearly worse at 99.5%)")
    tag = "_no" + "+".join(without) if without else "_wo"
    d = df.drop(columns="checked").rename(columns={"mean_diff_vs_REF": "vs_REF", "mean_diff_vs_PCA": "vs_PCA", "mean_diff_vs_H": "vs_H",
                                                  "clearly_worse_benchmarks_99.5": "worse99.5", "vs_REF_wo": "vs_REF" + tag,
                                                  "vs_PCA_wo": "vs_PCA" + tag, "worse_wo": "worse" + tag, "PASS_wo": "PASS" + tag})
    d["teacher"] = [short(t) for t in d.teacher]
    print(d.round(3).to_string(index=False))
    print("(vs_X = average difference over benchmarks to X; worse99.5 = benchmarks clearly worse than the reference; checked: %s)"
          % df["checked"].iloc[0])


def seed_spread(head, out):
    """Teacher-training noise: teachers named <base>_s1, <base>_s2, ... are repeats of one configuration; the spread of their headline
    number (average difference to the reference, and to H) is what a claimed gain has to be compared with."""
    import re
    h = head[head.rowset == "all"]
    groups = {}
    for r in h.rep.unique():
        m = re.fullmatch(r"(teacher_.+)_s(\d+)", str(r))
        if m:
            groups.setdefault(m.group(1), []).append(r)
    rows = []
    for base, members in groups.items():
        if len(members) < 2:
            continue
        for versus in ("REF", "H"):
            for l in sorted(h.learner.unique()):
                x = h[(h.versus == versus) & (h.learner == l) & h.rep.isin(members)].sort_values("rep")
                if len(x) < 2:
                    continue
                v = x.mean_diff.to_numpy()
                w_ = x.mean_diff_wo.to_numpy() if "mean_diff_wo" in x.columns else v
                rows.append({"teacher": base, "versus": versus, "learner": l, "runs": len(v), "mean": v.mean(), "sd": v.std(ddof=1),
                             "min": v.min(), "max": v.max(), "mean_wo": w_.mean(), "sd_wo": w_.std(ddof=1), "per_run": " ".join("%s %.4f" % (r.rep.rsplit("_", 1)[1], r.mean_diff) for r in x.itertuples())})
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "seed_spread.csv"), index=False)
    print("\n=== TEACHER-SEED SPREAD: the headline number (average difference over benchmarks) of repeated trainings of one configuration ===")
    d = df.drop(columns="per_run")
    d["teacher"] = [short(t) for t in d.teacher]
    print(d.round(4).to_string(index=False))
    print("(rule: one final run is acceptable if this spread is smaller than the gain you plan to claim; otherwise train 3 seeds)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", nargs="*", default=None)
    ap.add_argument("--teachers", nargs="*", default=[], help="teacher names: adds teacher_<name> to the representations")
    ap.add_argument("--bench", nargs="*", default=BENCHES)
    ap.add_argument("--learners", nargs="*", default=["linear", "gbt", "mlp"], choices=["linear", "gbt", "mlp", "kernel"])
    ap.add_argument("--seeds", type=int, default=5, help="learner seeds for gbt and mlp (linear is deterministic)")
    ap.add_argument("--select", choices=["val", "none"], default="val",
                    help="how the reference single input is chosen: on a validation split of the training data (val), or fixed to H (none)")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap draws for the intervals (0 = none); the 99.5%% interval needs >= 1000")
    ap.add_argument("--no-e-pass", action="store_true")
    ap.add_argument("--without", nargs="*", default=["PAMPA"],
                    help="benchmarks left out of the second ('without') headline, pass mark and ranks (default PAMPA: its R2 scale dwarfs the others)")
    ap.add_argument("--workers", type=int, default=1,
                    help="benchmarks run in this many processes; the thread allowance (OMP_NUM_THREADS) is divided between them (default 1)")
    ap.add_argument("--out", default="harness")
    a = ap.parse_args()
    reps = list(a.reps) if a.reps else list(DEFAULT_REPS)
    reps += ["teacher_" + t for t in a.teachers if "teacher_" + t not in reps]
    st = V.load_stats() if os.path.exists(V.STATS_FILE) else None
    pca = PB.load() if os.path.exists(PB.PCA_FILE) else None
    if "PCA" in reps and (st is None or pca is None):
        print("NOTE: PCA is skipped: run `views_v3.py --stats` and `pca_baseline.py --fit` first")
        reps.remove("PCA")
    out = os.path.join(RESULTS_DIR, a.out)
    os.makedirs(out, exist_ok=True)
    print("representations:", reps, "| learners", a.learners, "| seeds", a.seeds, "| select", a.select)
    rows, keep = [], {}
    if a.workers > 1 and len(a.bench) > 1:
        rows, keep = run_parallel(a, reps, st, pca)
    else:
        for b in a.bench:
            t0 = time.time()
            r, k = run_bench(b, reps, a.learners, a.seeds, a.select, st, pca, not a.no_e_pass)
            rows += r
            keep.update(k)
            print("== %s done (%.0f s)" % (b, time.time() - t0), flush=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(out, "raw.csv"), index=False)
    metric_cols = [c for c in raw.columns if c in ("MCC", "AUROC", "ACC", "F1", "R2", "Spearman", "Pearson", "MAE", "RMSE")]
    raw["primary"] = [row.get(PRIMARY[row["benchmark"]], np.nan) for row in raw.to_dict("records")]
    summ = raw.groupby(["benchmark", "rowset", "subset", "rep", "learner"])[["primary"] + metric_cols].agg(["mean", "std"])
    summ.to_csv(os.path.join(out, "summary.csv"))

    pbs, heads, per_in, bench_cis = [], [], [], []
    for rowset in [x for x in ("all", "has_E") if (raw.rowset == x).any()]:
        part = raw[(raw.rowset == rowset) & (raw.subset == "all")]
        seedmean = part.groupby(["benchmark", "learner", "rep"]).primary.mean().reset_index()
        ref = seedmean[seedmean.rep == "REF"].set_index(["benchmark", "learner"]).primary
        seedmean["diff"] = [r.primary - ref.get((r.benchmark, r.learner), np.nan) for r in seedmean.itertuples()]
        per_bench = seedmean[seedmean.rep != "REF"].copy()
        pbs.append(per_bench.assign(rowset=rowset))
        what = "all usable items" if rowset == "all" else "items that have E (the only items on which E can be used)"
        print("\n=== %s: primary metric, mean over learners and seeds (rows: representation; columns: benchmark) ===" % what)
        tab = per_bench.groupby(["rep", "benchmark"]).primary.mean().unstack("benchmark")
        tab = tab[[b_ for b_ in a.bench if b_ in tab.columns]]
        tab["AVERAGE"] = tab.mean(1)
        refrow = seedmean[seedmean.rep == "REF"].groupby("benchmark").primary.mean()
        sing = [s_ for s_ in SINGLES if s_ in tab.index]
        bestrow = tab.loc[sing].drop(columns="AVERAGE").max() if sing else pd.Series(dtype=float)       # best single input ON THE TEST DATA
        extra_rows = [pd.DataFrame([{**refrow.to_dict(), "AVERAGE": refrow.mean()}], index=["REF (chosen on validation)"])]
        if len(bestrow):
            extra_rows.append(pd.DataFrame([{**bestrow.to_dict(), "AVERAGE": bestrow.mean()}], index=["BEST single (on test)"]))
        full = pd.concat([tab] + extra_rows)
        full.index = [short(i) for i in full.index]
        bcols = [c for c in full.columns if c != "AVERAGE"]
        for i in range(0, len(bcols), 5):
            print(full[bcols[i:i + 5] + ["AVERAGE"]].round(3).to_string())
            print()
        hs = seedmean[seedmean.rep == "H"].set_index(["benchmark", "learner"]).primary
        per_bench["diff_H"] = [r.primary - hs.get((r.benchmark, r.learner), np.nan) for r in per_bench.itertuples()]
        best = per_bench[per_bench.rep.isin(SINGLES)].groupby(["benchmark", "learner"]).primary.max()        # best single input ON THE TEST DATA
        per_bench["diff_BEST"] = [r.primary - best.get((r.benchmark, r.learner), np.nan) for r in per_bench.itertuples()]
        without = [b_ for b_ in a.without if b_ in set(per_bench.benchmark)]
        head, cis = [], []
        for versus in ("REF", "H", "BEST"):
            if versus == "H" and "H" not in set(per_bench.rep):
                continue
            col = {"REF": "diff", "H": "diff_H", "BEST": "diff_BEST"}[versus]
            for lname in a.learners + ["ALL"]:
                for r in [x for x in reps if x in set(per_bench.rep) and x != versus]:
                    pb = per_bench[(per_bench.rep == r) & ((per_bench.learner == lname) if lname != "ALL" else True)]
                    d = pb.groupby("benchmark")[col].mean()
                    d_wo = d.drop(labels=[b_ for b_ in without if b_ in d.index])
                    row = {"rowset": rowset, "versus": versus, "learner": lname, "rep": r, "benchmarks": len(d), "mean_diff": d.mean(),
                           "mean_diff_wo": d_wo.mean()}
                    if a.boot and lname != "ALL" and versus != "BEST":
                        bd = boot_diffs(keep, r, lname, a.boot, rowset, 0, versus)
                        if bd:
                            avg = np.mean(np.stack(list(bd.values())), axis=0)
                            pc = lambda v, q: float(np.percentile(v, q))                                    # noqa: E731
                            row.update(ci_lo=pc(avg, 2.5), ci_hi=pc(avg, 97.5), ci995_lo=pc(avg, 0.25), ci995_hi=pc(avg, 99.75),
                                       wins=sum(pc(v, 2.5) > 0 for v in bd.values()),
                                       losses=sum(pc(v, 97.5) < 0 for v in bd.values()),
                                       ties=sum(pc(v, 2.5) <= 0 <= pc(v, 97.5) for v in bd.values()),
                                       better995=sum(pc(v, 0.25) > 0 for v in bd.values()),
                                       worse995=sum(pc(v, 99.75) < 0 for v in bd.values()))
                            kw = {b_: v for b_, v in bd.items() if b_ not in without}                         # the same without the left-out benchmarks
                            if kw:
                                avg_wo = np.mean(np.stack(list(kw.values())), axis=0)
                                row.update(ci_lo_wo=pc(avg_wo, 2.5), ci_hi_wo=pc(avg_wo, 97.5),
                                           worse995_wo=sum(pc(v, 99.75) < 0 for v in kw.values()))
                            for b_, v in bd.items():
                                cis.append({"rowset": rowset, "versus": versus, "learner": lname, "rep": r, "benchmark": b_,
                                            "diff": float(d.get(b_, np.nan)), "lo95": pc(v, 2.5), "hi95": pc(v, 97.5),
                                            "lo995": pc(v, 0.25), "hi995": pc(v, 99.75)})
                    head.append(row)
        head = pd.DataFrame(head)
        heads.append(head)
        bench_cis.append(pd.DataFrame(cis))
        print("\n=== HEADLINE (%s): average over benchmarks of (score - score of the comparison) ===" % rowset)
        wo = "no" + "+".join(without) if without else "wo"
        print("(versus REF = best single input among PepLand / PeptideCLM-2 / ECFP, chosen per benchmark on validation data;")
        print(" versus H = ECFP alone; versus BEST = the best single input ON THE TEST DATA, a strict bar without interval;")
        print(" diff_%s = the same average without %s; interval = paired bootstrap over test items;" % (wo, " and ".join(without) or "nothing"))
        print(" W/L/T = benchmarks above / below / around 0 at 95%; worse99.5 = benchmarks entirely below 0 at 99.5%; ALL = mean over learners)")
        h_ = head[~head.rep.isin(SINGLES)]
        if len(a.learners) == 1:
            h_ = h_[h_.learner != "ALL"]
        nan = pd.Series(np.nan, index=h_.index)
        gc = lambda k: h_[k] if k in h_.columns else nan                                           # noqa: E731
        print(pd.DataFrame({
            "versus": h_.versus, "learner": h_.learner, "rep": [short(r) for r in h_.rep], "n": h_.benchmarks, "diff": h_.mean_diff.round(3),
            "95% interval": ["" if pd.isna(x) else "[%.3f, %.3f]" % (x, y) for x, y in zip(gc("ci_lo"), gc("ci_hi"))],
            "diff_" + wo: h_.mean_diff_wo.round(3),
            "interval_" + wo: ["" if pd.isna(x) else "[%.3f, %.3f]" % (x, y) for x, y in zip(gc("ci_lo_wo"), gc("ci_hi_wo"))],
            "W/L/T": ["" if pd.isna(w) else "%d/%d/%d" % (w, l, t) for w, l, t in zip(gc("wins"), gc("losses"), gc("ties"))],
            "worse99.5": ["" if pd.isna(x) else "%d" % x for x in gc("worse995")]}).to_string(index=False))
        print("(the single inputs are left out here; headline.csv keeps every row and the 99.5% intervals)")
        # average rank among the inputs, scale-free: no benchmark can dominate because its metric has a wide range
        src = tab.drop(columns="AVERAGE")
        ranks = src.rank(ascending=False)
        rk = pd.DataFrame({"avg_rank": ranks.mean(1)})
        if without and len(src.columns) > len(without):
            rk["avg_rank_" + wo] = ranks[[c for c in src.columns if c not in without]].mean(1)
        rk = rk.sort_values("avg_rank")
        rk.index = [short(i) for i in rk.index]
        print("\n=== AVERAGE RANK of the %d inputs over the benchmarks (1 = best; lower is better; scale-free) ===" % len(src))
        print(rk.round(1).to_string())
        rk.assign(rowset=rowset).to_csv(os.path.join(out, "ranks_%s.csv" % rowset))
        for s_ in SINGLES:
            pi = per_bench[per_bench.rep == s_].set_index(["benchmark", "learner"]).primary
            for r in [x for x in reps if x in set(per_bench.rep)]:
                pr = per_bench[per_bench.rep == r].set_index(["benchmark", "learner"]).primary
                per_in.append(pd.DataFrame({"rowset": rowset, "rep": r, "versus": s_, "diff": (pr - pi)}).reset_index())
    if pbs:
        pd.concat(pbs).to_csv(os.path.join(out, "per_bench.csv"), index=False)
        pd.concat(heads).to_csv(os.path.join(out, "headline.csv"), index=False)
        if any(len(c) for c in bench_cis):
            pd.concat(bench_cis).to_csv(os.path.join(out, "per_bench_ci.csv"), index=False)
        pass_mark(pd.concat(pbs), pd.concat(heads), a.learners, out, [b_ for b_ in a.without if b_ in set(pd.concat(pbs).benchmark)])
        seed_spread(pd.concat(heads), out)
    if per_in:
        pd.concat(per_in).to_csv(os.path.join(out, "per_input.csv"), index=False)
    ty = raw[(raw.rowset == "all") & raw.subset.isin(["standard", "modified"]) & (raw.rep != "REF")]
    if len(ty):
        t = ty.groupby(["benchmark", "subset", "rep"]).primary.mean().unstack("rep").round(3)
        t.to_csv(os.path.join(out, "by_type.csv"))
        print("\n=== standard (E present) against modified items, primary metric, mean over learners and seeds ===")
        t.columns = [short(c) for c in t.columns]
        show(t, 6)
    print("\nwrote", out)

if __name__ == "__main__":
    main()

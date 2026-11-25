"""Kernel learners for the evaluation harness (learner name "kernel"): an RBF kernel SVM for classification and RBF kernel ridge
regression for regression. Same inputs, same splits and same metrics as linear / gbt / mlp (eval_single.learners wires them in).

    python src/teacher/eval_harness.py --teachers v3_base --learners kernel --reps P L H P+L+H PCA --no-e-pass --out final_kernel

What they do (all choices are made INSIDE the training data of each fit; the test fold is never used):
  * features: standardised with the training rows' mean and sd, clipped at +-10 (a rare ECFP bit with a tiny sd would otherwise dominate
    every distance); the kernel is exp(-gamma * squared distance)
  * gamma: m / median(squared distance between training points) (the "median heuristic"), m picked from GAMMA_MULT
  * classification (KernelSVMCV): C-support-vector machine on the precomputed kernel; (m, C) from 3 x 3 values picked by the mean AUROC of a
    stratified 3-fold CV inside the training data; the output is sigmoid(decision value), so MCC uses the SVM's own rule (decision >= 0)
    and AUROC / bootstrap work as for the other learners. Plain SVM: no class weights, as the other learners.
  * regression (KernelRidgeCV): kernel ridge on the z-scored target; (m, lambda) from 3 x 11 values picked by the EXACT leave-one-out error,
    which needs one eigendecomposition per m (closed form: e_i = alpha_i / [(K + lambda I)^-1]_ii). This is the posterior mean of a
    Gaussian process with the same kernel and noise level, so a Gaussian process would give the same predictions; we skip its marginal-
    likelihood search because the leave-one-out search is cheaper and measures prediction error directly.
  * large training sets: the search runs on a random subsample of at most TUNE_ROWS training rows (env KERNEL_TUNE_ROWS, default 4000),
    the final model is then fitted on ALL training rows with the chosen values.
Deterministic (no random seeds): the harness runs it once, like linear.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

GAMMA_MULT = (0.3, 1.0, 3.0)                                   # times the median heuristic
SVM_C = (0.5, 5.0, 50.0)
KRR_LAMBDA = tuple(float(x) for x in np.logspace(-3, 2, 11))   # ridge on a kernel with unit diagonal
TUNE_ROWS = int(os.environ.get("KERNEL_TUNE_ROWS", "4000"))
CLIP = 10.0
N_JOBS = int(os.environ.get("OMP_NUM_THREADS", "8"))


def _prep(X, scaler=None):
    X = np.asarray(X, np.float64)
    if scaler is None:
        scaler = StandardScaler().fit(X)
    return np.clip(scaler.transform(X), -CLIP, CLIP), scaler


def _sqdist(A, B):
    d = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * (A @ B.T)
    np.maximum(d, 0.0, out=d)
    return d


def _median_sqdist(Z, rng, m=1500):
    idx = rng.choice(len(Z), min(m, len(Z)), replace=False)
    d = _sqdist(Z[idx], Z[idx])
    return float(np.median(d[np.triu_indices(len(idx), 1)])) + 1e-12


def _subsample(n, y, rng, rows, stratify):
    """Indices of at most `rows` training rows for the hyper-parameter search (all rows if n <= rows)."""
    if n <= rows:
        return np.arange(n)
    if not stratify:
        return np.sort(rng.choice(n, rows, replace=False))
    pick = []
    for c in np.unique(y):
        ic = np.flatnonzero(y == c)
        pick.append(rng.choice(ic, max(3, int(round(rows * len(ic) / n))), replace=False) if len(ic) > 3 else ic)
    return np.sort(np.concatenate(pick))


class KernelRidgeCV:
    """RBF kernel ridge regression; (gamma multiplier, lambda) by exact leave-one-out inside the training data."""

    def __init__(self, gamma_mult=GAMMA_MULT, lambdas=KRR_LAMBDA, tune_rows=None):
        self.gamma_mult, self.lambdas = tuple(gamma_mult), tuple(lambdas)
        self.tune_rows = TUNE_ROWS if tune_rows is None else tune_rows

    def fit(self, X, y):
        y = np.asarray(y, np.float64)
        Z, self.scaler_ = _prep(X)
        self.mu_, self.sd_ = float(y.mean()), float(y.std() + 1e-12)
        t = (y - self.mu_) / self.sd_
        n = len(Z)
        rng = np.random.default_rng(0)
        self.g0_ = 1.0 / _median_sqdist(Z, rng)
        sub = _subsample(n, y, rng, self.tune_rows, False)
        D2 = _sqdist(Z[sub], Z[sub])
        ts = t[sub]
        best = (np.inf, None, None)
        self.table_ = []
        for m in self.gamma_mult:
            w, U = np.linalg.eigh(np.exp(-(m * self.g0_) * D2))
            w = np.maximum(w, 0.0)
            c, U2 = U.T @ ts, U * U
            for lam in self.lambdas:
                inv = 1.0 / (w + lam)
                alpha = U @ (c * inv)                              # (K + lambda I)^-1 t
                mse = float(np.mean((alpha / (U2 @ inv)) ** 2))    # leave-one-out residual = alpha_i / S_ii
                self.table_.append((m, lam, mse))
                if mse < best[0]:
                    best = (mse, m, lam)
        del D2
        _, self.mult_, self.lam_ = best
        self.gamma_ = self.mult_ * self.g0_
        K = np.exp(-self.gamma_ * _sqdist(Z, Z))
        K[np.diag_indices(n)] += self.lam_
        self.alpha_ = cho_solve(cho_factor(K, lower=True, check_finite=False), t, check_finite=False)
        self.Z_ = Z
        return self

    def predict(self, X):
        Zt, _ = _prep(X, self.scaler_)
        K = np.exp(-self.gamma_ * _sqdist(Zt, self.Z_))
        return K @ self.alpha_ * self.sd_ + self.mu_


class KernelSVMCV:
    """RBF kernel support-vector classifier; (gamma multiplier, C) by stratified 3-fold CV (AUROC) inside the training data."""

    def __init__(self, gamma_mult=GAMMA_MULT, cs=SVM_C, tune_rows=None):
        self.gamma_mult, self.cs = tuple(gamma_mult), tuple(cs)
        self.tune_rows = TUNE_ROWS if tune_rows is None else tune_rows

    def fit(self, X, y):
        y = np.asarray(y).astype(int)
        Z, self.scaler_ = _prep(X)
        n = len(Z)
        rng = np.random.default_rng(0)
        self.g0_ = 1.0 / _median_sqdist(Z, rng)
        sub = _subsample(n, y, rng, self.tune_rows, True)
        Zs, ys = Z[sub], y[sub]
        D2 = _sqdist(Zs, Zs)
        folds = list(StratifiedKFold(3, shuffle=True, random_state=0).split(Zs, ys))
        from joblib import Parallel, delayed
        self.table_, best = [], (-np.inf, 1.0, 5.0)
        for m in self.gamma_mult:
            K = np.exp(-(m * self.g0_) * D2)

            def one(tr, va, C, K=K):
                try:
                    svc = SVC(C=C, kernel="precomputed", cache_size=500).fit(K[np.ix_(tr, tr)], ys[tr])
                    return roc_auc_score(ys[va], svc.decision_function(K[np.ix_(va, tr)]))
                except Exception:
                    return np.nan

            res = Parallel(n_jobs=max(1, min(N_JOBS, len(folds) * len(self.cs))), prefer="threads")(
                delayed(one)(tr, va, C) for C in self.cs for tr, va in folds)
            res = np.array(res, np.float64).reshape(len(self.cs), len(folds))
            for ci, C in enumerate(self.cs):
                s = float(np.nanmean(res[ci])) if np.isfinite(res[ci]).any() else np.nan
                self.table_.append((m, C, s))
                if np.isfinite(s) and s > best[0]:
                    best = (s, m, C)
        del D2
        _, self.mult_, self.C_ = best
        self.gamma_ = self.mult_ * self.g0_
        self.svc_ = SVC(C=self.C_, kernel="precomputed", cache_size=500).fit(np.exp(-self.gamma_ * _sqdist(Z, Z)), y)
        self.Z_ = Z
        return self

    def decision_function(self, X):
        Zt, _ = _prep(X, self.scaler_)
        return self.svc_.decision_function(np.exp(-self.gamma_ * _sqdist(Zt, self.Z_)))

    def predict_proba(self, X):
        p = 1.0 / (1.0 + np.exp(-self.decision_function(X)))        # sigmoid: 0.5 <=> decision value 0 (the SVM's own class rule)
        return np.column_stack([1.0 - p, p])

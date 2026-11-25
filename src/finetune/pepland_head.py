"""PepLand's own downstream recipe, on cached PepLand embeddings (and, for an equal-treatment comparison,
the same head on other cached views).

    python src/finetune/pepland_head.py --bench CellPPD THPep PepMSND PAMPA AmpHGT --seeds 3
    python src/finetune/pepland_head.py --feat pepland_suite --bench PL_cCPP PL_cSol PL_ncCPP PL_cBinding PL_ncBinding
    python src/finetune/pepland_head.py --reps pepland pclm ecfp --bench CellPPD        # same head on three views

What PepLand's authors release for downstream use is `PropertyPredictor` (external/pepland/model/core.py):
the pretrained encoder is run under torch.no_grad() (so it is frozen whatever `freeze` says), the node
embeddings are pooled (avg by default; the paper reports a trainable GRU pooling as best), then
    Linear(300,256) ReLU Dropout(0.1) Linear(256,128) ReLU Dropout(0.1) Linear(128,1) Sigmoid.
The paper (arXiv 2311.04419, "Experimental setup") trains it with batch 32, 100 epochs (200 for the affinity
tasks), lr 1e-5 for the CPP permeability task and 5e-5 for the others. Classification: BCE; regression:
MSE on targets min-max scaled to [0,1] (the output is a sigmoid). Here:
  * the pooled embedding is the avg-pooled one already cached by featurize_pepland.py (GRU pooling needs
    node-level embeddings and is NOT done: a limitation, the paper found GRU best);
  * 10% of each training set is held out, the epoch with the best validation loss is kept (the paper shows
    validation curves, the selection rule is not stated); the test fold is never used for any choice;
  * inputs are z-scored with training statistics (needed for ECFP / PeptideCLM-2 vectors, harmless for PepLand's);
  * splits: the benchmarks' own (benchmarks.load); PepLand ships none, so PL_* use our fixed 5-fold protocol.
Results: $RESULTS_DIR/<out>/raw.csv and printed tables (mean over seeds).
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import benchmarks as BM  # noqa: E402
import eval_single as E  # noqa: E402
from paths import RESULTS_DIR  # noqa: E402

LOW_LR = ("PL_ncCPP", "PAMPA")                 # the paper's lr 1e-5 "CPP permeability" task (CycPeptMPDB)
LONG = ("PL_cBinding", "PL_ncBinding")         # the paper's 200-epoch affinity tasks


def mlp(d):
    return nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.ReLU(),
                         nn.Dropout(0.1), nn.Linear(128, 1))


def fit_predict(X, y, tr, te, task, bench, seed, a, dev):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    perm = rng.permutation(tr)
    nv = max(8, int(0.1 * len(tr)))
    va, fi = perm[:nv], perm[nv:]
    mu, sd = X[fi].mean(0), X[fi].std(0)
    sd = np.maximum(sd, 0.1 * np.sqrt((sd ** 2).mean()) + 1e-6)        # rare bits / near-constant dims are not blown up
    Xt = torch.from_numpy(np.clip((X - mu) / sd, -10, 10).astype(np.float32)).to(dev)
    clf = task == "clf"
    lo, hi = (0.0, 1.0) if clf else (float(y[fi].min()), float(y[fi].max()))
    yt = torch.from_numpy(((y - lo) / (hi - lo + 1e-8)).astype(np.float32)).to(dev)
    net = mlp(X.shape[1]).to(dev)
    lr = 1e-5 if bench in LOW_LR else 5e-5
    epochs = a.epochs or (200 if bench in LONG else 100)
    opt = torch.optim.Adam(net.parameters(), lr=lr)

    def loss(rows, train):
        out = net(Xt[rows]).squeeze(1)
        return F.binary_cross_entropy_with_logits(out, yt[rows]) if clf else F.mse_loss(torch.sigmoid(out), yt[rows])

    best, state = float("inf"), None
    va_t = torch.from_numpy(va).to(dev)
    for ep in range(epochs):
        net.train()
        order = torch.from_numpy(fi[rng.permutation(len(fi))]).to(dev)
        for i in range(0, len(order), a.batch):
            r = order[i:i + a.batch]
            if len(r) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss(r, True).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(loss(va_t, False))
        if v < best:
            best, state = v, copy.deepcopy(net.state_dict())
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        o = net(Xt[torch.from_numpy(te).to(dev)]).squeeze(1)
        p = torch.sigmoid(o).cpu().numpy()
    return p if clf else p * (hi - lo) + lo, epochs


def run(bench, rep, reps, a, dev):
    df = BM.load(bench).dropna(subset=["smiles"]).reset_index(drop=True)
    df["smiles"] = df.smiles.astype(str)
    task = BM.TASK[bench]
    idx, X, good = reps[rep]
    ii = np.array([idx[s] for s in df.smiles])
    ok = good[ii]
    df, X = df[ok].reset_index(drop=True), X[ii][ok].astype(np.float32)
    cols = [c for c in df.columns if c.startswith("extra_")]
    if cols:
        X = np.hstack([X, pd.get_dummies(df[cols].astype(str)).to_numpy(np.float32)])
    y = df.y.to_numpy(np.float32)
    cv = pd.api.types.is_integer_dtype(df.split)
    folds = sorted(df.split.unique()) if cv else ["test"]
    rows = []
    for seed in range(a.seeds):
        per, ep = [], 0
        for f in folds:
            te = np.flatnonzero((df.split == f).to_numpy() if cv else (df.split == "test").to_numpy())
            tr = np.setdiff1d(np.arange(len(df)), te)
            p, ep = fit_predict(X, y, tr, te, task, bench, seed, a, dev)
            per.append(E.score(task, y[te], p))
        rows.append({"benchmark": bench, "rep": rep, "seed": seed, "n": len(df), "coverage": round(ok.mean(), 4),
                     "epochs": ep, **{k: float(np.mean([s[k] for s in per])) for k in per[0]}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", nargs="*", default=None)
    ap.add_argument("--reps", nargs="*", default=["pepland"], help="cached views to run the head on")
    ap.add_argument("--feat", default=None, help="features subfolder under features/ (pepland_suite for PL_*)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=0, help="override the paper's 100 / 200")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.feat:
        E.FEAT = os.path.join(os.path.dirname(E.FEAT), a.feat)
    reps = E.load_reps()
    for r in a.reps:
        assert r in reps, "no cached features for '%s' in %s (have: %s)" % (r, E.FEAT, ", ".join(reps))
    benches = a.bench or BM.selected()
    out = os.path.join(RESULTS_DIR, a.out or ("pepland_head" + ("_" + a.feat if a.feat else "")))
    os.makedirs(out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for b in benches:
        for r in a.reps:
            t0 = time.time()
            rows += run(b, r, reps, a, dev)
            key = "MCC" if BM.TASK[b] == "clf" else "R2"
            print("%-13s %-8s done (%.0fs) | %s %.3f" % (b, r, time.time() - t0, key, rows[-1][key]), flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(out, "raw.csv"), index=False)
    raw = pd.DataFrame(rows)
    raw["headline"] = np.where(raw.benchmark.map(BM.TASK) == "clf", raw.get("MCC"), raw.get("R2"))
    print("\n=== PepLand-style head: MCC (classification) / R2 (regression), mean over seeds ===")
    print(raw.groupby(["benchmark", "rep"]).headline.mean().unstack("rep").round(3).to_string())
    for col, label in (("AUROC", "AUROC"), ("ACC", "accuracy"), ("F1", "F1"), ("Pearson", "Pearson r"), ("Spearman", "Spearman")):
        if col in raw and raw[col].notna().any():
            print("\n=== %s ===" % label)
            print(raw.dropna(subset=[col]).groupby(["benchmark", "rep"])[col].mean().unstack("rep").round(3).to_string())
    print("\nwrote", out)


if __name__ == "__main__":
    main()

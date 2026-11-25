"""Adapter vectors on a FROZEN teacher v3, trained per benchmark together with an MLP learner (GPU).

    python src/teacher/adapter_mlp_v3.py --name v3_base                       # PeptideCLM-2's five benchmarks, adapters + the control, 3 seeds
    python src/teacher/adapter_mlp_v3.py --name v3_base --suite both          # ... and PepLand's five tasks
    python src/teacher/adapter_mlp_v3.py --name v3_base --bench CellPPD --seeds 1 --modes adapt     # a quick look

The idea (2026-11-19): for every input view the teacher makes a 512-d vector (encoder + LayerNorm). Here each of the K views gets an ADAPTER
vector a_k (512-d, starts at all ones, so at the start the teacher is exactly the frozen v3 teacher) that multiplies that view's 512-d vector
element by element; the K adapted vectors are concatenated, go through the FROZEN trunk to the 1024-d latent (standardised with the run's
latent statistics as in the harness), and an MLP learner is trained on that latent. The adapters exist only during benchmark testing: they are
learned on the training part of each benchmark together with the MLP (same loss), and thrown away. Everything else (the encoders, the norms,
the trunk) is frozen; the encoder outputs are computed once per benchmark.

The MLP is the harness MLP (eval_single.learners "mlp") re-written in PyTorch so that gradients can reach the adapters, and run on the GPU:
one hidden layer of 512 ReLU units, Glorot-uniform initialisation, Adam (lr 1e-3, batches of min(200, n)), L2 penalty alpha = 1e-3 as in
scikit-learn, 10% of the training rows held out (stratified for classification) for early stopping on accuracy (classification) or R2
(regression), patience 10, at most 300 epochs, best epoch kept; inputs standardised with the training rows' mean / sd of the UNADAPTED latent;
regression targets z-scored. One model per outer fold and seed (as the harness). tests/ in the docstring of mlp_epoch: one epoch equals
scikit-learn's to float precision.

Modes (all with the same recipe, splits and seeds):
    adapt   the adapters + the MLP trained together                              <- the experiment
    plain   the same PyTorch MLP on the frozen latent, no adapters               <- the control that isolates the adapters
(the scikit-learn MLP numbers of the harness, results/final_mlp, are printed next to them when that folder exists)
Output: $RESULTS_DIR/adapter_<name>[_pepland]/raw.csv; adapt_dev_<view> = mean |a - 1| of the learned adapters (how far they moved).
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import benchmarks as BM  # noqa: E402
import eval_single as E  # noqa: E402
import finetune_v3 as FV  # noqa: E402
import train_teacher_v3 as V3  # noqa: E402
from paths import FEAT_DIR, RESULTS_DIR  # noqa: E402

PRIMARY = {"CellPPD": "MCC", "THPep": "MCC", "PepMSND": "MCC", "PAMPA": "R2", "AmpHGT": "MCC", "PL_cCPP": "AUROC", "PL_cSol": "AUROC",
           "PL_ncCPP": "Spearman", "PL_cBinding": "Spearman", "PL_ncBinding": "Spearman"}
ALPHA, HIDDEN, LR, MAX_EPOCHS, PATIENCE, TOL, VAL_FRAC = 1e-3, 512, 1e-3, 300, 10, 1e-4, 0.1


# ------------------------------------------------------------------------------------------- the MLP (scikit-learn's, in PyTorch)
class MLP(nn.Module):
    """d_in -> 512 ReLU -> 1; weights AND biases Glorot-uniform (bound sqrt(6 / (fan_in + fan_out))), as scikit-learn."""

    def __init__(self, d_in, hidden=HIDDEN):
        super().__init__()
        self.l1, self.l2 = nn.Linear(d_in, hidden), nn.Linear(hidden, 1)
        for lin in (self.l1, self.l2):
            b = math.sqrt(6.0 / (lin.in_features + lin.out_features))
            nn.init.uniform_(lin.weight, -b, b)
            nn.init.uniform_(lin.bias, -b, b)

    def forward(self, x):
        return self.l2(F.relu(self.l1(x))).squeeze(1)

    def l2_sum(self):
        return (self.l1.weight ** 2).sum() + (self.l2.weight ** 2).sum()


def mlp_loss(out, y, clf, net, n_batch):
    """scikit-learn's loss: mean log-loss (clf) or half the mean squared error (reg), plus 0.5 * alpha / n_batch * sum of squared weights."""
    base = F.binary_cross_entropy_with_logits(out, y) if clf else 0.5 * ((out - y) ** 2).mean()
    return base + 0.5 * ALPHA / n_batch * net.l2_sum()


def mlp_epoch(net, opt, feats, y, clf, order, batch):
    """One epoch over the rows `order` in that order, batches of `batch` (the last one may be shorter). feats(idx) -> [len(idx), d] input.
    Test (laptop): with shuffle off and the same initial weights, the weights after one epoch equal scikit-learn MLPClassifier /
    MLPRegressor (solver adam, alpha 1e-3, batch_size = batch) after max_iter=1 to about 1e-6."""
    net.train()
    for i in range(0, len(order), batch):
        b = order[i:i + batch]                                   # a LongTensor on the device
        loss = mlp_loss(net(feats(b)), y[b], clf, net, len(b))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()


@torch.no_grad()
def predict(net, feats, idx, chunk=4096):
    net.eval()
    return torch.cat([net(feats(idx[i:i + chunk])) for i in range(0, len(idx), chunk)])


def val_score(out, y, clf):
    """scikit-learn's early-stopping score: accuracy (MLPClassifier) or R2 (MLPRegressor)."""
    if clf:
        return float(((out >= 0).float() == y).float().mean())
    return float(1.0 - ((out - y) ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-12))


def split_fit_val(tr, y, clf, seed):
    """10% of the training rows for early stopping (stratified for classification), as scikit-learn's train_test_split."""
    rng = np.random.RandomState(seed)
    if clf:
        val = []
        for c in np.unique(y[tr]):
            ic = tr[y[tr] == c]
            val.append(rng.choice(ic, max(1, int(round(VAL_FRAC * len(ic)))), replace=False))
        val = np.concatenate(val)
    else:
        val = rng.choice(tr, max(1, int(round(VAL_FRAC * len(tr)))), replace=False)
    return np.setdiff1d(tr, val), np.sort(val)


# ------------------------------------------------------------------------------------------- one fit
class Frozen:
    """The frozen part of the teacher for one benchmark: the per-view vectors H [N, K, 512], the trunk and the latent standardisation."""

    def __init__(self, model0, lat, xs, dev, chunk=4096):
        self.model, self.dev = model0, dev
        self.mu = torch.as_tensor(np.asarray(lat["mu"], np.float32), device=dev) if lat is not None else None
        self.sd = torch.as_tensor(np.asarray(lat["sd"], np.float32), device=dev) if lat is not None else None
        with torch.no_grad():
            self.H = torch.cat([torch.stack(model0.encode_views([x[i:i + chunk] for x in xs]), 1) for i in range(0, len(xs[0]), chunk)])
            self.Z0 = torch.cat([self.latent(self.H[i:i + chunk], None) for i in range(0, len(self.H), chunk)])    # latent with no adapters

    def latent(self, h, adapters):
        """h [B, K, 512] (+ adapters [K, 512]) -> standardised latent [B, 1024] through the frozen trunk (gradients flow through it)."""
        if adapters is not None:
            h = h * adapters
        z = self.model.trunk(h.flatten(1))                    # concatenation of the K view vectors, as in v3_base
        return z if self.mu is None else (z - self.mu) / self.sd


def fit_fold(mode, fr, extra, y, ynp, tr, te, clf, seed, a):
    """-> (predictions on te [len(te)], info). mode: adapt | plain. y: float tensor on the device, ynp: the same as numpy; tr, te: numpy row indices."""
    dev = fr.dev
    torch.manual_seed(seed)
    fit, val = split_fit_val(tr, ynp, clf, seed)
    T = lambda idx: torch.as_tensor(np.asarray(idx), dtype=torch.long, device=dev)       # noqa: E731
    tr_t, fit_t, val_t, te_t = T(tr), T(fit), T(val), T(te)
    ex = extra
    if ex.shape[1]:
        ex = (ex - ex[tr_t].mean(0)) / (ex[tr_t].std(0) + 1e-6)
    f0 = torch.cat([fr.Z0, ex], 1)                           # the unadapted features; the scaler is fitted on ALL training rows, as the pipeline
    mean, sd = f0[tr_t].mean(0), f0[tr_t].std(0, unbiased=False)
    sd = torch.where(sd < 1e-8, torch.ones_like(sd), sd)
    if clf:
        yt, mu_y, sd_y = y, 0.0, 1.0
    else:
        mu_y, sd_y = float(y[tr_t].mean()), float(y[tr_t].std(unbiased=False) + 1e-8)
        yt = (y - mu_y) / sd_y
    adapters = nn.Parameter(torch.ones(fr.H.shape[1], fr.H.shape[2], device=dev)) if mode == "adapt" else None
    net = MLP(f0.shape[1]).to(dev)
    if mode == "adapt":
        def feats(idx):
            return (torch.cat([fr.latent(fr.H[idx], adapters), ex[idx]], 1) - mean) / sd
        opt = torch.optim.Adam([{"params": list(net.parameters()), "lr": LR}, {"params": [adapters], "lr": a.adapter_lr}])
    else:
        x0 = (f0 - mean) / sd

        def feats(idx):
            return x0[idx]
        opt = torch.optim.Adam(net.parameters(), lr=LR)
    batch = min(200, len(fit))
    rng = np.random.RandomState(seed + 12345)
    best, best_state, bad, ep = -np.inf, None, 0, 0
    for ep in range(1, MAX_EPOCHS + 1):
        mlp_epoch(net, opt, feats, yt, clf, T(rng.permutation(fit)), batch)
        sc = val_score(predict(net, feats, val_t), yt[val_t], clf)
        bad = bad + 1 if sc < best + TOL else 0                      # scikit-learn: no improvement of at least tol
        if sc > best:
            best = sc
            best_state = ({k: v.detach().clone() for k, v in net.state_dict().items()}, None if adapters is None else adapters.detach().clone())
        if bad > PATIENCE:
            break
    net.load_state_dict(best_state[0])
    if adapters is not None:
        adapters.data.copy_(best_state[1])
    out = predict(net, feats, te_t)
    p = torch.sigmoid(out) if clf else out * sd_y + mu_y
    info = {"epochs": ep, "val_score": best}
    if adapters is not None:
        dev_a = (adapters.detach() - 1.0).abs().mean(1).cpu().numpy()
        info.update({"adapt_dev_%d" % k: float(d) for k, d in enumerate(dev_a)})
    return p.cpu().numpy(), info


# ------------------------------------------------------------------------------------------- benchmark loop
def run_bench(bench, mode, fr, data, a, dev, inputs):
    df = data["df"]
    task = E.TASK[bench]
    clf = task == "clf"
    y = torch.as_tensor(df.y.to_numpy().astype(np.float32), device=dev)
    ynp = df.y.to_numpy().astype(np.float32)
    cv = pd.api.types.is_integer_dtype(df.split)
    folds = sorted(df.split.unique()) if cv else ["test"]
    rows = []
    for seed in range(a.seeds):
        per, infos, allp = [], [], np.full(len(df), np.nan)
        for fi, fo in enumerate(folds):
            te = np.flatnonzero((df.split == fo).to_numpy())
            tr = np.flatnonzero(~(df.split == fo).to_numpy())
            p, info = fit_fold(mode, fr, data["extra"], y, ynp, tr, te, clf, seed * 1000 + fi, a)
            allp[te] = p
            per.append(E.score(task, ynp[te], p))
            infos.append(info)
        m = ~np.isnan(allp)
        row = {"benchmark": bench, "mode": mode, "seed": seed, "n": len(df),
               **{k: float(np.mean([s[k] for s in per])) for k in per[0]}}
        if cv:
            row.update({"pooled_" + k: float(v) for k, v in E.score(task, ynp[m], allp[m]).items()})
        row["epochs"] = float(np.mean([i["epochs"] for i in infos]))
        for k in range(len(inputs)):
            if "adapt_dev_%d" % k in infos[0]:
                row["adapt_dev_" + V3.LABEL[inputs[k]]] = float(np.mean([i["adapt_dev_%d" % k] for i in infos]))
        rows.append(row)
    return rows


def primary(row):
    """The harness's primary score: the pooled score for cluster-CV / CV benchmarks, the test-set score for fixed splits."""
    m = PRIMARY[row["benchmark"]]
    v = row.get("pooled_" + m, np.nan)
    return row.get(m, np.nan) if pd.isna(v) else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v3_base", help="the teacher run (checkpoints/teacher/<name>); it stays frozen")
    ap.add_argument("--suite", choices=("main", "pepland", "both"), default="main")
    ap.add_argument("--bench", nargs="*", default=None)
    ap.add_argument("--modes", nargs="*", default=["plain", "adapt"], choices=("plain", "adapt"))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--adapter-lr", type=float, default=1e-2,
                    help="learning rate of the adapter vectors (the MLP itself uses 1e-3 as scikit-learn). Early stopping ends a fit after ~100-1000 steps, so 1e-3 would "
                         "let an adapter move by at most ~0.1-1; run 1e-3 / 1e-2 / 3e-2 as a sensitivity check and report all, do not pick the best on the test data")
    ap.add_argument("--out", default=None)
    ap.add_argument("--use-last", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = False             # exact float32: the adapters and the scikit-learn comparison need no shortcut
    model0, st, lat = V3.load_model(a.name, dev, a.use_last)
    for p in model0.parameters():
        p.requires_grad_(False)
    model0.eval()
    inputs = list(model0.inputs)
    print("teacher %s (frozen): inputs %s, fusion %s, latent %d-d | device %s | modes %s | seeds %d | adapters %d x %d, lr %g"
          % (a.name, [V3.LABEL[v] for v in inputs], model0.fuse, model0.cfg["latent"], dev, a.modes, a.seeds, len(inputs),
             model0.cfg["enc_dim"], a.adapter_lr), flush=True)
    assert model0.fuse == "concat", "the adapters multiply the vectors that are concatenated: a concat teacher is needed"
    suites = ["main", "pepland"] if a.suite == "both" else [a.suite]
    for suite in suites:
        pl = suite == "pepland"
        feat_dir = os.path.join(os.path.dirname(FEAT_DIR), "pepland_suite") if pl else FEAT_DIR
        raw, esm = FV.load_raw(feat_dir, inputs)
        benches = a.bench or (list(BM.PL_SUITE) if pl else E.selected())
        out = os.path.join(RESULTS_DIR, (a.out or "adapter_%s" % a.name) + ("_pepland" if pl else ""))      # one folder per suite
        os.makedirs(out, exist_ok=True)
        rows = []
        for b in benches:
            data = FV.bench_data(b, raw, esm, inputs, st, dev, False)
            t0 = time.time()
            fr = Frozen(model0, lat, data["xs"], dev)
            for m in a.modes:
                t1 = time.time()
                new = run_bench(b, m, fr, data, a, dev, inputs)
                rows += new
                r = new[-1]
                dv = " | adapter movement (mean |a-1|) " + " ".join("%s %.3f" % (V3.LABEL[v], r.get("adapt_dev_" + V3.LABEL[v], float("nan")))
                                                                   for v in inputs) if m == "adapt" else ""
                print("%-13s %-6s done (%.0fs) | %s %.3f (mean over %d seeds) | epochs %.0f%s"
                      % (b, m, time.time() - t1, PRIMARY[b], np.mean([primary(x) for x in new]), a.seeds, np.mean([x["epochs"] for x in new]), dv), flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(out, "raw.csv"), index=False)
        res = pd.DataFrame(rows)
        res["primary"] = [primary(r) for r in res.to_dict("records")]
        tab = res.groupby(["benchmark", "mode"]).primary.mean().unstack("mode")[[m for m in a.modes if m in res["mode"].values]]
        ref = os.path.join(RESULTS_DIR, "final_mlp", "raw.csv")
        if os.path.exists(ref):
            r0 = pd.read_csv(ref)
            r0 = r0[(r0.rep == "teacher_" + a.name) & (r0.subset == "all") & (r0.rowset == "all") & (r0.learner == "mlp")]
            r0["primary"] = [row.get(PRIMARY[row["benchmark"]], np.nan) for row in r0.to_dict("records")]
            tab["sklearn MLP (harness)"] = r0.groupby("benchmark").primary.mean()
        tab = tab.loc[[b for b in benches if b in tab.index]]
        nop = [b for b in tab.index if b != "PAMPA"]
        print("\n=== %s: primary metric (MCC / pooled R2 / AUROC / Spearman as in the harness; pooled for cluster / CV benchmarks), mean over %d seeds ==="
              % (suite, a.seeds))
        print(tab.round(3).to_string())
        avg = pd.DataFrame({"average, all": tab.mean(), "average, no PAMPA": tab.loc[nop].mean()})
        print(avg.round(3).T.to_string())
        if "adapt" in tab and "plain" in tab:
            d = (tab["adapt"] - tab["plain"])
            print("\nadapters minus the same MLP without adapters, per benchmark: " + " | ".join("%s %+.3f" % (b, d[b]) for b in d.index)
                  + " | mean %+.3f, wins %d of %d" % (d.mean(), (d > 0).sum(), len(d)))
        print("wrote", out, flush=True)


if __name__ == "__main__":
    main()

"""Fine-tune the TEACHER on the five PeptideCLM-2 benchmarks and compare it with its frozen embedding.

    python src/teacher/finetune.py --name t8                                 # 5 benchmarks, the teacher frozen and fine-tuned, 1 seed
    python src/teacher/finetune.py --name t8 --suite pepland                 # PepLand's five tasks
    python src/teacher/finetune.py --name t8 --bench PAMPA --seeds 3
    python src/teacher/finetune.py --name t8 --modes frozen ft scratch ecfp  # + the opt-in controls
    python src/teacher/finetune.py --name t8 --resume                        # skip (benchmark, variant) pairs already in raw.csv

Only the teacher's own networks (the SCAs and the SEA) are trained. The three encoders (PeptideCLM-2, PepLand, ESM-2) stay
frozen: their vectors come from features/single_modality/{pclm,pepland,esm}.npz, exactly as the teacher saw them in
training (standardised with the teacher's own statistics; an absent optional view, i.e. ESM of a non-canonical peptide, is
filled with p_E by the teacher itself, as in training). Teachers without optional views (t1-t5) work too.

Variants (one head and one recipe for all, so differences come from the encoder only). The default runs only the first two,
i.e. only the teacher; the other two are opt-in controls (--modes):
    frozen   teacher weights fixed, head trained on the 1024-d (or 512-d) embedding
    ft       teacher from the checkpoint, ALL its weights trained at a LOW learning rate (--lr-backbone 1e-5, the lr of the
             PeptideCLM-2 paper's full fine-tuning), the new head at --lr-head 1e-3
    scratch  (opt-in) the same architecture from random weights at --lr-scratch 1e-3: what pretraining adds (its SCA_esm is
             only trained through the task loss, so its p_E fill is not an ESM prediction)
    ecfp     (opt-in) a 2048-1024-d MLP on log1p(ECFP counts), standardised with the benchmark's own statistics (no labels):
             a single-view control

The recipe follows the authors' (PeptideCLM-2 repo: classification_finetuning_v2.py and finetune_ensemble.py; the student and
PCLM-2 itself go through finetune_smiles.py with the same ensemble structure):
    head      Linear(d + extra, d) - SiLU - Dropout(0.1) - Linear(d, 1) (BCE-with-logits; MSE on z-scored targets for PAMPA)
    optimiser AdamW, weight decay 1e-3, gradient clip 1.0, batch 64, linear warm-up over 10% of the steps then linear decay
              to 0 over --epochs, at most 100 epochs, early stopping after 10 epochs without a better validation loss
              (the best epoch is kept; the test fold is never used for any choice)
    ensemble  every test fold is predicted by the MEAN of several models (mean logit for classification):
              AmpHGT (has a validation file): one model; PAMPA: the authors' scheme, 5 models per held-out cluster, each
              trained on 4 of the other clusters and validated on the 5th; the others: --inner (5) random inner folds of the
              training part.
Splits are those of benchmarks.py: CellPPD and THPep fixed train/test, AmpHGT train/val/test, PepMSND 10 cluster folds, PAMPA
6 leave-one-cluster-out folds (PAMPA_clusters.csv is byte-identical to the authors' file).

Metrics: eval_single's (MCC, AUROC; R2, Spearman). Folds are averaged (the other scripts' convention) AND the predictions of all
folds are pooled into ONE score (pooled_*): for PAMPA the pooled R2 is the authors' number (their script concatenates the
six ensemble predictions). Every score is also computed separately for the test molecules WITH an ESM view (canonical) and
WITHOUT (columns head_esm / head_noesm, MCC or R2): the p_E fill is only exercised on the second group.
Output: $RESULTS_DIR/finetune_<name>/raw.csv and preds_<benchmark>_<variant>.csv (smiles, y, fold, seed, pred, esm).
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
import train_teacher as T  # noqa: E402
from paths import BENCH_DIR, FEAT_DIR, RESULTS_DIR  # noqa: E402

MODES = ("frozen", "ft", "scratch", "ecfp")


# ------------------------------------------------------------------------------------------- data
def load_ft(bench):
    """smiles, y, split ('train'/'test', integer folds, or train/val/test for AmpHGT), extra_* columns."""
    if bench == "AmpHGT":                                   # keep the authors' own validation file
        parts = [pd.read_csv(os.path.join(BENCH_DIR, "amp_%s.csv" % s)).assign(split=s) for s in ("train", "val", "test")]
        return pd.concat(parts, ignore_index=True).rename(columns={"label": "y"})[["smiles", "y", "split"]]
    if bench in BM.THPEP_OFF:                               # the PeptideCLM-2 authors' own splits: 389 train / 98 val / 122 test, one model per run
        return BM.thpep_official(bench, keep_val=True)
    return E.load(bench)


def load_views(feat_dir=FEAT_DIR):
    """The teacher's views for the benchmark molecules: view -> (smiles index, array as the cache holds it, ok)."""
    out = {}
    for v in T.VIEWS:
        z = np.load(os.path.join(feat_dir, v + ".npz"), allow_pickle=True)
        out[v] = (dict(zip(z["smiles"].tolist(), range(len(z["smiles"])))), T.bench_array(v, z), z["ok"].astype(bool))
    return out


def take(xs, b):
    """Rows b of a Views object (keeps the `present` mask)."""
    out = T.Views(x[b] for x in xs)
    out.present = None if xs.present is None else xs.present[b]
    return out


def bench_data(bench, views, st, dev, feat_dir=FEAT_DIR, want_ecfp=False):
    """-> dict: df (usable rows), xs (the teacher's standardised views, T.Views), ecfp (only with want_ecfp: the control's
    input), extra, coverage."""
    df = load_ft(bench).dropna(subset=["smiles"]).reset_index(drop=True)
    df["smiles"] = df.smiles.astype(str)
    ii = {v: np.array([views[v][0][s] for s in df.smiles]) for v in T.VIEWS}
    good = {v: views[v][2][ii[v]] for v in T.VIEWS}
    ok = np.logical_and.reduce([good[v] for v in T.VIEWS if v not in T.OPTIONAL])     # every REQUIRED view fine
    df, ii = df[ok].reset_index(drop=True), {v: i[ok] for v, i in ii.items()}
    xs = T.Views()
    pres = np.ones((len(df), len(T.VIEWS)), bool)
    for k, v in enumerate(T.VIEWS):
        t = T.prep(v, torch.from_numpy(np.ascontiguousarray(views[v][1][ii[v]])).to(dev), st)
        if v in T.OPTIONAL:                                  # absent: zero, as in the cache (the teacher fills it)
            pres[:, k] = good[v][ok]
            t = t * torch.from_numpy(pres[:, k:k + 1]).to(dev).to(t.dtype)
        xs.append(t)
    xs.present = torch.from_numpy(pres).to(dev) if T.OPTIONAL else None
    # the ECFP control: log1p counts standardised with this benchmark's own molecules (no labels)
    ecfp = None
    if want_ecfp:                                            # only for the opt-in `ecfp` control
        z = np.load(os.path.join(feat_dir, "ecfp.npz"), allow_pickle=True)
        idx = dict(zip(z["smiles"].tolist(), range(len(z["smiles"]))))
        X = np.log1p(np.minimum(z["X"][np.array([idx[s] for s in df.smiles])], 255).astype(np.float32))
        mu, sd = X.mean(0), X.std(0)
        sd = np.maximum(sd, 0.1 * np.sqrt((sd ** 2).mean()) + 1e-6)
        ecfp = T.Views([torch.from_numpy(((X - mu) / sd).astype(np.float32)).clamp_(-10, 10).to(dev)])
        ecfp.present = None
    cols = [c for c in df.columns if c.startswith("extra_")]
    extra = torch.from_numpy(df[cols].to_numpy(np.float32)).to(dev) if cols else torch.zeros((len(df), 0), device=dev)
    return {"df": df, "xs": xs, "ecfp": ecfp, "extra": extra, "cov": float(ok.mean()),
            "esm": None if not T.OPTIONAL else pres[:, T.VIEWS.index(T.OPTIONAL[0])]}


# ------------------------------------------------------------------------------------------- splits
def inner_splits(df, bench, fold, test_idx, inner, seed):
    """-> list of (fit_idx, val_idx) for one outer test fold (see the module docstring)."""
    if (df.split == "val").any():                                           # a validation file: one model
        return [(np.flatnonzero((df.split == "train").to_numpy()), np.flatnonzero((df.split == "val").to_numpy()))]
    if bench == "PAMPA":                                                    # the authors' scheme: validate on another cluster
        others = [f for f in sorted(df.split.unique()) if f != fold]
        return [(np.flatnonzero(~df.split.isin([fold, v]).to_numpy()), np.flatnonzero((df.split == v).to_numpy()))
                for v in others]
    pool = np.setdiff1d(np.arange(len(df)), test_idx)
    perm = np.random.default_rng(seed).permutation(pool)
    if inner <= 1:
        nv = max(8, int(0.1 * len(pool)))
        return [(perm[nv:], perm[:nv])]
    parts = np.array_split(perm, inner)
    return [(np.concatenate([p for j, p in enumerate(parts) if j != k]), parts[k]) for k in range(inner)]


# ------------------------------------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(self, mode, ck, n_extra, head_dropout):
        super().__init__()
        self.mode = mode
        d = ck["args"]["sea_dim"]                                           # 512 for t1-t3, 1024 for t5 and t8
        if mode == "frozen":
            self.enc = None                                                 # the embedding is cached
        elif mode == "ecfp":
            self.enc = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, d))
        else:
            self.enc = T.Teacher.from_cfg(ck["args"])
            if mode == "ft":
                self.enc.load_state_dict(ck["model"])
        self.head = nn.Sequential(nn.Linear(d + n_extra, d), nn.SiLU(), nn.Dropout(head_dropout), nn.Linear(d, 1))

    def forward(self, xs, extra, cached=None):
        if cached is not None:
            s = cached
        elif self.mode == "ecfp":
            s = self.enc(xs[0])
        else:
            s = self.enc.encode(xs)
        return self.head(torch.cat([s, extra], 1)).squeeze(1)


def fit_one(mode, task, data, y, fit, val, pred, seed, ck, a, dev, cache):
    """Train one model on rows `fit`, stop on rows `val`, return its raw outputs (logits / values) on rows `pred`."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    inp = data["ecfp"] if mode == "ecfp" else data["xs"]
    ex = data["extra"]
    if ex.shape[1]:
        ex = (ex - ex[fit].mean(0)) / (ex[fit].std(0) + 1e-6)
    clf = task == "clf"
    mu_y, sd_y = (0.0, 1.0) if clf else (float(y[fit].mean()), float(y[fit].std() + 1e-8))
    yt = torch.from_numpy(((y - mu_y) / sd_y).astype(np.float32)).to(dev)

    net = Net(mode, ck, ex.shape[1], a.head_dropout).to(dev)
    if mode == "frozen":
        groups = [{"params": net.head.parameters(), "lr": a.lr_head}]
    elif mode == "ft":
        groups = [{"params": net.head.parameters(), "lr": a.lr_head}, {"params": net.enc.parameters(), "lr": a.lr_backbone}]
    else:
        groups = [{"params": net.parameters(), "lr": a.lr_scratch}]
    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    per_epoch = max(1, math.ceil(len(fit) / a.batch))
    total = per_epoch * a.epochs
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda i: (i + 1) / warm if i < warm else max(0.0, (total - i) / max(1, total - warm)))

    def out(idx):
        b = torch.from_numpy(idx).to(dev)
        if mode == "frozen":
            return net(None, ex[b], cache[b]).float(), yt[b]
        return net(take(inp, b), ex[b]).float(), yt[b]

    def loss_of(o, t):
        return F.binary_cross_entropy_with_logits(o, t) if clf else F.mse_loss(o, t)

    best, best_state, bad, best_ep, stopped = float("inf"), None, 0, 0, False
    for ep in range(a.epochs):
        net.train()
        order = fit[rng.permutation(len(fit))]
        for i in range(0, len(order), a.batch):
            if len(order[i:i + a.batch]) < 2:
                continue
            loss = loss_of(*out(order[i:i + a.batch]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            v = sum(float(loss_of(*out(val[i:i + 1024]))) * len(val[i:i + 1024]) for i in range(0, len(val), 1024)) / len(val)
        if v < best - 1e-4:
            best, bad, best_ep = v, 0, ep + 1
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                stopped = True
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        o = torch.cat([out(pred[i:i + 1024])[0] for i in range(0, len(pred), 1024)]).cpu().numpy()
    return (o if clf else o * sd_y + mu_y), {"epochs": ep + 1, "best_epoch": best_ep, "stopped": stopped}


def headline(task, yy, pp):
    """MCC (classification, pp = probabilities) or R2 (regression); NaN when the subset cannot be scored."""
    try:
        return float(E.score(task, yy, pp)["MCC" if task == "clf" else "R2"]) if len(yy) >= 10 else float("nan")
    except Exception:
        return float("nan")


def run_bench(bench, mode, data, ck, a, dev, cache, out_dir):
    df = data["df"]
    task = E.TASK[bench]
    clf = task == "clf"
    y = df.y.to_numpy().astype(np.float32)
    cv = pd.api.types.is_integer_dtype(df.split)
    folds = sorted(df.split.unique()) if cv else ["test"]
    esm = data["esm"]
    rows, preds, t0 = [], [], time.time()
    for seed in range(a.seeds):
        per, infos = [], []
        allp = np.full(len(df), np.nan)
        for fi, fo in enumerate(folds):
            te = np.flatnonzero((df.split == fo).to_numpy())
            outs = []
            for j, (fit, val) in enumerate(inner_splits(df, bench, fo, te, a.inner, seed)):
                o, info = fit_one(mode, task, data, y, fit, val, te, seed * 1000 + fi * 100 + j, ck, a, dev, cache)
                outs.append(o)
                infos.append(info)
            ens = np.mean(outs, 0)                                          # mean logit / mean prediction
            p = 1 / (1 + np.exp(-ens)) if clf else ens
            allp[te] = p
            s = E.score(task, y[te], p)
            if bench == "AmpHGT":
                short = df.loc[te, "smiles"].map(E.n_residues).to_numpy() <= 30
                s.update({k + "_le30": v for k, v in E.score(task, y[te][short], p[short]).items()})
            per.append(s)
            if len(folds) > 1:
                print("   %s %s seed %d fold %d/%d (%d molecules, %d models) %.0fs" %
                      (bench, mode, seed, fi + 1, len(folds), len(te), len(outs), time.time() - t0), flush=True)
        m = ~np.isnan(allp)
        row = {"benchmark": bench, "mode": mode, "seed": seed, "n": len(df), "coverage": round(data["cov"], 4),
               **{k: float(np.mean([s[k] for s in per])) for k in per[0]}}
        if cv:                                                              # one score on the pooled predictions of all folds
            row.update({"pooled_" + k: float(v) for k, v in E.score(task, y[m], allp[m]).items()})
        if esm is not None:
            row.update({"head_esm": headline(task, y[m & esm], allp[m & esm]), "n_esm": int((m & esm).sum()),
                        "head_noesm": headline(task, y[m & ~esm], allp[m & ~esm]), "n_noesm": int((m & ~esm).sum())})
        row.update({"epochs": float(np.mean([i["epochs"] for i in infos])),
                    "best_epoch": float(np.mean([i["best_epoch"] for i in infos])),
                    "hit_cap": int(sum(not i["stopped"] for i in infos)), "fits": len(infos)})
        rows.append(row)
        preds.append(pd.DataFrame({"smiles": df.smiles[m], "y": y[m], "fold": df.split[m].astype(str), "seed": seed,
                                   "pred": allp[m], "esm": -1 if esm is None else esm[m].astype(int)}))
    pd.concat(preds).to_csv(os.path.join(out_dir, "preds_%s_%s.csv" % (bench, mode)), index=False)
    return rows


# ------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="the teacher run to start from (checkpoints/teacher/<name>)")
    ap.add_argument("--bench", nargs="*", default=None)
    ap.add_argument("--suite", choices=("main", "pepland"), default="main",
                    help="main = PeptideCLM-2's five benchmarks (features/single_modality); pepland = PepLand's five tasks "
                         "(PL_cCPP PL_cSol PL_ncCPP PL_cBinding PL_ncBinding, features/pepland_suite, our 5-fold protocol)")
    ap.add_argument("--feat", default=None, help="another features subfolder under features/ (default: the suite's own)")
    ap.add_argument("--out", default=None, help="results folder under results/ (default finetune_<name>[_pepland])")
    ap.add_argument("--modes", nargs="*", default=["frozen", "ft"], choices=MODES,
                    help="default: the teacher frozen and fine-tuned; scratch (random weights) and ecfp (ECFP MLP) are opt-in controls")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--inner", type=int, default=5, help="inner folds per test fold (PAMPA and AmpHGT use their own scheme)")
    ap.add_argument("--epochs", type=int, default=100, help="the limit (the lr schedule decays over this many epochs)")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr-head", type=float, default=1e-3, help="the new head (frozen, ft)")
    ap.add_argument("--lr-backbone", type=float, default=1e-5, help="the teacher's pretrained weights (ft): low")
    ap.add_argument("--lr-scratch", type=float, default=1e-3, help="everything trained from random weights (scratch, ecfp)")
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--head-dropout", type=float, default=0.1)
    ap.add_argument("--use-last", action="store_true", help="start from last.pt instead of best.pt")
    ap.add_argument("--resume", action="store_true", help="skip the (benchmark, variant) pairs already in raw.csv")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    model0, st, ck = T.load_model(a, dev)               # also activates the checkpoint's views, optional views and descriptor names
    pl = a.suite == "pepland"
    feat = a.feat or ("pepland_suite" if pl else None)
    feat_dir = os.path.join(os.path.dirname(FEAT_DIR), feat) if feat else FEAT_DIR
    views = load_views(feat_dir)
    benches = a.bench or (list(BM.PL_SUITE) if pl else E.selected())
    out = os.path.join(RESULTS_DIR, a.out or ("finetune_%s%s" % (a.name, "_pepland" if pl else "")))
    os.makedirs(out, exist_ok=True)
    print("features: %s | results: %s" % (feat_dir, out), flush=True)
    print("teacher %s: views %s, optional %s, fill %s, latent %d-d | variants %s | benchmarks %s\n"
          "recipe: AdamW wd %g, batch %d, <= %d epochs, patience %d, warm-up 10%% + linear decay, clip 1.0, head dropout %g | "
          "lr: head %g, teacher %g (ft), from scratch %g | inner folds %d | seeds %d"
          % (a.name, list(T.VIEWS), list(T.OPTIONAL), ck["args"].get("fill", "token"), ck["args"]["sea_dim"], a.modes, benches,
             a.wd, a.batch, a.epochs, a.patience, a.head_dropout, a.lr_head, a.lr_backbone, a.lr_scratch, a.inner, a.seeds),
          flush=True)
    raw_path = os.path.join(out, "raw.csv")
    rows = pd.read_csv(raw_path).to_dict("records") if (a.resume and os.path.exists(raw_path)) else []
    done = {(r["benchmark"], r["mode"]) for r in rows if sum(1 for q in rows if q["benchmark"] == r["benchmark"]
                                                              and q["mode"] == r["mode"]) >= a.seeds}
    for b in benches:
        data = bench_data(b, views, st, dev, feat_dir, "ecfp" in a.modes)
        cache = None
        if "frozen" in a.modes:                          # the weights never change: embed once per benchmark
            with torch.no_grad():
                cache = torch.cat([model0.encode(take(data["xs"], torch.arange(i, min(i + 4096, len(data["df"])), device=dev)))
                                   for i in range(0, len(data["df"]), 4096)])
            print("%s: %d usable molecules (%.1f%% of the file)%s | frozen embedding std per dimension %.3f"
                  % (b, len(data["df"]), 100 * data["cov"],
                     "" if data["esm"] is None else ", ESM present for %.1f%%" % (100 * data["esm"].mean()),
                     float(cache.std(0).mean())), flush=True)
        for m in a.modes:
            if (b, m) in done:
                print("%-8s %-8s already in raw.csv: skipped" % (b, m), flush=True)
                continue
            t0 = time.time()
            new = run_bench(b, m, data, ck, a, dev, cache, out)
            rows += new
            key = "MCC" if E.TASK[b] == "clf" else "R2"
            r = new[-1]
            print("%-8s %-8s done (%.0fs) | %s %.3f%s | epochs run %.0f, best at %.0f, hit the %d-epoch cap in %d of %d fits"
                  % (b, m, time.time() - t0, key, r[key],
                     (" (pooled %.3f)" % r["pooled_" + key]) if "pooled_" + key in r else "",
                     r["epochs"], r["best_epoch"], a.epochs, r["hit_cap"], r["fits"]), flush=True)
            pd.DataFrame(rows).to_csv(raw_path, index=False)
    raw = pd.DataFrame(rows)
    order = [m for m in MODES if m in raw["mode"].values]
    col = lambda n: raw[n] if n in raw else pd.Series(np.nan, index=raw.index)                    # noqa: E731
    is_clf = raw.benchmark.map(E.TASK) == "clf"
    raw["headline"] = np.where(is_clf, col("MCC"), col("R2"))
    print("\n=== headline: MCC (classification) / R2 (PAMPA), mean over folds and seeds ===")
    print(raw.groupby(["benchmark", "mode"]).headline.mean().unstack("mode").round(3)[order].to_string())
    if "pooled_R2" in raw or "pooled_MCC" in raw:
        raw["pooled"] = np.where(is_clf, col("pooled_MCC"), col("pooled_R2"))
        pv = raw.dropna(subset=["pooled"])
        print("\n=== the same on the POOLED predictions of all folds (cluster-CV benchmarks only; PAMPA pooled R2 = the authors' metric) ===")
        print(pv.groupby(["benchmark", "mode"]).pooled.mean().unstack("mode").round(3)[[m for m in order if m in pv["mode"].values]].to_string())
    if "AUROC" in raw:
        print("\n=== AUROC (classification) ===")
        print(raw.dropna(subset=["AUROC"]).groupby(["benchmark", "mode"]).AUROC.mean().unstack("mode").round(3)[order].to_string())
    if "head_esm" in raw:
        print("\n=== test molecules WITH an ESM view (canonical) / WITHOUT (the p_E fill): MCC or R2 ===")
        print(raw.groupby(["benchmark", "mode"])[["n_esm", "head_esm", "n_noesm", "head_noesm"]].mean().round(3).to_string())
    print("\n=== convergence: epochs run / epoch of the best validation loss (means), and fits that hit the epoch cap ===")
    print(raw.groupby(["benchmark", "mode"]).agg(run=("epochs", "mean"), best=("best_epoch", "mean"),
                                                 capped=("hit_cap", "sum"), fits=("fits", "sum")).round(1).to_string())
    if "MCC_le30" in raw:
        print("\nAmpHGT test restricted to <=30 residues (MCC):")
        print(raw[raw.benchmark == "AmpHGT"].groupby("mode").MCC_le30.mean().round(3).to_string())
    print("\nwrote", out)


if __name__ == "__main__":
    main()

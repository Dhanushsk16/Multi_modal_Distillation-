"""Fine-tune the SMILES-only student on the ten benchmarks, and the paired control with the student frozen (an MLP head on its fixed output).

    python src/student/finetune_student.py --run st6f --epoch 2                  # the snapshot epoch_02.pt of run st6f (0 = best.pt)
    python src/student/finetune_student.py --run st6f --bench CellPPD --epochs 3 --inner 2 --modes ft       # a quick smoke test
    python src/student/finetune_student.py --run st6f --epoch 2 --resume         # skip the (benchmark, mode) pairs already in raw.csv

Two variants (the recipe, the splits, the ensembles and the metrics are those of src/teacher/finetune_v3.py, which did the same for the teacher):
    ft       the student (embedding, 8 blocks, pooling, output layer) is trained END TO END with the labels at a LOW learning rate (--lr-backbone 1e-5,
             the lr of the PeptideCLM-2 paper's fine-tuning and of the teacher's), the new MLP head at a HIGH one (--lr-head 1e-3)
    frozen   the student's weights never change; the same head is trained on its fixed 1024-d output (cached once per benchmark): the paired control.
             (The sklearn MLP of the evaluation harness is a second, different MLP on the same frozen output.)
Input of the head: the student's 1024-d output `z` (--feature z, the vector that was distilled from the teacher's latent) or its 1280-d pooled vector `pre`.
Head: LayerNorm - Linear(d + extra, 1024) - SiLU - Dropout 0.1 - Linear(1024, 1). The teacher's head had no LayerNorm because its latent is standardised with
saved statistics; the student's output has no such statistics, so a LayerNorm makes the head independent of the output's scale (the one change).
Recipe: AdamW weight decay 1e-3, gradient clip 1.0, batch 64, linear warm-up over 10% of the steps then linear decay over --epochs (100), early stopping after
10 epochs without a better validation loss (best epoch kept; the test fold is NEVER used for any choice); classification: BCE, regression: MSE on the
standardised label. Every test fold is predicted by the MEAN of several models (--inner 5 random inner folds; AmpHGT: its own validation file, one model;
PAMPA: the authors' scheme, one model per other cluster; --max-models N keeps only the first N of them, for speed). Splits: src/benchmarks.py.
Input: the benchmark SMILES, neutralised and canonicalised like everything the student reads (normalise.py), tokenised with the run's tokenizer.
Score: the benchmark's primary metric (MCC / AUROC / R2 / Spearman, as in the harness) on the PREDICTIONS POOLED over all folds, plus all other metrics and the
mean over folds. Rows the student cannot read (no tokens) are dropped and counted. Output: $RESULTS_DIR/<--out>/{raw.csv, summary.csv, preds_*.csv}.
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import benchmarks as BM  # noqa: E402
import eval_single as E  # noqa: E402
import normalise as NM  # noqa: E402
import tokenizer as T  # noqa: E402
from model import PAD, Student  # noqa: E402
from paths import BENCH_DIR, RESULTS_DIR  # noqa: E402
from train_student import STUDENT_DIR  # noqa: E402

PRIMARY = {"CellPPD": "MCC", "THPep": "MCC", "PepMSND": "MCC", "PAMPA": "R2", "AmpHGT": "MCC", "PL_cCPP": "AUROC", "PL_cSol": "AUROC",
           "PL_ncCPP": "Spearman", "PL_cBinding": "Spearman", "PL_ncBinding": "Spearman"}
PRIMARY.update({n: "MCC" for n in BM.THPEP_OFF})                                         # THPep on the authors' three splits (benchmarks.py)
SEVEN = ["CellPPD", "THPep", "PepMSND", "AmpHGT", "PL_cCPP", "PL_cSol", "PL_ncCPP"]        # no PAMPA, no binding (as show_bench.py)
MODES = ("ft", "frozen")


# ------------------------------------------------------------------------------------------- the student
def load_student(run, epoch, dev, out_dir=STUDENT_DIR):
    """-> student (inference part only), tokenizer, max length, epoch of the weights. epoch 0 = best.pt, N = epoch_NN.pt (train_student.py --save-every)."""
    path = os.path.join(out_dir, run, "best.pt" if epoch <= 0 else "epoch_%02d.pt" % epoch)
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg, args = ck["cfg"], ck["args"]
    model = Student(**cfg)
    model.load_state_dict(ck["model"])
    del model.train_only                                    # the view heads, the masked-token bias and the temperature are not part of the student
    tok = T.load_tokenizer(args["tok"])
    max_len = args.get("canon_max_len", args.get("max_len", cfg["max_len"]))
    return model.to(dev), tok, max_len, ck.get("epoch", "?"), path


def tokenise(smiles, tok, max_len, workers, dev):
    """SMILES -> (token ids [N, L] right-padded with PAD on dev, usable [N] bool, number the normaliser could not parse)."""
    norm, bad = NM.normalise_many([str(s) for s in smiles], workers)
    ids = [tok.with_special(e, max_len) for e in tok.encode_batch(norm)]
    ok = np.array([len(x) >= 3 for x in ids])
    arr = np.zeros((len(ids), max(len(x) for x in ids)), np.int64)
    for i, x in enumerate(ids):
        arr[i, :len(x)] = x
    return torch.from_numpy(arr).to(dev), ok, bad


def features(student, x, feat):
    """Student output for token ids x [B, L] -> [B, 1024] (z) or [B, 1280] (pre), float32. Trims the padding to the longest row of the batch."""
    keep = x != PAD
    L = int(keep.sum(1).max())
    x, keep = x[:, :L], keep[:, :L]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
        o = student(x, keep)
    return (o["z"] if feat == "z" else o["pre"]).float()


def feat_dim(student, feat):
    return student.cfg["out"] if feat == "z" else student.cfg["n_query"] * student.cfg["d"]


# ------------------------------------------------------------------------------------------- data and splits
def load_ft(bench):
    """smiles, y, split (integer folds, 'train'/'test', or train/val/test for AmpHGT), extra_* columns (as src/teacher/finetune.py)."""
    if bench == "AmpHGT":                                   # keep the authors' own validation file
        parts = [pd.read_csv(os.path.join(BENCH_DIR, "amp_%s.csv" % s)).assign(split=s) for s in ("train", "val", "test")]
        return pd.concat(parts, ignore_index=True).rename(columns={"label": "y"})[["smiles", "y", "split"]]
    if bench in BM.THPEP_OFF:                               # the PeptideCLM-2 authors' own splits: 389 train / 98 val / 122 test, one model per run
        return BM.thpep_official(bench, keep_val=True)
    return BM.load(bench)


def inner_splits(df, bench, fold, test_idx, inner, seed):
    """-> list of (fit_idx, val_idx) for one outer test fold (copied from src/teacher/finetune.py)."""
    if (df.split == "val").any():                                           # a validation file: one model
        return [(np.flatnonzero((df.split == "train").to_numpy()), np.flatnonzero((df.split == "val").to_numpy()))]
    if bench == "PAMPA":                                                    # the authors' scheme: validate on another cluster
        others = [f for f in sorted(df.split.unique()) if f != fold]
        return [(np.flatnonzero(~df.split.isin([fold, v]).to_numpy()), np.flatnonzero((df.split == v).to_numpy())) for v in others]
    pool = np.setdiff1d(np.arange(len(df)), test_idx)
    perm = np.random.default_rng(seed).permutation(pool)
    if inner <= 1:
        nv = max(8, int(0.1 * len(pool)))
        return [(perm[nv:], perm[:nv])]
    parts = np.array_split(perm, inner)
    return [(np.concatenate([p for j, p in enumerate(parts) if j != k]), parts[k]) for k in range(inner)]


def bench_data(bench, tok, max_len, workers, dev):
    df = load_ft(bench).dropna(subset=["smiles"]).reset_index(drop=True)
    df["smiles"] = df.smiles.astype(str)
    ids, ok, bad = tokenise(df.smiles, tok, max_len, workers, dev)
    n0 = len(df)
    df = df[ok].reset_index(drop=True)
    ids = ids[torch.from_numpy(np.flatnonzero(ok)).to(dev)]
    cols = [c for c in df.columns if c.startswith("extra_")]
    extra = torch.from_numpy(df[cols].to_numpy(np.float32)).to(dev) if cols else torch.zeros((len(df), 0), device=dev)
    return {"df": df, "ids": ids, "extra": extra, "cov": len(df) / max(1, n0), "bad": bad}


# ------------------------------------------------------------------------------------------- model
class Head(nn.Module):
    def __init__(self, d_in, n_extra, hidden, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.mlp = nn.Sequential(nn.Linear(d_in + n_extra, hidden), nn.SiLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, f, extra):
        return self.mlp(torch.cat([self.norm(f), extra], 1)).squeeze(1)


class Net(nn.Module):
    """student (None for the frozen control: the head reads cached features) + head."""

    def __init__(self, student, feat, d_in, n_extra, hidden, dropout):
        super().__init__()
        self.student, self.feat = student, feat
        self.head = Head(d_in, n_extra, hidden, dropout)

    def forward(self, x, extra, cached=None):
        f = cached if cached is not None else features(self.student, x, self.feat)
        return self.head(f, extra)


@torch.no_grad()
def embed_all(student, ids, feat, bs=512):
    student.eval()
    return torch.cat([features(student, ids[i:i + bs], feat) for i in range(0, len(ids), bs)])


def fit_one(mode, task, data, y, fit, val, pred, seed, base, a, dev, cache):
    """Train one model on rows `fit`, stop on rows `val`, return its raw outputs (logits / values) on rows `pred`."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ex = data["extra"]
    if ex.shape[1]:
        ex = (ex - ex[fit].mean(0)) / (ex[fit].std(0) + 1e-6)
    clf = task == "clf"
    mu_y, sd_y = (0.0, 1.0) if clf else (float(y[fit].mean()), float(y[fit].std() + 1e-8))
    yt = torch.from_numpy(((y - mu_y) / sd_y).astype(np.float32)).to(dev)
    d_in = feat_dim(base, a.feature)
    if mode == "ft":
        net = Net(copy.deepcopy(base), a.feature, d_in, ex.shape[1], a.hidden, a.head_dropout).to(dev)
        groups = [{"params": list(net.head.parameters()), "lr": a.lr_head}, {"params": list(net.student.parameters()), "lr": a.lr_backbone}]
    else:
        net = Net(None, a.feature, d_in, ex.shape[1], a.hidden, a.head_dropout).to(dev)
        groups = [{"params": list(net.head.parameters()), "lr": a.lr_head}]
    clip_params = [p for g in groups for p in g["params"]]
    opt = torch.optim.AdamW(groups, weight_decay=a.wd)
    per_epoch = max(1, math.ceil(len(fit) / a.batch))
    total = per_epoch * a.epochs
    warm = max(1, int(0.1 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: (i + 1) / warm if i < warm else max(0.0, (total - i) / max(1, total - warm)))

    def out(idx):
        b = torch.from_numpy(idx).to(dev)
        if mode == "frozen":
            return net(None, ex[b], cache[b]).float(), yt[b]
        return net(data["ids"][b], ex[b]).float(), yt[b]

    def loss_of(o, t):
        return F.binary_cross_entropy_with_logits(o, t) if clf else F.mse_loss(o, t)

    best, best_state, bad, best_ep, stopped, ep = float("inf"), None, 0, 0, False, 0
    for ep in range(a.epochs):
        net.train()
        order = fit[rng.permutation(len(fit))]
        for i in range(0, len(order), a.batch):
            if len(order[i:i + a.batch]) < 2:
                continue
            loss = loss_of(*out(order[i:i + a.batch]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(clip_params, 1.0)
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            v = sum(float(loss_of(*out(val[i:i + 512]))) * len(val[i:i + 512]) for i in range(0, len(val), 512)) / len(val)
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
        o = torch.cat([out(pred[i:i + 512])[0] for i in range(0, len(pred), 512)]).cpu().numpy()
    return (o if clf else o * sd_y + mu_y), {"epochs": ep + 1, "best_epoch": best_ep, "stopped": stopped}


def safe_score(task, y, p):
    try:
        return E.score(task, y, p)
    except Exception:
        return {}


def run_bench(bench, mode, data, base, a, dev, cache, out_dir):
    df = data["df"]
    task = BM.TASK[bench]
    clf = task == "clf"
    y = df.y.to_numpy().astype(np.float32)
    cv = pd.api.types.is_integer_dtype(df.split)
    folds = sorted(df.split.unique()) if cv else ["test"]
    rows, preds, t0 = [], [], time.time()
    for seed in range(a.seeds):
        per, infos = [], []
        allp = np.full(len(df), np.nan)
        for fi, fo in enumerate(folds):
            te = np.flatnonzero((df.split == fo).to_numpy())
            outs = []
            splits = inner_splits(df, bench, fo, te, a.inner, seed)
            if a.max_models:
                splits = splits[:a.max_models]
            for j, (fit, val) in enumerate(splits):
                o, info = fit_one(mode, task, data, y, fit, val, te, seed * 1000 + fi * 100 + j, base, a, dev, cache)
                outs.append(o)
                infos.append(info)
            ens = np.mean(outs, 0)                                          # mean logit / mean prediction
            p = 1 / (1 + np.exp(-ens)) if clf else ens
            allp[te] = p
            per.append(safe_score(task, y[te], p))
            if len(folds) > 1:
                print("   %s %s seed %d fold %d/%d (%d molecules, %d models) %.0fs" % (bench, mode, seed, fi + 1, len(folds), len(te), len(outs),
                                                                                  time.time() - t0), flush=True)
        m = ~np.isnan(allp)
        pooled = safe_score(task, y[m], allp[m])
        row = {"benchmark": bench, "mode": mode, "seed": seed, "n": len(df), "coverage": round(data["cov"], 4), "metric": PRIMARY[bench],
               "primary": pooled.get(PRIMARY[bench], np.nan),
               "primary_foldmean": float(np.nanmean([s.get(PRIMARY[bench], np.nan) for s in per])) if per else np.nan,
               **{k: float(v) for k, v in pooled.items()},
               "epochs": float(np.mean([i["epochs"] for i in infos])), "best_epoch": float(np.mean([i["best_epoch"] for i in infos])),
               "hit_cap": int(sum(not i["stopped"] for i in infos)), "fits": len(infos)}
        rows.append(row)
        preds.append(pd.DataFrame({"smiles": df.smiles[m], "y": y[m], "fold": df.split[m].astype(str), "seed": seed, "pred": allp[m]}))
    pd.concat(preds).to_csv(os.path.join(out_dir, "preds_%s_%s.csv" % (bench, mode)), index=False)
    return rows


# ------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="the student run (checkpoints/student_distill/<run>)")
    ap.add_argument("--epoch", type=int, default=0, help="0 = best.pt; N = the snapshot epoch_NN.pt (train_student.py --save-every)")
    ap.add_argument("--bench", nargs="*", default=None, help="default: the ten benchmarks (the five PeptideCLM-2 ones and the five PepLand ones)")
    ap.add_argument("--modes", nargs="*", default=list(MODES), choices=MODES)
    ap.add_argument("--feature", choices=("z", "pre"), default="z", help="z = the 1024-d output, pre = the 1280-d pooled vector")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--inner", type=int, default=5, help="inner folds (= models) per test fold (PAMPA and AmpHGT use their own scheme)")
    ap.add_argument("--max-models", type=int, default=0, help="at most this many models per test fold (0 = all; PAMPA has one per other cluster)")
    ap.add_argument("--epochs", type=int, default=100, help="the limit (the lr schedule decays over this many epochs)")
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr-head", type=float, default=1e-3, help="the new MLP head: high")
    ap.add_argument("--lr-backbone", type=float, default=1e-5, help="the student's own weights (ft): low")
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--head-dropout", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=4, help="processes of the SMILES normaliser")
    ap.add_argument("--out", default=None, help="results folder under results/ (default finetune_student_<run>[_eNN])")
    ap.add_argument("--resume", action="store_true", help="skip the (benchmark, mode) pairs already in raw.csv")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base, tok, max_len, ep_w, path = load_student(a.run, a.epoch, dev)
    for p in base.parameters():
        p.requires_grad_(True)
    name = a.run if a.epoch <= 0 else "%s_e%02d" % (a.run, a.epoch)
    benches = a.bench or (BM.selected() + BM.PL_SUITE)
    out = os.path.join(RESULTS_DIR, a.out or "finetune_student_%s" % name)
    os.makedirs(out, exist_ok=True)
    print("student %s: %s | weights of epoch %s | %.3fM parameters | feature %s (%d-d) | modes %s | benchmarks %s\n"
          "recipe: AdamW wd %g, batch %d, <= %d epochs, patience %d, warm-up 10%% + linear decay, clip 1.0, head dropout %g, hidden %d | lr: head %g, student %g (ft) | "
          "inner folds %d (max models %s) | seeds %d | device %s\nresults: %s"
          % (name, path, ep_w, sum(p.numel() for p in base.parameters()) / 1e6, a.feature, feat_dim(base, a.feature), a.modes, benches,
             a.wd, a.batch, a.epochs, a.patience, a.head_dropout, a.hidden, a.lr_head, a.lr_backbone, a.inner, a.max_models or "all", a.seeds, dev, out), flush=True)
    raw_path = os.path.join(out, "raw.csv")
    rows = pd.read_csv(raw_path).to_dict("records") if (a.resume and os.path.exists(raw_path)) else []
    done = {(r["benchmark"], r["mode"]) for r in rows if sum(1 for q in rows if q["benchmark"] == r["benchmark"] and q["mode"] == r["mode"]) >= a.seeds}
    for b in benches:
        data = bench_data(b, tok, max_len, a.workers, dev)
        cache = None
        if "frozen" in a.modes:                              # the weights never change: embed once per benchmark
            cache = embed_all(base, data["ids"], a.feature)
        print("%s: %d usable molecules (%.1f%% of the file, %d the normaliser could not parse)" % (b, len(data["df"]), 100 * data["cov"], data["bad"]), flush=True)
        for m in a.modes:
            if (b, m) in done:
                print("%-12s %-7s already in raw.csv: skipped" % (b, m), flush=True)
                continue
            t0 = time.time()
            new = run_bench(b, m, data, base, a, dev, cache, out)
            rows += new
            r = new[-1]
            print("%-12s %-7s done (%.0fs) | %s %.3f (mean of folds %.3f) | epochs run %.0f, best at %.0f, hit the %d-epoch cap in %d of %d fits"
                  % (b, m, time.time() - t0, r["metric"], r["primary"], r["primary_foldmean"], r["epochs"], r["best_epoch"], a.epochs, r["hit_cap"], r["fits"]), flush=True)
            pd.DataFrame(rows).to_csv(raw_path, index=False)
    res = pd.DataFrame(rows)
    summ = res.groupby(["benchmark", "mode"]).agg(metric=("metric", "first"), primary=("primary", "mean"), sd=("primary", "std"), n=("n", "first"),
                                                  primary_foldmean=("primary_foldmean", "mean")).reset_index()
    summ.to_csv(os.path.join(out, "summary.csv"), index=False)
    tab = summ.pivot(index="benchmark", columns="mode", values="primary")
    tab.insert(0, "metric", [PRIMARY[i] for i in tab.index])
    order = [b for b in (BM.selected() + BM.PL_SUITE + BM.THPEP_OFF) if b in tab.index]
    tab = tab.loc[order]
    mcols = [m for m in a.modes if m in tab.columns]
    avg = {"AVERAGE (all)": tab[mcols].mean()}
    s7 = [b for b in SEVEN if b in tab.index]
    if len(s7) == len(SEVEN):
        avg["AVERAGE of 7 (no PAMPA, no binding)"] = tab.loc[s7, mcols].mean()
    print("\n=== primary metric on the pooled predictions (mean over seeds) ===")
    print(pd.concat([tab, pd.DataFrame(avg).T]).round(3).to_string())
    print("\nwrote", out)


if __name__ == "__main__":
    main()

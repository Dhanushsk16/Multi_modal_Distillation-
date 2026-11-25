"""Fine-tune a SMILES transformer on the benchmarks the way PeptideCLM-2's authors do (not frozen features
with simple learners). Works for PeptideCLM-2 itself, and later for our student and its MLM-only control.

    python src/finetune/finetune_smiles.py --name pclm2 --bench CellPPD --seeds 101        # reproduction check
    python src/finetune/finetune_smiles.py --name pclm2 --bench PL_cCPP PL_cSol --seeds 101
    python src/finetune/finetune_smiles.py --name pclm2 --bench PL_ncBinding --folds 0 --inner 1 --epochs 2   # quick look

Recipe "lora" (default) = what the authors' RELEASED code does for CellPPD, THPep and AmpHGT
(training/02_classification_benchmarks_training_code/scripts/classification_finetuning_v2.py), which is
also what their published numbers came from (reproduced in the earlier Distillation project; the paper's
text says full fine-tuning at lr 1e-5, the code does not):
    backbone frozen except LoRA adapters (r=16, alpha=32, dropout 0.1) on every attention `qkv_proj`,
    mean-pooled embedding -> Linear(d, d) -> SiLU -> Dropout(0.1) -> Linear(d, 1); BCE-with-logits
    (classification) or SmoothL1 (regression, targets z-scored here); AdamW lr 3e-4; linear warmup over 10% of
    the steps then linear decay; batch 16; at most 10 epochs; validation every half epoch; early stopping
    after 5 validations without a better val loss; the best checkpoint is kept.
    No validation file -> inner 5-fold CV on the training part and the test prediction is the MEAN LOGIT of
    the 5 models (the authors' fallback and aggregation); a val file (AmpHGT) -> one model.
Recipe "full" (--mode full) = every weight trainable, lr 1e-5 (the paper's text and the authors' PAMPA script).
Differences from the authors' code, all numerical or practical: plain PyTorch (no peft / lightning), bf16
autocast, batches padded to their own longest sequence (they pad every batch to the dataset's longest;
identical results up to rounding), regression targets z-scored, the model is loaded through
featurize.load_pclm (rebuilds the rotary buffers). The authors' PepMSND script early-stops
on the TEST fold; this harness never does (inner validation only).

Splits come from benchmarks.load (the benchmarks' own, or our documented 5-fold protocol for PL_*).
Output: $RESULTS_DIR/finetune_<name>/preds_<bench>_s<seed>_f<fold>.csv (one per outer fold: finished folds
are skipped on re-run), models.csv (one row per inner model), metrics.csv and a printed table.
"""
from __future__ import annotations

import argparse
import copy
import math
import os
import signal
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
import featurize  # noqa: E402
from paths import BENCH_DIR, PCLM_DIR, RESULTS_DIR  # noqa: E402


# ------------------------------------------------------------------------------------------- data
def load_ft(bench):
    """smiles, y, split ('train'/'test', integer folds, or train/val/test for AmpHGT), extra_* columns."""
    if bench == "AmpHGT":                                   # keep the authors' own validation file
        parts = [pd.read_csv(os.path.join(BENCH_DIR, "amp_%s.csv" % s)).assign(split=s) for s in ("train", "val", "test")]
        return pd.concat(parts, ignore_index=True).rename(columns={"label": "y"})[["smiles", "y", "split"]]
    return BM.load(bench)


def outer_folds(df):
    if pd.api.types.is_integer_dtype(df.split):
        return [(int(f), np.flatnonzero((df.split == f).to_numpy())) for f in sorted(df.split.unique())]
    return [("test", np.flatnonzero((df.split == "test").to_numpy()))]


def inner_splits(df, test_idx, inner, seed):
    """-> list of (train_idx, val_idx) for one outer fold."""
    if (df.split == "val").any():                           # a validation file exists: one model
        return [(np.flatnonzero((df.split == "train").to_numpy()), np.flatnonzero((df.split == "val").to_numpy()))]
    pool = np.setdiff1d(np.arange(len(df)), test_idx)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(pool)
    if inner <= 1:
        nv = max(8, int(0.1 * len(pool)))
        return [(perm[nv:], perm[:nv])]
    parts = np.array_split(perm, inner)
    return [(np.concatenate([p for j, p in enumerate(parts) if j != k]), parts[k]) for k in range(inner)]


def make_batch(ids, rows, pad, dev):
    L = max(len(ids[i]) for i in rows)
    x = torch.full((len(rows), L), pad, dtype=torch.long)
    m = torch.zeros((len(rows), L), dtype=torch.long)
    for j, i in enumerate(rows):
        x[j, :len(ids[i])] = torch.tensor(ids[i])
        m[j, :len(ids[i])] = 1
    return x.to(dev), m.to(dev)


# ------------------------------------------------------------------------------------------ model
class LoRALinear(nn.Module):
    """peft-style LoRA: y = W x + (alpha / r) * B(A(dropout(x))), W frozen, A kaiming, B zero."""

    def __init__(self, base: nn.Linear, r, alpha, dropout):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad = False
        self.A = nn.Parameter(torch.empty(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        self.scale = alpha / r
        self.drop = nn.Dropout(dropout)
        self.reset()

    def reset(self):
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        return self.base(x) + (self.drop(x) @ self.A.t().to(x.dtype) @ self.B.t().to(x.dtype)) * self.scale


def add_lora(model, r, alpha, dropout, target="qkv_proj"):
    for p in model.parameters():
        p.requires_grad = False
    sites = [(m, n) for _, m in model.named_modules() for n, c in m.named_children()
             if n == target and isinstance(c, nn.Linear)]
    for m, n in sites:
        setattr(m, n, LoRALinear(getattr(m, n), r, alpha, dropout))
    assert sites, "no '%s' Linear found: has the model code changed?" % target
    return len(sites)


class Net(nn.Module):
    def __init__(self, backbone, dim, n_extra):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(nn.Linear(dim + n_extra, dim), nn.SiLU(), nn.Dropout(0.1), nn.Linear(dim, 1))

    def reset(self, mode, orig_state):
        if mode == "full":
            self.backbone.load_state_dict(orig_state)
        else:
            for m in self.backbone.modules():
                if isinstance(m, LoRALinear):
                    m.reset()
        for m in self.head:
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def forward(self, ids, mask, extra):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=ids.is_cuda):
            pooled = self.backbone(input_ids=ids, attention_mask=mask).mean_pool
        return self.head(torch.cat([pooled.float(), extra], 1)).squeeze(1)


def trainable(net):
    return [(n, p) for n, p in net.named_parameters() if p.requires_grad]


# ---------------------------------------------------------------------------------------- training
@torch.no_grad()
def predict(net, ids, extra, rows, pad, dev, bs=64):
    net.eval()
    out = []
    for i in range(0, len(rows), bs):
        r = rows[i:i + bs]
        x, m = make_batch(ids, r, pad, dev)
        out.append(net(x, m, extra[r]).float().cpu())
    return torch.cat(out).numpy()


def loss_fn(task, out, y):
    return F.binary_cross_entropy_with_logits(out, y) if task == "clf" else F.smooth_l1_loss(out, y)


def train_one(net, ids, extra, y, tr, va, task, a, pad, dev, seed):
    """Authors' loop: AdamW, warmup 10% then linear decay, validation every half epoch, patience, best weights."""
    torch.manual_seed(seed)
    net.reset(a.mode, a.orig_state)
    lr = a.lr
    steps_per_epoch = max(1, len(tr) // a.batch)
    total = steps_per_epoch * a.epochs
    warm = int(0.1 * total)
    params = [p for _, p in trainable(net)]
    opt = torch.optim.AdamW(params, lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s / max(1, warm)) if s < warm else max(0.0, (total - s) / max(1, total - warm)))
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    rng = np.random.default_rng(seed)
    half = max(1, steps_per_epoch // 2)
    best, best_state, bad, step, nval = float("inf"), None, 0, 0, 0

    def validate():
        net.eval()
        with torch.no_grad():
            tot, n = 0.0, 0
            for i in range(0, len(va), 64):
                r = va[i:i + 64]
                x, m = make_batch(ids, r, pad, dev)
                tot += float(loss_fn(task, net(x, m, extra[r]).float(), yt[r])) * len(r)
                n += len(r)
        net.train()
        return tot / n

    net.train()
    stop = False
    for ep in range(a.epochs):
        order = tr[rng.permutation(len(tr))]
        for b in range(steps_per_epoch):
            r = order[b * a.batch:(b + 1) * a.batch]
            x, m = make_batch(ids, r, pad, dev)
            loss = loss_fn(task, net(x, m, extra[r]).float(), yt[r])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            if (b + 1) == half or (b + 1) == steps_per_epoch:          # every half epoch, like the authors
                v = validate()
                nval += 1
                if v < best:
                    best, bad = v, 0
                    best_state = {n: p.detach().clone() for n, p in trainable(net)}
                else:
                    bad += 1
                    if bad >= a.patience:
                        stop = True
                        break
        if stop:
            break
    for n, p in trainable(net):
        p.data.copy_(best_state[n])
    return {"val_loss": best, "epochs_run": ep + 1, "validations": nval, "stopped_early": stop}


def _term(sig, frame):
    raise KeyboardInterrupt


def auroc(y, p):
    from sklearn.metrics import roc_auc_score
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


# --------------------------------------------------------------------------------------------- main
def run_bench(bench, seed, a, tok, net, dev, out_dir, log):
    df = load_ft(bench).dropna(subset=["smiles"]).reset_index(drop=True)
    df["smiles"] = df.smiles.astype(str)
    if a.max_rows:
        df = df.sample(min(a.max_rows, len(df)), random_state=0).reset_index(drop=True)
    task = BM.TASK[bench]
    ids = tok(df.smiles.tolist(), add_special_tokens=True)["input_ids"]
    pad = tok.pad_token_id
    cols = [c for c in df.columns if c.startswith("extra_")]
    extra = (pd.get_dummies(df[cols].astype(str)).to_numpy(np.float32) if cols
             else np.zeros((len(df), 0), np.float32))
    extra = torch.from_numpy(extra).to(dev)
    if net.head[0].in_features != net.backbone.config.embed_dim + extra.shape[1]:
        net.head[0] = nn.Linear(net.backbone.config.embed_dim + extra.shape[1], net.backbone.config.embed_dim).to(dev)
    y_all = df.y.to_numpy(np.float32)
    for fold, te in outer_folds(df):
        if a.folds is not None and str(fold) not in a.folds:
            continue
        path = os.path.join(out_dir, "preds_%s_s%d_f%s.csv" % (bench, seed, fold))
        if os.path.exists(path):
            print("%s seed %d fold %s: done earlier" % (bench, seed, fold), flush=True)
            continue
        t0 = time.time()
        logits = []
        for k, (tr, va) in enumerate(inner_splits(df, te, a.inner, seed)):
            mu, sd = (0.0, 1.0) if task == "clf" else (float(y_all[tr].mean()), float(y_all[tr].std() + 1e-8))
            y = ((y_all - mu) / sd).astype(np.float32)
            t1 = time.time()
            info = train_one(net, ids, extra, y, tr, va, task, a, pad, dev, seed + 17 * k)
            pv = predict(net, ids, extra, va, pad, dev)
            collapsed = bool(task == "clf" and auroc(y_all[va], pv) < 0.55)
            pt = predict(net, ids, extra, te, pad, dev)
            logits.append(pt if task == "clf" else pt * sd + mu)
            log.append({"benchmark": bench, "seed": seed, "fold": fold, "inner": k, "val_loss": info["val_loss"],
                        "epochs_run": info["epochs_run"], "stopped_early": info["stopped_early"],
                        "collapsed": collapsed, "seconds": round(time.time() - t1)})
            print("  %s seed %d fold %s inner %d/%d | val loss %.4f | epochs %d%s | %.0f s%s"
                  % (bench, seed, fold, k + 1, a.inner if not (df.split == "val").any() else 1, info["val_loss"],
                     info["epochs_run"], " (early stop)" if info["stopped_early"] else "", time.time() - t1,
                     " | COLLAPSED (val AUROC < 0.55)" if collapsed else ""), flush=True)
        L = np.stack(logits)
        pd.DataFrame({"smiles": df.smiles.iloc[te].to_numpy(), "y": y_all[te], "pred": L.mean(0),
                      **{"m%d" % k: L[k] for k in range(len(L))}}).to_csv(path + ".tmp", index=False)
        os.replace(path + ".tmp", path)
        print("%s seed %d fold %s finished: %d test rows, %d model(s), %.1f min"
              % (bench, seed, fold, len(te), len(L), (time.time() - t0) / 60), flush=True)
    return task


def collect(out_dir, benches, seeds):
    rows = []
    for b in benches:
        task = BM.TASK[b]
        for s in seeds:
            files = sorted(f for f in os.listdir(out_dir) if f.startswith("preds_%s_s%d_f" % (b, s)) and f.endswith(".csv"))
            if not files:
                continue
            per = []
            for f in files:
                d = pd.read_csv(os.path.join(out_dir, f))
                p = 1 / (1 + np.exp(-d.pred.to_numpy())) if task == "clf" else d.pred.to_numpy()
                per.append(E.score(task, d.y.to_numpy(), p))
            rows.append({"benchmark": b, "seed": s, "folds": len(files), **{k: float(np.mean([x[k] for x in per])) for k in per[0]}})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run name (results/finetune_<name>)")
    ap.add_argument("--model", default=PCLM_DIR, help="HF model folder (PeptideCLM-2 mlm-large by default)")
    ap.add_argument("--bench", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[101])
    ap.add_argument("--inner", type=int, default=5, help="inner CV models per outer fold when there is no val file (1 = one 90/10 split)")
    ap.add_argument("--mode", choices=("lora", "full"), default="lora")
    ap.add_argument("--lr", type=float, default=None, help="default 3e-4 (lora) or 1e-5 (full)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--folds", nargs="*", default=None, help="only these outer folds (debug)")
    ap.add_argument("--max-rows", type=int, default=0, help="subsample the benchmark (debug)")
    a = ap.parse_args()
    a.lr = a.lr or (3e-4 if a.mode == "lora" else 1e-5)
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, signal.default_int_handler)

    out_dir = os.path.join(RESULTS_DIR, "finetune_" + a.name)
    os.makedirs(out_dir, exist_ok=True)
    tok, backbone, dev = featurize.load_pclm(a.model)
    dim = backbone.config.embed_dim
    a.orig_state = None
    if a.mode == "lora":
        n = add_lora(backbone, a.rank, a.alpha, 0.1)
        print("LoRA on %d qkv_proj layers (r=%d, alpha=%d)" % (n, a.rank, a.alpha), flush=True)
    else:
        for p in backbone.parameters():
            p.requires_grad = True
        a.orig_state = {k: v.detach().cpu().clone() for k, v in backbone.state_dict().items()}
    net = Net(backbone, dim, 0).to(dev)
    nt = sum(p.numel() for _, p in trainable(net))
    print("trainable parameters: %s of %s (%.2f%%) | mode %s lr %g batch %d epochs <=%d patience %d | device %s"
          % (format(nt, ","), format(sum(p.numel() for p in net.parameters()), ","),
             100 * nt / sum(p.numel() for p in net.parameters()), a.mode, a.lr, a.batch, a.epochs, a.patience, dev), flush=True)
    log = []
    try:
        for b in a.bench:
            for s in a.seeds:
                run_bench(b, s, a, tok, net, dev, out_dir, log)
    except KeyboardInterrupt:
        print("interrupted: finished folds are kept, run the same command again to continue", flush=True)
    if log:
        old = os.path.join(out_dir, "models.csv")
        pd.concat([pd.read_csv(old), pd.DataFrame(log)]).to_csv(old, index=False) if os.path.exists(old) \
            else pd.DataFrame(log).to_csv(old, index=False)
    m = collect(out_dir, a.bench, a.seeds)
    if len(m):
        m.to_csv(os.path.join(out_dir, "metrics.csv"), index=False)
        print("\n=== %s: test metrics per benchmark and seed (mean over outer folds) ===" % a.name)
        print(m.round(3).to_string(index=False))
        mm = pd.read_csv(os.path.join(out_dir, "models.csv")) if os.path.exists(os.path.join(out_dir, "models.csv")) else None
        if mm is not None:
            g = mm.groupby("benchmark").agg(models=("inner", "size"), collapsed=("collapsed", "sum"),
                                            early=("stopped_early", "mean"), epochs=("epochs_run", "mean")).round(2)
            print("\n=== training health: models trained, collapsed heads (val AUROC < 0.55), share stopped early, mean epochs ===")
            print(g.to_string())


if __name__ == "__main__":
    main()

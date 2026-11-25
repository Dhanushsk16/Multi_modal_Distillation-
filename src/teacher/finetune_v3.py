"""Fine-tune the VERSION 3 teacher on the benchmarks and compare it with its frozen embedding.
finetune.py is the same experiment for the older teachers t1-t11, whose network class and checkpoint format differ; the recipe, the
splits, the ensembles, the metrics and the output files here are the same as there (read its docstring for the reasons).

    python src/teacher/finetune_v3.py --name v3_base --seeds 3                          # PeptideCLM-2's five benchmarks: the teacher FINE-TUNED
    python src/teacher/finetune_v3.py --name v3_base --suite pepland --seeds 3          # PepLand's five tasks
    python src/teacher/finetune_v3.py --name v3_base --bench CellPPD --epochs 3 --inner 2         # a quick smoke test
    python src/teacher/finetune_v3.py --name v3_base --resume                           # skip (benchmark, variant) pairs already in raw.csv
By default only `ft` runs. The frozen-latent probes (linear / gbt / MLP) are already in the evaluation harness; `--modes frozen scratch ecfp`
are opt-in controls that this run does not need.

What is trained: only the teacher's own networks (the three input encoders, their LayerNorms and the trunk). The frozen encoders
(PeptideCLM-2, PepLand) are NOT touched: their vectors come from features/<suite>/{pepland,pclm,ecfp}.npz, standardised with the
statistics stored in the teacher's checkpoint, exactly as in training. ESM-2 is a reconstruction target of the teacher only: its
decoders are neither used nor trained here, and ESM-2 is never an input. The latent (1024-d) is standardised with the run's
latent_stats.npz, the same numbers as in features/<suite>/teacher_<name>.npz.

Variants (one head and one recipe for all):
    ft       (the default) teacher from the checkpoint, its encoders and trunk trained END TO END with the labels at a LOW learning rate
             (--lr-backbone 1e-5, the lr of the PeptideCLM-2 paper's full fine-tuning), the new head at --lr-head 1e-3; view dropout is off
    frozen   (opt-in) teacher weights fixed, the same head trained on the standardised 1024-d latent: the paired control of `ft`
    scratch  (opt-in) the same architecture from random weights at --lr-scratch 1e-3: what the teacher's training adds
    ecfp     (opt-in) a 2048-1024-d MLP on log1p(ECFP counts) standardised with the benchmark's own statistics: a single-view control
Recipe (as finetune.py): head Linear(d + extra, d) - SiLU - Dropout(0.1) - Linear(d, 1); AdamW, weight decay 1e-3, gradient clip 1.0,
batch 64, linear warm-up over 10% of the steps then linear decay, at most 100 epochs, early stopping after 10 epochs without a better
validation loss (best epoch kept; the test fold is never used for any choice); every test fold is predicted by the MEAN of several models
(--inner 5 random inner folds; AmpHGT: its own validation file, one model; PAMPA: the authors' 5-models-per-cluster scheme).
Splits: those of benchmarks.py. Metrics: eval_single's, averaged over folds AND pooled over all folds (pooled_*); every score is also
given for the test molecules WITH an ESM-2 vector (standard chains) and WITHOUT (modified chains): head_esm / head_noesm.
Output: $RESULTS_DIR/finetune_<name>[_pepland]/raw.csv and preds_<benchmark>_<variant>.csv.
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
import finetune as FT  # noqa: E402            (load_ft, inner_splits, headline, MODES: shared with the older teachers)
import train_teacher_v3 as V3  # noqa: E402
from paths import FEAT_DIR, RESULTS_DIR  # noqa: E402

MODES = FT.MODES


# ------------------------------------------------------------------------------------------- data
def load_raw(feat_dir, inputs):
    """view -> (smiles index, array as the teacher saw it in embed_benchmarks, ok); plus the ESM-2 flag of the benchmark molecules."""
    raw = {}
    for v in inputs:
        z = np.load(os.path.join(feat_dir, v + ".npz"), allow_pickle=True)
        X = np.minimum(z["X"], 255).astype(np.uint8) if v == "ecfp" else z["X"].astype(np.float16)
        raw[v] = (dict(zip(z["smiles"].tolist(), range(len(z["smiles"])))), X, z["ok"].astype(bool))
    esm = None
    p = os.path.join(feat_dir, "esm.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        esm = (dict(zip(z["smiles"].tolist(), range(len(z["smiles"])))), z["ok"].astype(bool))
    return raw, esm


def bench_data(bench, raw, esm, inputs, st, dev, want_ecfp=False):
    """-> dict: df (usable rows), xs (list of standardised input tensors, in the teacher's input order), ecfp (the control's input),
    extra (context columns), esm (bool per row: standard chain), cov (share of usable rows)."""
    df = FT.load_ft(bench).dropna(subset=["smiles"]).reset_index(drop=True)
    df["smiles"] = df.smiles.astype(str)
    ii = {v: np.array([raw[v][0][s] for s in df.smiles]) for v in inputs}
    ok = np.logical_and.reduce([raw[v][2][ii[v]] for v in inputs])
    df, ii = df[ok].reset_index(drop=True), {v: i[ok] for v, i in ii.items()}
    xs = [V3.prep_t(v, torch.from_numpy(np.ascontiguousarray(raw[v][1][ii[v]])).to(dev), st) for v in inputs]
    ecfp = None
    if want_ecfp:                                            # log1p counts standardised with this benchmark's own molecules (no labels)
        assert "ecfp" in raw, "the ecfp control needs the ecfp view among the teacher's inputs"
        X = np.log1p(np.minimum(raw["ecfp"][1][ii["ecfp"]], 255).astype(np.float32))
        mu, sd = X.mean(0), X.std(0)
        sd = np.maximum(sd, 0.1 * np.sqrt((sd ** 2).mean()) + 1e-6)
        ecfp = [torch.from_numpy(((X - mu) / sd).astype(np.float32)).clamp_(-10, 10).to(dev)]
    cols = [c for c in df.columns if c.startswith("extra_")]
    extra = torch.from_numpy(df[cols].to_numpy(np.float32)).to(dev) if cols else torch.zeros((len(df), 0), device=dev)
    flag = None if esm is None else np.array([bool(esm[1][esm[0][s]]) if s in esm[0] else False for s in df.smiles])
    return {"df": df, "xs": xs, "ecfp": ecfp, "extra": extra, "cov": float(ok.mean()), "esm": flag}


def take(xs, b):
    return [x[b] for x in xs]


# ------------------------------------------------------------------------------------------- model
class Net(nn.Module):
    def __init__(self, mode, model0, n_extra, head_dropout, lat, dev):
        super().__init__()
        self.mode = mode
        cfg = model0.cfg
        d = cfg["latent"]
        self.enc = None
        if mode == "ecfp":
            self.enc = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, d))
        elif mode in ("ft", "scratch"):
            self.enc = V3.TeacherV3.from_cfg(cfg, 0.0)
            if mode == "ft":
                self.enc.load_state_dict({k: t.detach().clone().cpu() for k, t in model0.state_dict().items()})
        use_lat = lat is not None and mode in ("frozen", "ft")          # a random network has no trained latent scale to undo
        self.register_buffer("mu", torch.from_numpy(np.asarray(lat["mu"], np.float32)) if use_lat else torch.zeros(0))
        self.register_buffer("sd", torch.from_numpy(np.asarray(lat["sd"], np.float32)) if use_lat else torch.zeros(0))
        self.head = nn.Sequential(nn.Linear(d + n_extra, d), nn.SiLU(), nn.Dropout(head_dropout), nn.Linear(d, 1))

    def latent_std(self, z):
        return (z - self.mu) / self.sd if self.mu.numel() else z

    def forward(self, xs, extra, cached=None):
        if cached is not None:
            s = cached
        elif self.mode == "ecfp":
            s = self.enc(xs[0])
        else:
            s = self.latent_std(self.enc.latent(xs))
        return self.head(torch.cat([s, extra], 1)).squeeze(1)

    def backbone_params(self):
        """The teacher's trained parts: encoders, norms and trunk (the decoders are not in the graph)."""
        if self.mode in ("ft", "scratch"):
            return [p for n, p in self.enc.named_parameters() if not n.startswith("dec.") and not n.startswith("hide")]
        return list(self.enc.parameters())


def fit_one(mode, task, data, y, fit, val, pred, seed, model0, lat, a, dev, cache):
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

    net = Net(mode, model0, ex.shape[1], a.head_dropout, lat, dev).to(dev)
    head_p = list(net.head.parameters())
    if mode == "frozen":
        groups = [{"params": head_p, "lr": a.lr_head}]
    elif mode == "ft":
        groups = [{"params": head_p, "lr": a.lr_head}, {"params": net.backbone_params(), "lr": a.lr_backbone}]
    else:
        groups = [{"params": head_p + net.backbone_params(), "lr": a.lr_scratch}]
    clip_params = [p for g in groups for p in g["params"]]
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
            nn.utils.clip_grad_norm_(clip_params, 1.0)
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


def run_bench(bench, mode, data, model0, lat, a, dev, cache, out_dir):
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
            for j, (fit, val) in enumerate(FT.inner_splits(df, bench, fo, te, a.inner, seed)):
                o, info = fit_one(mode, task, data, y, fit, val, te, seed * 1000 + fi * 100 + j, model0, lat, a, dev, cache)
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
            row.update({"head_esm": FT.headline(task, y[m & esm], allp[m & esm]), "n_esm": int((m & esm).sum()),
                        "head_noesm": FT.headline(task, y[m & ~esm], allp[m & ~esm]), "n_noesm": int((m & ~esm).sum())})
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
    ap.add_argument("--name", required=True, help="the version 3 teacher run to start from (checkpoints/teacher/<name>)")
    ap.add_argument("--bench", nargs="*", default=None)
    ap.add_argument("--suite", choices=("main", "pepland"), default="main",
                    help="main = PeptideCLM-2's five benchmarks (features/single_modality); pepland = PepLand's five tasks "
                         "(PL_cCPP PL_cSol PL_ncCPP PL_cBinding PL_ncBinding, features/pepland_suite, our 5-fold protocol)")
    ap.add_argument("--feat", default=None, help="another features subfolder under features/ (default: the suite's own)")
    ap.add_argument("--out", default=None, help="results folder under results/ (default finetune_<name>[_pepland])")
    ap.add_argument("--modes", nargs="*", default=["ft"], choices=MODES,
                    help="default: ft, the teacher fine-tuned; frozen (same head on the fixed latent), scratch (random weights) and ecfp (ECFP MLP) are opt-in controls")
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
    model0, st, lat = V3.load_model(a.name, dev, a.use_last)
    for p in model0.parameters():
        p.requires_grad_(False)
    inputs = list(model0.inputs)
    pl = a.suite == "pepland"
    feat = a.feat or ("pepland_suite" if pl else None)
    feat_dir = os.path.join(os.path.dirname(FEAT_DIR), feat) if feat else FEAT_DIR
    raw, esm = load_raw(feat_dir, inputs)
    benches = a.bench or (list(BM.PL_SUITE) if pl else E.selected())
    out = os.path.join(RESULTS_DIR, a.out or ("finetune_%s%s" % (a.name, "_pepland" if pl else "")))
    os.makedirs(out, exist_ok=True)
    print("features: %s | results: %s | latent statistics: %s" % (feat_dir, out, "yes" if lat is not None else "NO (raw latent)"), flush=True)
    print("teacher %s: inputs %s, latent %d-d, %.1fM parameters | variants %s | benchmarks %s\n"
          "recipe: AdamW wd %g, batch %d, <= %d epochs, patience %d, warm-up 10%% + linear decay, clip 1.0, head dropout %g | "
          "lr: head %g, teacher %g (ft), from scratch %g | inner folds %d | seeds %d"
          % (a.name, inputs, model0.cfg["latent"], sum(p.numel() for p in model0.parameters()) / 1e6, a.modes, benches,
             a.wd, a.batch, a.epochs, a.patience, a.head_dropout, a.lr_head, a.lr_backbone, a.lr_scratch, a.inner, a.seeds), flush=True)
    raw_path = os.path.join(out, "raw.csv")
    rows = pd.read_csv(raw_path).to_dict("records") if (a.resume and os.path.exists(raw_path)) else []
    done = {(r["benchmark"], r["mode"]) for r in rows if sum(1 for q in rows if q["benchmark"] == r["benchmark"]
                                                              and q["mode"] == r["mode"]) >= a.seeds}
    for b in benches:
        data = bench_data(b, raw, esm, inputs, st, dev, "ecfp" in a.modes)
        cache = None
        if "frozen" in a.modes:                          # the weights never change: embed once per benchmark
            net0 = Net("frozen", model0, 0, 0.0, lat, dev).to(dev)
            with torch.no_grad():
                cache = torch.cat([net0.latent_std(model0.latent(take(data["xs"], torch.arange(i, min(i + 4096, len(data["df"])), device=dev))))
                                   for i in range(0, len(data["df"]), 4096)]).float()
            print("%s: %d usable molecules (%.1f%% of the file)%s | frozen embedding std per dimension %.3f"
                  % (b, len(data["df"]), 100 * data["cov"],
                     "" if data["esm"] is None else ", standard chains %.1f%%" % (100 * data["esm"].mean()),
                     float(cache.std(0).mean())), flush=True)
        for m in a.modes:
            if (b, m) in done:
                print("%-8s %-8s already in raw.csv: skipped" % (b, m), flush=True)
                continue
            t0 = time.time()
            new = run_bench(b, m, data, model0, lat, a, dev, cache, out)
            rows += new
            key = "MCC" if E.TASK[b] == "clf" else "R2"
            r = new[-1]
            print("%-8s %-8s done (%.0fs) | %s %.3f%s | epochs run %.0f, best at %.0f, hit the %d-epoch cap in %d of %d fits"
                  % (b, m, time.time() - t0, key, r[key],
                     (" (pooled %.3f)" % r["pooled_" + key]) if "pooled_" + key in r else "",
                     r["epochs"], r["best_epoch"], a.epochs, r["hit_cap"], r["fits"]), flush=True)
            pd.DataFrame(rows).to_csv(raw_path, index=False)
    res = pd.DataFrame(rows)
    order = [m for m in MODES if m in res["mode"].values]
    col = lambda n: res[n] if n in res else pd.Series(np.nan, index=res.index)                    # noqa: E731
    is_clf = res.benchmark.map(E.TASK) == "clf"
    res["headline"] = np.where(is_clf, col("MCC"), col("R2"))
    print("\n=== headline: MCC (classification) / R2 (PAMPA), mean over folds and seeds ===")
    print(res.groupby(["benchmark", "mode"]).headline.mean().unstack("mode").round(3)[order].to_string())
    if "pooled_R2" in res or "pooled_MCC" in res:
        res["pooled"] = np.where(is_clf, col("pooled_MCC"), col("pooled_R2"))
        pv = res.dropna(subset=["pooled"])
        print("\n=== the same on the POOLED predictions of all folds (cluster-CV benchmarks only) ===")
        print(pv.groupby(["benchmark", "mode"]).pooled.mean().unstack("mode").round(3)[[m for m in order if m in pv["mode"].values]].to_string())
    if "AUROC" in res:
        print("\n=== AUROC (classification) ===")
        print(res.dropna(subset=["AUROC"]).groupby(["benchmark", "mode"]).AUROC.mean().unstack("mode").round(3)[order].to_string())
    if "head_esm" in res:
        print("\n=== test molecules that are standard chains (ESM-2 vector exists) / modified chains: MCC or R2 ===")
        print(res.groupby(["benchmark", "mode"])[["n_esm", "head_esm", "n_noesm", "head_noesm"]].mean().round(3).to_string())
    print("\n=== convergence: epochs run / epoch of the best validation loss (means), and fits that hit the epoch cap ===")
    print(res.groupby(["benchmark", "mode"]).agg(run=("epochs", "mean"), best=("best_epoch", "mean"),
                                                 capped=("hit_cap", "sum"), fits=("fits", "sum")).round(1).to_string())
    if "MCC_le30" in res:
        print("\nAmpHGT test restricted to <=30 residues (MCC):")
        print(res[res.benchmark == "AmpHGT"].groupby("mode").MCC_le30.mean().round(3).to_string())
    print("\nwrote", out)


if __name__ == "__main__":
    main()

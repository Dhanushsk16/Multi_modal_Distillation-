"""PepMSND with KAN networks: the benchmark's own recipe, with the frozen teacher (and every baseline input) as the embedding branch.

    python src/teacher/pepmsnd_kan_v3.py --teacher v3_base                       # all inputs, modes kan / emb / fuse / desc_gbt, 3 seeds
    python src/teacher/pepmsnd_kan_v3.py --teacher v3_base --kan-impl pykan      # the official pykan package instead of our own KAN layer
    python src/teacher/pepmsnd_kan_v3.py --teacher v3_base --reps teacher_v3_base --seeds 1 --epochs 5     # a quick look

Where it comes from. PepMSND (Hu et al., Digital Discovery 2025; bioRxiv 2024.12.12.628290) predicts peptide blood stability (stable = more than 50% intact
after 1 h) on 635 peptides with 10-fold cross-validation. Its 0D branch is a Kolmogorov-Arnold Network (KAN) on 140 RDKit descriptors chosen by a random
forest, plus one-hot species and one-hot blood environment; "a series of learnable weights" combines the branches into one joint vector, followed by a shared
layer. Without the KAN its MCC drops from 0.7259 to 0.5370. The journal PeptideCLM-2 paper (J. Chem. Inf. Model. 2026, 66, 8110) puts PeptideCLM-2 embeddings
into the same KAN + feature-fusion recipe: MCC 0.729 / 0.741 / 0.749 (MLM / Hybrid / MTR); PeptideCLM-2 without the KAN is about 0.6 (read off its Fig. 5B).
This script was written BEFORE the PeptideCLM-2 authors' own PepMSND training scripts were found (github.com/AaronFeller/PeptideCLM-2, copied verbatim to
external/peptideclm2_authors/): the KAN width, grid and training settings below are OURS, not theirs, and the embedding is frozen (only the heads train). The
authors' recipe, with the backbone trained, is what src/teacher/pepmsnd_ft_kan_v3.py runs with our teacher as the backbone.

Data: data/benchmarks/PepMSND_clustered_data.csv = the benchmark's own 10 folds (column cluster) with exactly the 140 selected descriptors, Species,
Environment and the label; 640 rows. Every mode uses the same rows and the same folds. The embeddings are read like the evaluation harness reads them
(eval_harness.matrix), so ECFP / PeptideCLM-2 / PepLand / the three joined / PCA / the teacher are the numbers of the earlier tables.

Modes (per embedding input; one run per seed; every fold: 10% of the training rows held out for early stopping):
    kan        descriptors + one-hot species / environment -> KAN                   (the authors' 0D branch alone; no embedding)
    emb        embedding + one-hot species / environment -> small MLP branch         (no descriptors, no KAN: the "without KAN" analogue)
    fuse       both branches, joined by learnable branch weights                      (the PepMSND recipe)
    desc_gbt   gradient-boosted trees on the descriptors + species / environment      (descriptors without a KAN, for reference)
Network: branch -> D-dim vector; joint vector = [w_k * KAN vector, w_e * embedding vector] (learnable scalars w, start at 1); shared layer
Linear(n*D, D) - ReLU - Dropout(0.1) - Linear(D, 1); BCE loss; AdamW; early stopping on the validation loss.
Our KAN (--kan-impl ours): two KANLinear layers (descriptors -> hidden -> D), each = a SiLU "base" path + a cubic B-spline path on a fixed grid
(--grid intervals on [-3, 3], order 3) with one learnable spline per input-output pair, with a LayerNorm between the layers so the second layer's inputs
stay inside its grid. Descriptors are z-scored with the training rows and clipped at +-3. --kan-impl pykan uses the official package instead
(pip install --no-deps the wheel; widths [in, hidden, D], same grid / order / range).

Score: MCC at 0.5 on the pooled out-of-fold predictions of the 10 folds (the harness convention; the fold mean is in raw.csv too), AUROC, ACC, F1;
mean +- sd over --seeds. Output: $RESULTS_DIR/<out>/raw.csv and summary.csv.
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
import eval_harness as H  # noqa: E402
import eval_single as E  # noqa: E402
from paths import BENCH_DIR, RESULTS_DIR  # noqa: E402

META = ["ID", "PMID", "SMILES", "Length", "Species", "Environment", "SE-3", "label", "cluster"]
NAMES = {"H": "ECFP counts", "L": "PeptideCLM-2 (frozen)", "P": "PepLand (frozen)", "P+L+H": "the three joined", "PCA": "PCA of the three"}
PUBLISHED = [("PepMSND paper, its full model with KAN (its own protocol)", 0.726), ("PepMSND paper, without its KAN module", 0.537),
             ("PeptideCLM-2 MLM + KAN (journal paper)", 0.729), ("PeptideCLM-2 Hybrid + KAN", 0.741), ("PeptideCLM-2 MTR + KAN", 0.749),
             ("PeptideCLM-2 without KAN (read off its Fig. 5B, about)", 0.60)]


# ------------------------------------------------------------------------------------------- data
def load_pepmsnd():
    d = pd.read_csv(os.path.join(BENCH_DIR, "PepMSND_clustered_data.csv"))
    cols = [c for c in d.columns if c not in META]
    assert len(cols) == 140, "expected the 140 selected descriptors, found %d columns" % len(cols)
    desc = np.nan_to_num(d[cols].to_numpy(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    env = np.stack([d.Species == 0, d.Species == 1, d.Environment == 0, d.Environment == 1], 1).astype(np.float64)
    return d.SMILES.astype(str).to_numpy(), d.label.to_numpy().astype(np.float32), d.cluster.to_numpy().astype(int), desc, env


def embeddings(smiles, reps):
    """rep -> (features [N, d], ok [N]) exactly as the harness builds them (no context columns: species / environment are added here)."""
    store = H.Store(H.FEAT_DIR)
    st = H.V.load_stats() if os.path.exists(H.V.STATS_FILE) else None
    pca = H.PB.load() if os.path.exists(H.PB.PCA_FILE) else None
    df = pd.DataFrame({"smiles": smiles})
    out = {}
    for r in reps:
        if r == "PCA" and (st is None or pca is None):
            print("PCA skipped: run views_v3.py --stats and pca_baseline.py --fit first", flush=True)
            continue
        m = H.matrix(store, df, r, st, pca)
        if m is None:
            print("%s skipped: a feature file is missing" % r, flush=True)
            continue
        out[r] = (m[0].astype(np.float64), m[1])
    return out


# ------------------------------------------------------------------------------------------- KAN
class KANLinear(nn.Module):
    """One KAN layer: out_j = SiLU(x) @ base_j + sum_i spline_ji(x_i), each spline a cubic B-spline on a fixed grid of `grid` intervals on [-lim, lim]
    (extended by `order` knots on both sides), with its own learnable coefficients."""

    def __init__(self, d_in, d_out, grid=5, order=3, lim=3.0, noise=0.1):
        super().__init__()
        h = 2.0 * lim / grid
        self.register_buffer("knots", (torch.arange(-order, grid + order + 1) * h - lim).float())     # grid + 2 * order + 1 knots
        self.order = order
        self.base = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.kaiming_uniform_(self.base, a=math.sqrt(5))
        self.spline = nn.Parameter(torch.randn(d_out, d_in, grid + order) * noise / grid)

    def bases(self, x):
        """x [B, d_in] -> B-spline basis values [B, d_in, grid + order] (Cox - de Boor recursion on the shared knots)."""
        t = self.knots
        x = x.unsqueeze(-1)
        b = ((x >= t[:-1]) & (x < t[1:])).to(x.dtype)
        for k in range(1, self.order + 1):
            b = (x - t[:-(k + 1)]) / (t[k:-1] - t[:-(k + 1)]) * b[..., :-1] + (t[k + 1:] - x) / (t[k + 1:] - t[1:-k]) * b[..., 1:]
        return b

    def forward(self, x):
        return F.linear(F.silu(x), self.base) + F.linear(self.bases(x).flatten(1), self.spline.flatten(1))


class KANStack(nn.Module):
    def __init__(self, d_in, hidden, d_out, grid, order, lim):
        super().__init__()
        self.l1, self.norm, self.l2 = KANLinear(d_in, hidden, grid, order, lim), nn.LayerNorm(hidden), KANLinear(hidden, d_out, grid, order, lim)

    def forward(self, x):
        return self.l2(self.norm(self.l1(x)))


class PyKANStack(nn.Module):
    """The official pykan package (pip install --no-deps pykan-0.2.8-py3-none-any.whl), widths [d_in, hidden, d_out]."""

    def __init__(self, d_in, hidden, d_out, grid, order, lim, seed, dev):
        super().__init__()
        from kan import KAN
        self.m = KAN(width=[d_in, hidden, d_out], grid=grid, k=order, seed=seed, device=str(dev), auto_save=False, grid_range=[-lim, lim])
        if hasattr(self.m, "speed"):
            self.m.speed()                                   # no symbolic branch, no activation saving: plain training speed

    def forward(self, x):
        return self.m(x)


def make_kan(a, d_in, seed, dev):
    if a.kan_impl == "pykan":
        return PyKANStack(d_in, a.kan_hidden, a.branch_dim, a.grid, a.order, a.lim, seed, dev)
    return KANStack(d_in, a.kan_hidden, a.branch_dim, a.grid, a.order, a.lim)


class Net(nn.Module):
    def __init__(self, mode, d_desc, d_emb, a, seed, dev):
        super().__init__()
        D = a.branch_dim
        self.kan = make_kan(a, d_desc, seed, dev) if mode in ("kan", "fuse") else None
        self.emb = nn.Sequential(nn.Linear(d_emb, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, D)) if mode in ("emb", "fuse") else None
        n = (self.kan is not None) + (self.emb is not None)
        self.w = nn.Parameter(torch.ones(n))                 # the learnable branch weights
        self.head = nn.Sequential(nn.Linear(n * D, D), nn.ReLU(), nn.Dropout(0.1), nn.Linear(D, 1))

    def forward(self, xd, xe):
        hs, i = [], 0
        if self.kan is not None:
            hs.append(self.w[i] * self.kan(xd))
            i += 1
        if self.emb is not None:
            hs.append(self.w[i] * self.emb(xe))
        return self.head(torch.cat(hs, 1)).squeeze(1)


# ------------------------------------------------------------------------------------------- one fit
def split_fit_val(tr, y, seed, frac=0.1):
    rng = np.random.RandomState(seed)
    val = np.concatenate([rng.choice(tr[y[tr] == c], max(1, int(round(frac * (y[tr] == c).sum()))), replace=False) for c in np.unique(y[tr])])
    return np.setdiff1d(tr, val), np.sort(val)


def zscore(X, rows, clip=None):
    mu, sd = X[rows].mean(0), X[rows].std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Z = (X - mu) / sd
    return np.clip(Z, -clip, clip) if clip else Z


def fit_predict(mode, Xd, Xe, y, tr, te, seed, a, dev):
    """-> probabilities on te, epochs. Xd: descriptors + one-hot (KAN input), Xe: embedding + one-hot (embedding branch input)."""
    fit, val = split_fit_val(tr, y, seed)
    Td = torch.as_tensor(zscore(Xd, tr, clip=a.lim), dtype=torch.float32, device=dev)
    Te = torch.as_tensor(zscore(Xe, tr), dtype=torch.float32, device=dev) if Xe is not None else Td[:, :1]
    yt = torch.as_tensor(y, dtype=torch.float32, device=dev)
    torch.manual_seed(seed)
    net = Net(mode, Td.shape[1], Te.shape[1], a, seed, dev).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=a.wd)
    rng = np.random.RandomState(seed + 7)
    val_t = torch.as_tensor(val, device=dev)
    best, best_state, bad, ep = float("inf"), None, 0, 0
    for ep in range(1, a.epochs + 1):
        net.train()
        order = rng.permutation(fit)
        for i in range(0, len(order), a.batch):
            b = torch.as_tensor(order[i:i + a.batch], device=dev)
            if len(b) < 2:
                continue
            loss = F.binary_cross_entropy_with_logits(net(Td[b], Te[b]), yt[b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(F.binary_cross_entropy_with_logits(net(Td[val_t], Te[val_t]), yt[val_t]))
        if v < best - 1e-4:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(Td[torch.as_tensor(te, device=dev)], Te[torch.as_tensor(te, device=dev)])).cpu().numpy()
    return p, ep


def gbt_predict(Xd, y, tr, te, seed):
    m = E.learners("clf", seed)["gbt"]
    m.fit(Xd[tr], y[tr])
    return m.predict_proba(Xd[te])[:, 1], 0


# ------------------------------------------------------------------------------------------- run
def run_one(mode, Xd, Xe, y, folds, seed, a, dev):
    allp = np.full(len(y), np.nan)
    per, eps = [], []
    for f in sorted(set(folds)):
        te = np.flatnonzero(folds == f)
        tr = np.flatnonzero(folds != f)
        p, ep = gbt_predict(Xd, y, tr, te, seed) if mode == "desc_gbt" else fit_predict(mode, Xd, Xe, y, tr, te, seed, a, dev)
        allp[te] = p
        per.append(E.score("clf", y[te], p)["MCC"])
        eps.append(ep)
    s = E.score("clf", y, allp)
    return {"MCC": float(s["MCC"]), "MCC_foldmean": float(np.mean(per)), "AUROC": float(s["AUROC"]), "ACC": float(s["ACC"]), "F1": float(s["F1"]),
            "epochs": float(np.mean(eps))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", default="v3_base", help="the teacher whose frozen embedding (features/<suite>/teacher_<name>.npz) is one input")
    ap.add_argument("--reps", nargs="*", default=None, help="embedding inputs (default: H L P P+L+H PCA and the teacher)")
    ap.add_argument("--modes", nargs="*", default=["kan", "emb", "fuse", "desc_gbt"], choices=["kan", "emb", "fuse", "desc_gbt"])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--kan-impl", choices=("ours", "pykan"), default="ours")
    ap.add_argument("--grid", type=int, default=5, help="spline grid intervals")
    ap.add_argument("--order", type=int, default=3, help="spline order")
    ap.add_argument("--lim", type=float, default=3.0, help="grid range [-lim, lim] (descriptors are z-scored and clipped to it)")
    ap.add_argument("--kan-hidden", type=int, default=32)
    ap.add_argument("--branch-dim", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="pepmsnd_kan")
    a = ap.parse_args()
    dev = torch.device(a.device)
    smiles, y, folds, desc, env = load_pepmsnd()
    reps = a.reps or ["H", "L", "P", "P+L+H", "PCA", "teacher_" + a.teacher]
    emb = embeddings(smiles, [r for r in reps if "emb" in a.modes or "fuse" in a.modes])
    ok = np.isfinite(desc).all(1)
    for r, (_, k) in emb.items():
        ok &= k
    keep = np.flatnonzero(ok)
    print("PepMSND: %d of %d molecules usable in every input, %d folds | descriptors %d + one-hot %d | KAN %s (grid %d, order %d, hidden %d, branch %d) | "
          "modes %s | inputs %s | seeds %d | device %s" % (len(keep), len(ok), len(set(folds[keep])), desc.shape[1], env.shape[1], a.kan_impl, a.grid, a.order,
                                                           a.kan_hidden, a.branch_dim, a.modes, list(emb), a.seeds, dev), flush=True)
    y, folds, desc, env = y[keep], folds[keep], desc[keep], env[keep]
    Xd = np.concatenate([desc, env], 1)
    out = os.path.join(RESULTS_DIR, a.out)
    os.makedirs(out, exist_ok=True)
    jobs = []
    if "kan" in a.modes:
        jobs.append(("kan", "-", None))
    if "desc_gbt" in a.modes:
        jobs.append(("desc_gbt", "-", None))
    for r in emb:
        for m in ("emb", "fuse"):
            if m in a.modes:
                jobs.append((m, r, np.concatenate([emb[r][0][keep], env], 1)))
    rows = []
    for mode, rep, Xe in jobs:
        for seed in range(a.seeds):
            t0 = time.time()
            res = run_one(mode, Xd, Xe, y, folds, seed, a, dev)
            rows.append({"mode": mode, "rep": rep, "seed": seed, **res})
            print("%-9s %-22s seed %d | MCC %.3f (fold mean %.3f) AUROC %.3f | epochs %.0f | %.0fs" % (mode, rep, seed, res["MCC"], res["MCC_foldmean"],
                                                                                                    res["AUROC"], res["epochs"], time.time() - t0), flush=True)
            pd.DataFrame(rows).to_csv(os.path.join(out, "raw.csv"), index=False)
    raw = pd.DataFrame(rows)
    summ = raw.groupby(["mode", "rep"])[["MCC", "MCC_foldmean", "AUROC", "ACC", "F1"]].agg(["mean", "std"])
    summ.to_csv(os.path.join(out, "summary.csv"))
    f = lambda m, s: "%.3f +- %.3f" % (m, 0.0 if pd.isna(s) else s)               # noqa: E731
    g = raw.groupby(["mode", "rep"]).MCC.agg(["mean", "std"])
    label = lambda r: NAMES.get(r, "%s (frozen)" % r[len("teacher_"):] if r.startswith("teacher_") else r)     # noqa: E731
    print("\n=== PepMSND, 10-fold cluster CV, MCC pooled over the folds, mean +- sd over %d seeds ===" % a.seeds)
    tab = {}
    for r in emb:
        tab[label(r)] = {m: f(*g.loc[(m, r)]) if (m, r) in g.index else "-" for m in ("emb", "fuse")}
    if tab:
        t = pd.DataFrame(tab).T
        t["KAN gain (fuse - emb)"] = ["%+.3f" % (g.loc[("fuse", r), "mean"] - g.loc[("emb", r), "mean"]) if ("fuse", r) in g.index and ("emb", r) in g.index else "-" for r in emb]
        t.columns = ["embedding only (no KAN)", "embedding + KAN (fuse)", "KAN gain (fuse - emb)"]
        print(t.to_string())
    for m, title in (("kan", "descriptors + species / environment, KAN alone"), ("desc_gbt", "descriptors + species / environment, gradient-boosted trees")):
        if (m, "-") in g.index:
            print("%-62s %s" % (title, f(*g.loc[(m, "-")])))
    print("\npublished, for comparison (their protocols; the first is the PepMSND paper's own 10-fold average):")
    for n, v in PUBLISHED:
        print("  %-62s %.3f" % (n, v))
    print("\nwrote", out, flush=True)


if __name__ == "__main__":
    main()

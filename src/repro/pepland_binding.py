"""PepLand's two binding tasks (c-binding, nc-binding) the way the PepLand authors say they ran them.

    python src/repro/pepland_binding.py --protein-file features/pepland_suite/protein_esm_t12.npz [--seeds 10] [--gru]

WHAT IS KNOWN (sources, so nothing below is a guess presented as a fact)
  * Protein features: ESM-2 t12 (esm2_t12_35M_UR50D, 480-d). The authors, GitHub issue #6 of zhangruochi/pepland (2024-07-18):
    "for properties related to protein-peptide interactions, such as affinity prediction, we utilized ESM2_t12 to extract protein features.
    To ensure fairness ..., the protein feature extraction model remained consistent across all comparisons." (The paper's "largest ESM2,
    48 layers" is about the ESM2 BASELINE rows, not about these protein features.)
  * Splits: "the division of the training set, validation set and test set ... is done randomly" (same issue; also: "random division can
    indeed lead to some data leakage"). The repository's own function is splitters.random_split: python `random.seed(seed)`, shuffle the
    row numbers, 80 % train / 10 % valid / 10 % test. Copied here (split_random). The ratio is the function's default; the paper gives none.
  * Training (paper, Experimental setup): "we only connected two simple linear layers to the representations", batch size 32, 200 epochs
    for the affinity tasks, learning rate 5e-5 (1e-5 only for the CPP tasks), Adam; GRU pooling beat max and average pooling on all tasks.
  * Metrics: Spearman (SCC) and Pearson (PCC); published (Table 1 / Table 5 of arXiv 2311.04419): c-binding SCC 0.503, PCC 0.45,
    MAE 1.146, RMSE 1.493; nc-binding SCC 0.768, PCC 0.873, MAE 0.762, RMSE 1.045. ESM2 alone on c-binding: SCC 0.452; PepLand + ESM: 0.454 / 0.530.
  * NOT stated anywhere (the repository ships no downstream code; issue #3 "code for downstream tasks" is still open): how the protein and
    peptide vectors are combined, the hidden width, the loss, the model-selection rule, the number of repeats. Choices here: concatenation,
    width 256, MSE on min-max scaled targets behind a sigmoid (as the released PropertyPredictor), the epoch with the lowest validation
    loss, inputs z-scored with training statistics.
  * Peptide vectors: the cached frozen PepLand avg-pooled 300-d vectors (padding_mode="exclude": only real nodes). The paper's era code
    averaged over padded batches ("legacy", batch-dependent; issues #12/#13): that cannot be reproduced exactly and is not a better
    representation. With --gru the trainable GRU pooling of the released Node_GRU design is run on the cached node vectors
    (features/pepland_suite/pepland_nodes.npz; src/finetune/pool_probe.py).

WHAT IS RUN (every cell: mean and sd over --seeds random splits)
  feature sets: protein alone | peptide alone | both joined (| both with the GRU pooling, --gru);  heads: two = Linear(d,256) ReLU Linear(256,1)
  (the paper's "two linear layers"), released = the repository's PropertyPredictor MLP (256-128, dropout 0.1);
  splits: random (the paper's) and protein (all rows of a protein in one part: no test protein seen in training, the leak-free control).
Output: $RESULTS_DIR/<out>/raw.csv and a printed table next to the published numbers.
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..")
sys.path.insert(0, SRC)
from paths import BENCH_DIR, FEAT_DIR, RESULTS_DIR  # noqa: E402

SUITE = os.path.join(os.path.dirname(FEAT_DIR), "pepland_suite")
TASKS = {"c-binding": "c-binding.csv", "nc-binding": "nc-binding.csv"}
PUB = {"c-binding": "SCC 0.503, PCC 0.450, MAE 1.146, RMSE 1.493 (ESM2 baseline SCC 0.452; PepLand + ESM 0.454 / PCC 0.530)",
       "nc-binding": "SCC 0.768, PCC 0.873, MAE 0.762, RMSE 1.045"}


# ------------------------------------------------------------------------------------------------ data
def load_task(name):
    d = pd.read_csv(os.path.join(BENCH_DIR, "pepland", TASKS[name]), low_memory=False, index_col=0)
    d = d.rename(columns={"Merge_SMILES": "smiles", "affinity": "y"})[["smiles", "y", "seq1"]].dropna().reset_index(drop=True)
    d["smiles"], d["seq1"] = d.smiles.astype(str), d.seq1.astype(str).str.strip()
    return d


def load_npz(path):
    z = np.load(path, allow_pickle=True)
    return z["smiles"].tolist() if "smiles" in z.files else z["seqs"].tolist(), z["X"].astype(np.float32), z["ok"].astype(bool)


def split_random(n, seed, fr_train=0.8, fr_valid=0.1):
    """splitters.random_split of the PepLand repository (row level, 80 / 10 / 10)."""
    random.seed(seed)
    idx = list(range(n))
    random.shuffle(idx)
    a, b = int(fr_train * n), int(fr_valid * n)
    return np.array(idx[:a]), np.array(idx[a:a + b]), np.array(idx[a + b:])


def split_groups(groups, seed, fr_train=0.8, fr_valid=0.1):
    """All rows of a group (a protein) go to the same part; groups are shuffled, parts filled in order up to 80 % / 90 % of the rows."""
    rng = np.random.RandomState(seed)
    g = pd.Series(groups)
    u = g.unique()
    rng.shuffle(u)
    n, cnt = len(g), g.value_counts()
    part = {}
    acc = 0
    for k in u:
        part[k] = 0 if acc < fr_train * n else (1 if acc < (fr_train + fr_valid) * n else 2)
        acc += int(cnt[k])
    p = g.map(part).to_numpy()
    return np.flatnonzero(p == 0), np.flatnonzero(p == 1), np.flatnonzero(p == 2)


# ------------------------------------------------------------------------------------------------ heads
def make_head(kind, d):
    if kind == "two":
        return nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 1))
    return nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, 1))


def standardise(X, fit_rows):
    mu, sd = X[fit_rows].mean(0), X[fit_rows].std(0)
    sd = np.maximum(sd, 0.1 * np.sqrt((sd ** 2).mean()) + 1e-6)
    return np.clip((X - mu) / sd, -10, 10).astype(np.float32)


def metrics(y, p):
    return {"SCC": float(spearmanr(y, p)[0]), "PCC": float(pearsonr(y, p)[0]),
            "MAE": float(np.mean(np.abs(y - p))), "RMSE": float(np.sqrt(np.mean((y - p) ** 2)))}


def fit_mlp(X, y, tr, va, te, head, seed, a, dev):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    Xs = torch.from_numpy(standardise(X, tr)).to(dev)
    lo, hi = float(y[tr].min()), float(y[tr].max())
    yt = torch.from_numpy(((y - lo) / (hi - lo + 1e-8)).astype(np.float32)).to(dev)
    net = make_head(head, X.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    trt, vat, tet = (torch.from_numpy(i).to(dev) for i in (tr, va, te))
    best, state, best_ep = float("inf"), None, 0
    for ep in range(a.epochs):
        net.train()
        order = trt[torch.randperm(len(trt), generator=gen).to(dev)]
        for i in range(0, len(order), a.batch):
            r = order[i:i + a.batch]
            if len(r) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            F.mse_loss(torch.sigmoid(net(Xs[r]).squeeze(1)), yt[r]).backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(F.mse_loss(torch.sigmoid(net(Xs[vat]).squeeze(1)), yt[vat]))
        if v < best:
            best, state, best_ep = v, copy.deepcopy(net.state_dict()), ep + 1
    net.load_state_dict(state)
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(Xs[tet]).squeeze(1)).cpu().numpy()
    return metrics(y[te], p * (hi - lo) + lo), best_ep


def fit_gru(PP, V, rows, P, y, tr, va, te, seed, a, dev):
    """Trainable GRU pooling over the cached PepLand node vectors + the protein vector -> MLP (pool_probe.fit_predict)."""
    trva = np.concatenate([tr, va])
    ex = standardise(P, trva)
    pa = SimpleNamespace(lr=a.lr, epochs=a.epochs, batch=a.batch, patience=10 ** 9, attn_examples=0)
    p, ep, _ = PP.fit_predict(V, rows, ex, y, trva, te, "reg", "gru", seed, pa, dev)
    return metrics(y[te], p), ep


# ------------------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein-file", nargs="+", required=True,
                    help="protein vectors as NAME=PATH (featurize_esm.py --proteins ...); a bare PATH is named after its file")
    ap.add_argument("--peptide-file", default=os.path.join(SUITE, "pepland.npz"))
    ap.add_argument("--nodes-file", default=os.path.join(SUITE, "pepland_nodes.npz"))
    ap.add_argument("--tasks", nargs="+", default=list(TASKS))
    ap.add_argument("--splits", nargs="+", default=["random", "protein"], choices=["random", "protein"])
    ap.add_argument("--features", nargs="+", default=["protein", "peptide", "both"], choices=["protein", "peptide", "both"])
    ap.add_argument("--heads", nargs="+", default=["two", "released"], choices=["two", "released"])
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--gru", action="store_true", help="also: trainable GRU pooling on the cached node vectors + protein (needs --nodes-file)")
    ap.add_argument("--gru-seeds", type=int, default=3)
    ap.add_argument("--device", default="cpu", help="the MLP heads are tiny: cpu is fine; --gru uses cuda when there is one")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--max-rows", type=int, default=0, help="smoke test: a random subset of rows")
    ap.add_argument("--out", default="repro_pepland_binding")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    dev = torch.device(a.device)
    gdev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    smi, PX, pok = load_npz(a.peptide_file)
    prow = {s: i for i, s in enumerate(smi)}
    prots = {}
    for spec in a.protein_file:
        name, path = spec.split("=", 1) if "=" in spec else (os.path.splitext(os.path.basename(spec))[0], spec)
        if not os.path.exists(path):
            print("(no %s: %s skipped)" % (path, name))
            continue
        prots[name] = load_npz(path)
    assert prots, "no protein file found"
    PP = V = None
    first_prot = next(iter(prots))                     # the GRU runs are slow: only for the first protein vector (t12)
    if a.gru:
        if not os.path.exists(a.nodes_file):
            print("(no %s: GRU pooling skipped; python src/featurize_pepland_nodes.py --feat pepland_suite --bench PL_cBinding PL_ncBinding in the pepland env)" % a.nodes_file)
        else:
            sys.path.insert(0, os.path.join(SRC, "finetune"))
            import pool_probe as PP
            V = PP.load_pepland_nodes(a.nodes_file, gdev)

    out = os.path.join(RESULTS_DIR, a.out)
    os.makedirs(out, exist_ok=True)
    rows = []
    for task in a.tasks:
        d = load_task(task)
        if a.max_rows and len(d) > a.max_rows:
            d = d.sample(a.max_rows, random_state=0).reset_index(drop=True)
        miss = [s for s in d.smiles if s not in prow]
        assert not miss, "%d %s peptides are not in %s (e.g. %s)" % (len(miss), task, a.peptide_file, miss[0])
        ii = np.array([prow[s] for s in d.smiles])
        good = pok[ii]
        for pname, (pseq, PM, pk) in prots.items():
            pidx = {s: i for i, s in enumerate(pseq)}
            miss = [s for s in d.seq1 if s not in pidx]
            assert not miss, "%d proteins of %s are not in the %s file" % (len(miss), task, pname)
            jj = np.array([pidx[s] for s in d.seq1])
            good2 = good & pk[jj]
            dd = d[good2].reset_index(drop=True)
            y = dd.y.to_numpy(np.float32)
            A, P = PX[ii[good2]], PM[jj[good2]]
            print("\n%s | protein vectors %s (%d-d) | %d rows (%d dropped: no usable vector) | %d distinct proteins, %d distinct peptides"
                  % (task, pname, P.shape[1], len(dd), int((~good2).sum()), dd.seq1.nunique(), dd.smiles.nunique()), flush=True)
            feats = {"protein": P, "peptide": A, "both": np.hstack([A, P])}
            for split in a.splits:
                for seed in range(a.seeds):
                    tr, va, te = split_random(len(dd), seed) if split == "random" else split_groups(dd.seq1.to_numpy(), seed)
                    for fs in a.features:
                        for head in a.heads:
                            t0 = time.time()
                            m, ep = fit_mlp(feats[fs], y, tr, va, te, head, seed, a, dev)
                            rows.append({"task": task, "protein_vec": pname, "split": split, "features": fs, "head": head, "seed": seed,
                                         "n_test": len(te), "best_epoch": ep, **m})
                    if V is not None and seed < a.gru_seeds and pname == first_prot:
                        vr = np.array([V.row[s] for s in dd.smiles])
                        okv = V.ok[vr]
                        if okv.all():
                            t0 = time.time()
                            m, ep = fit_gru(PP, V, vr, P, y, tr, va, te, seed, a, gdev)
                            rows.append({"task": task, "protein_vec": pname, "split": split, "features": "both_gru", "head": "released",
                                         "seed": seed, "n_test": len(te), "best_epoch": ep, **m})
                            print("   %s %s seed %d  both_gru SCC %.3f (%.0fs)" % (pname, split, seed, m["SCC"], time.time() - t0), flush=True)
                        else:
                            print("   GRU skipped: %d peptides have no node vectors" % int((~okv).sum()))
                    pd.DataFrame(rows).to_csv(os.path.join(out, "raw.csv"), index=False)
                if rows:
                    r = pd.DataFrame(rows)
                    r = r[(r.task == task) & (r.protein_vec == pname) & (r.split == split)]
                    print("   %s %s: %d seeds done | %s" % (pname, split, a.seeds, "; ".join(
                        "%s/%s SCC %.3f" % (f, h, g.SCC.mean()) for (f, h), g in r.groupby(["features", "head"]))), flush=True)

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(out, "raw.csv"), index=False)
    print("\n" + "=" * 110)
    for task in a.tasks:
        t = raw[raw.task == task]
        if t.empty:
            continue
        print("\n%s | published (PepLand): %s" % (task, PUB[task]))
        g = t.groupby(["split", "protein_vec", "features", "head"])
        s = g[["SCC", "PCC", "MAE", "RMSE"]].mean().round(3)
        s["SCC sd"] = g.SCC.std().round(3)
        s["seeds"] = g.size()
        print(s.to_string())
    print("\nreading guide: 'random' is the paper's split (rows of one protein or one peptide may sit on both sides: a leak the authors "
          "acknowledge); 'protein' keeps every protein in one part. Compare 'both' with 'protein' alone: the gap is what the peptide adds.")
    print("wrote", out)


if __name__ == "__main__":
    main()

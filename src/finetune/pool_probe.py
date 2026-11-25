"""Does the pooling matter?  Mean / max / attention (/ GRU) pooling over the token or node vectors of a FROZEN
encoder, with the same small head and the same splits, trained WITH LABELS on the benchmark molecules only.

This is a pilot for the question "does average pooling throw away information?". It does not train the teacher.
If a learned pooler clearly beats the mean here, it is worth the (large) step of re-caching the corpus with it;
if not, it is not.

    # PeptideCLM-2: token vectors from the last layer of the frozen model, extracted on the fly (GPU, minutes)
    python src/finetune/pool_probe.py --model pclm --bench CellPPD THPep PepMSND PAMPA --out pool_pclm_main
    python src/finetune/pool_probe.py --model pclm --bench PL_cCPP PL_cSol PL_ncCPP PL_cBinding PL_ncBinding --out pool_pclm_pl
    # PepLand: needs the node vectors first (pepland env, CPU): python src/featurize_pepland_nodes.py ...
    python src/finetune/pool_probe.py --model pepland --feat pepland_suite --bench PL_cCPP ... --out pool_pepland_pl

Poolers (all followed by the same MLP d -> 256 -> 128 -> 1 as PepLand's head, dropout 0.1):
  mean     masked mean over the real tokens (what the model's own `mean_pool` and our caches use)
  max      masked max
  meanmax  [mean ; max], the cheap "more statistics" alternative
  attn     additive attention, one learned query: score_t = v . tanh(W h_t), softmax over real tokens, sum
  avg      (PepLand) mean over the union of real atom and fragment nodes = the released `pooling="avg"`
  gru      (PepLand) the released Node_GRU design: a bidirectional GRU over the atom nodes and one over the
           fragment nodes, last output of each, concatenated (4 x 300), Linear -> 300
Token vectors are z-scored per feature with the statistics of the training molecules before pooling, so every
pooler sees the same inputs. Per run: Adam lr 3e-4 (all parameters), batch 32, <= 40 epochs, 10% of the training
part held out for early stopping (patience 6, best epoch kept); classification BCE; regression MSE on targets
min-max scaled to [0,1] behind a sigmoid, as in pepland_head.py. The test fold is never used for any choice.
Splits are the benchmarks' own (benchmarks.load). Not the paper's recipe (lr 5e-5, 100 epochs, no early stop):
this compares poolers against each other, and `mean` here is the control. Expect `mean` to land near the
pclm / pepland rows of head_main and head_pepland, not on them.

Output: $RESULTS_DIR/<out>/raw.csv (one row per benchmark x pooler x seed; finished pairs are skipped on re-run),
attn_examples.csv (token weights of a few test molecules, `attn` only), and printed tables with the gain over
the baseline pooler (paired by seed).
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
from paths import FEAT_DIR, PCLM_DIR, RESULTS_DIR  # noqa: E402

DEFAULTS = {"pclm": ["mean", "max", "meanmax", "attn"], "pepland": ["avg", "max", "attn", "gru"]}
BASELINE = {"pclm": "mean", "pepland": "avg"}


# ------------------------------------------------------------------------------------------ vectors
class Vectors:
    """The per-molecule sets of vectors of one encoder, flat on the device.
    flat [Ntok, D] fp16; start/length per molecule; n_atoms per molecule (PepLand: atoms come first, then fragments)."""

    def __init__(self, smiles, flat, start, length, ok, dev, n_atoms=None, ids=None, tok=None):
        self.row = {s: i for i, s in enumerate(smiles)}
        self.ok = np.asarray(ok, bool)
        self.D = flat.shape[1]
        self.flat = flat.to(dev)
        self.start = torch.as_tensor(np.asarray(start, np.int64), device=dev)
        self.length = torch.as_tensor(np.asarray(length, np.int64), device=dev)
        self.n_atoms = None if n_atoms is None else torch.as_tensor(np.asarray(n_atoms, np.int64), device=dev)
        self.ids, self.tok = ids, tok


def extract_pclm(smiles, model_dir, dev, max_tokens=8192):
    """Last-layer token vectors of the frozen PeptideCLM-2 (loaded through featurize.load_pclm: rotary buffers
    rebuilt and verified), real tokens only (id != pad, the mask the model's own mean_pool uses)."""
    import featurize
    tok, model, mdev = featurize.load_pclm(model_dir)
    pad = tok.pad_token_id
    assert pad == 0, "the model's mean_pool masks id 0; this tokenizer pads with %r" % pad
    est = np.array([len(x) + 2 for x in tok(smiles, add_special_tokens=True)["input_ids"]])
    order = np.argsort(est)
    n = len(smiles)
    pieces, id_pieces = [None] * n, [None] * n
    ok = np.ones(n, bool)
    first, i, t0 = True, 0, time.time()
    with torch.no_grad():
        while i < n:
            j = i
            while j < n and est[order[j]] * (j - i + 1) <= max_tokens:
                j += 1
            j = max(j, i + 1)
            idx = order[i:j]
            b = tok([smiles[k] for k in idx], return_tensors="pt", padding=True).to(mdev)
            with torch.autocast(mdev, dtype=torch.bfloat16, enabled=(mdev == "cuda")):
                out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
            H = out.last_hidden_state.float()
            m = b["input_ids"] != pad
            if first:                                   # our masked mean must be the model's own mean_pool
                mine = (H * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1)
                cos = F.cosine_similarity(mine, out.mean_pool.float(), dim=1).min().item()
                print("check: masked mean of the token vectors vs the model's mean_pool, min cosine %.5f" % cos, flush=True)
                assert cos > 0.999, "token mask differs from the model's mean_pool mask"
                first = False
            for r, k in enumerate(idx):
                h = H[r][m[r]]
                ok[k] = bool(torch.isfinite(h).all()) and h.abs().max() < 6e4
                pieces[k] = h.to(torch.float16).cpu()
                id_pieces[k] = b["input_ids"][r][m[r]].cpu().numpy().astype(np.int32)
            i = j
    length = np.array([p.shape[0] for p in pieces])
    start = np.concatenate([[0], np.cumsum(length)[:-1]])
    flat = torch.cat(pieces)
    print("extracted %d molecules, %d tokens, %.1f GB fp16, %.1f min, unusable %d"
          % (n, len(flat), flat.numel() * 2 / 1e9, (time.time() - t0) / 60, (~ok).sum()), flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return Vectors(smiles, flat, start, length, ok, dev, ids=np.concatenate(id_pieces), tok=tok)


def load_pepland_nodes(path, dev):
    """features/<feat>/pepland_nodes.npz (featurize_pepland_nodes.py) -> union of atom and fragment nodes per molecule."""
    z = np.load(path, allow_pickle=True)
    smiles, ok = z["smiles"].tolist(), z["ok"].astype(bool)
    ao, fo = z["a_off"], z["f_off"]
    na, nf = np.diff(ao), np.diff(fo)
    ok &= (na > 0) & (nf > 0)
    length = na + nf
    start = np.concatenate([[0], np.cumsum(length)[:-1]])
    flat = np.zeros((int(length.sum()), z["a_flat"].shape[1]), np.float16)
    for i in range(len(smiles)):
        if length[i]:
            flat[start[i]:start[i] + na[i]] = z["a_flat"][ao[i]:ao[i + 1]]
            flat[start[i] + na[i]:start[i] + length[i]] = z["f_flat"][fo[i]:fo[i + 1]]
    print("loaded %s: %d molecules, %d nodes, usable %d" % (path, len(smiles), len(flat), ok.sum()), flush=True)
    return Vectors(smiles, torch.from_numpy(flat), start, length, ok, dev, n_atoms=na)


def gather(V, start, length, mu, sd):
    """Padded, z-scored [B, L, D] float32 plus the boolean mask [B, L]; padding is exactly zero."""
    L = int(length.max())
    ar = torch.arange(L, device=start.device)
    mask = ar[None] < length[:, None]
    idx = (start[:, None] + ar[None]).clamp_(max=V.flat.shape[0] - 1)
    x = (V.flat[idx].float() - mu) / sd
    return x * mask.unsqueeze(-1), mask


def vector_stats(V, rows, dev, chunk=256):
    s = torch.zeros(V.D, dtype=torch.float64, device=dev)
    s2 = torch.zeros_like(s)
    n = 0
    one, zero = torch.zeros(V.D, device=dev), torch.ones(V.D, device=dev)
    for i in range(0, len(rows), chunk):
        r = torch.as_tensor(rows[i:i + chunk], device=dev)
        x, m = gather(V, V.start[r], V.length[r], one, zero)
        x = x[m].double()
        s += x.sum(0)
        s2 += (x * x).sum(0)
        n += x.shape[0]
    mu = s / n
    sd = (s2 / n - mu * mu).clamp(min=0).sqrt()
    sd = torch.maximum(sd, 0.1 * (sd ** 2).mean().sqrt() + 1e-6)
    return mu.float(), sd.float()


# ----------------------------------------------------------------------------------------- poolers
class Pool(nn.Module):
    def __init__(self, kind, d):
        super().__init__()
        self.kind, self.d = kind, d
        self.out_dim = 2 * d if kind == "meanmax" else d
        self.last_w = None
        if kind == "attn":
            self.W = nn.Linear(d, 128)
            self.v = nn.Linear(128, 1, bias=False)
        elif kind == "gru":
            self.ag = nn.GRU(d, d, batch_first=True, bidirectional=True)
            self.fg = nn.GRU(d, d, batch_first=True, bidirectional=True)
            self.proj = nn.Linear(4 * d, d)

    @staticmethod
    def _last(gru, x, n):
        pk = nn.utils.rnn.pack_padded_sequence(x, n.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = gru(pk)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        return out[torch.arange(len(n), device=x.device), n - 1]      # output at the last real step

    def forward(self, b):
        if self.kind == "gru":
            la = self._last(self.ag, b["A"], b["na"])
            lf = self._last(self.fg, b["F"], b["nf"])
            return self.proj(torch.cat([la, lf], 1))
        H, m = b["H"], b["m"]
        mm = m.unsqueeze(-1)
        if self.kind in ("mean", "avg"):
            return H.sum(1) / m.sum(1, keepdim=True).clamp(min=1)
        mx = H.masked_fill(~mm, float("-inf")).max(1).values
        if self.kind == "max":
            return mx
        if self.kind == "meanmax":
            return torch.cat([H.sum(1) / m.sum(1, keepdim=True).clamp(min=1), mx], 1)
        s = self.v(torch.tanh(self.W(H))).squeeze(-1).masked_fill(~m, float("-inf"))      # attn
        w = torch.softmax(s, 1)
        self.last_w = w.detach()
        return (w.unsqueeze(-1) * H).sum(1)


class Net(nn.Module):
    def __init__(self, kind, d, n_extra):
        super().__init__()
        self.pool = Pool(kind, d)
        din = self.pool.out_dim + n_extra
        self.mlp = nn.Sequential(nn.Linear(din, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 128), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(128, 1))

    def forward(self, b, extra):
        z = self.pool(b)
        if extra is not None:
            z = torch.cat([z, extra], 1)
        return self.mlp(z).squeeze(1)


def make_batch(V, rows, mu, sd, kind, dev):
    r = torch.as_tensor(rows, device=dev)
    st, ln = V.start[r], V.length[r]
    if kind == "gru":
        na = V.n_atoms[r]
        A, _ = gather(V, st, na, mu, sd)
        Fr, _ = gather(V, st + na, ln - na, mu, sd)
        return {"A": A, "na": na, "F": Fr, "nf": ln - na}
    H, m = gather(V, st, ln, mu, sd)
    return {"H": H, "m": m}


# ----------------------------------------------------------------------------------------- training
def fit_predict(V, rows, X_extra, y, tr, te, task, kind, seed, a, dev, want_attn=False):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    perm = rng.permutation(tr)
    nv = max(8, int(0.1 * len(tr)))
    va, fi = perm[:nv], perm[nv:]
    mu, sd = vector_stats(V, rows[fi], dev)
    clf = task == "clf"
    lo, hi = (0.0, 1.0) if clf else (float(y[fi].min()), float(y[fi].max()))
    yt = torch.from_numpy(((y - lo) / (hi - lo + 1e-8)).astype(np.float32)).to(dev)
    ex = None if X_extra is None else torch.from_numpy(X_extra).to(dev)
    net = Net(kind, V.D, 0 if ex is None else ex.shape[1]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)

    def fwd(idx):                                   # idx: positions in the benchmark's own (filtered) table
        b = make_batch(V, rows[idx], mu, sd, kind, dev)
        return net(b, None if ex is None else ex[torch.as_tensor(idx, device=dev)])

    def loss_of(idx):
        out = fwd(idx)
        t = yt[torch.as_tensor(idx, device=dev)]
        return F.binary_cross_entropy_with_logits(out, t) if clf else F.mse_loss(torch.sigmoid(out), t)

    def eval_loss(idx):
        net.eval()
        tot = 0.0
        with torch.no_grad():
            for i in range(0, len(idx), 128):
                part = idx[i:i + 128]
                tot += float(loss_of(part)) * len(part)
        return tot / len(idx)

    best, state, bad, ep = float("inf"), None, 0, 0
    for ep in range(1, a.epochs + 1):
        net.train()
        order = fi[rng.permutation(len(fi))]
        for i in range(0, len(order), a.batch):
            r = order[i:i + a.batch]
            if len(r) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss_of(r).backward()
            opt.step()
        v = eval_loss(va)
        if v < best - 1e-5:
            best, state, bad = v, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= a.patience:
                break
    net.load_state_dict(state if state is not None else net.state_dict())
    net.eval()
    ps, attn = [], []
    with torch.no_grad():
        for i in range(0, len(te), 128):
            part = te[i:i + 128]
            ps.append(torch.sigmoid(fwd(part)).cpu().numpy())
            if want_attn and kind == "attn" and len(attn) < a.attn_examples:
                w = net.pool.last_w.cpu().numpy()
                for k, pos in enumerate(part):
                    attn.append((int(pos), w[k][: int(V.length[int(rows[pos])])]))
    p = np.concatenate(ps)
    return (p if clf else p * (hi - lo) + lo), ep, attn[: a.attn_examples]


def load_bench(bench, max_rows=0):
    df = BM.load(bench).dropna(subset=["smiles"]).reset_index(drop=True)
    df["smiles"] = df.smiles.astype(str)
    if max_rows and len(df) > max_rows:                 # smoke tests only; the same rows every time
        df = df.sample(max_rows, random_state=0).reset_index(drop=True)
    return df


def run_one(bench, kind, V, a, dev, attn_rows):
    df = load_bench(bench, a.max_rows)
    task = BM.TASK[bench]
    miss = [s for s in df.smiles if s not in V.row]
    assert not miss, "%d %s molecules are not in the vector file (e.g. %s): built for another benchmark set?"         % (len(miss), bench, miss[0])
    ii = np.array([V.row[s] for s in df.smiles])
    ok = V.ok[ii]
    df, rows = df[ok].reset_index(drop=True), ii[ok]
    y = df.y.to_numpy(np.float32)
    cols = [c for c in df.columns if c.startswith("extra_")]
    X_extra = pd.get_dummies(df[cols].astype(str)).to_numpy(np.float32) if cols else None
    cv = pd.api.types.is_integer_dtype(df.split)
    folds = sorted(df.split.unique()) if cv else ["test"]
    out = []
    for seed in range(a.seeds):
        per, eps, t0 = [], [], time.time()
        for f in folds:
            te = np.flatnonzero((df.split == f).to_numpy() if cv else (df.split == "test").to_numpy())
            tr = np.setdiff1d(np.arange(len(df)), te)
            if len(te) < 2 or len(tr) < 20:
                continue
            p, ep, attn = fit_predict(V, rows, X_extra, y, tr, te, task, kind, seed, a, dev,
                                      want_attn=(seed == 0 and f == folds[-1]))
            per.append(E.score(task, y[te], p))
            eps.append(ep)
            for pos, w in attn:
                attn_rows.append((bench, df.smiles[pos], w, rows[pos]))
        if not per:
            continue
        out.append({"benchmark": bench, "pooling": kind, "seed": seed, "n": len(df), "coverage": round(float(ok.mean()), 4),
                    "epochs": float(np.mean(eps)), "seconds": round(time.time() - t0, 1),
                    **{k: float(np.mean([s[k] for s in per])) for k in per[0]}})
    return out


def dump_attention(attn_rows, V, path):
    recs = []
    for bench, smi, w, row in attn_rows:
        if V.ids is not None:
            st = int(V.start[row])
            toks = V.tok.convert_ids_to_tokens(V.ids[st:st + len(w)].tolist())
        else:
            na = int(V.n_atoms[row])
            toks = ["atom%d" % k if k < na else "frag%d" % (k - na) for k in range(len(w))]
        for k, (t, x) in enumerate(zip(toks, w)):
            recs.append({"benchmark": bench, "smiles": smi, "pos": k, "token": t, "weight": float(x),
                         "uniform": 1.0 / len(w)})
    pd.DataFrame(recs).to_csv(path, index=False)


def summarise(raw, model):
    base = BASELINE[model]
    raw = raw.copy()
    nan = pd.Series(np.nan, index=raw.index)
    raw["headline"] = np.where(raw.benchmark.map(BM.TASK) == "clf", raw.get("MCC", nan), raw.get("Spearman", nan))
    print("\n=== headline: MCC (classification) / Spearman (regression), mean +- std over seeds ===")
    g = raw.groupby(["benchmark", "pooling"]).headline.agg(["mean", "std"]).round(3)
    print(g.assign(v=lambda d: d["mean"].map("{:.3f}".format) + " +- " + d["std"].fillna(0).map("{:.3f}".format))
          .v.unstack("pooling").to_string())
    for col, label in (("AUROC", "AUROC (classification)"), ("R2", "R2 (regression)")):
        if col in raw and raw[col].notna().any():
            print("\n=== %s, mean over seeds ===" % label)
            print(raw.dropna(subset=[col]).groupby(["benchmark", "pooling"])[col].mean().unstack("pooling").round(3).to_string())
    if base in set(raw.pooling):
        piv = raw.pivot_table(index=["benchmark", "seed"], columns="pooling", values="headline")
        print("\n=== gain over '%s' (paired by seed): mean, and seeds with a positive gain ===" % base)
        rows = {}
        for k in piv.columns:
            if k == base:
                continue
            d = (piv[k] - piv[base]).dropna()
            rows[k] = d.groupby(level=0).agg(lambda x: "%+.3f (%d/%d)" % (x.mean(), (x > 0).sum(), len(x)))
        print(pd.DataFrame(rows).to_string())
        print("(a gain under about 0.03 is inside the noise of these single splits)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["pclm", "pepland"], required=True)
    ap.add_argument("--bench", nargs="*", default=None)
    ap.add_argument("--poolings", nargs="*", default=None)
    ap.add_argument("--feat", default=None, help="features subfolder holding pepland_nodes.npz (default single_modality)")
    ap.add_argument("--nodes", default=None, help="explicit path of the pepland_nodes.npz")
    ap.add_argument("--pclm-dir", default=PCLM_DIR)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-rows", type=int, default=0, help="random subsample per benchmark (smoke tests only)")
    ap.add_argument("--attn-examples", type=int, default=30)
    ap.add_argument("--out", default=None)
    ap.add_argument("--redo", action="store_true", help="recompute benchmark x pooler pairs that are already in raw.csv")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    benches = a.bench or BM.selected()
    poolings = a.poolings or DEFAULTS[a.model]
    for p in poolings:
        assert p in (DEFAULTS["pclm"] + DEFAULTS["pepland"]), p
        assert not (a.model == "pclm" and p in ("avg", "gru")) and not (a.model == "pepland" and p in ("mean", "meanmax")), \
            "pooler %s does not belong to --model %s" % (p, a.model)
    out_dir = os.path.join(RESULTS_DIR, a.out or ("pool_" + a.model))
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "raw.csv")
    rows = pd.read_csv(raw_path).to_dict("records") if os.path.exists(raw_path) and not a.redo else []
    done = {(r["benchmark"], r["pooling"]) for r in rows}
    todo = [b for b in benches if any((b, p) not in done for p in poolings)]

    if todo:
        if a.model == "pclm":
            smiles = sorted(set(pd.concat([load_bench(b, a.max_rows).smiles for b in todo])))
            V = extract_pclm(smiles, a.pclm_dir, dev)
        else:
            path = a.nodes or os.path.join(os.path.dirname(FEAT_DIR), a.feat or os.path.basename(FEAT_DIR), "pepland_nodes.npz")
            assert os.path.exists(path), "%s is missing: run featurize_pepland_nodes.py in the pepland env first" % path
            V = load_pepland_nodes(path, dev)
    attn_rows = []
    for b in todo:
        for k in poolings:
            if (b, k) in done:
                continue
            t0 = time.time()
            new = run_one(b, k, V, a, dev, attn_rows)
            rows += new
            if new:
                key = "MCC" if BM.TASK[b] == "clf" else "Spearman"
                print("%-13s %-8s done (%.0fs) | %s %.3f over %d seed(s), %.0f epochs"
                      % (b, k, time.time() - t0, key, np.mean([r[key] for r in new]), len(new), np.mean([r["epochs"] for r in new])), flush=True)
            pd.DataFrame(rows).to_csv(raw_path, index=False)
    if attn_rows:
        dump_attention(attn_rows, V, os.path.join(out_dir, "attn_examples.csv"))
    if rows:
        summarise(pd.DataFrame(rows), a.model)
    print("\nwrote", out_dir)


if __name__ == "__main__":
    main()

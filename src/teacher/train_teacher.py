"""CDI-style fusion teacher on the cached views: one leave-one-out SCA per view and one SEA.

    python src/teacher/train_teacher.py --name t1                    # every cached shard, 40 epochs
    python src/teacher/train_teacher.py --name t1 --first 4 --epochs 5    # quick look: shards 0-3
    python src/teacher/train_teacher.py --name t1 --resume           # continue a stopped run
    python src/teacher/train_teacher.py --name t1 --embed-pool       # distillation targets, every shard
    # the gate (teacher vs ECFP alone, same learners and splits as the single-modality check):
    python src/eval_single.py --reps ecfp teacher_t1 --out teacher_t1
    # teacher v2: RDKit descriptors instead of PepLand, 1024-d latent, early stopping (see "Teacher v2" below)
    python src/teacher/train_teacher.py --name t5 --views ecfp pclm rdkit --sea-input both --sea-dim 1024 \
        --sea-hidden 2048 --arch res --blocks 2 --dropout 0.1 --view-dropout 0.1 \
        --epochs 200 --patience 20 --sched plateau

Follows Kumar et al., CDI (Nat Commun 2026). SCA_j reads the concatenation of the OTHER views;
its latent p_j has the size of the held-out view j (ECFP 2048, PeptideCLM-2 1024, PepLand 300, RDKit ~206), and
    L_SCA_j = MSE(decoder(p_j), input)   [RE: keep what the other views say]
            + MSE(p_j, view_j)           [predict the held-out view from them]
The SEA compresses concat(p_1 .. p_K) to a LINEAR latent (512-d for t1-t3), the teacher embedding,
and reconstructs it. L = mean_j L_SCA_j + L_SEA, all trained together.
Differences from CDI: AdamW and tens of epochs instead of 600 epochs of SGD; views standardised with
fixed statistics (CDI's learnable per-feature weights also scale the reconstruction targets, which
lets them shrink towards zero); the output layers (SCA latent, decoders, SEA latent) are linear.

--sea-input both  feeds the SEA concat(latents, standardised views) as well: the fallback if the CDI
form loses the fingerprint detail (the bottleneck sees ECFP only through the SCAs).

Teacher v2 (all of it optional; the defaults reproduce teachers t1-t3):
--views      the view list, any of ecfp pclm pepland rdkit (two or more). rdkit = RDKit 2D descriptors
             (src/rdkit_desc.py, cached by corpus/cache_views.py --view rdkit): signed log, standardised,
             a missing value becomes the mean. Without pepland every cached molecule is usable (no PepLand gap).
--sea-dim    teacher latent size; --sea-hidden the SEA's hidden width (default = --hidden).
--arch res   residual pre-norm blocks (LayerNorm-Linear-GELU-Dropout-Linear, --blocks of them, --dropout) in every
             encoder / decoder instead of the 2-layer ReLU MLP.
--view-dropout p   denoising: every view of every training sample is zeroed with probability p (at least one is
             kept) on the INPUT side of the SCAs and the SEA; the reconstruction and prediction targets stay clean.
--epochs N --patience P   stop when the validation loss has not improved by --min-delta for P epochs, at most
             N epochs; best.pt is the best epoch and is what every embedding comes from (--use-last: last.pt).
--sched plateau   linear warm-up, then the learning rate is multiplied by --plateau-factor every
             --plateau-patience epochs without improvement (the cosine schedule needs the epoch count in advance).

Teacher v3: an OPTIONAL view (all of it optional; the defaults reproduce teachers t1-t5):
--views ... esm   ESM-2 embedding (src/featurize_esm.py, cached by corpus/cache_views.py --view esm). ESM reads only
             canonical peptides, so for the other molecules the view is ABSENT (ok = False in its cache).
--optional-views esm   such a view may be absent for a molecule. Then
             (1) --fill imputed (the default): on the input side of the OTHER SCAs the absent view is replaced by p_esm, what its own
                 SCA predicts from the other views (PCLM + PepLand); no gradient flows through the fill. --fill token: a learned
                 MASK TOKEN (one vector per optional view, initialised at the mean) instead (the ablation arm, also used on the
                 raw-view half of the SEA input with --sea-input both);
             (2) its reconstruction / prediction losses skip the molecules that lack it;
             (3) SCA_esm reads only the OTHER views, so its prediction p_esm exists for every molecule: the SEA always sees p_esm
                 (real ESM or not). With --sea-input latents (the default) that is all the SEA sees: no raw view, no token.
--cons-weight w   IMPUTATION CONSISTENCY: for molecules that do have the optional view, the latent computed with the view replaced by
             its stand-in (--fill) is pulled towards the latent computed with the real view (target: no gradient; loss = squared
             error relative to the latent's variance, so w ~ 1 is meaningful). This makes the latent of a non-canonical peptide mean the same
             thing as that of a canonical one, which is what the SMILES-only student has to be distilled from.
--cons-frac f     fraction of the eligible molecules of a batch that get the extra masked pass (cost: about 1 + f/2 forward passes).
The validation line reports `cons`: the relative error between the masked and the full latent on canonical validation
molecules (0 = identical, 1 = as bad as predicting the mean) and their cosine similarity. finetune.py does not support
optional views.

Files: checkpoints $CKPT_DIR/teacher/<name>/last.pt (+ best.pt), history $RESULTS_DIR/teacher/<name>/history.csv,
benchmark embeddings features/single_modality/teacher_<name>.npz (read by eval_single.py), pool
embeddings features/corpus/teacher_<name>/shard_NNN.npz (--embed-pool).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import CACHE_DIR, FEAT_DIR, RESULTS_DIR, TEACHER_DIR  # noqa: E402

KNOWN_DIMS = {"ecfp": 2048, "pclm": 1024, "pepland": 300}       # rdkit, esm: from the cache (view_dim)
DEFAULT_VIEWS = ("ecfp", "pclm", "pepland")
VIEWS = DEFAULT_VIEWS         # the active view list: set_views() changes it. train(), the embedding functions and
DIMS = (2048, 1024, 300)      # finetune.py all read these two, so call set_views() before using them
OPTIONAL = ()                 # views that may be absent for a molecule (set_views; esm: only canonical peptides have one)
DESC_NAMES = None             # the rdkit column names of the cache, saved in the checkpoint
VAL_MOD = 100                 # a pool row is validation when row % 100 == 0 (the pool is shuffled: ~1%)


def say(*a):
    print(*a, flush=True)


def view_dim(v):
    if v in KNOWN_DIMS:
        return KNOWN_DIMS[v]
    if v == "rdkit":
        p = os.path.join(CACHE_DIR, "rdkit", "names.json")
        if os.path.exists(p):
            with open(p) as fh:
                return len(json.load(fh))
        import rdkit_desc
        return rdkit_desc.n_desc()
    if v == "esm":
        p = os.path.join(CACHE_DIR, "esm", "dim.json")
        if not os.path.exists(p):
            raise ValueError("no ESM cache yet (%s is missing): run corpus/cache_views.py --view esm" % p)
        with open(p) as fh:
            return int(json.load(fh)["dim"])
    raise ValueError("unknown view %r (ecfp pclm pepland rdkit esm)" % v)


def set_views(views=None, dims=None, optional=None):
    global VIEWS, DIMS, OPTIONAL
    VIEWS = tuple(views) if views else DEFAULT_VIEWS
    DIMS = tuple(int(d) for d in dims) if dims else tuple(view_dim(v) for v in VIEWS)
    OPTIONAL = tuple(optional) if optional else ()
    assert len(VIEWS) == len(DIMS) >= 2 and len(set(VIEWS)) == len(VIEWS), "--views: two or more different views"
    assert all(v in VIEWS for v in OPTIONAL), "--optional-views %s: every optional view must be in --views %s" % (list(OPTIONAL), list(VIEWS))
    assert len(OPTIONAL) < len(VIEWS), "at least one view must be required (not optional)"


# ------------------------------------------------------------------------------- the cache
def shard_file(view, s):
    return os.path.join(CACHE_DIR, view, "shard_%03d.npz" % s)


def _shard_ids(view, prefix="shard_"):
    d = os.path.join(CACHE_DIR, view)
    names = os.listdir(d) if os.path.isdir(d) else []
    return {int(f[len(prefix):len(prefix) + 3]) for f in names
            if f.startswith(prefix) and f.endswith(".npz") and ".tmp" not in f}


def cached_shards(strict=True):
    """Shards cached in every active view. strict=False also takes a shard whose PepLand view is still
    being computed, using the finished prefix kept in part_NNN.npz (cache_views.py)."""
    done = [_shard_ids(v) for v in VIEWS]
    if not strict and "pepland" in VIEWS:
        i = VIEWS.index("pepland")
        done[i] = done[i] | _shard_ids("pepland", "part_")
    return sorted(set.intersection(*done))


def _view_rows(v, s):
    """Rows of view v in shard s -> (X, ok, lo): the finished shard, else (PepLand) the finished prefix of the
    partial file, whose lo is None."""
    cands = [(shard_file(v, s), False)]
    if v == "pepland":
        cands.append((os.path.join(CACHE_DIR, "pepland", "part_%03d.npz" % s), True))
    for path, partial in cands:
        try:
            z = np.load(path)
            n = int(z["n"]) if partial else len(z["ok"])
            return z["X"][:n], z["ok"][:n], (None if partial else int(z["lo"]))
        except FileNotFoundError:               # finished (or replaced) while we looked: try the next one
            continue
    return None


def _store_dtype(v):
    return torch.uint8 if v == "ecfp" else torch.float32 if v == "rdkit" else torch.float16


def load_rows(shards, device):
    """-> data {view: tensor [N, d] on device (ecfp uint8, rdkit float32, others float16)}, ok [N] (every REQUIRED
    view fine, numpy), rows [N] (pool row ids, numpy). A shard whose PepLand view is only partly cached
    contributes the rows PepLand has finished (the shards are shuffled, so a prefix is a random sample).
    With optional views (OPTIONAL) data also holds "_present" [N, K] bool: False = that view is absent for that
    molecule (its rows are zero); such molecules stay usable, they are not dropped."""
    spans = []
    for s in shards:
        got = [_view_rows(v, s) for v in VIEWS]
        if any(g is None for g in got):
            continue
        m = min(len(g[1]) for g in got)
        if m == 0:
            continue
        los = {g[2] for g in got if g[2] is not None}
        assert len(los) == 1, "shard %d covers different rows in different views" % s
        spans.append((s, los.pop(), m, got))
    n = sum(m for _, _, m, _ in spans)
    data = {v: torch.empty((n, d), dtype=_store_dtype(v), device=device) for v, d in zip(VIEWS, DIMS)}
    ok, rows, o = np.zeros(n, bool), np.zeros(n, np.int64), 0
    pres = np.ones((n, len(VIEWS)), bool) if OPTIONAL else None
    for s, lo, m, got in spans:
        good = np.ones(m, bool)
        for k, (v, (X, okv, _)) in enumerate(zip(VIEWS, got)):
            assert X.shape[1] == data[v].shape[1], \
                "%s shard %d has %d columns, expected %d" % (v, s, X.shape[1], data[v].shape[1])
            Xv = np.ascontiguousarray(X[:m])
            if v in OPTIONAL:                                   # absent is allowed: remember it, zero the row
                pres[o:o + m, k] = okv[:m]
                if not okv[:m].all():
                    Xv = Xv.copy()
                    Xv[~okv[:m]] = 0
            else:
                good &= okv[:m]
            data[v][o:o + m] = torch.from_numpy(Xv).to(device)
        ok[o:o + m], rows[o:o + m] = good, np.arange(lo, lo + m)
        o += m
    if OPTIONAL:
        data["_present"] = torch.from_numpy(pres).to(device)
    return data, ok, rows


def _slog(x):
    return torch.sign(x) * torch.log1p(x.abs())


def prep(view, x, st):
    """raw cached rows -> standardised float32 (ECFP counts go through log1p first; RDKit descriptors through a
    signed log, and a missing descriptor becomes 0 = the mean)."""
    x = x.float()
    if view == "ecfp":
        x = torch.log1p(x)
    elif view == "rdkit":
        x = _slog(x)
    z = (x - st[view]["mu"]) / st[view]["sd"]
    if view == "rdkit":
        z = torch.nan_to_num(z, nan=0.0, posinf=10.0, neginf=-10.0)
    return z.clamp_(-10.0, 10.0)


def fit_stats(data, idx, dev, n=200_000, seed=0):
    """Per-feature mean / std of a random training subset; std floored at 10% of the RMS std so rare
    bits and near-constant dimensions are not blown up."""
    rng = np.random.default_rng(seed)
    sub = torch.from_numpy(np.sort(rng.choice(idx, min(n, len(idx)), replace=False))).to(data[VIEWS[0]].device)
    pres = data.get("_present")
    st = {}
    for k, v in enumerate(VIEWS):
        sv = sub
        if v in OPTIONAL and pres is not None:                 # statistics over the molecules that HAVE the view
            sv = sub[pres[sub, k]]
            assert len(sv) >= 1000, "only %d sampled molecules have a %s view: too few to standardise it" % (len(sv), v)
        x = data[v][sv].to(dev).float()
        if v == "ecfp":
            x = torch.log1p(x)
        if v == "rdkit":                                   # statistics over the finite entries only
            x = _slog(x)
            fin = torch.isfinite(x)
            cnt = fin.sum(0).clamp(min=1)
            mu = torch.where(fin, x, torch.zeros_like(x)).sum(0) / cnt
            sd = (torch.where(fin, (x - mu) ** 2, torch.zeros_like(x)).sum(0) / cnt).sqrt()
        else:
            mu, sd = x.mean(0), x.std(0)
        st[v] = {"mu": mu, "sd": torch.maximum(sd, 0.1 * sd.pow(2).mean().sqrt())}
    return st


class Views(list):
    """The standardised views of a batch (a list of K tensors) plus `present` [B, K] bool (None = every view present).
    An absent view (OPTIONAL views only) is zero in its tensor; the model puts its mask token there."""
    present = None


def batch(data, idx, st, dev):
    xs = Views(prep(v, data[v][idx].to(dev), st) for v in VIEWS)
    pres = data.get("_present")
    if pres is not None:
        pr = pres[idx].to(dev)
        for k, v in enumerate(VIEWS):
            if v in OPTIONAL:                                  # prep() of a zero row is not zero: zero it again
                xs[k] = xs[k] * pr[:, k:k + 1].to(xs[k].dtype)
        xs.present = pr
    return xs


# ------------------------------------------------------------------------------- the model
class Block(nn.Module):
    """Pre-norm residual block: x + Linear(Dropout(GELU(Linear(LayerNorm(x)))))."""

    def __init__(self, h, dropout):
        super().__init__()
        self.f = nn.Sequential(nn.LayerNorm(h), nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, h))

    def forward(self, x):
        return x + self.f(x)


def mlp(i, h, o, arch="mlp", blocks=1, dropout=0.0):
    """Linear output layer in both forms (the embedding code reads net[-1].out_features)."""
    if arch == "mlp":
        return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o))        # the t1-t3 form
    return nn.Sequential(nn.Linear(i, h), *[Block(h, dropout) for _ in range(blocks)],
                         nn.LayerNorm(h), nn.GELU(), nn.Linear(h, o))


class Teacher(nn.Module):
    def __init__(self, hidden=1024, sea_dim=512, sea_input="latents", sea_hidden=None, arch="mlp", blocks=1,
                 dropout=0.0, view_dropout=0.0, fill="token"):
        super().__init__()
        dims = DIMS
        self.K = len(dims)
        tot = sum(dims)
        self.sea_input = sea_input
        self.view_dropout = view_dropout
        self.opt = [VIEWS.index(v) for v in OPTIONAL]                       # positions of the optional views
        assert fill in ("token", "imputed"), "fill: token or imputed"
        assert fill == "token" or len(self.opt) <= 1, \
            "--fill imputed handles one optional view (its SCA reads the other views, which must all be present)"
        self.fill_mode = fill        # what stands in for an absent optional view on the SCA inputs: its own SCA's prediction, or a token
        # the learned mask tokens (0 = the view mean) exist only in token mode
        self.mask_tok = nn.ParameterList([nn.Parameter(torch.zeros(dims[k])) for k in self.opt] if fill == "token" else [])
        self.cons_weight, self.cons_frac = 0.0, 0.5                         # training only: train() sets them
        net = lambda i, h, o: mlp(i, h, o, arch, blocks, dropout)           # noqa: E731
        self.enc = nn.ModuleList([net(tot - d, hidden, d) for d in dims])   # SCA_j: other views -> p_j
        self.dec = nn.ModuleList([net(d, hidden, tot - d) for d in dims])   # mirrored decoder
        sin = tot * (2 if sea_input == "both" else 1)
        sh = sea_hidden or hidden
        self.sea_enc, self.sea_dec = net(sin, sh, sea_dim), net(sea_dim, sh, sin)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    @classmethod
    def from_cfg(cls, cfg):
        """The model of a checkpoint's args (older checkpoints lack the v2 keys: their defaults are the t1-t3 form)."""
        return cls(cfg["hidden"], cfg["sea_dim"], cfg["sea_input"], cfg.get("sea_hidden"), cfg.get("arch", "mlp"),
                   cfg.get("blocks", 1), cfg.get("dropout", 0.0), 0.0, cfg.get("fill", "token"))

    # ---- optional views: what stands in for an absent view -----------------------------------------------------------
    def impute(self, xs, k):
        """Optional view k as its own SCA predicts it from the OTHER views (p_k of those views), without gradient: the
        stand-in for the view where it is absent. The other views are never absent (one optional view at most)."""
        return self.enc[k](torch.cat([xs[m] for m in range(self.K) if m != k], 1)).detach()

    def _standin(self, xs, t, k):
        if self.fill_mode == "imputed":
            return self.impute(xs, k)
        return self.mask_tok[t].to(xs[k].dtype).expand_as(xs[k])

    def fill(self, xs, pres):
        """xs with the stand-in in every slot of an absent optional view (computed from xs's own other views)."""
        out = list(xs)
        if pres is not None:
            for t, k in enumerate(self.opt):
                out[k] = torch.where(pres[:, k:k + 1], xs[k], self._standin(xs, t, k))
        return out

    def masked(self, xs):
        """xs with every optional view replaced by its stand-in (what a non-canonical peptide looks like)."""
        out = list(xs)
        for t, k in enumerate(self.opt):
            out[k] = self._standin(xs, t, k)
        return out

    def latent(self, xs):
        """SCAs + SEA on K input tensors as they are (no dropout, no filling) -> the teacher latent."""
        p = [self.enc[j](torch.cat([xs[k] for k in range(self.K) if k != j], 1)) for j in range(self.K)]
        return self.sea_enc(torch.cat(p + (list(xs) if self.sea_input == "both" else []), 1))

    def corrupt(self, xs, pres=None):
        """View dropout (training only): zero whole views per sample, at least one PRESENT view stays; an absent view is
        left alone (it already holds its mask token). -> (inputs, keep [B, K] or None)."""
        if not (self.training and self.view_dropout > 0):
            return xs, None
        B = xs[0].shape[0]
        keep = torch.rand(B, self.K, device=xs[0].device) >= self.view_dropout
        if pres is None:
            none = ~keep.any(1)
            if bool(none.any()):
                keep[none, torch.randint(0, self.K, (int(none.sum()),), device=keep.device)] = True
        else:
            none = ~(keep & pres).any(1)
            if bool(none.any()):
                pick = (torch.rand(B, self.K, device=keep.device) * pres).argmax(1)
                keep[none, pick[none]] = True
            keep = keep | ~pres
        return [x * keep[:, k:k + 1].to(x.dtype) for k, x in enumerate(xs)], keep

    def consistency(self, xin, s, pres, keep):
        """Imputation consistency (training): on molecules whose optional views are all present (and were not dropped by view
        dropout) the latent computed with those views replaced by their stand-ins is pulled towards the latent computed
        with the real views (`s`, the target: no gradient). Squared error relative to the latent's variance."""
        rows = pres[:, self.opt].all(1)
        if keep is not None:
            rows = rows & keep[:, self.opt].all(1)
        rows = rows & (torch.rand(s.shape[0], device=s.device) < self.cons_frac)
        if int(rows.sum()) < 8:
            return s.sum() * 0.0
        sm = self.latent(self.masked([x[rows] for x in xin]))
        tgt = s[rows].detach()
        return F.mse_loss(sm, tgt) / (tgt.var(0).mean() + 1e-6)

    @staticmethod
    def _wmse(x, y, w):
        return (((x - y) ** 2) * w).sum() / w.sum().clamp(min=1.0)

    def run(self, xs):
        pres = getattr(xs, "present", None)
        xin, keep = self.corrupt(xs, pres)        # view dropout first: the stand-in is computed from what the SCAs get to see
        same = xin is xs
        if pres is not None:
            xs = self.fill(xs, pres)
            xin = xs if same else self.fill(xin, pres)
        a, p, rec = [], [], []
        for j in range(self.K):
            aj = torch.cat([xs[k] for k in range(self.K) if k != j], 1)                     # the clean target
            ai = aj if xin is xs else torch.cat([xin[k] for k in range(self.K) if k != j], 1)
            pj = self.enc[j](ai)
            a.append(aj), p.append(pj), rec.append(self.dec[j](pj))
        both = self.sea_input == "both"
        z = torch.cat(p + (list(xs) if both else []), 1)
        zi = z if xin is xs else torch.cat(p + (list(xin) if both else []), 1)
        s = self.sea_enc(zi)
        o = {"a": a, "p": p, "rec": rec, "z": z, "s": s, "zr": self.sea_dec(s), "pres": pres}
        if pres is not None and self.training and self.cons_weight > 0:
            o["cons"] = self.consistency(xin, s, pres, keep)
        return o

    def loss(self, xs, o):
        pres = o.get("pres")
        if pres is None:                                    # every view present for every molecule: the t1-t5 loss
            re = [F.mse_loss(o["rec"][j], o["a"][j]) for j in range(self.K)]
            ms = [F.mse_loss(o["p"][j], xs[j]) for j in range(self.K)]
            sea = F.mse_loss(o["zr"], o["z"])
            return sum(re[j] + ms[j] for j in range(self.K)) / self.K + sea, re, ms, sea
        B = pres.shape[0]
        re, ms = [], []
        for j in range(self.K):
            w = torch.cat([pres[:, k:k + 1].expand(B, DIMS[k]) for k in range(self.K) if k != j], 1).to(o["a"][j].dtype)
            re.append(self._wmse(o["rec"][j], o["a"][j], w))            # absent views are not reconstructed
            rows = pres[:, j]
            ms.append(F.mse_loss(o["p"][j][rows], xs[j][rows]) if bool(rows.any()) else o["p"][j].sum() * 0.0)
        parts = [torch.ones(B, d, device=pres.device) for d in DIMS]
        if self.sea_input == "both":
            parts += [pres[:, k:k + 1].expand(B, DIMS[k]).float() for k in range(self.K)]
        sea = self._wmse(o["zr"], o["z"], torch.cat(parts, 1).to(o["z"].dtype))
        total = sum(re[j] + ms[j] for j in range(self.K)) / self.K + sea
        if "cons" in o:
            total = total + self.cons_weight * o["cons"]
        return total, re, ms, sea

    def encode(self, xs):
        """The teacher embedding (SEA latent) without the decoders; differentiable (for fine-tuning)."""
        return self.latent(self.fill(xs, getattr(xs, "present", None)))

    @torch.no_grad()
    def embed(self, xs):
        return self.run(xs)["s"]


@torch.no_grad()
def evaluate(model, data, idx, st, dev, bs=8192):
    """Validation losses and R2 (1 - MSE / variance) of every part over the pool rows idx."""
    model.eval()
    K = len(VIEWS)
    z64 = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)   # noqa: E731
    se_re, se_ms = z64(K), z64(K)
    n_re, n_ms = z64(K), z64(K)                  # elements / molecules behind each sum (absent optional views are skipped)
    sx, sxx = [z64(d) for d in DIMS], [z64(d) for d in DIMS]
    se_sea, sz, szz, nz, n = z64(1), None, None, None, 0
    c_se, c_cos, c_n, c_s, c_ss = z64(1), z64(1), 0, None, None          # consistency (optional views only)
    for i in range(0, len(idx), bs):
        xs = batch(data, idx[i:i + bs], st, dev)
        o = model.run(xs)
        pres = o["pres"]
        B = len(xs[0])
        for j in range(K):
            if pres is None:                                   # every view present: no masks, no copies
                se_re[j] += ((o["rec"][j] - o["a"][j]) ** 2).double().sum()
                n_re[j] += B * (sum(DIMS) - DIMS[j])
                rows, cnt = slice(None), B
            else:
                w = torch.cat([pres[:, k:k + 1].expand(B, DIMS[k]) for k in range(K) if k != j], 1).float()
                se_re[j] += (((o["rec"][j] - o["a"][j]) ** 2) * w).double().sum()
                n_re[j] += w.double().sum()
                rows = pres[:, j]
                cnt = int(rows.sum())
            se_ms[j] += ((o["p"][j][rows] - xs[j][rows]) ** 2).double().sum()
            n_ms[j] += cnt
            sx[j] += xs[j][rows].double().sum(0)
            sxx[j] += (xs[j][rows].double() ** 2).sum(0)
        parts = [torch.ones(B, d, device=dev) for d in DIMS]
        if model.sea_input == "both":
            parts += [(pres[:, k:k + 1].expand(B, DIMS[k]).float() if pres is not None else torch.ones(B, DIMS[k], device=dev))
                      for k in range(K)]
        wz = torch.cat(parts, 1).double()
        z = o["z"].double()
        se_sea += (((o["zr"].double() - z) ** 2) * wz).sum()
        sz = (z * wz).sum(0) if sz is None else sz + (z * wz).sum(0)
        szz = ((z ** 2) * wz).sum(0) if szz is None else szz + ((z ** 2) * wz).sum(0)
        nz = wz.sum(0) if nz is None else nz + wz.sum(0)
        n += B
        if pres is not None and model.opt:                 # the masked latent against the full latent, canonical molecules
            rows = pres[:, model.opt].all(1)
            if bool(rows.any()):
                sf = o["s"][rows].double()
                sm = model.latent(model.masked([x[rows] for x in xs])).double()
                c_se += ((sm - sf) ** 2).sum()
                c_cos += F.cosine_similarity(sm, sf, dim=1).sum()
                c_n += int(rows.sum())
                c_s = sf.sum(0) if c_s is None else c_s + sf.sum(0)
                c_ss = (sf ** 2).sum(0) if c_ss is None else c_ss + (sf ** 2).sum(0)
    tot = sum(DIMS)
    out = {}
    for j, name in enumerate(VIEWS):
        nj = float(n_ms[j])
        var = (sxx[j] / nj - (sx[j] / nj) ** 2).mean()
        out["mse_" + name] = float(se_ms[j] / (nj * DIMS[j]))
        out["re_" + name] = float(se_re[j] / n_re[j] * 1.0)
        out["r2_" + name] = float(1 - out["mse_" + name] / var)           # predicting the held-out view
    out["sea"] = float(se_sea / nz.sum())
    out["r2_sea"] = float(1 - out["sea"] / (szz / nz - (sz / nz) ** 2).mean())
    out["loss"] = sum(out["re_" + v] + out["mse_" + v] for v in VIEWS) / K + out["sea"]
    if c_n:
        var_s = (c_ss / c_n - (c_s / c_n) ** 2).mean()
        out["cons_rel"] = float(c_se / (c_n * c_s.numel()) / var_s)
        out["cons_cos"] = float(c_cos / c_n)
        out["loss"] += model.cons_weight * out["cons_rel"]
    model.train()
    return out


# ----------------------------------------------------------------------------- checkpoints
def run_dir(a):
    assert re.fullmatch(r"[A-Za-z0-9_\-]+", a.name), "--name: letters, digits, _ and - only"
    return os.path.join(TEACHER_DIR, a.name)


def ckpt_path(name, last=False):
    """best.pt when the run has one (early stopping), else last.pt."""
    d = os.path.join(TEACHER_DIR, name)
    best = os.path.join(d, "best.pt")
    return best if (os.path.exists(best) and not last) else os.path.join(d, "last.pt")


def save_ckpt(path, **kw):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(kw, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_model(a, dev):
    """-> model (eval), stats, checkpoint. Also activates the checkpoint's view list (set_views)."""
    global DESC_NAMES
    ck = torch.load(ckpt_path(a.name, getattr(a, "use_last", False)), map_location=dev, weights_only=False)
    cfg = ck["args"]
    set_views(cfg.get("views"), cfg.get("dims"), cfg.get("optional"))
    DESC_NAMES = ck.get("desc_names")
    model = Teacher.from_cfg(cfg).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    st = {v: {k: t.to(dev) for k, t in ck["stats"][v].items()} for v in VIEWS}
    return model, st, ck


# ------------------------------------------------------------------------------- embedding
@torch.no_grad()
def embed_rows(model, data, st, dev, bs=8192):
    n = len(data[VIEWS[0]])
    out = np.zeros((n, model.sea_enc[-1].out_features), np.float32)
    for i in range(0, n, bs):
        idx = torch.arange(i, min(i + bs, n), device=data[VIEWS[0]].device)
        out[i:i + bs] = model.embed(batch(data, idx, st, dev)).cpu().numpy()
    return out


def bench_array(v, z):
    """Benchmark feature file -> the array the cache would hold for those molecules (ECFP counts clipped at 255,
    RDKit descriptors float32, the others through float16)."""
    if v == "ecfp":
        return np.minimum(z["X"], 255).astype(np.uint8)
    if v == "rdkit":
        if DESC_NAMES is not None and "names" in z.files:
            assert list(z["names"]) == list(DESC_NAMES), "the benchmark rdkit.npz has other descriptors than the teacher's"
        return z["X"].astype(np.float32)
    return z["X"].astype(np.float16)


def embed_benchmarks(a, model, st, dev, feat_dir=FEAT_DIR):
    """Teacher embedding of every benchmark molecule from the benchmark feature files, treated exactly like the
    cache. feat_dir is features/single_modality (the 5 main benchmarks) or features/pepland_suite (PepLand's 5 tasks)."""
    zs = []
    for v in VIEWS:
        p = os.path.join(feat_dir, v + ".npz")
        if not os.path.exists(p):
            say("benchmark embeddings skipped: %s is missing" % p)
            return
        zs.append(np.load(p, allow_pickle=True))
    smiles = zs[0]["smiles"]
    for v, z in zip(VIEWS, zs):
        assert np.array_equal(z["smiles"], smiles), "%s.npz lists the benchmark molecules in another order" % v
    req = [z["ok"] for v, z in zip(VIEWS, zs) if v not in OPTIONAL]
    ok = np.logical_and.reduce(req)                      # every REQUIRED view fine; an absent optional view is not a failure
    data = {v: torch.from_numpy(np.ascontiguousarray(bench_array(v, z))).to(dev) for v, z in zip(VIEWS, zs)}
    extra = {}
    if OPTIONAL:
        pres = np.ones((len(smiles), len(VIEWS)), bool)
        for k, (v, z) in enumerate(zip(VIEWS, zs)):
            if v in OPTIONAL:
                pres[:, k] = z["ok"]
                data[v][torch.from_numpy(~pres[:, k]).to(dev)] = 0
        data["_present"] = torch.from_numpy(pres).to(dev)
        extra["present"] = pres
        for v in OPTIONAL:
            say("benchmark molecules with a %s view: %d of %d (%.1f%%); the others use %s"
                % (v, int(pres[:, VIEWS.index(v)].sum()), len(smiles), 100 * pres[:, VIEWS.index(v)].mean(),
                   "the view predicted by its SCA from the other views" if model.fill_mode == "imputed" else "the mask token"))
    E = embed_rows(model, data, st, dev)
    E[~ok] = 0
    out = os.path.join(feat_dir, "teacher_%s.npz" % a.name)
    np.savez_compressed(out + ".tmp.npz", smiles=smiles, X=E, ok=ok, **extra)
    os.replace(out + ".tmp.npz", out)
    say("benchmark embeddings: %d molecules, %d-d, usable %d (%.1f%%) -> %s"
        % (len(smiles), E.shape[1], int(ok.sum()), 100 * ok.mean(), out))


def embed_pool(a, dev):
    model, st, _ = load_model(a, dev)
    out = os.path.join(CACHE_DIR, "teacher_" + a.name)
    os.makedirs(out, exist_ok=True)
    shards = [s for s in cached_shards(strict=True) if not a.first or s < a.first]     # finished shards only
    for s in shards:
        path = os.path.join(out, "shard_%03d.npz" % s)
        if os.path.exists(path):
            continue
        data, ok, rows = load_rows([s], dev)
        E = embed_rows(model, data, st, dev)
        E[~ok] = 0
        extra = {"present": data["_present"].cpu().numpy()} if "_present" in data else {}
        np.savez(path + ".tmp.npz", X=E.astype(np.float16), ok=ok, lo=int(rows[0]), hi=int(rows[-1]) + 1, **extra)
        os.replace(path + ".tmp.npz", path)
        say("teacher_%s shard %3d | %s molecules | unusable %d" % (a.name, s, format(len(rows), ","), int((~ok).sum())))
    say("done: %s" % out)


# --------------------------------------------------------------------------------- training
def train(a, dev):
    global DESC_NAMES
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_views(a.views, None, a.optional_views)
    a.views, a.dims, a.optional = list(VIEWS), list(DIMS), list(OPTIONAL)
    assert a.cons_weight == 0 or OPTIONAL, "--cons-weight needs --optional-views"
    if "rdkit" in VIEWS:
        npath = os.path.join(CACHE_DIR, "rdkit", "names.json")
        if os.path.exists(npath):
            with open(npath) as fh:
                DESC_NAMES = json.load(fh)
            assert len(DESC_NAMES) == DIMS[VIEWS.index("rdkit")]
    shards = [s for s in cached_shards(strict=a.strict) if not a.first or s < a.first]
    assert shards, "no shard is cached in all of the views %s under %s" % (list(VIEWS), CACHE_DIR)
    if "pepland" in VIEWS:
        partial = [s for s in shards if s not in _shard_ids("pepland")]
        say("shards: %d, numbers %d..%d (%d complete, %d with PepLand still partly cached: only its finished rows are used)"
            % (len(shards), shards[0], shards[-1], len(shards) - len(partial), len(partial)))
    else:
        say("shards: %d, numbers %d..%d, views %s" % (len(shards), shards[0], shards[-1], list(VIEWS)))
    t0 = time.time()
    ddev = dev if a.data_on == "gpu" else torch.device("cpu")
    try:
        data, ok, rows = load_rows(shards, ddev)
    except torch.cuda.OutOfMemoryError:
        say("the cache does not fit on the GPU: keeping it in RAM")
        torch.cuda.empty_cache()
        ddev = torch.device("cpu")
        data, ok, rows = load_rows(shards, ddev)
    val = np.flatnonzero(ok & (rows % VAL_MOD == 0))
    tr = np.flatnonzero(ok & (rows % VAL_MOD != 0))
    say("molecules: %s cached, %s usable (%s ok): train %s, validation %s | data on %s | %.0f s"
        % (format(len(ok), ","), format(int(ok.sum()), ","),
           "all views" if not OPTIONAL else "every required view", format(len(tr), ","), format(len(val), ","),
           ddev, time.time() - t0))
    for v in OPTIONAL:
        pk = data["_present"][:, VIEWS.index(v)].cpu().numpy()
        say("optional view %s: present for %s of the usable molecules (%.1f%% of train, %.1f%% of validation); "
            "the others get %s%s" % (v, format(int(pk[ok].sum()), ","), 100 * pk[tr].mean(), 100 * pk[val].mean(),
                                     ("p_%s, the view predicted by its own SCA from the other views" % v) if a.fill == "imputed"
                                     else "the mask token",
                                     (" + consistency loss, weight %g on %g of the eligible molecules"
                                      % (a.cons_weight, a.cons_frac)) if a.cons_weight > 0 else ""))
    assert len(tr) >= a.batch and len(val) > 0, "too few usable molecules"

    d = run_dir(a)
    ckp, bestp = os.path.join(d, "last.pt"), os.path.join(d, "best.pt")
    model = Teacher(a.hidden, a.sea_dim, a.sea_input, a.sea_hidden or None, a.arch, a.blocks, a.dropout,
                    a.view_dropout, a.fill).to(dev)
    model.cons_weight, model.cons_frac = a.cons_weight, a.cons_frac
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = len(tr) // a.batch
    total = a.epochs * steps
    warm = min(500, total // 10) if a.sched == "cosine" else 500
    scale = {"lr": 1.0}                          # the plateau schedule's multiplier (changed between epochs)
    if a.sched == "cosine":
        lam = lambda i: min(1.0, (i + 1) / max(1, warm)) * (0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * min(1.0, i / total))))  # noqa: E731
    else:
        lam = lambda i: min(1.0, (i + 1) / warm) * scale["lr"]                                                            # noqa: E731
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    epoch, hist = 0, []
    es = {"best": float("inf"), "best_epoch": 0, "bad": 0}
    if a.resume and os.path.exists(ckp):
        ck = torch.load(ckp, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        st = {v: {k: t.to(dev) for k, t in ck["stats"][v].items()} for v in VIEWS}
        epoch, hist = ck["epoch"], ck["hist"]
        es = ck.get("es", es)
        scale["lr"] = es.get("scale", 1.0)
        say("resumed %s at epoch %d" % (ckp, epoch))
    else:
        assert a.resume or not os.path.exists(ckp), "%s exists: use --resume, or another --name" % ckp
        st = fit_stats(data, tr, dev)
    n_par = sum(p.numel() for p in model.parameters())
    say("teacher: %s trainable parameters (%d SCAs + SEA, views %s, arch %s, sea-input %s, SEA latent %d, hidden %d/%d)"
        % (format(n_par, ","), len(VIEWS), list(VIEWS), a.arch, a.sea_input, a.sea_dim, a.hidden, a.sea_hidden or a.hidden))
    say("schedule: %s | epochs <= %d | patience %s (min-delta %g)%s"
        % (a.sched, a.epochs, a.patience or "off", a.min_delta,
           (" | lr x %g every %d epochs without improvement" % (a.plateau_factor, a.plateau_patience))
           if a.sched == "plateau" else ""))

    tr_t = torch.from_numpy(tr).to(ddev)
    val_t = torch.from_numpy(val).to(ddev)
    rdir = os.path.join(RESULTS_DIR, "teacher", a.name)
    os.makedirs(rdir, exist_ok=True)
    with open(os.path.join(rdir, "config.json"), "w") as fh:
        json.dump(vars(a), fh, indent=1)
    meta = dict(args=vars(a), stats={k: {n_: t.cpu() for n_, t in s_.items()} for k, s_ in st.items()},
                desc_names=DESC_NAMES)
    model.train()
    stopped = False
    for ep in range(epoch, a.epochs):
        if a.patience and es["bad"] >= a.patience:      # a resumed run that had already stopped early
            stopped = True
            say("already stopped early (best epoch %d, val %.4f)" % (es["best_epoch"], es["best"]))
            break
        t1 = time.time()
        perm = torch.randperm(len(tr_t), device=ddev)
        run = torch.zeros((), device=dev)
        for i in range(steps):
            xs = batch(data, tr_t[perm[i * a.batch:(i + 1) * a.batch]], st, dev)
            loss, *_ = model.loss(xs, model.run(xs))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            run += loss.detach()
        train_loss = float(run) / steps
        assert np.isfinite(train_loss), "the loss is not finite: lower --lr"
        v = evaluate(model, data, val_t, st, dev)
        improved = v["loss"] < es["best"] - a.min_delta
        if improved:
            es.update(best=v["loss"], best_epoch=ep + 1, bad=0)
        else:
            es["bad"] += 1
            if a.sched == "plateau" and es["bad"] % a.plateau_patience == 0:
                scale["lr"] *= a.plateau_factor
        es["scale"] = scale["lr"]
        hist.append({"epoch": ep + 1, "train_loss": train_loss, "lr": opt.param_groups[0]["lr"],
                     "seconds": round(time.time() - t1, 1), **v, "best": int(improved), "bad": es["bad"]})
        state = dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep + 1,
                     hist=hist, es=dict(es), **meta)
        save_ckpt(ckp, **state)
        if improved:                                  # the best epoch: weights only (no optimiser state)
            save_ckpt(bestp, **{k: x for k, x in state.items() if k not in ("opt", "sched")})
        with open(os.path.join(rdir, "history.csv"), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(hist[0]))
            w.writeheader(), w.writerows(hist)
        r2 = " ".join("%s %.3f" % (n_, v["r2_" + n_]) for n_ in VIEWS)
        cons = (" | cons (masked vs full latent) rel.err %.3f cos %.3f" % (v["cons_rel"], v["cons_cos"])) if "cons_rel" in v else ""
        say("epoch %3d/%d | %5.1f s | lr %.1e | train %.4f | val %.4f%s | R2 of the held-out view predicted from "
            "the others: %s | SEA R2 %.3f%s%s"
            % (ep + 1, a.epochs, time.time() - t1, opt.param_groups[0]["lr"], train_loss, v["loss"],
               " *" if improved else "", r2, v["r2_sea"], cons,
               (" | no gain for %d/%d" % (es["bad"], a.patience)) if a.patience and es["bad"] else ""))
        if a.patience and es["bad"] >= a.patience:
            stopped = True
            say("early stop: no improvement of the validation loss by %g for %d epochs; best epoch %d (val %.4f)"
                % (a.min_delta, a.patience, es["best_epoch"], es["best"]))
            break
    if not stopped:
        say("reached the epoch limit (%d); best epoch %d (val %.4f)" % (a.epochs, es["best_epoch"], es["best"]))
    if os.path.exists(bestp):                       # everything downstream uses the best epoch
        model.load_state_dict(torch.load(bestp, map_location=dev, weights_only=False)["model"])
    model.eval()
    if not a.no_bench:
        embed_benchmarks(a, model, st, dev)
    say("done. checkpoint %s | next: python src/eval_single.py --reps ecfp teacher_%s --out teacher_%s"
        % (bestp if os.path.exists(bestp) else ckp, a.name, a.name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--views", nargs="+", default=list(DEFAULT_VIEWS), help="ecfp pclm pepland rdkit esm (two or more)")
    ap.add_argument("--optional-views", nargs="*", default=[],
                    help="views that may be absent for a molecule (esm: only canonical peptides have one): a stand-in on the "
                         "input side (--fill) + skipped losses (see the docstring, Teacher v3)")
    ap.add_argument("--fill", choices=("imputed", "token"), default="imputed",
                    help="what stands in for an absent optional view on the inputs of the other SCAs: imputed = the view "
                         "predicted by its own SCA from the other views (p_esm; default), token = a learned mask token")
    ap.add_argument("--cons-weight", type=float, default=0.0,
                    help="imputation-consistency loss weight (masked latent -> full latent); needs --optional-views; ~1 is a start")
    ap.add_argument("--cons-frac", type=float, default=0.5,
                    help="fraction of the eligible molecules of a batch that get the masked pass (cost ~ 1 + frac/2 forwards)")
    ap.add_argument("--epochs", type=int, default=40, help="the limit")
    ap.add_argument("--patience", type=int, default=0, help="early stopping: epochs without improvement (0 = off)")
    ap.add_argument("--min-delta", type=float, default=1e-4, help="smallest validation-loss decrease that counts")
    ap.add_argument("--sched", choices=("cosine", "plateau"), default="cosine")
    ap.add_argument("--plateau-patience", type=int, default=6)
    ap.add_argument("--plateau-factor", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--sea-dim", type=int, default=512)
    ap.add_argument("--sea-hidden", type=int, default=0, help="SEA hidden width (0 = --hidden)")
    ap.add_argument("--sea-input", choices=("latents", "both"), default="latents",
                    help="latents = the SEA reads only the SCA predictions (CDI; default); both = also the standardised raw views")
    ap.add_argument("--arch", choices=("mlp", "res"), default="mlp")
    ap.add_argument("--blocks", type=int, default=1, help="residual blocks per network (--arch res)")
    ap.add_argument("--dropout", type=float, default=0.0, help="inside the residual blocks (--arch res)")
    ap.add_argument("--view-dropout", type=float, default=0.0, help="probability of zeroing a whole input view")
    ap.add_argument("--first", type=int, default=0, help="only shards below this number")
    ap.add_argument("--strict", action="store_true",
                    help="only shards finished in every view (default: also the finished prefix of PepLand shards)")
    ap.add_argument("--data-on", choices=("gpu", "cpu"), default="gpu", help="where the cache lives while training")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-bench", action="store_true", help="skip the benchmark embeddings at the end")
    ap.add_argument("--embed-pool", action="store_true", help="write the teacher embedding of every cached shard")
    ap.add_argument("--use-last", action="store_true", help="embed with last.pt instead of best.pt")
    ap.add_argument("--embed-bench-from", default=None, metavar="SUBFOLDER",
                    help="no training: embed features/SUBFOLDER/<view>.npz with the finished teacher "
                         "--name and write features/SUBFOLDER/teacher_<name>.npz (SUBFOLDER = pepland_suite)")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.embed_bench_from:
        assert re.fullmatch(r"[A-Za-z0-9_\-]+", a.embed_bench_from), "--embed-bench-from: a folder name under features/"
        model, st, _ = load_model(a, dev)
        return embed_benchmarks(a, model, st, dev, os.path.join(os.path.dirname(FEAT_DIR), a.embed_bench_from))
    if a.embed_pool:
        return embed_pool(a, dev)
    train(a, dev)


if __name__ == "__main__":
    main()

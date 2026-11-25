"""Teacher v3: a multi-view autoencoder with E as a reconstruction target only (design of 2026-11-17).

    python src/teacher/train_teacher_v3.py --name v3a --first 5 --epochs 60 --patience 8     # 10% ablation run (5 of 51 shards)
    python src/teacher/train_teacher_v3.py --name v3 --epochs 200 --patience 10 --resume         # the full-pool run
    python src/teacher/train_teacher_v3.py --name v3 --embed-bench-from pepland_suite             # benchmarks of PepLand's suite
    python src/teacher/train_teacher_v3.py --name v3 --embed-pool                                 # the student's distillation targets

Views:  P pepland (300), L pclm (1024), H ecfp (2048 counts, log1p) are INPUTS, present for every molecule.
        E esm (2560) is a TARGET only, never an input, and exists for standard chains only. No mask token, no fill-in, no
        consistency loss: the t8-t11 machinery for a missing input is gone.
Network (blocks and checkpoint helpers come from train_teacher.py):
        one encoder per input view -> 512, LayerNorm on its output;
        view dropout (training only): per molecule, with probability --view-dropout, ONE of the input views (uniformly) has its 512-d
        encoder OUTPUT replaced by a learned vector (one per view); never more than one;
        trunk: concat of the 512-d vectors -> residual MLP -> a LINEAR 1024-d latent (the student's regression target);
        decoders (shallow: latent -> 1024 -> view) for P, L, H and E.
Loss:   mean squared error per view (mean over the elements), summed with equal weights (--w-p/--w-l/--w-h/--w-e scale them); P, L, H on
        every molecule (also when the view was hidden), E on the molecules that have it.
Batches: half standard (E present) and half modified chains (--balance 0.5). An epoch is (training molecules // batch) steps, as in
        t1-t11. Each group is shuffled separately and walked through WITHOUT replacement (reshuffled when used up), so in an epoch a
        standard molecule is seen ~0.62 times and a modified one ~2.6 times (the epoch log prints the measured numbers).
Early stopping: the validation loss is the plain average of the loss on the standard validation molecules and on the modified ones
        (log: loss_std, loss_mod, loss_nat = natural mix), so the 81% group does not decide the stopping point.
Output: view dropout off; the latent standardised per dimension with training-row statistics (saved with the checkpoint and in
        $CKPT_DIR/teacher/<name>/latent_stats.npz). The sd of near-dead dimensions is floored (--latent-floor x the median sd) so
        they are not inflated into noise; the per-dimension sd is logged each epoch (min / median / dims below 10% of the median).
Ablations (one at a time, on the 10% subset with --first 5): --no-h (H neither input nor target), --view-dropout 0 / 0.3, --w-h 2,
        --no-e (no E decoder), --balance 0.19 (the pool's own mix).
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

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import train_teacher as TT  # noqa: E402
import views_v3 as V  # noqa: E402
from paths import CACHE_DIR, FEAT_DIR, RESULTS_DIR, TEACHER_DIR  # noqa: E402

say = TT.say
ALL_VIEWS = list(V.VIEWS)                                   # pepland pclm ecfp esm
LABEL = {"pepland": "P", "pclm": "L", "ecfp": "H", "esm": "E", "pclm_mtr": "M"}


# ------------------------------------------------------------------------------- the model
class AttnPool(nn.Module):
    """Attention pooling of the K per-view vectors of a molecule (Perceiver / Set-Transformer "pooling by multihead attention").
    n_queries LEARNED query vectors (the same for every molecule) each attend over the K view vectors (softmax over the views, attn_heads
    heads) and each returns one pooled 512-d vector; the Q pooled vectors are concatenated for the trunk. The view vectors get a learned
    per-view embedding added (zero at the start) so the attention can tell which view is which; the encoders already keep them apart,
    the embedding only makes it easy. n_queries = 1 is an "attention sum" (one weighted sum of the views). Each pooled vector gets a LayerNorm."""

    def __init__(self, d, n_views, n_queries=4, heads=8):
        super().__init__()
        self.q = nn.Parameter(torch.randn(n_queries, d) * d ** -0.5)
        self.view_emb = nn.Parameter(torch.zeros(n_views, d))
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.norm = nn.LayerNorm(d)

    def forward(self, h, weights=False):
        kv = torch.stack(h, 1) + self.view_emb.to(h[0].dtype)                              # [B, K, d]
        q = self.q.to(kv.dtype).unsqueeze(0).expand(kv.shape[0], -1, -1)                    # [B, Q, d]
        out, w = self.attn(q, kv, kv, need_weights=weights, average_attn_weights=True)      # w: [B, Q, K] (mean over heads)
        out = self.norm(out).flatten(1)                                                     # [B, Q*d]
        return (out, w) if weights else out


class TeacherV3(nn.Module):
    def __init__(self, inputs=("pepland", "pclm", "ecfp"), e_target=True, enc_dim=512, enc_hidden=1024, trunk_hidden=2048, latent=1024,
                 dec_hidden=1024, blocks=1, dropout=0.1, view_dropout=0.1, fuse="concat", n_queries=4, attn_heads=8):
        super().__init__()
        assert fuse in ("concat", "attn"), "fuse: concat or attn"
        self.inputs, self.e_target, self.fuse = list(inputs), bool(e_target), fuse
        self.targets = self.inputs + (["esm"] if self.e_target else [])
        self.view_dropout = view_dropout
        net = lambda i, h, o: TT.mlp(i, h, o, "res", blocks, dropout)          # noqa: E731
        self.enc = nn.ModuleList([net(V.DIM[v], enc_hidden, enc_dim) for v in self.inputs])
        self.norm = nn.ModuleList([nn.LayerNorm(enc_dim) for _ in self.inputs])
        self.hide = nn.ParameterList([nn.Parameter(torch.zeros(enc_dim)) for _ in self.inputs])
        if fuse == "attn":                                       # attention pooling of the view vectors with n_queries learned query vectors
            self.pool = AttnPool(enc_dim, len(self.inputs), n_queries, attn_heads)
            fused = enc_dim * n_queries
        else:                                                    # the original design: concatenation of the view vectors
            fused = enc_dim * len(self.inputs)
        self.trunk = net(fused, trunk_hidden, latent)            # the last layer is linear: the latent is linear
        self.dec = nn.ModuleDict({v: nn.Sequential(nn.Linear(latent, dec_hidden), nn.GELU(), nn.Linear(dec_hidden, V.DIM[v]))
                                  for v in self.targets})
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        self.cfg = dict(inputs=self.inputs, e_target=self.e_target, enc_dim=enc_dim, enc_hidden=enc_hidden, trunk_hidden=trunk_hidden,
                        latent=latent, dec_hidden=dec_hidden, blocks=blocks, dropout=dropout, fuse=fuse, n_queries=n_queries,
                        attn_heads=attn_heads)

    @classmethod
    def from_cfg(cls, cfg, view_dropout=0.0):
        """Rebuild a model from a checkpoint's model_cfg (checkpoints from before the fuse option have no fuse keys: concatenation)."""
        return cls(cfg["inputs"], cfg["e_target"], cfg["enc_dim"], cfg["enc_hidden"], cfg["trunk_hidden"], cfg["latent"],
                   cfg["dec_hidden"], cfg["blocks"], cfg["dropout"], view_dropout, cfg.get("fuse", "concat"), cfg.get("n_queries", 4),
                   cfg.get("attn_heads", 8))

    def encode_views(self, xs, drop=False, hide=None):
        """xs: list of standardised input tensors (in self.inputs order) -> list of 512-d vectors. drop: training view dropout;
        hide: index of a view to hide for EVERY molecule (diagnostics)."""
        h = [n(e(x)) for e, n, x in zip(self.enc, self.norm, xs)]
        B, K = h[0].shape[0], len(h)
        if hide is not None:
            h[hide] = self.hide[hide].to(h[hide].dtype).expand_as(h[hide])
        elif drop and self.view_dropout > 0:
            hit = torch.rand(B, device=h[0].device) < self.view_dropout
            which = torch.randint(0, K, (B,), device=h[0].device)
            for k in range(K):
                m = (hit & (which == k)).unsqueeze(1)
                h[k] = torch.where(m, self.hide[k].to(h[k].dtype).expand_as(h[k]), h[k])
        return h

    def fuse_views(self, h):
        """The K view vectors [B, 512] each -> one vector for the trunk: concatenation [B, K*512], or attention pooling [B, Q*512]."""
        return torch.cat(h, 1) if self.fuse == "concat" else self.pool(h)

    def latent(self, xs, drop=False, hide=None):
        return self.trunk(self.fuse_views(self.encode_views(xs, drop, hide)))

    def forward(self, xs, drop=False, hide=None):
        z = self.latent(xs, drop, hide)
        return z, {v: self.dec[v](z) for v in self.targets}


def view_losses(model, rec, targets, has_e, w):
    """-> (total, {view: mse}). targets {view: standardised [B, d]}; E only on the rows with has_e."""
    parts = {}
    for v in model.targets:
        if v == "esm":
            if not bool(has_e.any()):
                parts[v] = rec[v].sum() * 0.0
                continue
            parts[v] = ((rec[v][has_e] - targets[v][has_e]) ** 2).mean()
        else:
            parts[v] = ((rec[v] - targets[v]) ** 2).mean()
    return sum(w[v] * parts[v] for v in parts), parts


class Cycler:
    """Hands out indices from `idx`: shuffled, without replacement; when all are used the group is reshuffled."""

    def __init__(self, idx, dev):
        self.idx, self.dev, self.passes = idx, dev, 0
        self._shuffle()

    def _shuffle(self):
        self.perm = self.idx[torch.randperm(len(self.idx), device=self.dev)]
        self.pos = 0

    def next(self, n):
        out = []
        while n > 0:
            if self.pos >= len(self.perm):
                self.passes += 1
                self._shuffle()
            k = min(n, len(self.perm) - self.pos)
            out.append(self.perm[self.pos:self.pos + k])
            self.pos += k
            n -= k
        return out[0] if len(out) == 1 else torch.cat(out)

    def seen(self):
        """How many times the group has been walked through so far (fractional)."""
        return self.passes + self.pos / len(self.perm)


# ------------------------------------------------------------------------------- data
def load_rows_v3(shards, dev_map):
    """-> data {view: tensor [N, d] on dev_map[view]} (ecfp uint8, the others float16), ok [N] (P, L, H fine), has_e [N] bool, rows [N].
    Reads the shards cached in all four views (TT._view_rows), keeps molecules whose P, L and H are fine."""
    spans = []
    for s in shards:
        got = {v: TT._view_rows(v, s) for v in ALL_VIEWS}
        if any(g is None for g in got.values()):
            continue
        m = min(len(g[1]) for g in got.values())
        los = {g[2] for g in got.values() if g[2] is not None}
        assert m > 0 and len(los) == 1, "shard %d: empty or inconsistent" % s
        spans.append((s, los.pop(), m, got))
    n = sum(m for _, _, m, _ in spans)
    data = {v: torch.empty((n, V.DIM[v]), dtype=torch.uint8 if v == "ecfp" else torch.float16, device=dev_map[v]) for v in ALL_VIEWS}
    ok, has_e, rows, o = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, np.int64), 0
    for s, lo, m, got in spans:
        good = np.ones(m, bool)
        for v in ALL_VIEWS:
            X, okv, _ = got[v]
            assert X.shape[1] == V.DIM[v], "%s shard %d has %d columns, expected %d" % (v, s, X.shape[1], V.DIM[v])
            Xv = np.ascontiguousarray(X[:m])
            if v == V.TARGET_ONLY:
                has_e[o:o + m] = okv[:m]
                if not okv[:m].all():
                    Xv = Xv.copy()
                    Xv[~okv[:m]] = 0
            else:
                good &= okv[:m]
            data[v][o:o + m] = torch.from_numpy(Xv).to(dev_map[v])
        ok[o:o + m], rows[o:o + m] = good, np.arange(lo, lo + m)
        o += m
    return data, ok, has_e & ok, rows


def prep_t(view, x, st):
    """Raw rows -> standardised float32 (the same formulas as views_v3.prep)."""
    x = x.float()
    if view == "ecfp":
        x = torch.log1p(x)
    return ((x - st[view]["mu"]) / st[view]["sd"]).clamp_(-V.CLIP, V.CLIP)


def fit_stats_t(data, idx, has_e, dev, chunk=65536):
    """Mean and spread of ALL the training rows idx (E: of the rows that have it), accumulated chunk by chunk in float64
    (views_v3.Running; the same formulas and floors as views_v3.fit_stats) -> torch on dev."""
    acc = {v: V.Running() for v in ALL_VIEWS}
    for i in range(0, len(idx), chunk):
        sub = idx[i:i + chunk]
        for v in ALL_VIEWS:
            rows = sub[has_e[sub]] if v == V.TARGET_ONLY else sub
            if len(rows) == 0:
                continue
            x = data[v][torch.from_numpy(rows).to(data[v].device)].to(dev).double()
            if v == "ecfp":
                x = torch.log1p(x)
            mu = x.mean(0)
            acc[v].merge(len(rows), mu.cpu().numpy(), ((x - mu) ** 2).sum(0).cpu().numpy())
    for v in ALL_VIEWS:
        assert acc[v].n >= 1000, "only %d rows to standardise %s: too few" % (acc[v].n, v)
    st = {v: V.finish_stats(v, *acc[v].mean_sd()) for v in ALL_VIEWS}
    return {v: {k: torch.from_numpy(a_).to(dev) for k, a_ in d.items()} for v, d in st.items()}, st


def take(data, idx, dev):
    """Rows idx (a tensor on any device) of every view -> raw tensors on dev."""
    out = {}
    for v in ALL_VIEWS:
        ii = idx if data[v].device == idx.device else idx.to(data[v].device)
        out[v] = data[v][ii].to(dev, non_blocking=True)
    return out


def standardised(raw, st):
    return {v: prep_t(v, raw[v], st) for v in raw}


# ------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, data, idx, has_e, st, w, dev, bs=8192):
    """Validation, in two groups scored separately: "std" (E present) and "mod" (E absent: modified chains).
    Per group: mean squared error and R2 of every view with every input visible ("mse_P_std", "r2_L_mod", ...), the R2 of each input view
    when ITS OWN encoder output is hidden ("r2hide_H_std", ...: how much the other views say about it), and the weighted loss
    ("loss_std", "loss_mod"; the E term only exists in std). "loss" = the plain average of the two group losses: this is the
    early-stopping criterion, so the larger group does not decide it. "loss_nat" = the loss on the natural mix of the validation rows.
    Also the spread of the latent per dimension over all validation rows (mean / min / median sd, dimensions below 10% of the median)."""
    model.eval()
    G = ("std", "mod")
    K = len(model.inputs)
    sums = {(g, v): 0.0 for g in G for v in model.targets}
    sums_h = {(g, k): 0.0 for g in G for k in range(K)}
    cnt = {g: 0 for g in G}
    s1 = {(g, v): torch.zeros(V.DIM[v], dtype=torch.float64, device=dev) for g in G for v in model.targets}
    s2 = {(g, v): torch.zeros(V.DIM[v], dtype=torch.float64, device=dev) for g in G for v in model.targets}
    zsum = zsq = None
    nz = 0
    aw_sum, n_aw = None, 0                                       # --fuse attn: mean attention weight of every (query, view) pair
    for i in range(0, len(idx), bs):
        ii = idx[i:i + bs]
        z_all = standardised(take(data, ii, dev), st)
        he = has_e[ii.to(has_e.device)].to(dev)
        xs = [z_all[v] for v in model.inputs]
        z, rec = model(xs)
        if model.fuse == "attn":
            aw = model.pool(model.encode_views(xs), weights=True)[1].double()          # [B, Q, K]
            aw_sum = aw.sum(0) if aw_sum is None else aw_sum + aw.sum(0)
            n_aw += len(aw)
        rec_h = [model(xs, hide=k)[1] for k in range(K)]
        for g, m in (("std", he), ("mod", ~he)):
            n = int(m.sum())
            if n == 0:
                continue
            cnt[g] += n
            for v in model.targets:
                if v == "esm" and g == "mod":                   # no E target on modified chains
                    continue
                tgt = z_all[v][m]
                sums[(g, v)] += float(((rec[v][m] - tgt) ** 2).sum())
                s1[(g, v)] += tgt.double().sum(0)
                s2[(g, v)] += (tgt.double() ** 2).sum(0)
            for k, v in enumerate(model.inputs):
                sums_h[(g, k)] += float(((rec_h[k][v][m] - z_all[v][m]) ** 2).sum())
        zd = z.double()
        zsum = zd.sum(0) if zsum is None else zsum + zd.sum(0)
        zsq = (zd ** 2).sum(0) if zsq is None else zsq + (zd ** 2).sum(0)
        nz += len(z)
    out = {}
    for g in G:
        n = cnt[g]
        out["loss_" + g] = float("nan")
        if n == 0:
            continue
        loss = 0.0
        for v in model.targets:
            if v == "esm" and g == "mod":
                continue
            mse = sums[(g, v)] / (n * V.DIM[v])
            var = float((s2[(g, v)] / n - (s1[(g, v)] / n) ** 2).mean())
            out["mse_%s_%s" % (LABEL[v], g)] = mse
            out["r2_%s_%s" % (LABEL[v], g)] = 1 - mse / var
            loss += w[v] * mse
        for k, v in enumerate(model.inputs):
            var = float((s2[(g, v)] / n - (s1[(g, v)] / n) ** 2).mean())
            out["r2hide_%s_%s" % (LABEL[v], g)] = 1 - (sums_h[(g, k)] / (n * V.DIM[v])) / var
        out["loss_" + g] = loss
    fin = [out["loss_" + g] for g in G if cnt[g]]
    out["loss"] = float(np.mean(fin))
    out["loss_nat"] = float(sum(cnt[g] * out["loss_" + g] for g in G if cnt[g]) / sum(cnt.values()))
    sd = (zsq / nz - (zsum / nz) ** 2).clamp(min=0).sqrt()
    out["latent_sd"] = float(sd.mean())
    out["latent_sd_min"] = float(sd.min())
    out["latent_sd_med"] = float(sd.median())
    out["latent_dead"] = int((sd < 0.1 * sd.median()).sum())
    out["n_std"], out["n_mod"] = cnt["std"], cnt["mod"]
    if aw_sum is not None:
        awm = (aw_sum / n_aw).cpu().numpy()
        for j in range(awm.shape[0]):
            for k, v in enumerate(model.inputs):
                out["attn_q%d_%s" % (j + 1, LABEL[v])] = float(awm[j, k])
    model.train()
    return out


# ------------------------------------------------------------------------------- embedding
@torch.no_grad()
def latent_rows(model, raw_get, n, st, dev, bs=8192):
    """Raw latent (no dropout) for n rows; raw_get(i, j) -> {view: raw tensor on dev} for rows i..j."""
    model.eval()
    out = np.zeros((n, model.cfg["latent"]), np.float32)
    for i in range(0, n, bs):
        j = min(i + bs, n)
        z = standardised(raw_get(i, j), st)
        out[i:j] = model.latent([z[v] for v in model.inputs]).float().cpu().numpy()
    return out


def run_dir(name):
    assert re.fullmatch(r"[A-Za-z0-9_\-]+", name), "--name: letters, digits, _ and - only"
    return os.path.join(TEACHER_DIR, name)


def load_model(name, dev, use_last=False):
    d = run_dir(name)
    path = os.path.join(d, "best.pt") if (os.path.exists(os.path.join(d, "best.pt")) and not use_last) else os.path.join(d, "last.pt")
    ck = torch.load(path, map_location=dev, weights_only=False)
    cfg = ck["model_cfg"]
    model = TeacherV3.from_cfg(cfg, 0.0).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    st = {v: {k: t.to(dev) for k, t in ck["stats"][v].items()} for v in ck["stats"]}
    lat = np.load(os.path.join(d, "latent_stats.npz")) if os.path.exists(os.path.join(d, "latent_stats.npz")) else None
    return model, st, lat


def standardise_latent(z, lat):
    return ((z - lat["mu"]) / lat["sd"]).astype(np.float32)


def embed_benchmarks(name, model, st, lat, dev, feat_dir=FEAT_DIR):
    """Latent of every benchmark molecule from the benchmark feature files, standardised with the training statistics.
    -> features/<suite>/teacher_<name>.npz (smiles, X, ok), read by the evaluation harness and eval_single.py"""
    zs = {}
    for v in model.inputs:
        p = os.path.join(feat_dir, v + ".npz")
        if not os.path.exists(p):
            say("benchmark embeddings skipped: %s is missing" % p)
            return
        zs[v] = np.load(p, allow_pickle=True)
    smiles = zs[model.inputs[0]]["smiles"]
    for v, z in zs.items():
        assert np.array_equal(z["smiles"], smiles), "%s.npz lists the benchmark molecules in another order" % v
    ok = np.logical_and.reduce([z["ok"] for z in zs.values()])
    raw = {v: torch.from_numpy(np.ascontiguousarray(np.minimum(z["X"], 255).astype(np.uint8) if v == "ecfp" else z["X"].astype(np.float16))).to(dev)
           for v, z in zs.items()}
    Z = latent_rows(model, lambda i, j: {v: raw[v][i:j] for v in raw}, len(smiles), st, dev)
    Z = standardise_latent(Z, lat) if lat is not None else Z
    Z[~ok] = 0
    out = os.path.join(feat_dir, "teacher_%s.npz" % name)
    np.savez_compressed(out + ".tmp.npz", smiles=smiles, X=Z, ok=ok)
    os.replace(out + ".tmp.npz", out)
    say("benchmark embeddings: %d molecules, %d-d, usable %d (%.1f%%) -> %s" % (len(smiles), Z.shape[1], int(ok.sum()), 100 * ok.mean(), out))


def embed_pool(a, dev):
    model, st, lat = load_model(a.name, dev, a.use_last)
    assert lat is not None, "no latent_stats.npz: the run did not finish; train to the end first"
    out = os.path.join(CACHE_DIR, "teacher_" + a.name)
    os.makedirs(out, exist_ok=True)
    have = set.intersection(*[set(V.shard_ids(v)) for v in model.inputs])
    for s in sorted(have):
        if a.first and s >= a.first:
            continue
        path = os.path.join(out, "shard_%03d.npz" % s)
        if os.path.exists(path):
            continue
        z = {v: np.load(os.path.join(CACHE_DIR, v, "shard_%03d.npz" % s)) for v in model.inputs}
        n = min(len(z[v]["ok"]) for v in model.inputs)
        ok = np.logical_and.reduce([z[v]["ok"][:n] for v in model.inputs])
        raw = {v: torch.from_numpy(np.ascontiguousarray(z[v]["X"][:n])).to(dev) for v in model.inputs}
        Z = standardise_latent(latent_rows(model, lambda i, j: {v: raw[v][i:j] for v in raw}, n, st, dev), lat)
        Z[~ok] = 0
        lo = int(z[model.inputs[0]]["lo"])
        np.savez(path + ".tmp.npz", X=Z.astype(np.float16), ok=ok, lo=lo, hi=lo + n)
        os.replace(path + ".tmp.npz", path)
        say("teacher_%s shard %3d | %s molecules | unusable %d" % (a.name, s, format(n, ","), int((~ok).sum())))
    say("done: %s" % out)


# ------------------------------------------------------------------------------- training
def train(a, dev):
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    inputs = [v for v, flag in (("pepland", not a.no_p), ("pclm", not a.no_l), ("ecfp", not a.no_h)) if flag] + list(a.extra_input)
    assert len(inputs) >= 2, "at least two input views"
    ALL_VIEWS[:] = [v for v in V.VIEWS if v != V.TARGET_ONLY] + list(a.extra_input) + [V.TARGET_ONLY]   # every view this run reads from the cache
    w = {"pepland": a.w_p, "pclm": a.w_l, "ecfp": a.w_h, "esm": a.w_e, "pclm_mtr": a.w_m}
    ddev = dev if a.data_on == "gpu" else torch.device("cpu")
    dev_map = {v: ddev for v in ALL_VIEWS}
    if a.e_on == "cpu":
        dev_map["esm"] = torch.device("cpu")
    per_view = {v: set(V.shard_ids(v)) for v in ALL_VIEWS}
    shards = sorted(set.intersection(*per_view.values()))
    if len({len(x) for x in per_view.values()}) > 1:
        say("WARNING: the views are cached for different numbers of shards: %s. Only the %d shards cached in ALL of them are used. "
            "Finish the caching first (cache_views.py --status), or this teacher trains on a part of the pool."
            % ({LABEL[v]: len(x) for v, x in per_view.items()}, len(shards)))
    shards = [s for s in shards if not a.first or s < a.first]
    assert shards, "no shard is cached in all of %s under %s" % (ALL_VIEWS, CACHE_DIR)
    say("shards: %d, numbers %d..%d | views %s | E stays on %s" % (len(shards), shards[0], shards[-1], ALL_VIEWS, dev_map["esm"]))
    t0 = time.time()
    data, ok, has_e, rows = load_rows_v3(shards, dev_map)
    val = np.flatnonzero(ok & (rows % V.VAL_MOD == 0))
    tr = np.flatnonzero(ok & (rows % V.VAL_MOD != 0))
    say("molecules: %s cached, %s usable: train %s, validation %s | E present for %.1f%% of train, %.1f%% of validation | %.0f s"
        % (format(len(ok), ","), format(int(ok.sum()), ","), format(len(tr), ","), format(len(val), ","),
           100 * has_e[tr].mean(), 100 * has_e[val].mean(), time.time() - t0))
    assert len(tr) >= a.batch and len(val) > 0 and has_e[tr].any() and (~has_e[tr]).any(), "too few usable molecules"

    d = run_dir(a.name)
    ckp, bestp = os.path.join(d, "last.pt"), os.path.join(d, "best.pt")
    model = TeacherV3(inputs, not a.no_e, a.enc_dim, a.enc_hidden, a.trunk_hidden, a.latent, a.dec_hidden, a.blocks, a.dropout,
                      a.view_dropout, a.fuse, a.n_queries, a.attn_heads).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps = len(tr) // a.batch
    scale = {"lr": 1.0}
    warm = 500
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / warm) * scale["lr"])
    epoch, hist, es = 0, [], {"best": float("inf"), "best_epoch": 0, "bad": 0}
    if a.resume and os.path.exists(ckp):
        ck = torch.load(ckp, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"])
        st = {v: {k: t.to(dev) for k, t in ck["stats"][v].items()} for v in ALL_VIEWS}
        epoch, hist, es = ck["epoch"], ck["hist"], ck.get("es", es)
        scale["lr"] = es.get("scale", 1.0)
        say("resumed %s at epoch %d" % (ckp, epoch))
    else:
        assert a.resume or not os.path.exists(ckp), "%s exists: use --resume, or another --name" % ckp
        t_s = time.time()
        st, _ = fit_stats_t(data, tr, has_e, dev)
        say("standardising statistics: ALL %s training molecules (E: the %s that have it) | %.0f s"
            % (format(len(tr), ","), format(int(has_e[tr].sum()), ","), time.time() - t_s))
    n_par = sum(p.numel() for p in model.parameters())
    say("teacher v3: %s trainable parameters | inputs %s, targets %s | fusion %s | latent %d | view dropout %g | balance %g | weights %s"
        % (format(n_par, ","), [LABEL[v] for v in model.inputs], [LABEL[v] for v in model.targets],
           "concatenation" if a.fuse == "concat" else "attention pooling (%d learned queries, %d heads)" % (a.n_queries, a.attn_heads),
           a.latent, a.view_dropout, a.balance, {LABEL[v]: w[v] for v in model.targets}))
    say("schedule: AdamW lr %g | plateau x%g every %d epochs without improvement | epochs <= %d | patience %s | %d steps per epoch"
        % (a.lr, a.plateau_factor, a.plateau_patience, a.epochs, a.patience or "off", steps))

    cdev = data["pepland"].device
    tr_t, val_t = torch.from_numpy(tr).to(cdev), torch.from_numpy(val).to(cdev)
    he_t = torch.from_numpy(has_e).to(cdev)
    std_idx = tr_t[he_t[tr_t]]
    mod_idx = tr_t[~he_t[tr_t]]
    nm = int(round(a.balance * a.batch))
    cyc_s = Cycler(std_idx, cdev)                                    # each group walked through without replacement
    cyc_m = Cycler(mod_idx, cdev)
    rdir = os.path.join(RESULTS_DIR, "teacher", a.name)
    os.makedirs(rdir, exist_ok=True)
    with open(os.path.join(rdir, "config.json"), "w") as fh:
        json.dump(vars(a), fh, indent=1)
    meta = dict(args=vars(a), model_cfg=model.cfg, stats={v: {k: t.cpu() for k, t in s_.items()} for v, s_ in st.items()})
    model.train()
    stopped = False
    for ep in range(epoch, a.epochs):
        if a.patience and es["bad"] >= a.patience:
            stopped = True
            say("already stopped early (best epoch %d, val %.4f)" % (es["best_epoch"], es["best"]))
            break
        t1 = time.time()
        perm = torch.randperm(len(tr_t), device=cdev) if nm == 0 else None
        run = torch.zeros((), device=dev)
        seen0 = (cyc_s.seen(), cyc_m.seen())
        for i in range(steps):
            if nm == 0:
                ii = tr_t[perm[i * a.batch:(i + 1) * a.batch]]
            else:                                                    # half standard, half modified, each group without replacement
                ii = torch.cat([cyc_s.next(a.batch - nm), cyc_m.next(nm)])
            raw = take(data, ii, dev)
            z_all = standardised(raw, st)
            he = he_t[ii].to(dev)
            _, rec = model([z_all[v] for v in model.inputs], drop=True)
            loss, _ = view_losses(model, rec, z_all, he, w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            run += loss.detach()
        train_loss = float(run) / steps
        assert np.isfinite(train_loss), "the loss is not finite: lower --lr"
        v = evaluate(model, data, val_t, he_t, st, w, dev)
        improved = v["loss"] < es["best"] - a.min_delta
        if improved:
            es.update(best=v["loss"], best_epoch=ep + 1, bad=0)
        else:
            es["bad"] += 1
            if es["bad"] % a.plateau_patience == 0:
                scale["lr"] *= a.plateau_factor
        es["scale"] = scale["lr"]
        hist.append({"epoch": ep + 1, "train_loss": train_loss, "lr": opt.param_groups[0]["lr"], "seconds": round(time.time() - t1, 1),
                     **v, "best": int(improved), "bad": es["bad"]})
        state = dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep + 1, hist=hist, es=dict(es), **meta)
        TT.save_ckpt(ckp, **state)
        if improved:
            TT.save_ckpt(bestp, **{k: x for k, x in state.items() if k not in ("opt", "sched")})
        with open(os.path.join(rdir, "history.csv"), "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(hist[0]))
            wr.writeheader(), wr.writerows(hist)
        def r2s(g, pre="r2_", views=model.targets):
            return " ".join("%s %.3f" % (LABEL[x], v[pre + LABEL[x] + "_" + g]) for x in views if pre + LABEL[x] + "_" + g in v)
        passes = "" if nm == 0 else " | passes per molecule std %.2f mod %.2f" % (cyc_s.seen() - seen0[0], cyc_m.seen() - seen0[1])
        say("epoch %3d/%d | %5.1f s | lr %.1e | train %.4f | val %.4f = (std %.4f + mod %.4f)/2, natural mix %.4f%s%s"
            % (ep + 1, a.epochs, time.time() - t1, opt.param_groups[0]["lr"], train_loss, v["loss"], v["loss_std"], v["loss_mod"],
               v["loss_nat"], " *" if improved else "", (" | no gain for %d/%d" % (es["bad"], a.patience)) if a.patience and es["bad"] else ""))
        say("           R2 std: %s | R2 mod: %s | own view hidden, std: %s | mod: %s | latent sd min %.3f med %.3f, %d dims < 10%% of median%s"
            % (r2s("std"), r2s("mod"), r2s("std", "r2hide_", model.inputs), r2s("mod", "r2hide_", model.inputs),
               v["latent_sd_min"], v["latent_sd_med"], v["latent_dead"], passes))
        if a.fuse == "attn":
            say("           attention weights (mean over validation molecules; views %s): %s"
                % ("/".join(LABEL[x] for x in model.inputs),
                   " | ".join("q%d %s" % (j + 1, " ".join("%.2f" % v["attn_q%d_%s" % (j + 1, LABEL[x])] for x in model.inputs))
                              for j in range(a.n_queries))))
        if a.patience and es["bad"] >= a.patience:
            stopped = True
            say("early stop: no improvement of the validation loss by %g for %d epochs; best epoch %d (val %.4f)"
                % (a.min_delta, a.patience, es["best_epoch"], es["best"]))
            break
    if not stopped:
        say("reached the epoch limit (%d); best epoch %d (val %.4f)" % (a.epochs, es["best_epoch"], es["best"]))
    if os.path.exists(bestp):
        model.load_state_dict(torch.load(bestp, map_location=dev, weights_only=False)["model"])
    model.eval()
    # latent statistics of the training rows (view dropout off): the student's targets are standardised with them
    run = V.Running()
    for i in range(0, len(tr), 65536):
        sub_t = torch.from_numpy(tr[i:i + 65536]).to(cdev)
        run.add(latent_rows(model, lambda x, y: take(data, sub_t[x:y], dev), len(sub_t), st, dev))
    mu_lat, sd_raw = run.mean_sd()
    floor = max(a.latent_floor * float(np.median(sd_raw)), 1e-6)          # near-dead dimensions are not inflated into noise
    lat = {"mu": mu_lat.astype(np.float32), "sd": np.maximum(sd_raw, floor).astype(np.float32), "sd_raw": sd_raw.astype(np.float32)}
    np.savez(os.path.join(d, "latent_stats.npz"), **lat)
    say("latent statistics (ALL %s training rows) -> %s" % (format(run.n, ","), os.path.join(d, "latent_stats.npz")))
    say("latent sd per dimension before standardising: min %.4g, median %.4g, max %.4g | %d of %d dimensions are below the floor %.4g "
        "(%g x the median) and come out with a standardised sd below 1"
        % (sd_raw.min(), np.median(sd_raw), sd_raw.max(), int((sd_raw < floor).sum()), len(sd_raw), floor, a.latent_floor))
    if not a.no_bench:
        embed_benchmarks(a.name, model, st, lat, dev)
    say("done. checkpoint %s | next: python src/teacher/eval_harness.py --teachers %s --out harness_%s" % (bestp if os.path.exists(bestp) else ckp, a.name, a.name))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--no-p", action="store_true", help="drop view P (input and target)")
    ap.add_argument("--no-l", action="store_true", help="drop view L (input and target)")
    ap.add_argument("--no-h", action="store_true", help="ablation: drop view H (neither input nor target)")
    ap.add_argument("--no-e", action="store_true", help="ablation: no E decoder (E is not reconstructed)")
    ap.add_argument("--w-p", type=float, default=1.0)
    ap.add_argument("--w-l", type=float, default=1.0)
    ap.add_argument("--w-h", type=float, default=1.0)
    ap.add_argument("--w-e", type=float, default=1.0)
    ap.add_argument("--w-m", type=float, default=1.0, help="weight of the extra input view pclm_mtr (input and target)")
    ap.add_argument("--extra-input", nargs="*", default=[], choices=list(V.EXTRA_INPUTS),
                    help="variation: add input views, each also a reconstruction target (pclm_mtr = PeptideCLM-2 MTR, 1024-d, cached in features/corpus/pclm_mtr)")
    ap.add_argument("--fuse", choices=("concat", "attn"), default="concat",
                    help="how the per-view 512-d vectors are combined: concat (the v3 design) or attn (attention pooling with --n-queries learned vectors)")
    ap.add_argument("--n-queries", type=int, default=4, help="--fuse attn: learned query vectors (1 = an attention sum)")
    ap.add_argument("--attn-heads", type=int, default=8, help="--fuse attn: attention heads (must divide --enc-dim)")
    ap.add_argument("--view-dropout", type=float, default=0.1, help="probability, per molecule, that ONE input view is hidden")
    ap.add_argument("--balance", type=float, default=0.5, help="share of modified chains (E absent) in a batch; 0 = the pool's own mix")
    ap.add_argument("--enc-dim", type=int, default=512)
    ap.add_argument("--enc-hidden", type=int, default=1024)
    ap.add_argument("--trunk-hidden", type=int, default=2048)
    ap.add_argument("--latent", type=int, default=1024)
    ap.add_argument("--dec-hidden", type=int, default=1024)
    ap.add_argument("--blocks", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--min-delta", type=float, default=1e-4)
    ap.add_argument("--plateau-patience", type=int, default=3)
    ap.add_argument("--plateau-factor", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--first", type=int, default=0, help="only shards below this number (5 = the 10%% ablation subset of the 51 shards)")
    ap.add_argument("--data-on", choices=("gpu", "cpu"), default="gpu", help="where the cache lives while training")
    ap.add_argument("--e-on", choices=("cpu", "gpu"), default="cpu", help="where view E lives (cpu keeps 17 GB off a shared GPU)")
    ap.add_argument("--latent-floor", type=float, default=0.1,
                    help="floor of the per-dimension latent sd when standardising, as a fraction of the median sd (near-dead dimensions)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--embed-pool", action="store_true", help="write the standardised latent of every cached shard (distillation targets)")
    ap.add_argument("--use-last", action="store_true")
    ap.add_argument("--embed-bench-from", default=None, metavar="SUBFOLDER", help="no training: embed features/SUBFOLDER/<view>.npz")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if a.embed_bench_from:
        assert re.fullmatch(r"[A-Za-z0-9_\-]+", a.embed_bench_from), "--embed-bench-from: a folder name under features/"
        model, st, lat = load_model(a.name, dev, a.use_last)
        return embed_benchmarks(a.name, model, st, lat, dev, os.path.join(os.path.dirname(FEAT_DIR), a.embed_bench_from))
    if a.embed_pool:
        return embed_pool(a, dev)
    train(a, dev)


if __name__ == "__main__":
    main()

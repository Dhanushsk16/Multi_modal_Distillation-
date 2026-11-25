"""Distil the frozen teacher (and its views) into the SMILES-only student (sizes fixed 2026-11-22).

    python src/student/train_student.py --name s1 --tok CKPT/student_tokenizer_norm/atombpe_v1024 --first 5 --targets latent
    ... --name s2 --targets latent --w-nce 1 --w-rel 1                                    step 2: + InfoNCE + soft similarity matrix
    ... --name s3 --targets latent,P,L,H,E --w-nce 1 --w-rel 1 --balance ema --teacher-drop 0.1   step 3: + the four view heads
    ... --name s4 --targets P,L,H,E --w-nce 1 --w-rel 1 --balance ema --teacher-drop 0.1          step 4: no teacher latent at all
    ... --name s5 <the s3 flags> --mlm-frac 0.2                                           step 5: + masked-token loss on 20% of the steps
    ... --name ctl --targets none --mlm-frac 1                                            the MLM-ONLY CONTROL (same architecture and data)

Inputs (all must exist; scripts/run_student.sh makes them):
    features/corpus/student_<tokenizer>/shard_NNN.npz      token ids              (make_student_data.py)
    features/corpus/teacher_<teacher>/shard_NNN.npz        standardised 1024-d latent of the frozen teacher (train_teacher_v3.py --embed-pool)
    features/corpus/{pepland,pclm,ecfp,esm}/shard_NNN.npz  the teacher's views, standardised here with features/corpus/views_v3_stats.npz
                                                           (H: log1p first; every value clipped at +-10) = exactly what the teacher saw

Rows: every molecule whose teacher inputs were fine (the teacher shard's `ok`), whatever the targets, so every run and the control train on the
same molecules. Validation = pool rows with row % 100 == 0 (the teacher's own validation rows); never a benchmark row. Batches are half
"standard" (ESM-2 vector present, 81% of the pool) and half "modified", each group walked through without replacement.

Losses (every one is switched by a flag):
    per target t (latent, P, L, H, E): w_cos * (1 - cosine) + w_l1 * smooth L1 on the standardised vectors. The latent is compared with the student's
        1024-d output directly; P, L, H, E through the training-only projector heads. E only on rows that have it; P only on molecules of at most
        --p-max-res residues (PepLand's range; the teacher had to swallow its garbage above 30, the student does not imitate it).
    --w-nce W : symmetric in-batch InfoNCE (learnable temperature, start 0.07) between prediction and target of every pair in --rel-on
    --w-rel W : soft similarity matrix: the teacher's batch cosine-similarity matrix, row-softmaxed, is matched by the student's (KL), so
                near-duplicates in the batch are not pushed apart the way hard InfoNCE does
    --balance ema : every loss term is divided by its running mean (decay 0.99) so P (300-d) and E (2560-d) pull equally; none = plain weighted sum
    --teacher-drop p : with probability p per step one view head's terms are left out (the student cannot lean on one view)
    --mlm-frac f : on a fraction f of the steps 15% of the tokens are masked (80% [MASK], 10% random, 10% kept), an MLM loss is added and the
        distillation losses are computed on that masked input; the other steps see clean strings (no train / inference mismatch). f = 1 with
        --targets none is the MLM-only control.
    --p-random p : RANDOM-SMILES AUGMENTATION (2026-11-22). With probability p a molecule is read in one of its K stored random spellings
        (make_student_data.py --k-random K; RDKit random atom / branch order, same molecule) instead of the canonical string; every target stays the
        teacher's vector of the CANONICAL string, so spelling-order artefacts of the teacher's language view are not copied. Validation and
        inference always read the canonical string. --w-consist W adds 1 - cosine between the outputs of two different spellings of the same molecule
        (on --consist-frac of each batch, an extra forward pass). The validation line reports `spell` = the mean cosine between the canonical and a
        random spelling's output (also for runs without augmentation, as the baseline).
Evaluation uses an exponential moving average (decay --ema) of the weights. The criterion for early stopping and for best.pt is the validation
sum over the active targets of (1 - cosine), averaged over the standard and the modified group (the MLM control: the validation masked-token loss).
NOTHING is selected on benchmark data.

Outputs: $CKPT_DIR/student_distill/<name>/{best.pt, last.pt, history.csv, meta.json} and, with --save-every N, epoch_NN.pt (the EMA weights after every N-th epoch)
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "teacher"))
sys.path.insert(0, HERE)
from paths import CACHE_DIR, CKPT_DIR  # noqa: E402
import views_v3 as V  # noqa: E402
import tokenizer as T  # noqa: E402
from model import MASK, PAD, VIEW_DIM, Student, count_params  # noqa: E402
from make_student_data import student_dir  # noqa: E402

STUDENT_DIR = f"{CKPT_DIR}/student_distill"
VIEW_OF = {"P": "pepland", "L": "pclm", "H": "ecfp", "E": "esm"}
VAL_MOD = V.VAL_MOD


def say(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------------------------------------- data
class Rows:
    """Everything the trainer reads, in memory: token ids on the GPU, targets on the device chosen per target."""


def load_rows(shards, tok_folder, teacher, targets, dev, dev_of, cache_dir=CACHE_DIR):
    """-> Rows with ids (int32, flat), off (int64 [N+1]), ln (int64 [N]), res, ok, has_e, rows (numpy), data {target: tensor}."""
    sdir = student_dir(tok_folder, cache_dir)
    tdir = os.path.join(cache_dir, "teacher_" + teacher)
    spans = []
    for s in shards:
        zi = np.load(os.path.join(sdir, "shard_%03d.npz" % s))
        zt = np.load(os.path.join(tdir, "shard_%03d.npz" % s))
        ze = np.load(os.path.join(cache_dir, "esm", "shard_%03d.npz" % s))
        lo = int(zi["lo"])
        assert lo == int(zt["lo"]) == int(ze["lo"]), "shard %d: the token, teacher and ESM caches start at different pool rows" % s
        m = min(len(zi["len"]), len(zt["ok"]), len(ze["ok"]))
        spans.append((s, lo, m))
    n = sum(m for _, _, m in spans)
    R = Rows()
    R.ok, R.has_e, R.rows, R.res = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, np.int64), np.zeros(n, np.int16)
    ln = np.zeros(n, np.int64)
    ids_parts, rids_parts, rlen_parts = [], [], []
    ks = set()
    data = {}
    for t in targets:
        if t == "latent":
            data[t] = torch.empty((n, 1024), dtype=torch.float16, device=dev_of[t])
        else:
            v = VIEW_OF[t]
            data[t] = torch.empty((n, V.DIM[v]), dtype=torch.uint8 if t == "H" else torch.float16, device=dev_of[t])
    o = 0
    for s, lo, m in spans:
        zi = np.load(os.path.join(sdir, "shard_%03d.npz" % s))
        zt = np.load(os.path.join(tdir, "shard_%03d.npz" % s))
        ze = np.load(os.path.join(cache_dir, "esm", "shard_%03d.npz" % s))
        l = zi["len"][:m].astype(np.int64)
        ids_parts.append(zi["ids"][: int(l.sum())])
        ln[o:o + m] = l
        ks.add(int(zi["rlen"].shape[1]) if "rlen" in zi.files else 0)
        if ks != {0}:
            assert len(ks) == 1, "the token shards hold different numbers of random spellings %s: run make_student_data.py --redo" % sorted(ks)
            rl = zi["rlen"][:m].astype(np.int64)
            rlen_parts.append(rl)
            rids_parts.append(zi["rids"][: int(rl.sum())])
        R.ok[o:o + m] = zt["ok"][:m] & (l >= 3)
        R.has_e[o:o + m] = ze["ok"][:m] & R.ok[o:o + m]
        R.rows[o:o + m] = np.arange(lo, lo + m)
        R.res[o:o + m] = zi["res"][:m]
        for t in targets:
            if t == "latent":
                X = zt["X"][:m]
            else:
                v = VIEW_OF[t]
                zv = np.load(os.path.join(cache_dir, v, "shard_%03d.npz" % s))
                assert int(zv["lo"]) == lo and len(zv["ok"]) >= m, "%s shard %d does not line up with the token shard" % (v, s)
                X = np.ascontiguousarray(zv["X"][:m])
                if t == "E":
                    X = X.copy()
                    X[~ze["ok"][:m]] = 0
            data[t][o:o + m] = torch.from_numpy(np.ascontiguousarray(X)).to(dev_of[t])
        o += m
    # token store: the canonical spellings first, then (if the shards have them) K random spellings per molecule. A "slot" = molecule * K1 + j,
    # j = 0 the canonical spelling, j = 1..K the random ones; ln_slot / off_slot give every slot's length and start in the store
    ids_c = np.concatenate(ids_parts)
    K = next(iter(ks)) if ks else 0
    off_c = np.concatenate([[0], np.cumsum(ln)[:-1]])
    if K:
        ids_r = np.concatenate(rids_parts)
        rl = np.concatenate(rlen_parts)
        off_r = len(ids_c) + np.concatenate([[0], np.cumsum(rl.reshape(-1))[:-1]]).reshape(n, K)
        store = np.concatenate([ids_c, ids_r])
        ln_slot, off_slot = np.concatenate([ln[:, None], rl], 1), np.concatenate([off_c[:, None], off_r], 1)
    else:
        store, ln_slot, off_slot = ids_c, ln[:, None], off_c[:, None]
    assert store.max() < 32768, "token ids do not fit int16"
    R.K1 = K + 1
    R.ids = torch.from_numpy(store.astype(np.int16)).to(dev)
    R.ln = torch.from_numpy(ln).to(dev)                               # tokens of the canonical spelling of each molecule
    R.ln_slot = torch.from_numpy(np.ascontiguousarray(ln_slot).reshape(-1)).to(dev)
    R.off_slot = torch.from_numpy(np.ascontiguousarray(off_slot).reshape(-1)).to(dev)
    R.data = data
    R.n = n
    R.res_t = torch.from_numpy(R.res.astype(np.int64)).to(dev)
    R.has_e_t = torch.from_numpy(R.has_e).to(dev)
    return R


def gather_ids(R, slots):
    """Token slots (molecule * K1 + spelling, long tensor on the device) -> right-padded ids [B, Lmax] (long) and the real-token mask."""
    ln = R.ln_slot[slots]
    L = int(ln.max())
    ar = torch.arange(L, device=slots.device)
    keep = ar[None, :] < ln[:, None]
    pos = (R.off_slot[slots][:, None] + ar[None, :]).clamp_max(len(R.ids) - 1)
    return R.ids[pos].long() * keep, keep


def pick_slots(R, idx, p_random):
    """The spelling each molecule is read in: with probability p_random one of its random spellings (uniform), else the canonical one."""
    if R.K1 == 1 or p_random <= 0:
        return idx * R.K1
    j = torch.randint(1, R.K1, (len(idx),), device=idx.device)
    use = torch.rand(len(idx), device=idx.device) < p_random
    return idx * R.K1 + torch.where(use, j, torch.zeros_like(j))


def mask_tokens(ids, keep, vocab, rate, gen=None):
    """BERT masking of real, non-special tokens. -> (new ids, labels with -100 where nothing is predicted)."""
    dev = ids.device
    cand = keep & (ids >= 5)
    sel = (torch.rand(ids.shape, device=dev, generator=gen) < rate) & cand
    r = torch.rand(ids.shape, device=dev, generator=gen)
    new = ids.clone()
    new[sel & (r < 0.8)] = MASK
    rnd = torch.randint(5, vocab, ids.shape, device=dev, generator=gen)
    pick = sel & (r >= 0.8) & (r < 0.9)
    new[pick] = rnd[pick]
    return new, torch.where(sel, ids, torch.full_like(ids, -100))


def run_student(model, R, slots, chunks, mlm_rate=0.0, gen=None, want=("z",)):
    """The student on token slots: sorted by length and cut into `chunks` pieces (each padded to its own longest molecule, so a batch of short
    peptides does not pay for one long one), outputs put back in the order of `slots`. -> dict z [B, 1024], (pre [B, 1280]), tapm, and the
    masked-token loss (sum, count) when mlm_rate > 0."""
    order = torch.argsort(R.ln_slot[slots])
    inv = torch.empty_like(order)
    inv[order] = torch.arange(len(order), device=order.device)
    outs = {"z": [], "pre": [], "tapm": []}
    ce, cnt = 0.0, 0
    for part in torch.chunk(order, max(1, min(chunks, len(order)))):
        ids, keep = gather_ids(R, slots[part])
        if mlm_rate > 0:
            ids_in, labels = mask_tokens(ids, keep, model.cfg["vocab"], mlm_rate, gen)
        else:
            ids_in = ids
        o = model(ids_in, keep)
        outs["z"].append(o["z"])
        if "pre" in want:
            outs["pre"].append(o["pre"])
        if o["tapm"] is not None:
            outs["tapm"].append(o["tapm"])
        if mlm_rate > 0:
            sel = labels != -100
            if bool(sel.any()):
                lg = model.mlm_logits(o["h"][sel]).float()
                ce = ce + F.cross_entropy(lg, labels[sel], reduction="sum")
                cnt += int(sel.sum())
    res = {k: torch.cat(v)[inv] for k, v in outs.items() if v}
    res["tapm"] = res.get("tapm")
    if mlm_rate > 0:
        res["mlm"] = (ce, cnt)
    return res


# ------------------------------------------------------------------------------------------------- targets and losses
def std_target(name, x, st):
    """Raw cached rows -> the standardised float32 target (the teacher latent is stored standardised already)."""
    x = x.float()
    if name == "latent":
        return x
    if name == "H":
        x = torch.log1p(x)
    s = st[VIEW_OF[name]]
    return ((x - s["mu"]) / s["sd"]).clamp_(-V.CLIP, V.CLIP)


def fetch(R, idx, names, st, dev):
    return {t: std_target(t, R.data[t][idx.to(R.data[t].device)].to(dev, non_blocking=True), st) for t in names}


def align(pred, tgt, m, a):
    """w_cos * (1 - cosine) + w_l1 * smooth L1 over the rows with m (None = all). -> scalar or None when no row is left."""
    if m is not None:
        if not bool(m.any()):
            return None
        pred, tgt = pred[m], tgt[m]
    pred = pred.float()
    return a.w_cos * (1 - F.cosine_similarity(pred, tgt, dim=-1).mean()) + a.w_l1 * F.smooth_l1_loss(pred, tgt)


def info_nce(p, t, scale):
    p, t = F.normalize(p.float(), dim=-1), F.normalize(t, dim=-1)
    lg = scale.exp().clamp(max=100.0) * p @ t.t()
    lab = torch.arange(len(p), device=p.device)
    return 0.5 * (F.cross_entropy(lg, lab) + F.cross_entropy(lg.t(), lab))


def soft_relational(p, t, tau):
    """The teacher's cosine-similarity matrix of the batch (self excluded), row-softmaxed at temperature tau, against the student's."""
    p, t = F.normalize(p.float(), dim=-1), F.normalize(t, dim=-1)
    eye = torch.eye(len(p), dtype=torch.bool, device=p.device)
    sp = (p @ p.t() / tau).masked_fill(eye, -1e4)
    st_ = (t @ t.t() / tau).masked_fill(eye, -1e4)
    return F.kl_div(F.log_softmax(sp, -1), F.softmax(st_, -1), reduction="batchmean")


def step_losses(model, out, tg, pm, has_e, a, drop_view):
    """Every loss term of one step as {name: scalar}. `pm`: rows P may be imitated on (<= p-max-res residues); `has_e`: rows with an ESM vector."""
    comp, pairs = {}, {}
    if "latent" in a.targets_l:
        comp["a_latent"] = align(out["z"], tg["latent"], None, a)
        pairs["latent"] = (out["z"], tg["latent"], None)
    for v in a.views_l:
        if v == drop_view:
            continue
        m = {"P": pm, "E": has_e}.get(v)
        pr = model.view_pred(v, out["z"], out["tapm"])
        l = align(pr, tg[v], m, a)
        if l is not None:
            comp["a_" + v] = l
        pairs[v] = (pr, tg[v], m)
    for name in a.rel_on_l:
        if name not in pairs:
            continue
        p, t, m = pairs[name]
        if m is not None:
            p, t = p[m], t[m]
        if len(p) < 8:
            continue
        if a.w_nce > 0:
            comp["nce_" + name] = info_nce(p, t, model.train_only["nce_scale"][0])
        if a.w_rel > 0:
            comp["rel_" + name] = soft_relational(p, t, a.tau_rel)
    return {k: v for k, v in comp.items() if v is not None}


def term_weight(name, a):
    if name == "a_latent":
        return a.w_latent
    if name.startswith("a_"):
        return a.w_view
    if name.startswith("nce_"):
        return a.w_nce
    if name.startswith("rel_"):
        return a.w_rel
    if name == "cons":
        return a.w_consist
    return a.w_mlm


# ------------------------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def evaluate(model, R, idx, st, a, dev, bs=4096):
    """Validation rows idx on CLEAN strings, split into std (ESM present) and mod. Mean cosine to every active target, the latent's R2, and the
    criterion (sum over active targets of 1 - cosine, mean of the two groups). With --mlm-frac > 0 also the masked-token loss (fixed masks)."""
    model.eval()
    he = R.has_e_t[idx.to(dev)]
    res = {}
    gen = torch.Generator(device=dev)
    gen.manual_seed(123)
    for grp, sel in (("std", he), ("mod", ~he)):
        ii = idx[sel.cpu()] if idx.device.type == "cpu" else idx[sel]
        if len(ii) == 0:
            continue
        cos_sum = {t: 0.0 for t in a.active}
        cnt = {t: 0 for t in a.active}
        sse = sst_n = 0.0
        lat_sum = torch.zeros(1024, device=dev, dtype=torch.float64)
        lat_sq = torch.zeros(1024, device=dev, dtype=torch.float64)
        zs, ts = [], []
        mlm_sum, mlm_n = 0.0, 0
        spell_sum = 0.0
        for i in range(0, len(ii), bs):
            b = ii[i:i + bs].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out = run_student(model, R, b * R.K1, a.len_chunks)             # always the canonical spelling: what the student reads at inference
                if R.K1 > 1:                                                     # spelling invariance: canonical vs the first random spelling
                    o_r = run_student(model, R, b * R.K1 + 1, a.len_chunks)
                    spell_sum += float(F.cosine_similarity(out["z"].float(), o_r["z"].float(), dim=-1).sum())
            tg = fetch(R, b, a.active, st, dev)
            pm = R.res_t[b] <= a.p_max_res
            hb = R.has_e_t[b]
            for t in a.active:
                if t == "latent":
                    pr = out["z"].float()
                    m = None
                else:
                    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                        pr = model.view_pred(t, out["z"], out["tapm"]).float()
                    m = {"P": pm, "E": hb}.get(t)
                c = F.cosine_similarity(pr, tg[t], dim=-1)
                if m is not None:
                    c = c[m]
                cos_sum[t] += float(c.sum())
                cnt[t] += len(c)
                if t == "latent":
                    zs.append(pr.double())
                    ts.append(tg[t].double())
            if a.mlm_frac > 0:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    o2 = run_student(model, R, b * R.K1, a.len_chunks, mlm_rate=a.mlm_rate, gen=gen)
                if "mlm" in o2:
                    mlm_sum += float(o2["mlm"][0])
                    mlm_n += o2["mlm"][1]
        r = {"n": len(ii)}
        for t in a.active:
            if cnt[t]:                                        # E has no row in the modified group, P none above the residue limit
                r["cos_" + t] = cos_sum[t] / cnt[t]
        if zs:
            Z, Tt = torch.cat(zs), torch.cat(ts)
            r["r2_latent"] = float(1 - ((Z - Tt) ** 2).sum() / ((Tt - Tt.mean(0)) ** 2).sum())
        if mlm_n:
            r["mlm"] = mlm_sum / mlm_n
        if R.K1 > 1:
            r["spell"] = spell_sum / len(ii)                      # mean cosine between the 1024-d outputs of two spellings of one molecule (not in the criterion)
        r["crit"] = sum(1 - x for k, x in r.items() if k.startswith("cos_")) if a.active else r.get("mlm", float("nan"))
        res[grp] = r
    crit = float(np.mean([r["crit"] for r in res.values()]))
    model.train()
    return crit, res


# ------------------------------------------------------------------------------------------------- the trainer
class Cycler:
    """Indices from `idx`, shuffled, without replacement; reshuffled when used up."""

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
        return self.passes + self.pos / len(self.perm)


@torch.no_grad()
def ema_update(em, m, decay):
    ep, mp = list(em.parameters()), list(m.parameters())
    torch._foreach_mul_(ep, decay)
    torch._foreach_add_(ep, mp, alpha=1 - decay)


def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--tok", required=True, help="learned tokenizer folder (…/atombpe_v1024); make_student_data.py must have run on it")
    ap.add_argument("--teacher", default="v3_base")
    ap.add_argument("--targets", default="latent", help="comma list of latent,P,L,H,E or 'none' (the MLM-only control)")
    ap.add_argument("--first", type=int, default=0, help="only the first N cache shards (5 = the fixed 10%% subset)")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=8)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--std-share", type=float, default=0.5, help="share of each batch drawn from the standard group")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=0.05, help="share of all steps")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--len-chunks", type=int, default=4)
    # shape
    ap.add_argument("--d", type=int, default=320)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=5)
    ap.add_argument("--ff", type=int, default=768)
    ap.add_argument("--queries", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=192, help="longest stored string (the positional table and the check on the token files; random spellings go up to 192)")
    ap.add_argument("--canon-max-len", type=int, default=128, help="where the canonical spelling is cut at inference (export_student.py); = make_student_data.py --max-len")
    ap.add_argument("--ladder", action="store_true", help="the H head reads a learned mix of the layer-4 mean state and the 1024-d output")
    ap.add_argument("--tap", type=int, default=4)
    # losses
    ap.add_argument("--w-cos", type=float, default=0.9, help="AM-RADIO's weights as I remember them (0.9 / 0.1): not verified")
    ap.add_argument("--w-l1", type=float, default=0.1)
    ap.add_argument("--w-latent", type=float, default=1.0)
    ap.add_argument("--w-view", type=float, default=1.0)
    ap.add_argument("--w-nce", type=float, default=0.0)
    ap.add_argument("--w-rel", type=float, default=0.0)
    ap.add_argument("--w-mlm", type=float, default=1.0)
    ap.add_argument("--rel-on", default="auto", help="targets InfoNCE / similarity matrix act on: auto = latent if it is a target, else every view")
    ap.add_argument("--tau-rel", type=float, default=0.1)
    ap.add_argument("--balance", choices=["ema", "none"], default="none")
    ap.add_argument("--teacher-drop", type=float, default=0.0)
    ap.add_argument("--p-max-res", type=int, default=30)
    ap.add_argument("--mlm-frac", type=float, default=0.0)
    ap.add_argument("--mlm-rate", type=float, default=0.15)
    ap.add_argument("--p-random", type=float, default=0.0, help="random-SMILES augmentation: probability that a molecule is read in one of its stored random "
                    "spellings instead of the canonical one (targets stay the teacher's vectors of the canonical string); needs make_student_data.py --k-random")
    ap.add_argument("--w-consist", type=float, default=0.0, help="weight of the spelling-consistency loss: 1 - cosine between the 1024-d outputs of two different "
                    "spellings of the same molecule (on --consist-frac of the batch, an extra forward pass)")
    ap.add_argument("--consist-frac", type=float, default=0.25)
    ap.add_argument("--save-every", type=int, default=0, help="also keep the EMA weights after every N-th epoch as epoch_NN.pt (same format as best.pt; 0 = off)")
    # data placement
    ap.add_argument("--data-on", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument("--e-on", choices=["gpu", "cpu"], default=None, help="where view E lives (default: --data-on)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--out-dir", default=STUDENT_DIR)
    a = ap.parse_args(argv)
    a.targets_l = [] if a.targets == "none" else a.targets.split(",")
    bad = [t for t in a.targets_l if t not in ("latent", "P", "L", "H", "E")]
    assert not bad, "--targets: latent,P,L,H,E (got %s)" % bad
    a.views_l = [t for t in a.targets_l if t != "latent"]
    a.active = list(a.targets_l)
    a.rel_on_l = (["latent"] if "latent" in a.targets_l else a.views_l) if a.rel_on == "auto" else [t for t in a.rel_on.split(",") if t]
    assert a.targets_l or a.mlm_frac > 0, "no target and no masked-token loss: nothing to train (the consistency loss alone would collapse to a constant)"
    assert re.fullmatch(r"[A-Za-z0-9_\-]+", a.name), "--name: letters, digits, _ and - only"
    return a


def main(argv=None):
    a = parse(argv)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    rdir = os.path.join(a.out_dir, a.name)
    os.makedirs(rdir, exist_ok=True)
    tok = T.load_tokenizer(a.tok)
    ddev = dev if a.data_on == "gpu" else torch.device("cpu")
    dev_of = {t: ddev for t in ("latent", "P", "L", "H", "E")}
    if a.e_on:
        dev_of["E"] = dev if a.e_on == "gpu" else torch.device("cpu")

    # ---- rows
    have = set(V.shard_ids("pclm", a.cache_dir))
    sdir, tdir = student_dir(a.tok, a.cache_dir), os.path.join(a.cache_dir, "teacher_" + a.teacher)

    def listed(d):
        return {int(f[6:9]) for f in os.listdir(d) if f.startswith("shard_") and f.endswith(".npz") and ".tmp" not in f} if os.path.isdir(d) else set()

    shards = sorted(have & listed(sdir) & listed(tdir) & set(V.shard_ids("esm", a.cache_dir)))
    assert shards, "no shard has token ids (%s), teacher latents (%s) and the view caches: run make_student_data.py / train_teacher_v3.py --embed-pool" % (sdir, tdir)
    if a.first:
        shards = shards[: a.first]
    t0 = time.time()
    R = load_rows(shards, a.tok, a.teacher, a.targets_l, dev, dev_of, a.cache_dir)
    val = np.flatnonzero(R.ok & (R.rows % VAL_MOD == 0))
    tr = np.flatnonzero(R.ok & (R.rows % VAL_MOD != 0))
    say("rows: %s usable of %s (%d shards) | train %s, validation %s | standard (ESM present) %.1f%% / %.1f%% | %.0f s" % (
        format(int(R.ok.sum()), ","), format(R.n, ","), len(shards), format(len(tr), ","), format(len(val), ","),
        100 * R.has_e[tr].mean(), 100 * R.has_e[val].mean(), time.time() - t0))
    assert len(tr) >= a.batch and len(val) > 0 and R.has_e[tr].any() and (~R.has_e[tr]).any(), "too few usable molecules for batch %d" % a.batch
    mx = int(R.ln_slot.max())
    assert mx <= a.max_len, "token files hold molecules of %d tokens, --max-len %d" % (mx, a.max_len)
    assert R.K1 > 1 or (a.p_random <= 0 and a.w_consist <= 0), "--p-random / --w-consist need random spellings in the token shards: make_student_data.py --k-random 8 --redo"
    say("tokens per molecule: mean %.1f, max %d | vocabulary %d | P is imitated on %.1f%% of the training molecules (<= %d residues)" % (
        float(R.ln.float().mean()), mx, tok.vocab_size, 100 * (R.res[tr] <= a.p_max_res).mean(), a.p_max_res))
    if R.K1 > 1:
        say("random spellings: %d per molecule, mean %.1f tokens against %.1f canonical | augmentation: read in a random spelling with probability %.2f | "
            "consistency loss weight %.2g on %.0f%% of each batch" % (R.K1 - 1, float(R.ln_slot.view(-1, R.K1)[:, 1:].float().mean()), float(R.ln.float().mean()),
                                                                     a.p_random, a.w_consist, 100 * a.consist_frac))

    st = {}
    if a.views_l:
        s0 = V.load_stats(os.path.join(a.cache_dir, "views_v3_stats.npz"))
        st = {v: {k: torch.from_numpy(x).to(dev) for k, x in d.items()} for v, d in s0.items()}

    # ---- model
    model = Student(tok.vocab_size, d=a.d, layers=a.layers, heads=a.heads, ff=a.ff, n_query=a.queries, max_len=max(a.max_len, 16),
                    tap=a.tap, views=a.views_l, ladder=a.ladder and "H" in a.views_l, mlm=a.mlm_frac > 0).to(dev)
    inf, extra = count_params(model)
    say("student: %.3fM parameters at inference (embedding %.3fM, blocks %.3fM, pooling + output %.3fM) | training-only %.2fM (discarded)" % (
        inf / 1e6, model.emb.weight.numel() / 1e6, sum(p.numel() for p in model.blocks.parameters()) / 1e6,
        (model.proj.weight.numel() + model.proj.bias.numel() + model.queries.numel()) / 1e6, extra / 1e6))
    ema_m = copy.deepcopy(model).eval()
    for p in ema_m.parameters():
        p.requires_grad_(False)
    decay_p = [p for n, p in model.named_parameters() if p.ndim >= 2 and not n.startswith("emb") and "queries" not in n]
    other_p = [p for n, p in model.named_parameters() if not (p.ndim >= 2 and not n.startswith("emb") and "queries" not in n)]
    opt = torch.optim.AdamW([{"params": decay_p, "weight_decay": a.wd}, {"params": other_p, "weight_decay": 0.0}], lr=a.lr, betas=(0.9, 0.98))
    steps_per_epoch = max(1, len(tr) // a.batch)
    total = a.epochs * steps_per_epoch
    warm = max(1, int(a.warmup * total))

    def lr_at(s):
        return a.lr * (s + 1) / warm if s < warm else a.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    # ---- state
    ep0, step, best, bad, hist = 0, 0, float("inf"), 0, []
    last = os.path.join(rdir, "last.pt")
    if a.resume and os.path.exists(last):
        ck = torch.load(last, map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"])
        ema_m.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        ep0, step, best, bad, hist = ck["epoch"], ck["step"], ck["best"], ck["bad"], ck["hist"]
        say("resumed from %s at epoch %d (best criterion %.4f)" % (last, ep0, best))
    elif a.resume:
        say("--resume: no last.pt in %s, starting from scratch" % rdir)

    he_tr = R.has_e[tr]
    std_idx = torch.from_numpy(tr[he_tr]).to(dev)
    mod_idx = torch.from_numpy(tr[~he_tr]).to(dev)
    cyc_s, cyc_m = Cycler(std_idx, dev), Cycler(mod_idx, dev)
    n_s = int(round(a.batch * a.std_share))
    val_t = torch.from_numpy(val)
    say("steps per epoch %d (batch %d: %d standard + %d modified) | %d epochs | lr %.1e | targets %s | rel on %s | nce %.2g rel %.2g | balance %s | drop %.2f | mlm %.2f | random spelling p %.2f consist %.2g" % (
        steps_per_epoch, a.batch, n_s, a.batch - n_s, a.epochs, a.lr, a.targets_l or "none", a.rel_on_l, a.w_nce, a.w_rel, a.balance, a.teacher_drop, a.mlm_frac,
        a.p_random, a.w_consist))
    json.dump({"args": {k: v for k, v in vars(a).items()}, "inference_params": inf, "training_only_params": extra, "shards": len(shards), "train_rows": int(len(tr)),
               "val_rows": int(len(val))}, open(os.path.join(rdir, "meta.json"), "w"), indent=1)

    ema_loss = {}
    model.train()
    for ep in range(ep0, a.epochs):
        t1 = time.time()
        acc, acc_n = {}, {}
        tok_acc = torch.zeros((), dtype=torch.long, device=dev)
        for _ in range(steps_per_epoch):
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            idx = torch.cat([cyc_s.next(n_s), cyc_m.next(a.batch - n_s)])
            slots = pick_slots(R, idx, a.p_random)                   # the spelling each molecule is read in this step
            masked = a.mlm_frac > 0 and random.random() < a.mlm_frac
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                out = run_student(model, R, slots, a.len_chunks, mlm_rate=a.mlm_rate if masked else 0.0)
            tg = fetch(R, idx, a.active, st, dev) if a.active else {}
            drop = random.choice(a.views_l) if (a.teacher_drop > 0 and len(a.views_l) >= 2 and random.random() < a.teacher_drop) else None
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                comp = step_losses(model, out, tg, R.res_t[idx] <= a.p_max_res, R.has_e_t[idx], a, drop) if a.active else {}
            if masked and "mlm" in out and out["mlm"][1] > 0:
                comp["mlm"] = out["mlm"][0] / out["mlm"][1]
            if a.w_consist > 0 and not masked:                       # two different spellings of the same molecule must give the same embedding
                sel = torch.randperm(len(idx), device=dev)[:max(8, int(a.consist_frac * len(idx)))]
                j1 = slots[sel] % R.K1
                j2 = (j1 + torch.randint(1, R.K1, (len(sel),), device=dev)) % R.K1
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                    out2 = run_student(model, R, idx[sel] * R.K1 + j2, a.len_chunks)
                comp["cons"] = (1 - F.cosine_similarity(out["z"][sel].float(), out2["z"].float(), dim=-1)).mean()
            assert comp, "no loss term on this step"
            total_loss = 0.0
            for k, l in comp.items():
                if a.balance == "ema":
                    e = ema_loss.get(k)
                    ema_loss[k] = l.detach() if e is None else 0.99 * e + 0.01 * l.detach()
                    total_loss = total_loss + term_weight(k, a) * l / ema_loss[k].clamp_min(1e-6)
                else:
                    total_loss = total_loss + term_weight(k, a) * l
                acc[k] = acc.get(k, 0.0) + l.detach()
                acc_n[k] = acc_n.get(k, 0) + 1
            opt.zero_grad(set_to_none=True)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
            ema_update(ema_m, model, min(a.ema, (1 + step) / (10 + step)))
            step += 1
            tok_acc += R.ln_slot[slots].sum()
        tok_n = int(tok_acc)
        train_str = " ".join("%s %.4f" % (k, float(v) / acc_n[k]) for k, v in acc.items())
        crit, v = evaluate(ema_m, R, val_t, st, a, dev)
        secs = time.time() - t1
        row = {"epoch": ep + 1, "seconds": round(secs, 1), "lr": opt.param_groups[0]["lr"], "crit": crit}
        row.update({"train_" + k: float(x) / acc_n[k] for k, x in acc.items()})
        for grp, r in v.items():
            row.update({"%s_%s" % (k, grp): x for k, x in r.items() if k != "n"})
        hist.append(row)
        vs = " | ".join("%s %s" % (grp, " ".join("%s %.3f" % (k, x) for k, x in r.items() if k not in ("n", "crit"))) for grp, r in v.items())
        improved = crit < best - 1e-4
        bad = 0 if improved else bad + 1
        say("epoch %3d/%d | %.0f s (%.0f tokens/s) | lr %.2e | train: %s | val crit %.4f (best %.4f, no gain %d/%d) | %s" % (
            ep + 1, a.epochs, secs, tok_n / secs, opt.param_groups[0]["lr"], train_str, crit, min(best, crit), bad, a.patience, vs))
        if improved:
            best = crit
            torch.save({"model": ema_m.state_dict(), "cfg": model.cfg, "args": vars(a), "tok": a.tok, "epoch": ep + 1, "crit": crit, "val": v},
                       os.path.join(rdir, "best.pt.tmp"))
            os.replace(os.path.join(rdir, "best.pt.tmp"), os.path.join(rdir, "best.pt"))
        torch.save({"model": model.state_dict(), "ema": ema_m.state_dict(), "opt": opt.state_dict(), "epoch": ep + 1, "step": step, "best": best, "bad": bad,
                    "hist": hist}, last + ".tmp")
        os.replace(last + ".tmp", last)
        if a.save_every > 0 and (ep + 1) % a.save_every == 0:       # a numbered copy of the EMA weights, same format as best.pt (export_student.py --epoch N reads it)
            snap = os.path.join(rdir, "epoch_%02d.pt" % (ep + 1))
            torch.save({"model": ema_m.state_dict(), "cfg": model.cfg, "args": vars(a), "tok": a.tok, "epoch": ep + 1, "crit": crit, "val": v}, snap + ".tmp")
            os.replace(snap + ".tmp", snap)
            say("saved %s (EMA weights after epoch %d)" % (snap, ep + 1))
        with open(os.path.join(rdir, "history.csv"), "w", newline="") as fh:
            keys = sorted({k for r in hist for k in r}, key=lambda k: (k not in ("epoch", "seconds", "lr", "crit"), k))
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(hist)
        if bad >= a.patience:
            say("early stop: no gain of the validation criterion for %d epochs; best epoch %d (criterion %.4f)" % (a.patience, ep + 1 - bad, best))
            break
    say("done. best.pt = the EMA weights of the best epoch (criterion %.4f) in %s | next: python src/student/export_student.py --name %s" % (best, rdir, a.name))


if __name__ == "__main__":
    main()

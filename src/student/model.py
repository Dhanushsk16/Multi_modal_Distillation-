"""The SMILES-only student (sizes fixed on 2026-11-22).

    tokens -> embedding (vocab x 320) -> 8 bidirectional Transformer blocks -> RMSNorm -> multi-query attentive pooling
           -> Linear 1280 -> 1024   (no activation, no normalisation on the 1024-d vector: it is regressed onto the teacher's latent)

Block (pre-norm): x + Attn(RMSNorm(x)), then x + SwiGLU(RMSNorm(x)); 5 heads of 64 dimensions, RoPE on queries and keys (no learned
positions), no biases in the blocks. SwiGLU hidden size 768. Pooling: 4 learned query vectors (320-d) attend over the token states
(keys = values = the states, one scaled dot-product head, padding masked); the four outputs are concatenated (1280-d) and one linear layer
makes the 1024-d embedding.

Everything under `train_only` is thrown away after distillation and is NOT counted in the size:
    * one projector per teacher view (P 300, L 1024, H 2048, E 2560): Linear 1024-1024, LayerNorm, GELU, Linear 1024-dim (AM-RADIO style)
    * the optional ladder shortcut for H (mean of the layer-`tap` token states -> Linear 320-1024, mixed with the final 1024-d vector by a
      learned weight)
    * the masked-token head (logits = states @ embedding^T + a bias: weight-tied, only the bias is extra)
    * the learnable InfoNCE temperature
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PAD, UNK, CLS, SEP, MASK = 0, 1, 2, 3, 4
VIEW_DIM = {"P": 300, "L": 1024, "H": 2048, "E": 2560}


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.w.to(x.dtype)


def rope_tables(max_len, dim, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    f = torch.outer(torch.arange(max_len).float(), inv)
    e = torch.cat([f, f], -1)
    return e.cos(), e.sin()                                    # [max_len, dim] each


def rotate_half(x):
    a, b = x.chunk(2, -1)
    return torch.cat([-b, a], -1)


def apply_rope(x, cos, sin):                                   # x [B, h, L, hd]; cos, sin [L, hd]
    return x * cos + rotate_half(x) * sin


class Block(nn.Module):
    def __init__(self, d, heads, ff):
        super().__init__()
        assert d % heads == 0
        self.h, self.hd = heads, d // heads
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.w1 = nn.Linear(d, ff, bias=False)                 # gate
        self.w3 = nn.Linear(d, ff, bias=False)                 # value
        self.w2 = nn.Linear(ff, d, bias=False)

    def forward(self, x, cos, sin, mask):                      # mask [B, 1, 1, L] bool, True = a real token
        B, L, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, L, 3, self.h, self.hd).permute(2, 0, 3, 1, 4)
        q = apply_rope(q, cos[:L], sin[:L]).to(v.dtype)
        k = apply_rope(k, cos[:L], sin[:L]).to(v.dtype)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = x + self.o(a.transpose(1, 2).reshape(B, L, D))
        y = self.n2(x)
        return x + self.w2(F.silu(self.w1(y)) * self.w3(y))


def projector(dim_out, d_in=1024):
    return nn.Sequential(nn.Linear(d_in, 1024), nn.LayerNorm(1024), nn.GELU(), nn.Linear(1024, dim_out))


class Student(nn.Module):
    def __init__(self, vocab, d=320, layers=8, heads=5, ff=768, n_query=4, out=1024, max_len=256, tap=4,
                 views=("P", "L", "H", "E"), ladder=False, mlm=True):
        super().__init__()
        self.cfg = dict(vocab=vocab, d=d, layers=layers, heads=heads, ff=ff, n_query=n_query, out=out, max_len=max_len, tap=tap,
                        views=list(views), ladder=ladder, mlm=mlm)
        self.d, self.tap = d, tap
        self.emb = nn.Embedding(vocab, d, padding_idx=PAD)
        self.blocks = nn.ModuleList([Block(d, heads, ff) for _ in range(layers)])
        self.norm_f = RMSNorm(d)
        self.queries = nn.Parameter(torch.randn(n_query, d) * 0.02)
        self.proj = nn.Linear(n_query * d, out)
        cos, sin = rope_tables(max_len, d // heads)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        # ---- training only
        t = nn.ModuleDict()
        for v in views:
            t["head_" + v] = projector(VIEW_DIM[v], out)
        if ladder:
            t["ladder_proj"] = nn.Linear(d, out)
            t["ladder_mix"] = nn.ParameterList([nn.Parameter(torch.zeros(()))])
        if mlm:
            t["mlm_bias"] = nn.ParameterList([nn.Parameter(torch.zeros(vocab))])
        t["nce_scale"] = nn.ParameterList([nn.Parameter(torch.tensor(math.log(1 / 0.07)))])
        self.train_only = t
        self._init(layers)

    def _init(self, layers):
        for n, m in self.named_modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
        with torch.no_grad():
            self.emb.weight[PAD].zero_()
            s = 0.02 / math.sqrt(2 * layers)                   # GPT-2 style: the residual-branch outputs start small
            for b in self.blocks:
                b.o.weight.normal_(std=s)
                b.w2.weight.normal_(std=s)

    # ------------------------------------------------------------------------------------------ forward
    def encode(self, ids, keep=None):
        """ids [B, L] (right-padded with 0) -> (states [B, L, d] after the final norm, tap states [B, L, d] or None, mask [B, L] bool)."""
        mask = ids != PAD if keep is None else keep
        x = self.emb(ids)
        m4 = mask[:, None, None, :]
        tap = None
        for i, b in enumerate(self.blocks):
            x = b(x, self.cos, self.sin, m4)
            if self.cfg["ladder"] and i + 1 == self.tap:
                tap = x
        return self.norm_f(x), tap, mask

    def pool(self, h, mask):
        """Multi-query attentive pooling. h [B, L, d] -> [B, n_query * d]."""
        s = torch.einsum("qd,bld->bql", self.queries.to(h.dtype), h).float() / math.sqrt(self.d)
        s = s.masked_fill(~mask[:, None, :], float("-inf"))
        a = torch.softmax(s, -1).to(h.dtype)
        return torch.einsum("bql,bld->bqd", a, h).flatten(1)

    def forward(self, ids, keep=None):
        """-> z [B, 1024] (the embedding), pre [B, 1280] (pooled, before the last linear layer), h [B, L, d] token states,
        tapm [B, d] mean of the layer-`tap` token states (only with the ladder), mask."""
        h, tap, mask = self.encode(ids, keep)
        pre = self.pool(h, mask)
        tapm = None
        if tap is not None:
            m = mask.unsqueeze(-1).to(tap.dtype)
            tapm = (tap * m).sum(1) / m.sum(1).clamp_min(1)
        return {"z": self.proj(pre), "pre": pre, "h": h, "tapm": tapm, "mask": mask}

    # ------------------------------------------------------------------------------------------ training-only parts
    def view_pred(self, view, z, tapm=None):
        """Prediction of a teacher view from the student's 1024-d output (the H head can read the ladder mix)."""
        t = self.train_only
        if view == "H" and self.cfg["ladder"] and tapm is not None:
            a = torch.sigmoid(t["ladder_mix"][0]).to(z.dtype)
            z = a * z + (1 - a) * t["ladder_proj"](tapm)
        return t["head_" + view](z)

    def mlm_logits(self, h):
        return h @ self.emb.weight.t().to(h.dtype) + self.train_only["mlm_bias"][0].to(h.dtype)


def count_params(model):
    """(parameters that run at inference, training-only parameters). The tied masked-token weight is the embedding (counted once)."""
    total = sum(p.numel() for p in model.parameters())
    extra = sum(p.numel() for p in model.train_only.parameters())
    return total - extra, extra

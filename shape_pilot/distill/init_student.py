"""Write a randomly initialised student as an ordinary HuggingFace directory.

Same architecture as the released checkpoints (their ChemPepMTR.MLM_model: pre-LN,
RoPE, SwiGLU, 405-token vocabulary) at a shape of our choosing, so train_kd.py and
every downstream script load it with AutoModel.from_pretrained exactly as they load
theirs. The code and tokenizer files are copied from the mlm-small snapshot, the
same way export_truncated.py builds the pruned model; only config.json and the
weights differ.

Parameter count of this architecture, used as a check on the build:

    per block = 4d^2 + 3df + 2f + 5d
        qkv + out_proj, no bias          4d^2
        SwiGLU d -> 2f -> d, with bias   3df + 2f + d
        two LayerNorms                   4d
    total = L * block + 812d + 405
        embedding 405d, LM head 405d + 405, final LayerNorm 2d

    python distill/init_student.py --embed-dim 384 --heads 6 --ffn 768 --blocks 7 \
        --out $DATA/checkpoints/students/student-s384x7
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import torch
from transformers import AutoModel

DATA = os.environ.get("BTP_DATA", "/mnt/wwn-0x5000c500e64c5cc0/dhanushsk/BTPData")
SMALL = f"{DATA}/checkpoints/peptideclm-2-mlm-small"


def expected_params(d, f, L, vocab=405):
    return L * (4 * d * d + 3 * d * f + 2 * f + 5 * d) + 2 * vocab * d + vocab + 2 * d


def build(src, out, d, heads, f, L, seed):
    assert d % heads == 0, "embed_dim must be divisible by heads"
    os.makedirs(out, exist_ok=True)
    for fn in os.listdir(src):
        p = os.path.join(src, fn)
        if fn not in ("model.safetensors", "config.json") and os.path.isfile(p):
            shutil.copy(p, os.path.join(out, fn))
    cfg = json.load(open(os.path.join(src, "config.json")))
    cfg.update(embed_dim=d, num_heads=heads, ffn_hidden_dim=f, num_blocks=L,
               size="student-%dx%d" % (d, L))
    cfg["init"] = {"from": "random", "seed": seed}
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)

    # Construct from the copied config so the weights are random, not loaded.
    torch.manual_seed(seed)
    from transformers import AutoConfig
    conf = AutoConfig.from_pretrained(out, trust_remote_code=True)
    model = AutoModel.from_config(conf, trust_remote_code=True)
    n = sum(p.numel() for p in model.parameters())
    want = expected_params(d, f, L, cfg["vocab_size"])
    assert n == want, "param count %d != formula %d" % (n, want)
    model.save_pretrained(out, safe_serialization=True)
    # save_pretrained rewrites config.json from the config object; put ours back
    # so auto_map and our init note survive.
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    return n


@torch.no_grad()
def verify(out, d):
    """Round-trip load, then one forward pass on real molecules."""
    from transformers import AutoTokenizer

    from student import patch_sdpa_dtype
    from train_kd import refresh_rope, verify_rope
    patch_sdpa_dtype()
    m = AutoModel.from_pretrained(out, trust_remote_code=True,
                                  use_safetensors=True).eval()
    refresh_rope(m, "cpu"); verify_rope(m, "student")
    tok = AutoTokenizer.from_pretrained(out, trust_remote_code=True)
    smis = ["CC(C)C[C@H](NC(=O)[C@@H](N)CCCNC(=N)N)C(=O)O",
            "NCCCC[C@H](NC(=O)CN)C(=O)N[C@@H](Cc1ccccc1)C(=O)O"]
    b = tok(smis, return_tensors="pt", padding=True)
    o = m(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
    assert o.logits.shape[:2] == b["input_ids"].shape and o.logits.shape[2] == 405
    assert o.mean_pool.shape == (2, d)
    assert torch.isfinite(o.logits).all() and torch.isfinite(o.mean_pool).all()
    print("   verified: reload OK, logits %s, mean_pool %s, finite"
          % (tuple(o.logits.shape), tuple(o.mean_pool.shape)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed-dim", type=int, required=True)
    ap.add_argument("--heads", type=int, required=True)
    ap.add_argument("--ffn", type=int, required=True, help="SwiGLU hidden dim")
    ap.add_argument("--blocks", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default=None, help="code/tokenizer source; default mlm-small")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    hits = [a.src] if a.src else glob.glob(SMALL)
    assert hits, "mlm-small snapshot not found under %s (pass --src)" % SMALL
    n = build(hits[0], a.out, a.embed_dim, a.heads, a.ffn, a.blocks, a.seed)
    print("%s: d=%d heads=%d ffn=%d blocks=%d -> %s params (%.2fM)"
          % (a.out, a.embed_dim, a.heads, a.ffn, a.blocks, "{:,}".format(n), n / 1e6))
    verify(a.out, a.embed_dim)


if __name__ == "__main__":
    main()

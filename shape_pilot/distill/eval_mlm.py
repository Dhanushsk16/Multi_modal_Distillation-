"""Score any MLM checkpoint on the SAME held-out molecules with the SAME masks.

The val numbers logged during training are not comparable across runs: they
score only the first 40 batches, and batches are filled to a token budget, so a
different --max-tokens scores a different (and differently composed) subset.
This script removes both effects:

  - every one of the 20,000 held-out molecules is scored, not the first 40 batches
  - batching and masking are a pure function of (seed, max_tokens), both fixed
    here, so every model sees byte-identical inputs and masked positions
  - results are split by molecule source, because long natural peptides are far
    easier to fill in than small molecules and the mix moves the overall number

Exactly the same masking code as training (data_live.LiveStream), replayed with
the molecule index kept so each masked token can be attributed to its source.

ONE MODEL PER PROCESS: two PeptideCLM-2 models in one interpreter corrupt each
other's rotary buffers. Run it once per model:

    python eval_mlm.py --model $CKPT_DIR/peptideclm-2-mlm-large --name mlm-large
    python eval_mlm.py --model $CKPT_DIR/students/student-s320x10 \
        --weights $CKPT_DIR/shape_pilot/s320x10/student_final.pt --name s320x10
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from data_live import LiveStream, build_batches
from student import patch_sdpa_dtype
from train_kd import refresh_rope, verify_rope

DATA = os.environ.get("BTP_DATA", "/mnt/wwn-0x5000c500e64c5cc0/dhanushsk/BTPData")
RESULTS_DIR = f"{DATA}/results"

SOURCES = {"ESMAtlas_clust30_clean.smiles": "peptide",
           "PubChem_clean4.smiles": "small_molecule",
           "LMSD_clean.smiles": "lipid"}


def batches_with_index(stream, epoch=0):
    """LiveStream.iter_epoch, replayed step for step, but also yielding the index
    of each molecule in the batch. Same RNG calls in the same order, so the masks
    are identical to what iter_epoch would produce."""
    rng = np.random.default_rng(stream.seed * 1000 + epoch)
    starts = list(range(0, len(stream.smiles), stream.chunk))
    rng.shuffle(starts)
    for s0 in starts:
        block = list(stream.smiles[s0:s0 + stream.chunk])
        enc = stream.tok(block, add_special_tokens=False)["input_ids"]
        lengths = np.array([len(x) + 2 for x in enc])
        for rows in build_batches(lengths, stream.max_tokens, rng):
            out = stream._collate(rows, enc, lengths, rng)
            if out is not None:
                yield out, s0 + np.asarray(rows)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF model dir (config + code + tokenizer)")
    ap.add_argument("--weights", default=None,
                    help="optional student_final.pt / latest.pt to load over --model")
    ap.add_argument("--name", required=True)
    ap.add_argument("--subset", default=f"{DATA}/data/pretrain_subset_2M/pretrain_subset_2M.parquet")
    ap.add_argument("--val-molecules", type=int, default=20_000)
    ap.add_argument("--max-tokens", type=int, default=16_384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=f"{RESULTS_DIR}/mlm_eval")
    ap.add_argument("--limit", type=int, default=None, help="debug only")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    patch_sdpa_dtype()
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    model = AutoModel.from_pretrained(a.model, trust_remote_code=True, use_safetensors=True)
    if a.weights:
        st = torch.load(a.weights, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"], strict=True)
    model = model.to(dev).eval()
    refresh_rope(model, dev); verify_rope(model, a.name)
    n_params = sum(p.numel() for p in model.parameters())

    stream = LiveStream(a.subset, tok, max_tokens=a.max_tokens, seed=a.seed,
                        val_molecules=a.val_molecules, split="val", limit=a.limit)
    src_all = pd.read_parquet(a.subset, columns=["source"])["source"].to_numpy()
    if a.limit:
        src_all = src_all[:a.limit]
    src_val = np.array([SOURCES.get(s, s) for s in src_all[-a.val_molecules:]])

    groups = ["all"] + sorted(set(src_val))
    nll = {g: 0.0 for g in groups}
    hit = {g: 0 for g in groups}
    cnt = {g: 0 for g in groups}
    t0 = time.time()
    for b, mol_idx in batches_with_index(stream):
        b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=(dev != "cpu")):
            out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        logits = (out["logits"] if isinstance(out, dict) else out.logits)
        logits = logits[b["rows"], b["cols"]].float()
        ce = F.cross_entropy(logits, b["labels"], reduction="none").cpu().numpy()
        ok = (logits.argmax(-1) == b["labels"]).cpu().numpy()
        kind = src_val[mol_idx[b["rows"].cpu().numpy()]]
        for g in groups:
            m = slice(None) if g == "all" else (kind == g)
            nll[g] += float(ce[m].sum()); hit[g] += int(ok[m].sum()); cnt[g] += int(np.size(ce[m]))

    res = {"name": a.name, "params_M": round(n_params / 1e6, 2),
           "max_tokens": a.max_tokens, "seed": a.seed, "minutes": round((time.time() - t0) / 60, 2)}
    for g in groups:
        res[g] = {"loss": round(nll[g] / max(cnt[g], 1), 4),
                  "acc": round(hit[g] / max(cnt[g], 1), 4),
                  "masked_tokens": cnt[g]}
    print(json.dumps(res, indent=1))
    os.makedirs(a.out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(a.out_dir, a.name + ".json"), "w"), indent=1)


if __name__ == "__main__":
    main()

"""Inference cost of the 337M teacher against the 84.8M pruned student.

Measured on the real CellPPD test molecules rather than synthetic fixed-length
input, because encoder cost scales with sequence length and padding to a uniform
max_length would flatter whichever model is being timed on short sequences.

ONE MODEL PER PROCESS, deliberately. Loading both in the same interpreter
corrupts the non-persistent rotary buffer (the same bug that made the bi_drop8
export verification report a 0.90 cosine), and peak-memory numbers are only
meaningful when nothing else is resident.

Reports throughput, per-molecule latency and peak memory. Run as:

    python distill/bench_latency.py --model teacher --device cuda
    python distill/bench_latency.py --model student --device cuda
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

DATA = os.environ.get("BTP_DATA", "/mnt/wwn-0x5000c500e64c5cc0/dhanushsk/BTPData")
CKPT_DIR = f"{DATA}/checkpoints"
RESULTS_DIR = f"{DATA}/results"

PATHS = {
    "teacher": f"{CKPT_DIR}/peptideclm-2-mlm-large",
    "student": f"{CKPT_DIR}/peptideclm-2-mlm-bisel8",
}


def resolve(name):
    p = PATHS[name]
    hits = glob.glob(p)
    if not hits:
        raise SystemExit("no model at %s" % p)
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="'teacher', 'student', or a label for --path")
    ap.add_argument("--path", default=None, help="any model dir; --model is then its label")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16", "bf16"])
    ap.add_argument("--csv", default=f"{DATA}/data/benchmarks/CellPPD_test.csv")
    ap.add_argument("--out-dir", default=f"{RESULTS_DIR}/latency")
    a = ap.parse_args()

    dev = torch.device(a.device)
    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[a.dtype]
    src = a.path or resolve(a.model)

    smiles = pd.read_csv(a.csv).smiles.tolist()

    if dtype != torch.float32:
        # Their attention builds an fp32 padding mask; SDPA rejects it against
        # fp16/bf16 queries on CUDA. Same wrapper train_kd.py uses.
        from student import patch_sdpa_dtype
        patch_sdpa_dtype()

    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=True)
    model = AutoModel.from_pretrained(src, trust_remote_code=True,
                                      torch_dtype=dtype).to(dev).eval()

    n_params = sum(p.numel() for p in model.parameters())

    # Pre-tokenise so tokenizer cost never lands inside the timed region.
    batches = []
    for i in range(0, len(smiles), a.batch):
        enc = tok(smiles[i:i + a.batch], return_tensors="pt", padding=True,
                  truncation=True, max_length=a.max_length)
        batches.append({k: v.to(dev) for k, v in enc.items()})
    n_tokens = int(sum(b["attention_mask"].sum().item() for b in batches))

    def sweep():
        with torch.no_grad():
            for b in batches:
                model(**b)

    for _ in range(a.warmup):
        sweep()
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(a.repeats):
        t0 = time.perf_counter()
        sweep()
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    t = np.array(times)
    peak = (torch.cuda.max_memory_allocated() / 2**20) if dev.type == "cuda" else float("nan")

    out = dict(model=a.model, device=a.device, dtype=a.dtype, batch=a.batch,
               params_M=round(n_params / 1e6, 1), n_molecules=len(smiles),
               n_real_tokens=n_tokens,
               sweep_s_mean=round(float(t.mean()), 4),
               sweep_s_sd=round(float(t.std(ddof=1)), 4),
               ms_per_molecule=round(1000 * float(t.mean()) / len(smiles), 3),
               molecules_per_s=round(len(smiles) / float(t.mean()), 1),
               tokens_per_s=round(n_tokens / float(t.mean()), 1),
               peak_mem_MB=round(peak, 1) if peak == peak else None)

    print(json.dumps(out, indent=2))
    os.makedirs(a.out_dir, exist_ok=True)
    with open(os.path.join(a.out_dir, "%s_%s_%s.json" % (a.model, a.device, a.dtype)), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()

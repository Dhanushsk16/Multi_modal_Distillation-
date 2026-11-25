"""Is a small SMILES-only student faster than ECFP on THIS machine? (efficiency claim: "a model that runs on a laptop")

    python src/student/time_laptop.py                       # everything, CellPPD molecules
    python src/student/time_laptop.py --lengths 20 40 70    # token lengths of the student's input

What is timed, per molecule:
  ECFP     the fingerprint call only (count ECFP, radius 2, 2048), molecules ALREADY parsed: RDKit parsing is common to both sides and left out.
           1 thread, and N worker processes (throughput = molecules / wall time of the slowest worker).
  tokenise the student's own extra CPU step (our AtomBPE: a Python prep + the Rust tokenizer); a small tokenizer is trained on the benchmark
           SMILES only to time it (compression numbers are indicative, not the real tokenizer's).
  student  the forward pass of an UNTRAINED SMILES-only transformer of ~3M and ~10M parameters (random weights: the cost depends on shapes
           only), plain PyTorch, no compile / no export: GPU fp16 and fp32, CPU fp32. Two figures on the GPU: compute only (ids already on
           the GPU, output stays there) and with the transfers (ids host -> GPU, 1024-d vector GPU -> host).
Normalisation / parsing by RDKit (needed by both sides) is NOT counted anywhere. Results: runs/time_laptop/results.csv.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))


def load_smiles(bench):
    try:
        import benchmarks as BM
        return [str(s) for s in BM.load(bench).smiles]
    except Exception as e:                                         # no benchmark files here: synthetic peptides
        print("benchmark %s not available (%s): using synthetic peptides" % (bench, str(e)[:70]))
        from rdkit import Chem
        rng = np.random.default_rng(0)
        aa = list("ACDEFGHIKLMNPQRSTVWY")
        return [Chem.MolToSmiles(Chem.MolFromSequence("".join(rng.choice(aa, rng.integers(8, 31))))) for _ in range(1500)]


# ------------------------------------------------------------------------------------------------ ECFP
def _fp_worker(blobs, reps=3):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mols = [Chem.Mol(b) for b in blobs]
    t = time.perf_counter()
    for _ in range(reps):
        for m in mols:
            gen.GetCountFingerprintAsNumPy(m)
    return time.perf_counter() - t, len(mols) * reps


def part_ecfp(smiles, workers):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mols = [m for m in (Chem.MolFromSmiles(s) for s in smiles) if m is not None]
    for m in mols[:50]:
        gen.GetCountFingerprintAsNumPy(m)
    ts = []
    for _ in range(3):
        for m in mols:
            t = time.perf_counter_ns()
            gen.GetCountFingerprintAsNumPy(m)
            ts.append(time.perf_counter_ns() - t)
    ts = np.array(ts) / 1e6
    ha = np.array([m.GetNumHeavyAtoms() for m in mols])
    out = {"ecfp_1thread_mean_ms": float(ts.mean()), "ecfp_1thread_median_ms": float(np.median(ts))}
    print("ECFP fingerprint only (parsing excluded), %d molecules (median %d heavy atoms), one thread: mean %.3f ms, median %.3f ms"
          % (len(mols), np.median(ha), ts.mean(), np.median(ts)))
    blobs = [m.ToBinary() for m in mols]
    import concurrent.futures as cf
    import multiprocessing as mp
    for w in workers:
        with cf.ProcessPoolExecutor(max_workers=w, mp_context=mp.get_context("spawn")) as ex:
            list(ex.map(_fp_worker, [blobs[:100]] * w))                       # warm-up: start the processes
            res = list(ex.map(_fp_worker, [blobs] * w))
        wall = max(r[0] for r in res)
        n = sum(r[1] for r in res)
        out["ecfp_%dproc_ms" % w] = 1000.0 * wall / n                            # effective ms per molecule over all workers together
        print("ECFP fingerprint only, %2d processes: %.0f molecules/s = %.4f ms per molecule effective" % (w, n / wall, out["ecfp_%dproc_ms" % w]))
    return out


# ------------------------------------------------------------------------------------------------ tokenising
def part_tokenise(smiles_bench, all_smiles):
    from student.tokenizer import AtomBPE
    tk = AtomBPE.train(sorted(set(all_smiles)), vocab_size=1024, min_atom_count=5, min_frequency=10)
    sub = smiles_bench
    ids = [tk.encode(s) for s in sub[:200]]
    lens = np.array([len(tk.encode(s)) for s in sub])
    t = time.perf_counter()
    for _ in range(3):
        for s in sub:
            tk.encode(s)
    one = (time.perf_counter() - t) / (3 * len(sub)) * 1000
    t = time.perf_counter()
    for _ in range(3):
        tk.encode_batch(sub)
    bat = (time.perf_counter() - t) / (3 * len(sub)) * 1000
    print("tokenising (AtomBPE, small vocab %d): %.4f ms per molecule one by one, %.4f ms per molecule in a batch call | tokens per molecule: median %d, p95 %d (indicative)"
          % (tk.vocab_size, one, bat, np.median(lens), np.percentile(lens, 95)))
    return {"tokenise_ms": one, "tokenise_batch_ms": bat, "tokens_median": float(np.median(lens)), "tokens_p95": float(np.percentile(lens, 95))}


# ------------------------------------------------------------------------------------------------ student forward
def build(torch, d, layers, heads, ff, vocab=1024, out=1024, maxlen=256):
    nn = torch.nn

    class Student(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok, self.pos = nn.Embedding(vocab, d), nn.Embedding(maxlen, d)
            layer = nn.TransformerEncoderLayer(d, heads, ff, dropout=0.0, activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, out)

        def forward(self, ids):
            x = self.tok(ids) + self.pos(torch.arange(ids.shape[1], device=ids.device))
            return self.head(self.norm(self.enc(x)).mean(1))

    return Student().eval()


def timed(fn, sync, budget):
    t = time.perf_counter(); fn(); sync(); first = time.perf_counter() - t       # warm-up
    if first > 5 * budget:
        return first
    fn(); sync()
    t = time.perf_counter(); fn(); sync(); one = time.perf_counter() - t
    reps = int(min(max(budget / max(one, 1e-7), 3), 60))
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def part_student(args):
    import torch
    shapes = {"3M": dict(d=192, layers=6, heads=4, ff=768), "10M": dict(d=320, layers=10, heads=5, ff=960)}
    rows = []
    has_gpu = torch.cuda.is_available()
    print("\ntorch %s | GPU: %s" % (torch.__version__, torch.cuda.get_device_name(0) if has_gpu else "none"))
    settings = []
    if has_gpu:
        settings += [("GPU fp16", "cuda", torch.float16, None), ("GPU fp32", "cuda", torch.float32, None)]
    settings += [("CPU fp32 %dthr" % t, "cpu", torch.float32, t) for t in args.cpu_threads]
    for sname, cfg in shapes.items():
        base = build(torch, **cfg)
        npar = sum(p.numel() for p in base.parameters())
        print("\n=== student %s: %.2f M parameters (d=%d, %d layers, %d heads, ff %d, vocab 1024, 1024-d output) ===" % (sname, npar / 1e6, cfg["d"], cfg["layers"], cfg["heads"], cfg["ff"]))
        for label, dev, dt, thr in settings:
            if thr:
                torch.set_num_threads(thr)
            m = build(torch, **cfg).to(dev).to(dt)
            sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
            batches = args.gpu_batches if dev == "cuda" else args.cpu_batches
            print("\n%s: ms per molecule%s" % (label, " (compute only | with host<->GPU transfers)" if dev == "cuda" else ""))
            print("  tokens " + "".join("%22s" % ("batch %d" % b) for b in batches))
            for L in args.lengths:
                cells = []
                for B in batches:
                    ids = torch.randint(5, 1024, (B, L))
                    with torch.inference_mode():
                        if dev == "cuda":
                            ids_g = ids.to(dev)
                            t_c = timed(lambda: m(ids_g), sync, args.budget)
                            ids_p = ids.pin_memory()
                            t_x = timed(lambda: m(ids_p.to(dev, non_blocking=True)).float().cpu(), sync, args.budget)
                            cells.append("%8.4f | %8.4f" % (t_c / B * 1000, t_x / B * 1000))
                            rows.append(dict(shape=sname, setting=label, tokens=L, batch=B, ms_per_mol=t_c / B * 1000, ms_per_mol_with_transfer=t_x / B * 1000))
                        else:
                            t_c = timed(lambda: m(ids), sync, args.budget)
                            cells.append("%8.4f" % (t_c / B * 1000))
                            rows.append(dict(shape=sname, setting=label, tokens=L, batch=B, ms_per_mol=t_c / B * 1000, ms_per_mol_with_transfer=np.nan))
                print("  %5d " % L + "".join("%22s" % c for c in cells), flush=True)
            m = None
            if dev == "cuda":
                torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default="CellPPD")
    ap.add_argument("--workers", type=int, nargs="*", default=[8, 16], help="ECFP worker processes to time")
    ap.add_argument("--lengths", type=int, nargs="*", default=[20, 40, 70, 111], help="student input lengths in tokens (pool median with the learned tokenizer ~18-20, p99 of benchmarks ~71, released tokenizer median 111)")
    ap.add_argument("--gpu-batches", type=int, nargs="*", default=[1, 16, 128, 1024])
    ap.add_argument("--cpu-batches", type=int, nargs="*", default=[1, 16, 128])
    ap.add_argument("--cpu-threads", type=int, nargs="*", default=[1, 8])
    ap.add_argument("--budget", type=float, default=1.0, help="seconds of repeats per measurement")
    ap.add_argument("--skip-student", action="store_true")
    args = ap.parse_args()
    import platform
    print("machine: %s | %d logical cores | Python %s" % (platform.processor()[:60], os.cpu_count(), platform.python_version()))
    smiles = load_smiles(args.bench)
    print("molecules (%s): %d\n" % (args.bench, len(smiles)))
    res = {}
    res.update(part_ecfp(smiles, args.workers))
    try:
        from benchmarks import MAIN, PL_SUITE, load
        allsm = []
        for b in list(MAIN) + list(PL_SUITE):
            try:
                allsm += [str(s) for s in load(b).smiles]
            except Exception:
                pass
    except Exception:
        allsm = smiles
    res.update(part_tokenise(smiles, allsm or smiles))
    rows = [] if args.skip_student else part_student(args)
    os.makedirs(os.path.join(os.path.dirname(__file__), "..", "..", "runs", "time_laptop"), exist_ok=True)
    import pandas as pd
    out = os.path.join(os.path.dirname(__file__), "..", "..", "runs", "time_laptop")
    pd.DataFrame(rows).to_csv(os.path.join(out, "student_forward.csv"), index=False)
    pd.DataFrame([res]).to_csv(os.path.join(out, "ecfp_tokenise.csv"), index=False)
    print("\nwrote", os.path.abspath(out))


if __name__ == "__main__":
    main()

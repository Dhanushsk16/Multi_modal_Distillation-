"""How fast is the student? Timing of the TRAINED models on real molecules (inference only, frozen weights).

    python src/student/time_models.py                                # paths from src/paths.py (the lab server)
    python src/student/time_models.py --student P/st6f/best.pt --tok P/atombpe_v1024 --teacher P/v3_base/best.pt --pclm P/peptideclm-2-mlm-large --out P/time_models

    python src/student/time_models.py --devices cuda --batches 1 16 64 256 --out $DATA/results/time_models_a6000      # the server's GPU only

BTP_DATA must point at a folder that holds data/benchmarks (the molecules to time are drawn from the benchmark files); --out defaults to
$BTP_DATA/results/time_models.

What is timed, per molecule, in milliseconds. Every number is the median of repeated passes over the same real molecules (a seeded mix of the
benchmark molecules), after a warm-up; the GPU is synchronised before each clock reading. Batches hold molecules of similar length, as in the
embedding scripts (export_student.py, featurize.py).

  student       RDKit normalisation + tokenising + forward pass of st6f (10.8 M parameters at inference); ids host -> device, 1024-d vector back to host.
  PeptideCLM-2  tokenising + forward pass of mlm-large (337 M parameters), mean-pooled 1024-d vector: the input model of the teacher, and the model
                the student replaces.
  teacher       the fusion network v3_base on CACHED view vectors (PepLand 300-d, PeptideCLM-2 1024-d, ECFP 2048-d; random values of the right shape,
                the cost depends on shapes only): what the teacher costs once its three views exist, as when the corpus caches are used.
  ECFP          RDKit parsing + count fingerprint (radius 2, 2048 bits), one thread.
  chain         the teacher for a NEW molecule, without caches = ECFP + PeptideCLM-2 + fusion network. PepLand (a graph network that needs DGL) is not
                run here, so the chain is a LOWER bound on the teacher's cost.

GPU runs use bfloat16 autocast (as the embedding scripts do), CPU runs fp32 with --threads threads. Plain PyTorch: no compile, no export, no
quantisation. The batch is a number of molecules (the embedding scripts cap the tokens per batch instead).
For a fair reading on a laptop: plug it in, choose the best-performance power mode, close other programs, and run it twice.
"""
from __future__ import annotations

import argparse
import csv
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.abspath(os.path.join(HERE, ".."))
sys.path[:0] = [SRC, HERE, os.path.join(SRC, "teacher")]

LOG = []


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    LOG.append(s)


def clock(run, sync, repeats):
    """Median wall time (s) of run(None) over `repeats` passes, after a warm-up (run(2) = the first two batches)."""
    run(2)
    sync()
    ts = []
    for _ in range(repeats):
        sync()
        t = time.perf_counter()
        run(None)
        sync()
        ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def sample_smiles(n, seed):
    import benchmarks as BM
    pool = []
    for b in list(BM.MAIN) + list(BM.PL_SUITE):
        try:
            pool += [str(s) for s in BM.load(b).smiles]
        except Exception as e:
            say("  benchmark %s not loaded (%s)" % (b, str(e)[:70]))
    pool = sorted(set(pool))
    assert pool, "no benchmark molecules found: set BTP_DATA to a folder that has data/benchmarks"
    idx = np.random.default_rng(seed).permutation(len(pool))[:min(n, len(pool))]
    return [pool[i] for i in idx]


def chunks(order, B):
    return [order[i:i + B] for i in range(0, len(order), B)]


def gpu_state():
    """-> (highest GPU utilisation in 5 samples over 2 s in %, memory used in MB, compute processes of OTHER programs) from nvidia-smi; None without it.
    On a shared GPU a timing taken while someone else computes is not a timing of this code: the state is printed before and after the runs."""
    import subprocess

    def q(args):
        return subprocess.run(["nvidia-smi"] + args, capture_output=True, text=True, timeout=30).stdout.strip()
    try:
        util = 0
        for _ in range(5):
            row = q(["--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits", "-i", "0"]).split("\n")[0].split(",")
            util = max(util, int(row[0]))
            time.sleep(0.5)
        apps = [l.split(",")[0].strip() for l in q(["--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"]).split("\n") if l.strip()]
        return util, int(row[1]), [p for p in apps if p != str(os.getpid())]
    except Exception:
        return None


def gpu_text(label, g):
    if g is None:
        return "%s: nvidia-smi is not available" % label
    flag = "" if (g[0] <= 10 and not g[2]) else "   <-- THE GPU IS IN USE BY SOMEONE ELSE: these timings are contaminated; repeat when it is idle"
    return "%s: utilisation (highest of 5 samples) %d %%, memory used %d MB, compute processes of other programs: %d%s" % (label, g[0], g[1], len(g[2]), flag)


def gv(g, i):
    return "" if g is None else (g[i] if i < 2 else len(g[2]))


# ------------------------------------------------------------------------------------------------ the student
def load_student(path, tok_dir):
    import torch
    import model as M
    import tokenizer as T
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = M.Student(**ck["cfg"])
    net.load_state_dict(ck["model"])
    del net.train_only                                  # the training-only view heads are not part of the deployed student
    net.eval()
    return net, T.load_tokenizer(tok_dir), M.PAD, int(ck["cfg"]["max_len"])


def student_batches(torch, tok, smiles, B, max_len, PAD):
    ids = [tok.with_special(e, max_len) for e in tok.encode_batch(smiles)]
    order = np.argsort([len(x) for x in ids], kind="stable")
    out = []
    for sel in chunks(order, B):
        x = torch.full((len(sel), max(len(ids[j]) for j in sel)), PAD, dtype=torch.long)
        for r, j in enumerate(sel):
            x[r, :len(ids[j])] = torch.tensor(ids[j])
        out.append(x)
    return out, np.array([len(x) for x in ids])


def time_student(torch, net, PAD, batches, dev, sync, repeats):
    def run(nb):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            for x in batches[:nb]:
                x = x.to(dev)
                net(x, x != PAD)["z"].float().cpu()
    return clock(run, sync, repeats)


# ------------------------------------------------------------------------------------------------ PeptideCLM-2
def pclm_batches(tok, smiles, B):
    lens = np.array([len(x) for x in tok(smiles, add_special_tokens=True)["input_ids"]])
    out = []
    for sel in chunks(np.argsort(lens, kind="stable"), B):
        b = tok([smiles[k] for k in sel], return_tensors="pt", padding=True)
        out.append((b["input_ids"], b["attention_mask"]))
    return out, lens


def time_pclm(torch, model, batches, dev, sync, repeats):
    def run(nb):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            for ids, am in batches[:nb]:
                model(input_ids=ids.to(dev), attention_mask=am.to(dev)).mean_pool.float().cpu()
    return clock(run, sync, repeats)


# ------------------------------------------------------------------------------------------------ the teacher (cached views)
def load_teacher(path):
    import torch
    import train_teacher_v3 as TV3
    ck = torch.load(path, map_location="cpu", weights_only=True)
    net = TV3.TeacherV3.from_cfg(ck["model_cfg"], 0.0)
    net.load_state_dict(ck["model"])
    net.eval()
    return net, ck["stats"], TV3


def time_teacher(torch, net, st, TV3, n, dev, B, sync, repeats):
    import views_v3 as V
    g = torch.Generator().manual_seed(0)
    raw = {v: torch.rand(n, V.DIM[v], generator=g) * 3 for v in net.inputs}        # non-negative, like counts; only the shapes matter
    st = {v: {k: t.to(dev) for k, t in st[v].items()} for v in net.inputs}
    starts = list(range(0, n, B))

    def run(nb):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
            for i in starts[:nb]:
                z = TV3.standardised({v: raw[v][i:i + B].to(dev) for v in net.inputs}, st)
                net.latent([z[v] for v in net.inputs]).float().cpu()
    return clock(run, sync, repeats)


# ------------------------------------------------------------------------------------------------ the small CPU parts
def time_ecfp(smiles, repeats):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mols = [m for m in (Chem.MolFromSmiles(s) for s in smiles) if m is not None]

    def parse_and_fp(nb):
        for s in smiles:
            m = Chem.MolFromSmiles(s)
            if m is not None:
                gen.GetCountFingerprintAsNumPy(m)

    def fp_only(nb):
        for m in mols:
            gen.GetCountFingerprintAsNumPy(m)
    nop = (lambda: None)
    return clock(parse_and_fp, nop, repeats) / len(smiles) * 1000, clock(fp_only, nop, repeats) / len(mols) * 1000


def per_mol_ms(fn, n, repeats):
    return clock(lambda nb: fn(), lambda: None, repeats) / n * 1000


def main():
    ap = argparse.ArgumentParser()
    import paths as P
    ap.add_argument("--student", default=os.path.join(P.CKPT_DIR, "student_distill", "st6f", "best.pt"))
    ap.add_argument("--tok", default=os.path.join(P.CKPT_DIR, "student_tokenizer_norm", "atombpe_v1024"))
    ap.add_argument("--teacher", default=os.path.join(P.TEACHER_DIR, "v3_base", "best.pt"))
    ap.add_argument("--pclm", default=P.PCLM_DIR)
    ap.add_argument("--out", default=os.path.join(P.RESULTS_DIR, "time_models"))
    ap.add_argument("--n", type=int, default=512, help="molecules per GPU pass")
    ap.add_argument("--n-cpu", type=int, default=64, help="molecules per CPU pass (PeptideCLM-2 is slow on a CPU)")
    ap.add_argument("--batches", type=int, nargs="*", default=[1, 16, 64], help="GPU batch sizes")
    ap.add_argument("--cpu-batches", type=int, nargs="*", default=[1, 16])
    ap.add_argument("--repeats", type=int, default=5, help="timed passes per measurement (CPU: one fewer)")
    ap.add_argument("--threads", type=int, default=8, help="CPU threads of the CPU runs")
    ap.add_argument("--devices", nargs="*", default=None, choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch
    from featurize import fix_rotary, load_pclm
    import normalise as NM
    torch.set_num_threads(a.threads)
    devices = a.devices or (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"])
    say("time_models | %s | torch %s | GPU: %s | %d logical cores, %d threads | %s" % (
        platform.processor()[:50], torch.__version__, torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none", os.cpu_count(), a.threads,
        time.strftime("%Y-%m-%d %H:%M")))

    smiles_all = sample_smiles(max(a.n, a.n_cpu), a.seed)
    say("molecules: %d drawn at random (seed %d) from the benchmark files; %d characters per SMILES on average" % (
        len(smiles_all), a.seed, np.mean([len(s) for s in smiles_all])))

    net_s, tok_s, PAD, max_len = load_student(a.student, a.tok)
    max_len = min(128, max_len)
    net_t, st_t, TV3 = load_teacher(a.teacher)
    tok_p, model_p, _ = load_pclm(a.pclm)
    model_p.to("cpu")
    p_s = sum(p.numel() for p in net_s.parameters())
    p_t = sum(p.numel() for n_, p in net_t.named_parameters() if not n_.startswith("dec."))      # decoders are training-only
    p_p = sum(p.numel() for p in model_p.parameters())
    say("parameters at inference: student %.2f M | teacher fusion network %.2f M | PeptideCLM-2 %.1f M" % (p_s / 1e6, p_t / 1e6, p_p / 1e6))

    rows = []
    for dev in devices:
        cuda = dev == "cuda"
        sync = torch.cuda.synchronize if cuda else (lambda: None)
        n = a.n if cuda else a.n_cpu
        reps = a.repeats if cuda else max(2, a.repeats - 1)
        batches = a.batches if cuda else a.cpu_batches
        sm = smiles_all[:n]
        say("")
        say("=" * 118)
        say("%s | %d molecules per pass, %d timed passes | batches %s" % (("GPU (bf16 autocast)" if cuda else "CPU (fp32, %d threads)" % a.threads), len(sm), reps, batches))

        ids_s, lens_s = student_batches(torch, tok_s, sm, 1, max_len, PAD)
        lens_p = np.array([len(x) for x in tok_p(sm, add_special_tokens=True)["input_ids"]])
        say("tokens per molecule: student median %d, p95 %d, max %d | PeptideCLM-2 median %d, p95 %d, max %d" % (
            np.median(lens_s), np.percentile(lens_s, 95), lens_s.max(), np.median(lens_p), np.percentile(lens_p, 95), lens_p.max()))

        # tokenising (CPU): one molecule at a time (batch 1) and one call for the whole list (batches)
        T = {}
        T["student", 1] = per_mol_ms(lambda: [tok_s.with_special(tok_s.encode(s), max_len) for s in sm], len(sm), reps)
        T["student", "batch"] = per_mol_ms(lambda: [tok_s.with_special(e, max_len) for e in tok_s.encode_batch(sm)], len(sm), reps)
        T["pclm", 1] = per_mol_ms(lambda: [tok_p(s) for s in sm], len(sm), reps)
        T["pclm", "batch"] = per_mol_ms(lambda: tok_p(sm), len(sm), reps)
        norm_ms = per_mol_ms(lambda: NM.normalise_many(sm, 1), len(sm), reps)
        ecfp_ms, ecfp_fp_ms = time_ecfp(sm, reps)
        say("CPU parts, ms per molecule: normalise (RDKit) %.4f | tokenise student %.4f (one by one) %.4f (batch call) | tokenise PeptideCLM-2 %.4f / %.4f | "
            "ECFP parse+fingerprint %.4f (fingerprint only %.4f)" % (norm_ms, T["student", 1], T["student", "batch"], T["pclm", 1], T["pclm", "batch"], ecfp_ms, ecfp_fp_ms))

        g0 = g1 = None
        if cuda:
            g0 = gpu_state()
            say(gpu_text("GPU before the timed runs", g0))
        fwd = {}
        peak = {}
        for who in ("student", "teacher", "pclm"):
            net = {"student": net_s, "teacher": net_t, "pclm": model_p}[who]
            net.to(dev)
            if who == "pclm":
                fix_rotary(net)                          # the rotary buffers must be rebuilt after every move (see featurize.fix_rotary)
            if cuda:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            for B in batches:
                try:
                    if who == "student":
                        bt, _ = student_batches(torch, tok_s, sm, B, max_len, PAD)
                        t = time_student(torch, net, PAD, bt, dev, sync, reps)
                    elif who == "pclm":
                        bt, _ = pclm_batches(tok_p, sm, B)
                        t = time_pclm(torch, net, bt, dev, sync, reps)
                    else:
                        t = time_teacher(torch, net, st_t, TV3, len(sm), dev, B, sync, reps)
                    fwd[who, B] = t / len(sm) * 1000
                except torch.cuda.OutOfMemoryError:
                    fwd[who, B] = float("nan")
                    say("   %s, batch %d: out of GPU memory" % (who, B))
                    torch.cuda.empty_cache()
                say("   %-8s batch %-3d forward %9.4f ms per molecule" % (who, B, fwd[who, B]))
            if cuda:
                peak[who] = torch.cuda.max_memory_allocated() / 2 ** 20
            net.to("cpu")
        if cuda:
            torch.cuda.empty_cache()
            g1 = gpu_state()
            say(gpu_text("GPU after the timed runs", g1))
            say("peak GPU memory (weights + activations, largest batch): student %.0f MB | teacher %.0f MB | PeptideCLM-2 %.0f MB" % (peak["student"], peak["teacher"], peak["pclm"]))

        say("")
        say("per molecule, in ms (student = normalise + tokenise + forward; PeptideCLM-2 = tokenise + forward; teacher cached = forward only; chain = ECFP + PeptideCLM-2 + fusion network)")
        say("%-6s | %9s | %12s | %13s | %14s | %9s | %s" % ("batch", "student", "PeptideCLM-2", "teacher cached", "teacher chain*", "ECFP", "student faster than PeptideCLM-2 | than the chain"))
        for B in batches:
            k = 1 if B == 1 else "batch"
            s_tot = norm_ms + T["student", k] + fwd["student", B]
            p_tot = T["pclm", k] + fwd["pclm", B]
            t_c = fwd["teacher", B]
            chain = ecfp_ms + p_tot + t_c
            say("%-6d | %9.3f | %12.3f | %13.3f | %14.3f | %9.3f | %6.2f ms (%5.1fx) | %6.2f ms (%5.1fx)" % (
                B, s_tot, p_tot, t_c, chain, ecfp_ms, p_tot - s_tot, p_tot / s_tot, chain - s_tot, chain / s_tot))
            rows.append(dict(device=dev, batch=B, molecules=len(sm), normalise_ms=norm_ms, student_tokenise_ms=T["student", k], student_forward_ms=fwd["student", B],
                             student_total_ms=s_tot, pclm_tokenise_ms=T["pclm", k], pclm_forward_ms=fwd["pclm", B], pclm_total_ms=p_tot, teacher_cached_forward_ms=t_c,
                             ecfp_ms=ecfp_ms, teacher_chain_ms=chain, student_mol_per_s=1000 / s_tot, pclm_mol_per_s=1000 / p_tot, chain_mol_per_s=1000 / chain,
                             speedup_vs_pclm=p_tot / s_tot, speedup_vs_chain=chain / s_tot, speedup_forward_only=fwd["pclm", B] / fwd["student", B],
                             gpu_util_before_pct=gv(g0, 0), gpu_util_after_pct=gv(g1, 0), other_gpu_processes_before=gv(g0, 2), other_gpu_processes_after=gv(g1, 2)))
        say("")
        say("the models alone (forward pass only, no RDKit, no tokenising), ms per molecule")
        say("%-6s | %9s | %12s | %13s | %s" % ("batch", "student", "PeptideCLM-2", "teacher cached", "student faster than PeptideCLM-2"))
        for B in batches:
            say("%-6d | %9.3f | %12.3f | %13.3f | %6.2f ms (%5.1fx)" % (B, fwd["student", B], fwd["pclm", B], fwd["teacher", B], fwd["pclm", B] - fwd["student", B], fwd["pclm", B] / fwd["student", B]))
        say("* the chain leaves out PepLand (it needs DGL), so it is a lower bound of the teacher's cost for a new molecule. The teacher on cached views is cheap, but its three")
        say("  input vectors must exist: PeptideCLM-2, ECFP and PepLand have to be run once per molecule, which is what the chain adds up.")

    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "results.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(a.out, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")
    say("")
    say("wrote %s" % os.path.abspath(a.out))


if __name__ == "__main__":
    main()

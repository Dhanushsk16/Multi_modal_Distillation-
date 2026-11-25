"""ESM-2 view: a protein language model reads the amino-acid SEQUENCE of a peptide.

ESM-2 knows 20 amino acids and nothing else, so only CANONICAL peptides (linear, the 20 standard L-residues, free termini:
corpus/peptide_filters.py, kind == 'canonical') get an embedding. Every other molecule is "absent" in this view
(ok = False, the row is zero); the teacher (train_teacher.py --optional-views esm) treats absence with a learned mask
token and an imputation-consistency loss.

    python src/featurize_esm.py --coverage                  # CPU, no weights needed: how much of each benchmark and of the
                                                            # pool ESM can read
    python src/featurize_esm.py --bench                     # GPU: features/single_modality/esm.npz (every benchmark molecule)
    python src/featurize_esm.py --bench --model /path/to/esm2_t33_650M_UR50D      # another ESM-2 size
    python src/featurize_esm.py --time 2000                 # embed 2000 benchmark molecules, print ms/molecule
    python src/featurize_esm.py --proteins                  # GPU: the protein partners of PepLand's binding tasks (1,227 sequences)
                                                            # -> features/pepland_suite/protein_esm.npz (read by benchmarks.py, PL_*Prot)
The pool is cached with  python src/corpus/cache_views.py --view esm.

Embedding = mean over the residue positions of ESM's last layer (after its final layer norm), special tokens excluded:
2560-d for esm2_t36_3B_UR50D (the default), 1280-d for esm2_t33_650M_UR50D. Stored as float16.
Weights: the lab server has no internet, so download them elsewhere (scripts/get_esm_laptop.py) and place the folder at
$CKPT_DIR/esm2_t36_3B_UR50D (the default; ESM_DIR in paths.py).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "corpus"))
from paths import ESM_DIR, FEAT_DIR, RESULTS_DIR  # noqa: E402

_ESM = None          # (tokenizer, model, device, model_dir), one per process


def say(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------------------------ sequences
def _seq(smi, max_res=None, caps=False, stereo=True):
    import peptide_filters
    try:
        return peptide_filters.sequence(smi, max_res or peptide_filters.MAX_RES, caps, stereo)
    except Exception:
        return None


def sequences(smiles, workers=1, skip=None, max_res=None, caps=False, stereo=True):
    """One-letter sequences of the canonical peptides among `smiles`, None elsewhere. skip: optional bool array, True =
    known not to be canonical (e.g. from the pool's `kind` column), not parsed. max_res: longest chain (default 50, the
    corpus window). caps: ignore an N-terminal acetyl and a C-terminal primary amide."""
    import functools
    smiles = [str(s) for s in smiles]
    todo = [i for i in range(len(smiles)) if skip is None or not skip[i]]
    out = [None] * len(smiles)
    fn = functools.partial(_seq, max_res=max_res, caps=caps, stereo=stereo)
    if workers > 1 and len(todo) > 2000:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(workers) as pool:
            res = pool.map(fn, [smiles[i] for i in todo], chunksize=256)
    else:
        res = [fn(smiles[i]) for i in todo]
    for i, r in zip(todo, res):
        out[i] = r
    return out


# ------------------------------------------------------------------------------------ the model
def load_esm(model_dir=None, device=None):
    import torch
    from transformers import AutoTokenizer, EsmModel
    model_dir = model_dir or ESM_DIR
    if not os.path.isdir(model_dir):
        raise SystemExit("no ESM-2 weights in %s: download them (scripts/get_esm_laptop.py) and place the folder there" % model_dir)
    dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t = time.time()
    say("esm: loading %s on %s (the 3B .bin shards are 11 GB: reading them takes minutes) ..." % (model_dir, dev))
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = EsmModel.from_pretrained(model_dir, add_pooling_layer=False).eval().to(dev)
    if dev.type == "cuda":
        model = model.half()
    p = next(model.parameters())
    say("esm: loaded in %.0f s | weights on %s, %s" % (time.time() - t, p.device, str(p.dtype).replace("torch.", "")))
    return tok, model, dev, model_dir


def esm_dim(model_dir=None):
    import json
    with open(os.path.join(model_dir or ESM_DIR, "config.json")) as fh:
        return int(json.load(fh)["hidden_size"])


def encode(tok, model, dev, seqs, max_tokens=16384):
    """Mean-pooled last-layer embeddings of the sequences (a list of str) -> float32 [n, d]. Sorted by length and batched
    by a token budget, so the many short peptides cost almost nothing."""
    import torch
    n = len(seqs)
    out = np.zeros((n, model.config.hidden_size), np.float32)
    order = sorted(range(n), key=lambda i: len(seqs[i]))
    i = 0
    while i < n:
        j, longest = i, 0
        while j < n and max(longest, len(seqs[order[j]]) + 2) * (j - i + 1) <= max_tokens:
            longest = max(longest, len(seqs[order[j]]) + 2)
            j += 1
        j = max(j, i + 1)
        idx = order[i:j]
        enc = tok([seqs[k] for k in idx], return_tensors="pt", padding=True, truncation=True, max_length=1022)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            h = model(**enc).last_hidden_state.float()
        am = enc["attention_mask"].clone()
        am[:, 0] = 0                                                  # <cls>
        lens = enc["attention_mask"].sum(1)
        am[torch.arange(len(idx), device=am.device), lens - 1] = 0    # <eos>
        pooled = (h * am.unsqueeze(-1)).sum(1) / am.sum(1, keepdim=True).clamp(min=1)
        out[idx] = pooled.cpu().numpy()
        i = j
    return out


def embed(smiles, workers=4, model_dir=None, skip=None, max_tokens=16384, caps=False):
    """-> X float16 [n, d] (zero where absent), ok bool [n] (True = a canonical peptide that was embedded; with caps=True a
    peptide with an N-terminal acetyl and / or a C-terminal amide counts too and is embedded without them)."""
    global _ESM
    seqs = sequences(smiles, workers, skip, caps=caps)
    ok = np.array([s is not None for s in seqs])
    d = esm_dim(model_dir)
    X = np.zeros((len(seqs), d), np.float32)
    if ok.any():
        if _ESM is None or (model_dir and _ESM[3] != model_dir):
            _ESM = load_esm(model_dir)
        idx = np.flatnonzero(ok)
        X[idx] = encode(*_ESM[:3], [seqs[i] for i in idx], max_tokens)
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
    good = ok & np.isfinite(X).all(1) & (np.abs(X).max(1) < 6e4)
    X[~good] = 0
    return X.astype(np.float16), good


# ------------------------------------------------------------------------------------ the benchmarks
def bench_smiles(feat_dir=None):
    """The benchmark molecules in the order of every features/single_modality/*.npz (or of another feature folder, e.g.
    features/pepland_suite)."""
    p = os.path.join(feat_dir or FEAT_DIR, "ecfp.npz")
    return [str(s) for s in np.load(p, allow_pickle=True)["smiles"]]


def coverage(workers, names=None):
    """names: benchmarks to check (default: the 5 main ones plus PepLand's 5, PL_*, three of which are canonical)."""
    import pandas as pd
    import benchmarks as BM
    import peptide_filters as PF
    if not names:
        names = list(BM.selected()) + [n for n in BM.PL_SUITE if n not in BM.selected()]
    rows = []
    say("ESM can read a molecule when it is a linear chain of the 20 standard L-residues: STRICT = also free termini (the pool's")
    say("'canonical'); +CAPS = an N-acetyl and / or C-amide is ignored; +LONG = chains of more than 50 residues (the corpus window);")
    say("+STEREO = readable only if stereochemistry is ignored (D-residues, or SMILES written without stereo flags).")
    say("%-12s %-6s %9s | %6s %6s %6s %6s | %6s %6s %6s %7s | %7s" % ("benchmark", "part", "molecules", "strict", "+caps", "+long",
                                                                    "+stereo", "strict%", "+caps%", "+long%", "+stereo%", "unread%"))
    for name in names:
        try:
            df = BM.load(name).dropna(subset=["smiles"])
        except Exception as e:                      # a benchmark whose files are not on this machine: say so, go on
            say("%-12s skipped (%s: %s)" % (name, type(e).__name__, str(e)[:80]))
            continue
        smi = df.smiles.tolist()
        strict = np.array([s is not None for s in sequences(smi, workers)])
        capsok = np.array([s is not None for s in sequences(smi, workers, caps=True)])
        anyok = np.array([s is not None for s in sequences(smi, workers, max_res=2000, caps=True)])
        flatok = np.array([s is not None for s in sequences(smi, workers, max_res=2000, caps=True, stereo=False)])
        parts = [("all", np.ones(len(df), bool))]
        if not pd.api.types.is_integer_dtype(df.split):      # fixed train / test split (strings: object or pandas-3 str dtype)
            parts += [(str(s), (df.split == s).to_numpy()) for s in sorted(df.split.unique())]
        for part, m in parts:
            n = int(m.sum())
            a = int(strict[m].sum())
            b = int(capsok[m].sum()) - a
            c = int(anyok[m].sum()) - a - b
            d = int(flatok[m].sum()) - a - b - c
            rows.append({"benchmark": name, "part": part, "molecules": n, "strict": a, "caps_only": b, "long_only": c,
                         "stereo_only": d, "unread": n - a - b - c - d})
            say("%-12s %-6s %9d | %6d %6d %6d %6d | %5.1f%% %5.1f%% %5.1f%% %6.1f%% | %6.1f%%"
                % (name, part, n, a, b, c, d, 100 * a / n, 100 * b / n, 100 * c / n, 100 * d / n, 100 * (n - a - b - c - d) / n))
    out = pd.DataFrame(rows)
    # the pool
    from paths import CORPUS_DIR
    pool_path = os.path.join(CORPUS_DIR, "pool.parquet")
    if os.path.exists(pool_path):
        pool = pd.read_parquet(pool_path, columns=["smiles", "kind", "residues", "source"])
        say("\npool: %s molecules" % format(len(pool), ","))
        for k, g in pool.groupby("kind"):
            say("  %-13s %10s  %5.1f%%" % (k, format(len(g), ","), 100 * len(g) / len(pool)))
        for src, g in pool.groupby("source"):
            say("  source %-22s canonical %5.1f%% of %s" % (src, 100 * (g.kind == "canonical").mean(), format(len(g), ",")))
        # does the sequence parser agree with the pool's own classification? (a sample)
        rng = np.random.default_rng(0)
        for kind in ("canonical", "noncanonical"):
            g = pool[pool.kind == kind]
            smp = g.smiles.iloc[rng.choice(len(g), min(1500, len(g)), replace=False)].tolist()
            got = np.array([s is not None for s in sequences(smp, workers)])
            say("  parser check on %d pool molecules labelled %-12s: sequence found for %.1f%% (expected %s)"
                % (len(smp), kind, 100 * got.mean(), "100%" if kind == "canonical" else "0%"))
    else:
        say("\n(no pool at %s: pool part skipped)" % pool_path)
    d = os.path.join(RESULTS_DIR, "esm_coverage")
    os.makedirs(d, exist_ok=True)
    out.to_csv(os.path.join(d, "coverage.csv"), index=False)
    say("wrote", os.path.join(d, "coverage.csv"))


def timing(n, workers, model_dir, max_tokens, caps):
    """--time N: the real workload, timed by stage. Draws a random sample of the pool (3N molecules), parses it to sequences
    (strict and with caps: the share readable per source), loads the model (timed on its own), embeds N readable sequences
    and extrapolates the encoding time to the whole pool."""
    import pandas as pd
    import torch
    from paths import CORPUS_DIR
    say("torch %s | cuda available: %s" % (torch.__version__, torch.cuda.is_available()))
    if torch.cuda.is_available():
        say("GPU: %s | CUDA_VISIBLE_DEVICES=%s" % (torch.cuda.get_device_name(0), os.environ.get("CUDA_VISIBLE_DEVICES", "(unset)")))
    rng = np.random.default_rng(0)
    pool_path = os.path.join(CORPUS_DIR, "pool.parquet")
    src, npool = None, None
    if os.path.exists(pool_path):
        pool = pd.read_parquet(pool_path, columns=["smiles", "source"])
        npool = len(pool)
        pick = rng.choice(npool, min(npool, 3 * n), replace=False)
        smi = pool.smiles.iloc[pick].astype(str).tolist()
        src = pool.source.iloc[pick].astype(str).to_numpy()
        say("sample: %d random molecules of the pool (%s)" % (len(smi), format(npool, ",")))
    else:
        allsmi = bench_smiles()
        pick = rng.choice(len(allsmi), min(len(allsmi), 3 * n), replace=False)
        smi = [allsmi[i] for i in pick]
        say("no pool at %s: sample of %d random benchmark molecules instead" % (pool_path, len(smi)))
    t = time.time()
    strict = sequences(smi, workers)
    t_strict = time.time() - t
    t = time.time()
    capped = sequences(smi, workers, caps=True)
    t_caps = time.time() - t
    rs = np.array([s is not None for s in strict])
    rc = np.array([s is not None for s in capped])
    say("parsing: %.1f ms/molecule strict, %.1f with caps (%d processes, wall clock) | readable: strict %.1f%%, with caps %.1f%%"
        % (1000 * t_strict / len(smi), 1000 * t_caps / len(smi), workers, 100 * rs.mean(), 100 * rc.mean()))
    if src is not None:
        for s in sorted(set(src)):
            m = src == s
            say("   %-22s strict %5.1f%% | with caps %5.1f%% | (%d in the sample)" % (s, 100 * rs[m].mean(), 100 * rc[m].mean(), m.sum()))
    seqs = [s for s in (capped if caps else strict) if s is not None][:n]
    if not seqs:
        return say("nothing readable in the sample")
    lens = np.array([len(s) for s in seqs])
    say("embedding %d sequences (%s) | length mean %.1f, max %d" % (len(seqs), "caps ignored" if caps else "strict", lens.mean(), lens.max()))
    t = time.time()
    tok, model, dev, _ = load_esm(model_dir)
    t_load = time.time() - t
    encode(tok, model, dev, seqs[:64], max_tokens)                    # warm-up (CUDA kernels)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    encode(tok, model, dev, seqs, max_tokens)
    if dev.type == "cuda":
        torch.cuda.synchronize()
    sec = time.time() - t
    ms = 1000 * sec / len(seqs)
    say("encoding: %d sequences in %.1f s = %.2f ms/sequence on %s | model load %.0f s (once per run, not per molecule)"
        % (len(seqs), sec, ms, dev, t_load))
    if dev.type == "cuda":
        say("GPU memory: %.1f GB peak in this process" % (torch.cuda.max_memory_allocated() / 1e9))
    if npool:
        share = rc.mean() if caps else rs.mean()
        say("pool estimate: %s readable sequences x %.2f ms = %.1f h of encoding (parsing not included; cache_views parses on %d processes)"
            % (format(int(share * npool), ","), ms, share * npool * ms / 3.6e6, workers))


# ------------------------------------------------------------------------------------ the protein partners of the binding tasks
PROTEIN_FILES = ("c-binding.csv", "nc-binding.csv")          # data/benchmarks/pepland: every row is a protein-peptide pair, the protein is column seq1


def protein_sequences():
    """The distinct protein sequences (column seq1) of PepLand's two binding files, sorted (a fixed order)."""
    import pandas as pd
    from paths import BENCH_DIR
    seqs = []
    for f in PROTEIN_FILES:
        d = pd.read_csv(os.path.join(BENCH_DIR, "pepland", f), usecols=["seq1"], low_memory=False)
        seqs += d.seq1.dropna().astype(str).str.strip().tolist()
    return sorted(set(seqs))


def protein_file():
    return os.path.join(os.path.dirname(FEAT_DIR), "pepland_suite", "protein_esm.npz")


def embed_proteins(model_dir=None, max_tokens=8192, out=None):
    """ESM-2 (3B by default) embedding of every protein partner of the binding tasks -> features/pepland_suite/protein_esm.npz
    (seqs [n] str, X [n, d] float16, ok [n] bool). Same embedding as every other ESM-2 vector of this project: the mean over the residue
    positions of the last layer, special tokens excluded. ESM-2 reads at most 1022 residues: a longer protein is cut at 1022 (one protein
    of the two files, two rows)."""
    seqs = protein_sequences()
    lens = np.array([len(s) for s in seqs])
    say("proteins: %d distinct sequences | length min %d, median %d, max %d | longer than 1022 (cut): %d | characters other than the "
        "20 amino acids: %d sequences (X = unknown residue, ESM-2 has a token for it)"
        % (len(seqs), lens.min(), np.median(lens), lens.max(), int((lens > 1022).sum()),
           sum(1 for s in seqs if set(s) - set("ACDEFGHIKLMNPQRSTVWY"))))
    t = time.time()
    tok, model, dev, _ = load_esm(model_dir)
    t_load = time.time() - t
    t = time.time()
    X = encode(tok, model, dev, seqs, max_tokens)
    ok = np.isfinite(X).all(1) & (np.abs(X).max(1) < 6e4)
    X[~ok] = 0
    out = out or protein_file()                     # --protein-file: another name for another ESM-2 size (e.g. protein_esm_t12.npz)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out + ".tmp.npz", seqs=np.array(seqs), X=X.astype(np.float16), ok=ok)
    os.replace(out + ".tmp.npz", out)
    say("esm: %d proteins, %d-d, usable %d | model load %.0f s, encoding %.0f s | per-dimension sd of the vectors: mean %.3f -> %s"
        % (len(seqs), X.shape[1], int(ok.sum()), t_load, time.time() - t, float(X[ok].std(0).mean()), out))
    try:
        import torch
        if dev.type == "cuda":
            say("GPU memory: %.1f GB peak in this process" % (torch.cuda.max_memory_allocated() / 1e9))
    except Exception:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proteins", action="store_true",
                    help="GPU: embed the protein partners of PepLand's binding tasks (column seq1 of c-binding.csv / nc-binding.csv) "
                         "-> features/pepland_suite/protein_esm.npz")
    ap.add_argument("--coverage", action="store_true", help="CPU only: share of canonical peptides in the benchmarks and the pool")
    ap.add_argument("--benchmarks", nargs="*", default=None,
                    help="with --coverage: the benchmarks to check (default: the 5 main + the 5 PepLand-suite ones, PL_*)")
    ap.add_argument("--bench", action="store_true", help="embed every benchmark molecule -> features/single_modality/esm.npz")
    ap.add_argument("--feat", default=None, metavar="SUBFOLDER",
                    help="with --bench: another feature folder under features/ (pepland_suite: PepLand's five tasks); it must already "
                         "hold ecfp.npz, which fixes the molecule order. Use the same --ignore-caps as for the pool cache")
    ap.add_argument("--time", type=int, default=0, help="embed this many benchmark molecules, print the speed, write nothing")
    ap.add_argument("--model", default=None, help="ESM-2 folder (default: ESM_DIR in paths.py)")
    ap.add_argument("--protein-file", default=None, help="with --proteins: the output .npz (default features/pepland_suite/protein_esm.npz); "
                    "the PepLand reproduction uses --model .../esm2_t12_35M_UR50D --protein-file .../protein_esm_t12.npz")
    ap.add_argument("--workers", type=int, default=8, help="processes for the sequence parsing (lab rule: at most 32)")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--ignore-caps", action="store_true",
                    help="embed peptides that only differ from canonical by an N-acetyl / C-amide too (without the caps)")
    a = ap.parse_args()
    assert a.workers <= 32, "lab rule: at most half of the 64 cores"
    if a.proteins:
        return embed_proteins(a.model, min(a.max_tokens, 8192), a.protein_file)
    if a.coverage:
        return coverage(a.workers, a.benchmarks)
    if a.time:
        return timing(a.time, a.workers, a.model, a.max_tokens, a.ignore_caps)
    feat_dir = os.path.join(os.path.dirname(FEAT_DIR), a.feat) if a.feat else FEAT_DIR
    if a.feat:
        assert a.bench, "--feat goes with --bench"
        assert os.path.exists(os.path.join(feat_dir, "ecfp.npz")), "%s/ecfp.npz is missing (bash run_cross_eval.sh makes the PepLand-suite features)" % feat_dir
    smi = bench_smiles(feat_dir)
    if a.bench:
        t = time.time()
        X, ok = embed(smi, a.workers, a.model, max_tokens=a.max_tokens, caps=a.ignore_caps)
        out = os.path.join(feat_dir, "esm.npz")
        np.savez_compressed(out + ".tmp.npz", smiles=np.array(smi), X=X, ok=ok)
        os.replace(out + ".tmp.npz", out)
        say("esm: %d benchmark molecules, %d-d, canonical (embedded) %d = %.1f%% | %.0f s -> %s"
            % (len(smi), X.shape[1], int(ok.sum()), 100 * ok.mean(), time.time() - t, out))
        return
    ap.print_help()


if __name__ == "__main__":
    main()

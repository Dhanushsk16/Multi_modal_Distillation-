"""Cache one frozen view of every pool molecule, shard by shard (resumable).

    python src/corpus/cache_views.py --export                     # once, env peptide: text copy of the pool
    python src/corpus/cache_views.py --view pclm                  # GPU, env peptide
    python src/corpus/cache_views.py --view pclm_mtr              # GPU, env peptide: PeptideCLM-2 MTR (same code, same tokenizer, other weights)
    python src/corpus/cache_views.py --view pepland --threads 1 --part 0 --parts 27   # CPU, env pepland
    python src/corpus/cache_views.py --view pepland --threads 1 --time 300   # check + timing, writes nothing
    python src/corpus/cache_views.py --view pclm --check          # only the check against the benchmark features
    python src/corpus/cache_views.py --view rdkit --workers 24    # CPU, env peptide: RDKit descriptors (src/rdkit_desc.py)
    python src/corpus/cache_views.py --view esm --workers 16      # GPU + CPU, env peptide: ESM-2 (src/featurize_esm.py);
                                                                  # ok = False for every non-canonical peptide (ESM cannot read it)
    python src/corpus/cache_views.py --status                     # progress per view

scripts/run_cache.sh starts PeptideCLM-2 and PepLand together.

The pool is read from $CORPUS_DIR/pool_smiles/ (plain text, one SMILES per line, written from
pool.parquet by --export) because the pepland env has no pyarrow. Output: $DATA/features/corpus/
<view>/shard_NNN.npz holding X (ecfp: uint8 counts, rdkit: float32 with NaN for a failed descriptor,
others float16), ok (False = the view failed
for that molecule, its row of X is zeros) and the pool rows lo:hi it covers. A finished shard is
skipped, so a killed job resumes; PepLand also keeps part_NNN.npz inside the shard it is working
on (written every 2 min and on Ctrl+C / SIGTERM), because one shard takes about an hour there.

Each model is loaded once per process. Before a real run the cache code embeds ~96 benchmark
molecules and compares them with features/single_modality/<view>.npz (cosine): the teacher is
trained on this cache and applied to those benchmark features, so both must be the same
function. A failed check stops the run before anything is written (--no-check skips it).
Feature code: featurize.py / featurize_pepland.py.

PepLand needs several processes to use the CPU: start P of them with --part 0..P-1 --parts P
(each takes the shards with shard % P == part) and keep P x --threads <= 32.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import CORPUS_DIR, DATA, EXTERNAL, FEAT_DIR, PCLM_DIR, PCLM_MTR_DIR, PEPLAND_MODEL  # noqa: E402

VIEWS = ("ecfp", "pclm", "pclm_mtr", "pepland", "rdkit", "esm")
CACHE = os.path.join(DATA, "features", "corpus")
POOL = os.path.join(CORPUS_DIR, "pool.parquet")
TXT = os.path.join(CORPUS_DIR, "pool_smiles")
_EXT = None                  # PepLand extractor, one per process
_PCLM = {}                   # model folder -> (tokenizer, model, device); a process runs one view, so it holds one model
_KINDS = None                # the pool's `kind` column (canonical / noncanonical / not_peptide), for the esm view


def say(*a):
    print(*a, flush=True)


def shard_path(view, s):
    return os.path.join(CACHE, view, "shard_%03d.npz" % s)


def part_path(view, s):
    return os.path.join(CACHE, view, "part_%03d.npz" % s)


# ------------------------------------------------------------------------------- the pool
def export_text():
    """Text copy of the pool: pool_smiles/shard_NNN.txt (one SMILES per line) + index.json."""
    import pandas as pd
    pool = pd.read_parquet(POOL, columns=["smiles", "shard"])
    sh = pool["shard"].to_numpy()
    assert (np.diff(sh) >= 0).all(), "pool rows are not ordered by shard"
    smi = pool["smiles"].tolist()
    assert not any(("\n" in x or "\r" in x) for x in smi), "a SMILES contains a line break"
    os.makedirs(TXT, exist_ok=True)
    shards = {}
    for s in sorted(set(sh.tolist())):
        lo, hi = int(np.searchsorted(sh, s, "left")), int(np.searchsorted(sh, s, "right"))
        path = os.path.join(TXT, "shard_%03d.txt" % s)
        with open(path + ".tmp", "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(smi[lo:hi]) + "\n")
        os.replace(path + ".tmp", path)
        shards[str(s)] = [lo, hi]
    idx = {"pool_bytes": os.path.getsize(POOL), "rows": len(smi), "shards": shards}
    with open(os.path.join(TXT, "index.json.tmp"), "w") as fh:
        json.dump(idx, fh)
    os.replace(os.path.join(TXT, "index.json.tmp"), os.path.join(TXT, "index.json"))
    say("pool_smiles: %s molecules in %d shards -> %s" % (format(len(smi), ","), len(shards), TXT))


def pool_index():
    """-> ({shard: (lo, hi)}, rows). Makes the text copy first when it is missing (needs pyarrow)."""
    path = os.path.join(TXT, "index.json")
    if not os.path.exists(path):
        try:
            export_text()
        except ImportError:
            sys.exit("%s does not exist yet and this env cannot read parquet.\nRun once in env peptide:\n"
                     "  python %s --export" % (TXT, os.path.abspath(__file__)))
    with open(path) as fh:
        idx = json.load(fh)
    if os.path.exists(POOL) and os.path.getsize(POOL) != idx["pool_bytes"]:
        sys.exit("pool.parquet changed since %s was written: delete that folder and run --export" % TXT)
    return {int(k): tuple(v) for k, v in idx["shards"].items()}, idx["rows"]


def shard_smiles(s, lo, hi):
    with open(os.path.join(TXT, "shard_%03d.txt" % s), encoding="utf-8") as fh:
        rows = fh.read().split("\n")
    assert rows[-1] == "" and len(rows) - 1 == hi - lo, "pool_smiles/shard_%03d.txt is damaged" % s
    return rows[:-1]


def first_smiles(index, k):
    out = []
    for s in sorted(index):
        out += shard_smiles(s, *index[s])
        if len(out) >= k:
            break
    return out[:k]


# ----------------------------------------------------------------------------- the views
def _ecfp_u8(smi):
    import featurize
    r = featurize._ecfp(smi)
    if r is None:
        return None
    return np.minimum(r, 255).astype(np.uint8), bool((r > 255).any())


def ecfp(smiles, workers):
    from multiprocessing import Pool
    with Pool(workers) as p:
        rows = p.map(_ecfp_u8, smiles, chunksize=256)
    X = np.zeros((len(smiles), 2048), np.uint8)
    ok = np.zeros(len(smiles), bool)
    clipped = 0
    for i, r in enumerate(rows):
        if r is not None:
            X[i], ok[i] = r[0], True
            clipped += r[1]
    if clipped:
        say("  note: %d molecules had an ECFP count above 255 (clipped)" % clipped)
    return X, ok


RDKIT_CHUNK = 16      # molecules per task
RDKIT_STALL = 90      # s without any finished task while tasks are outstanding: those tasks hold a stuck molecule
RDKIT_SOLO = 30       # s one molecule of a stuck task may take when it is retried alone; after that it counts as failed


def _rdkit_row(smi):
    import rdkit_desc
    try:
        return rdkit_desc.describe(smi)
    except Exception:
        return None


def _rdkit_chunk(job):
    return job[0], [_rdkit_row(s) for s in job[1]]


def _rdkit_worker_init():
    """main() installs a SIGTERM handler that Python only runs between bytecodes: a worker stuck inside RDKit's C++
    code would never die on Pool.terminate() (tested on Linux). Default action again; Ctrl-C is left to the parent."""
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _rdkit_child(conn, smi):
    conn.send(_rdkit_row(smi))


def _rdkit_alone(smi, limit):
    """One molecule in its own process, killed after `limit` s (RDKit's C++ code cannot be interrupted from Python,
    and a few pool molecules take over an hour in it). -> descriptor row, or None when failed or too slow."""
    import multiprocessing as mp
    a, b = mp.Pipe(duplex=False)
    pr = mp.Process(target=_rdkit_child, args=(b, smi), daemon=True)
    pr.start()
    b.close()
    row = None
    if a.poll(limit):
        try:
            row = a.recv()
        except EOFError:
            row = None
    if pr.is_alive():
        pr.kill()
    pr.join()
    a.close()
    return row


def rdkit(smiles, workers):
    """RDKit descriptors (src/rdkit_desc.py): float32, NaN where one descriptor failed, ok=False when the
    molecule does not parse, most descriptors failed, or it takes more than RDKIT_SOLO s (see below).
    Pool.map would wait for the slowest molecule: in the real pool one molecule held a shard up for 84 min while
    23 of 24 workers idled. So the tasks are collected as they finish; when nothing finishes for RDKIT_STALL s the
    pool is killed, and the molecules of the unfinished tasks are redone one by one with a time limit each."""
    from multiprocessing import Pool, TimeoutError as PoolTimeout
    import rdkit_desc
    n = len(smiles)
    X = np.zeros((n, rdkit_desc.n_desc()), np.float32)
    ok = np.zeros(n, bool)

    def put(i0, rows):
        for k, r in enumerate(rows):
            if r is not None:
                X[i0 + k], ok[i0 + k] = r, True

    starts = list(range(0, n, RDKIT_CHUNK))
    done = set()
    p = Pool(workers, initializer=_rdkit_worker_init)
    try:
        it = p.imap_unordered(_rdkit_chunk, [(i0, smiles[i0:i0 + RDKIT_CHUNK]) for i0 in starts])
        while len(done) < len(starts):
            try:
                i0, rows = it.next(timeout=RDKIT_STALL)
            except PoolTimeout:
                break
            put(i0, rows)
            done.add(i0)
    finally:
        p.terminate()
        p.join()
    for i0 in (s for s in starts if s not in done):
        for k, smi in enumerate(smiles[i0:i0 + RDKIT_CHUNK]):
            t = time.time()
            r = _rdkit_alone(smi, RDKIT_SOLO)
            if r is not None:
                X[i0 + k], ok[i0 + k] = r, True
            elif time.time() - t >= RDKIT_SOLO:
                say("  too slow (> %d s), counted as failed: molecule %d of this shard, %d characters: %s"
                    % (RDKIT_SOLO, i0 + k, len(smi), smi[:120]))
    return X, ok


def pclm(smiles, max_tokens, model_dir=PCLM_DIR):
    import featurize
    if model_dir not in _PCLM:
        _PCLM[model_dir] = featurize.load_pclm(model_dir)
    X, ok = featurize.pclm_encode(*_PCLM[model_dir], smiles, max_tokens=max_tokens)
    ok = ok & (np.abs(X).max(axis=1) < 6e4)
    X[~ok] = 0
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
    return X.astype(np.float16), ok


def esm(smiles, workers, model_dir=None, lohi=None, caps=False):
    """ESM-2 embedding of the canonical peptides; ok = False (row of zeros) for every other molecule, which ESM cannot read.
    lohi = (lo, hi) pool rows of `smiles`: the pool's own `kind` column then spares parsing the non-canonical 24%."""
    global _KINDS
    import featurize_esm
    skip = None
    if lohi is not None and os.path.exists(POOL) and not caps:       # with caps ignored a "non-canonical" row may still be readable
        if _KINDS is None:
            import pandas as pd
            _KINDS = pd.read_parquet(POOL, columns=["kind"])["kind"].to_numpy()
        skip = _KINDS[lohi[0]:lohi[1]] != "canonical"
        assert len(skip) == len(smiles)
    return featurize_esm.embed(smiles, workers, model_dir, skip, caps=caps)


def _save_part(path, X, ok, n):
    tmp = path + ".tmp.npz"
    np.savez(tmp, X=X[:n].astype(np.float16), ok=ok[:n], n=n)
    os.replace(tmp, path)


def pepland(smiles, threads, batch, part=None, label=""):
    """part = file that keeps the finished prefix of this shard (resume after a kill)."""
    global _EXT
    import torch
    if _EXT is None:
        torch.set_num_threads(threads)
        sys.path.insert(0, EXTERNAL)
        from pepland.model.core import PepLandFeatureExtractor
        _EXT = PepLandFeatureExtractor(model_path=PEPLAND_MODEL, pooling="avg", padding_mode="exclude")
        _EXT.eval()
    n = len(smiles)
    X = np.zeros((n, 300), np.float32)
    ok = np.zeros(n, bool)
    i = 0
    if part and os.path.exists(part):
        z = np.load(part)
        i = int(z["n"])
        X[:i], ok[:i] = z["X"][:i], z["ok"][:i]
        say("  %s resuming at molecule %s of %s" % (label, format(i, ","), format(n, ",")))
    start = i
    t0 = t_save = t_say = time.time()
    try:
        with torch.no_grad():
            while i < n:
                chunk = smiles[i:i + batch]
                try:
                    X[i:i + len(chunk)] = _EXT(chunk).float().cpu().numpy()
                    ok[i:i + len(chunk)] = True
                except Exception:
                    for j, s in enumerate(chunk):
                        try:
                            X[i + j] = _EXT([s]).float().cpu().numpy()[0]
                            ok[i + j] = True
                        except Exception:
                            pass
                i += len(chunk)
                now = time.time()
                if part and now - t_save > 120:
                    _save_part(part, X, ok, i)
                    t_save = now
                if part and now - t_say > 120:
                    per = (now - t0) / (i - start)
                    say("  %s %s/%s molecules | %.0f ms/molecule | about %.0f min to go"
                        % (label, format(i, ","), format(n, ","), 1000 * per, per * (n - i) / 60))
                    t_say = now
    except KeyboardInterrupt:
        if part:
            _save_part(part, X, ok, i)                   # i = first molecule not finished
            say("  %s interrupted at molecule %s, partial file kept" % (label, format(i, ",")))
        raise
    ok &= np.isfinite(X).all(axis=1)
    X[~ok] = 0
    return X.astype(np.float16), ok


def compute(a, smiles, part=None, label="", lohi=None):
    if a.view == "ecfp":
        return ecfp(smiles, a.workers)
    if a.view == "pclm":
        return pclm(smiles, a.max_tokens)
    if a.view == "pclm_mtr":
        return pclm(smiles, a.max_tokens, PCLM_MTR_DIR)
    if a.view == "rdkit":
        return rdkit(smiles, a.workers)
    if a.view == "esm":
        return esm(smiles, a.workers, a.esm_model, lohi, a.esm_caps)
    return pepland(smiles, a.threads, a.batch, part, label)


# ------------------------------------------------------------------ check, status, timing
def check(a, n=96):
    """Embed n benchmark molecules with the cache code and compare with the benchmark features."""
    if a.view == "ecfp":
        return True
    path = os.path.join(FEAT_DIR, a.view + ".npz")
    if not os.path.exists(path):
        say("check: %s not found, skipped" % path)
        return True
    z = np.load(path, allow_pickle=True)
    if a.view == "rdkit":
        import rdkit_desc
        if "names" not in z.files or list(z["names"]) != rdkit_desc.names():
            say("check: %s was made with another descriptor list than this RDKit gives (%d vs %d columns): FAIL"
                % (path, z["X"].shape[1], rdkit_desc.n_desc()))
            return False
    smi, ref, good = z["smiles"], z["X"], z["ok"]
    short = np.array([len(x) <= 600 for x in smi])
    cand = np.flatnonzero(good & short)
    pick = cand[np.linspace(0, len(cand) - 1, min(n, len(cand))).astype(int)]
    t = time.time()
    X, ok = compute(a, [str(x) for x in smi[pick]])
    X, R = np.nan_to_num(X.astype(np.float32)), np.nan_to_num(ref[pick].astype(np.float32))
    cos = (X * R).sum(1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(R, axis=1) + 1e-12)
    cos = np.where(ok, cos, 0.0)
    # same model, different batches / bf16: cosine >= 0.9999 (measured); a bad rotary buffer gives ~0.98 or less
    passed = bool(np.median(cos) >= 0.999 and (cos >= 0.99).mean() >= 0.97)
    say("check: %s cache code vs the benchmark features on %d molecules | cosine median %.5f, min %.5f | "
        "failed %d | %.0f s | %s" % (a.view, len(pick), np.median(cos), cos.min(), int((~ok).sum()),
                                     time.time() - t, "PASS" if passed else "FAIL"))
    if not passed and a.view in ("pclm", "pclm_mtr"):
        say("  The cache code rebuilds and verifies the rotary buffers (featurize.fix_rotary). If %s was made\n"
            "  without that, the stored features are the wrong ones: move it away, run\n"
            "    python src/featurize.py --rep %s      (GPU, a few minutes)\n"
            "  and start this again." % (path, a.view))
    return passed


def status():
    with open(POOL.replace(".parquet", ".json")) as fh:
        meta = json.load(fh)
    n, rows = meta["shards"], meta["rows"]
    for v in VIEWS:
        d = os.path.join(CACHE, v)
        names = os.listdir(d) if os.path.isdir(d) else []
        done = [f for f in names if f.startswith("shard_") and f.endswith(".npz") and ".tmp" not in f]
        parts = [f for f in names if f.startswith("part_") and f.endswith(".npz") and ".tmp" not in f]
        mol = 0
        for f in done:
            z = np.load(os.path.join(d, f))
            mol += int(z["hi"]) - int(z["lo"])
        for f in parts:
            try:
                mol += int(np.load(os.path.join(d, f))["n"])
            except Exception:                    # a partial file being replaced right now
                pass
        say("%-8s %3d / %d shards finished%s | %s of %s molecules (%.0f%%)"
            % (v, len(done), n, (" + %d in progress" % len(parts)) if parts else "",
               format(mol, ","), format(rows, ","), 100.0 * mol / rows))


def timing(a, index, rows):
    smi = first_smiles(index, a.time)
    if not check(a):
        say("(the check failed: see above)")
    compute(a, smi[:64])                         # warm-up: model load, CUDA start, first-call costs
    t = time.time()
    X, ok = compute(a, smi)
    sec = time.time() - t
    ms = 1000 * sec / len(smi)
    say("%s: %d molecules in %.1f s = %.2f ms/molecule | %d failed | whole pool (%s molecules): %.2f h"
        % (a.view, len(smi), sec, ms, int((~ok).sum()), format(rows, ","), ms * rows / 3.6e6))
    if a.view == "pepland":
        say("with %d thread(s) per process, if the processes do not slow each other down:" % a.threads)
        for p in (4, 9, 14, 27):
            if p * a.threads <= 32:
                say("  %2d processes (%2d cores): whole pool in %.1f h" % (p, p * a.threads, ms * rows / 3.6e6 / p))


def _term(sig, frame):
    raise KeyboardInterrupt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--view", choices=VIEWS)
    ap.add_argument("--workers", type=int, default=8, help="ecfp / rdkit processes")
    ap.add_argument("--threads", type=int, default=1, help="pepland torch threads per process")
    ap.add_argument("--batch", type=int, default=64, help="pepland molecules per call")
    ap.add_argument("--max-tokens", type=int, default=8192, help="pclm tokens per batch")
    ap.add_argument("--esm-model", default=None, help="ESM-2 folder (default: ESM_DIR in paths.py, the 3B model)")
    ap.add_argument("--esm-caps", action="store_true",
                    help="esm view: also embed peptides that only differ from canonical by an N-acetyl / C-amide (embedded without the caps); "
                         "then every pool molecule is parsed (~10 min more on 16 workers)")
    ap.add_argument("--first", type=int, default=0, help="only shards below this number")
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--parts", type=int, default=1)
    ap.add_argument("--time", type=int, default=0, help="check + time this many molecules, write nothing")
    ap.add_argument("--check", action="store_true", help="only compare with the benchmark features")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--export", action="store_true", help="write the text copy of the pool (env peptide)")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    if a.status:
        return status()
    if a.export:
        return export_text()
    assert a.view, "--view is required"
    assert a.workers <= 32 and a.threads <= 32, "lab rule: at most half of the 64 cores"

    index, rows = pool_index()
    if a.time:
        return timing(a, index, rows)
    if a.check:
        return check(a)
    signal.signal(signal.SIGINT, signal.default_int_handler)     # a shell can start us with SIGINT ignored
    signal.signal(signal.SIGTERM, _term)
    shards =[s for s in sorted(index) if (not a.first or s < a.first) and s % a.parts == a.part]
    todo = [s for s in shards if not os.path.exists(shard_path(a.view, s))]
    os.makedirs(os.path.join(CACHE, a.view), exist_ok=True)
    say("%s: %d shards for this process, %d to do" % (a.view, len(shards), len(todo)))
    if a.view == "rdkit":                          # the column names travel with the cache
        import rdkit_desc
        npath = os.path.join(CACHE, "rdkit", "names.json")
        if os.path.exists(npath):
            with open(npath) as fh:
                assert json.load(fh) == rdkit_desc.names(),                     "%s lists other descriptors than this RDKit gives: the cache was made with another version" % npath
        else:
            with open(npath, "w") as fh:
                json.dump(rdkit_desc.names(), fh)
    if a.view == "esm":                            # the embedding size travels with the cache (the teacher reads it)
        import featurize_esm
        dpath = os.path.join(CACHE, "esm", "dim.json")
        info = {"dim": featurize_esm.esm_dim(a.esm_model), "model": os.path.basename(a.esm_model or featurize_esm.ESM_DIR),
                "caps": bool(a.esm_caps)}
        if os.path.exists(dpath):
            with open(dpath) as fh:
                assert json.load(fh) == info, "%s was written for another ESM model: use that one or a new cache" % dpath
        else:
            with open(dpath, "w") as fh:
                json.dump(info, fh)
    if todo and not a.no_check and not check(a):
        sys.exit("the check failed: nothing was written (--no-check to skip it)")

    t0 = time.time()
    done = bad = 0
    try:
        for s in todo:
            lo, hi = index[s]
            t1 = time.time()
            X, ok = compute(a, shard_smiles(s, lo, hi), part_path(a.view, s), "%s shard %d:" % (a.view, s), (lo, hi))
            path = shard_path(a.view, s)
            tmp = path + ".tmp.npz"
            (np.savez_compressed if a.view == "ecfp" else np.savez)(tmp, X=X, ok=ok, lo=lo, hi=hi)
            os.replace(tmp, path)
            if os.path.exists(part_path(a.view, s)):
                os.remove(part_path(a.view, s))
            done += 1
            bad += int((~ok).sum())
            say("%s shard %3d | %s molecules | %s %d | %.1f min (run total %.1f min, %d shards done)"
                % (a.view, s, format(hi - lo, ","), "not canonical (absent)" if a.view == "esm" else "failed",
                   int((~ok).sum()), (time.time() - t1) / 60, (time.time() - t0) / 60, done))
    except KeyboardInterrupt:
        say("%s: stopped after %d shards (%.1f min); run the same command again to continue"
            % (a.view, done, (time.time() - t0) / 60))
        sys.exit(130)
    say("%s: %d shards written, %d molecules failed, %.1f min" % (a.view, done, bad, (time.time() - t0) / 60))


if __name__ == "__main__":
    main()

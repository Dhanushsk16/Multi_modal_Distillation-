"""Token ids of the pool for the student, one file per cache shard, row-aligned with the teacher's view caches.

    python src/student/make_student_data.py --tok CKPT_DIR/student_tokenizer_norm/atombpe_v1024 [--k-random 8] [--max-len 128] [--first N]

Reads pool.parquet (columns smiles, residues) and the normalised pool written by `normalise.py --write-pool` (column smiles_norm, the same row
order: neutralised charges, RDKit canonical = the form the student reads), cuts the rows of every shard the teacher cache has
(features/corpus/<view>/shard_NNN.npz holds `lo` and `hi`, the pool rows it covers), tokenises them and writes

    features/corpus/student_<tokenizer folder name>/shard_NNN.npz
        ids   uint16  the CANONICAL spellings, all tokens of the shard's molecules one after the other, each = [CLS] tokens [SEP], cut to --max-len
        len   int16   tokens per molecule (with CLS and SEP)
        rids  uint16  with --k-random K: K RANDOM valid spellings of every molecule (RDKit doRandom: random atom and branch order, same molecule,
                      stereo kept), tokenised the same way, molecule after molecule, spelling after spelling
        rlen  int16   [n, K] tokens of each random spelling
        res   int16   residues per molecule (pool column `residues`; the trainer masks PepLand's target above 30 with it)
        lo, hi        pool rows (as in the view caches)

Random-SMILES augmentation (decided 2026-11-22): the trainer feeds a random spelling instead of the canonical one on a fraction of the steps while the
targets stay the teacher's vectors of the canonical string. A random spelling that needs more than --max-len-random tokens is replaced by the canonical one
(a cut string would not describe the molecule the target belongs to); the limit for random spellings is --max-len-random (192), for canonical
ones --max-len (128). RDKit's random writer cannot be seeded: these files ARE the record of the
augmentation. A sample of each shard is parsed back and compared with the canonical SMILES (--verify).

A molecule whose normalised string could not be made (unparsable) was kept as it is by normalise.py; nothing is dropped here.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import CACHE_DIR, CORPUS_DIR  # noqa: E402
import normalise as NM  # noqa: E402
import tokenizer as T  # noqa: E402


def student_dir(tok_folder, cache_dir=CACHE_DIR):
    return os.path.join(cache_dir, "student_" + os.path.basename(os.path.normpath(tok_folder)))


def shard_spans(cache_dir=CACHE_DIR, views=("pclm", "pepland", "ecfp", "esm")):
    """{shard: (lo, hi)} from the view caches (only the two small members `lo` and `hi` are read)."""
    spans = {}
    for v in views:
        d = os.path.join(cache_dir, v)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if f.startswith("shard_") and f.endswith(".npz") and ".tmp" not in f:
                s = int(f[6:9])
                if s not in spans:
                    z = np.load(os.path.join(d, f))
                    spans[s] = (int(z["lo"]), int(z["hi"]))
    return spans


def pack(tok, enc, max_len):
    """Encoded molecules (lists of ids without specials) -> (flat ids uint16 with [CLS] / [SEP], tokens per molecule after the cut int16, before the cut)."""
    ids, lens, full = [], np.zeros(len(enc), np.int16), np.zeros(len(enc), np.int64)
    for i, e in enumerate(enc):
        w = tok.with_special(e, max_len)
        ids.append(np.asarray(w, np.uint16))
        lens[i], full[i] = len(w), len(e) + 2
    return (np.concatenate(ids) if ids else np.zeros(0, np.uint16)), lens, full


def tokenise(tok, smiles, max_len):
    return pack(tok, tok.encode_batch(list(smiles)), max_len)


def verify(canon, spellings, n):
    """Parse back the first n molecules' random spellings: how many do NOT give the canonical SMILES again."""
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    bad = tot = 0
    for c, sp in list(zip(canon, spellings))[:n]:
        for s in sp:
            m = Chem.MolFromSmiles(s)
            tot += 1
            if m is None or Chem.MolToSmiles(m, isomericSmiles=True, canonical=True) != c:
                bad += 1
    return bad, tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tok", required=True, help="the learned tokenizer folder (…/atombpe_v1024)")
    ap.add_argument("--pool", default=os.path.join(CORPUS_DIR, "pool.parquet"))
    ap.add_argument("--norm", default=os.path.join(CORPUS_DIR, "pool_norm.parquet"), help="normalise.py --write-pool output")
    ap.add_argument("--max-len", type=int, default=128, help="tokens per molecule with [CLS] and [SEP]; 99.9%% of the pool fits in 128 with the learned tokenizer")
    ap.add_argument("--max-len-random", type=int, default=192, help="longest random spelling kept (random spellings compress worse than canonical ones); a longer one "
                    "is replaced by the canonical spelling")
    ap.add_argument("--k-random", type=int, default=0, help="random spellings stored per molecule (0 = none; 8 is the planned value)")
    ap.add_argument("--verify", type=int, default=200, help="molecules per shard whose random spellings are parsed back and compared (0 = none)")
    ap.add_argument("--workers", type=int, default=8, help="processes for the random spellings (lab rule: at most half of the CPU)")
    ap.add_argument("--first", type=int, default=0, help="only the first N shards (a quick test)")
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--redo", action="store_true")
    a = ap.parse_args()
    import pyarrow.parquet as pq
    tok = T.load_tokenizer(a.tok)
    out = student_dir(a.tok, a.cache_dir)
    os.makedirs(out, exist_ok=True)
    spans = shard_spans(a.cache_dir)
    assert spans, "no view cache under %s" % a.cache_dir
    shards = sorted(spans)[: a.first] if a.first else sorted(spans)
    print("tokenizer %s (%d tokens) | %d shards | %d random spellings per molecule -> %s" % (tok.name, tok.vocab_size, len(shards), a.k_random, out), flush=True)
    res = pq.read_table(a.pool, columns=["residues"]).column("residues").to_numpy()
    sm = pq.read_table(a.norm, columns=["smiles_norm"]).column("smiles_norm").to_pylist()
    assert len(sm) == len(res), "pool_norm has %d rows, the pool %d: run normalise.py --write-pool on this pool" % (len(sm), len(res))
    t0, n_all, n_cut, n_unk, n_fall, n_bad, n_chk = time.time(), 0, 0, 0, 0, 0, 0
    sum_c = sum_r = 0
    K = a.k_random
    from multiprocessing import Pool
    pool = Pool(a.workers, initializer=NM._init) if K and a.workers > 1 else None
    try:
        for s in shards:
            path = os.path.join(out, "shard_%03d.npz" % s)
            if os.path.exists(path) and not a.redo:
                continue
            lo, hi = spans[s]
            enc_c = tok.encode_batch(sm[lo:hi])
            ids, lens, full = pack(tok, enc_c, a.max_len)
            n_cut += int((full > a.max_len).sum())
            n_unk += int(np.count_nonzero(ids == T.UNK))
            extra = {}
            if K:
                rs, unparsable = NM.random_spellings(sm[lo:hi], K, a.workers, pool=pool)
                flat = [x for row in rs for x in row]
                enc_r = tok.encode_batch(flat)
                for f, e in enumerate(enc_r):                         # a spelling that is too long for the student is replaced by the canonical one
                    if len(e) + 2 > a.max_len_random:
                        enc_r[f] = enc_c[f // K]
                        n_fall += 1
                rids, rlen, _ = pack(tok, enc_r, a.max_len_random)
                extra = {"rids": rids, "rlen": rlen.reshape(-1, K)}
                sum_c += int(lens.astype(np.int64).sum())
                sum_r += int(rlen.astype(np.int64).sum()) / K
                if a.verify:
                    b, t = verify(sm[lo:hi], rs, a.verify)
                    n_bad, n_chk = n_bad + b, n_chk + t
            n_all += hi - lo
            np.savez(path + ".tmp.npz", ids=ids, len=lens, res=res[lo:hi].astype(np.int16), lo=lo, hi=hi, **extra)
            os.replace(path + ".tmp.npz", path)
            print("shard %3d | rows %d..%d | tokens/molecule mean %.1f max %d%s | %.0f s" % (
                s, lo, hi, lens.mean(), lens.max(), (" | random spelling mean %.1f" % (extra["rlen"].mean())) if K else "", time.time() - t0), flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    print("done: %s molecules | cut at %d tokens: %d (%.3f%%) | [UNK] tokens: %d" % (
        format(n_all, ","), a.max_len, n_cut, 100 * n_cut / max(n_all, 1), n_unk))
    if K and n_all:
        print("random spellings: mean tokens %.1f against %.1f canonical (%.2fx) | replaced by the canonical one (longer than %d): %d | parsed back and compared: %d, NOT the same molecule: %d" % (
            sum_r / n_all, sum_c / n_all, (sum_r / max(sum_c, 1)), a.max_len_random, n_fall, n_chk, n_bad))


if __name__ == "__main__":
    main()

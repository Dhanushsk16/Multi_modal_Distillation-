"""Learn the student's tokenizer from the pool, then compare it with the released one. CPU only.

    python src/student/train_tokenizer.py                      # learn vocab 1024 / 2048 / 4096, then the report
    python src/student/train_tokenizer.py --vocab 2048 --n-train 100000
    python src/student/train_tokenizer.py --report-only        # A, B and the learned tokenizers already saved

Three tokenizers are compared (src/student/tokenizer.py):  A = PeptideCLM-2's released tokenizer, B = the same
405 ids with the digit defect fixed, C = atom-level BPE learned here (one per --vocab).
Learned on a sample of the POOL (every non-canonical molecule has a chance; --nc-frac of the sample is non-canonical).
Measured on (1) a held-out sample of the pool, drawn at the pool's natural mix and disjoint from the training sample,
and (2) the ten benchmarks, which the pool was cleaned against, so they are held out too.

Per tokenizer and dataset: tokens per residue, length percentiles (with [CLS] and [SEP]), the share of molecules longer than
128 / 192 / 256 / 384 / 512 tokens, the share with an [UNK], and for C the share with an atom that is spelled out letter by
letter, and the round-trip check decode(encode(s)) == s. Also the length that covers 99 / 99.9 / 100% of the pool and the
benchmarks: the student's maximum length has to be chosen from that.

Writes (under --tok-dir) one folder per learned tokenizer (tokenizer.json, atoms.json, meta.json, vocab.txt) and (under --out)
summary.csv, by_band.csv, token_use.csv, examples.txt, report.json. Decision rule for A / B / C: a distillation pilot at equal
compute (not part of this script); this report only says what each tokenizer does to the data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("RAYON_NUM_THREADS", "8")          # lab rule: at most 50% of the CPU (the Hugging Face trainer uses rayon)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import normalise as NM  # noqa: E402
import tokenizer as T  # noqa: E402
from paths import CORPUS_DIR, RESULTS_DIR  # noqa: E402

BENCH = ["CellPPD", "THPep", "AmpHGT", "PAMPA", "PepMSND", "PL_cCPP", "PL_cSol", "PL_ncCPP", "PL_cBinding", "PL_ncBinding"]
LIMITS = (128, 192, 256, 384, 512)
BANDS = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 100)]


def read_smiles(path, positions):
    """The `smiles` column at row `positions` (any order), reading one row group at a time."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    pf, want, got, start = pq.ParquetFile(path), np.sort(positions), {}, 0
    for rg in range(pf.num_row_groups):
        m = pf.metadata.row_group(rg).num_rows
        sel = want[(want >= start) & (want < start + m)] - start
        if len(sel):
            col = pf.read_row_group(rg, columns=["smiles"]).column("smiles")
            got.update(zip((sel + start).tolist(), col.take(pa.array(sel)).to_pylist()))
        start += m
    return [got[int(p)] for p in positions]


BAND_EDGES = [0, 4, 9, 14, 19, 24, 30, 40, 50, 10 ** 6]       # residues: build_pool's bands, plus everything above 50
BYTES_PER_CHAR = 85                                              # peak process memory per training character: 53 (60k synthetic molecules) to ~85 (real run, 200k)


def stratified_take(meta, idx, total, keys):
    """`idx` = pool rows in shuffled order. Equal quota per stratum (water-filling as in build_pool.py: a stratum smaller than its
    quota is taken whole and its unused share goes to the others); inside a stratum the rows come in the order of `idx`.
    -> (chosen rows, table of strata: available / taken)."""
    from corpus.build_pool import water_fill
    key = meta.iloc[idx][keys].astype(str).agg("|".join, axis=1).to_numpy()
    codes, uniq = pd.factorize(key)
    sizes = np.bincount(codes, minlength=len(uniq))
    quota = water_fill(sizes, total)
    order = np.argsort(codes, kind="stable")
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    out = [idx[order[st:st + int(q)]] for st, q in zip(starts, quota)]
    table = pd.DataFrame({"stratum": uniq, "available": sizes, "taken": quota.astype(int)})
    return (np.concatenate(out) if out else idx[:0]), table


def sample_pool(path, n_train, n_held, nc_frac, seed, fixed_heldout=False, stratified=False):
    """-> (train smiles, held-out smiles, held-out residues, held-out kinds, composition dict, strata tables or None).
    The two samples are disjoint. Held-out: the pool's own mix. Training, `random` (default): nc_frac non-canonical, the rest
    canonical, drawn at random. Training, `stratified`: the same split, but inside each half the quota is shared EQUALLY over the
    strata (non-canonical: source x cyclic x residue band; canonical: source x residue band), so rare chemistry (cyclic or long
    non-canonical peptides, small sources) is not drowned by the bulk of near-duplicate ESMAtlas peptides.
    fixed_heldout (implied by stratified): the held-out molecules are the first ones of the shuffled pool and the training
    molecules come after them, so the held-out set does NOT change with n_train or with the sampling (needed to compare runs)."""
    import pyarrow.parquet as pq
    have = set(pq.ParquetFile(path).schema.names)
    meta = pq.read_table(path, columns=[c for c in ("kind", "residues", "source", "cyclic") if c in have]).to_pandas()
    is_nc = (meta.kind == "noncanonical").to_numpy()
    nc, ca = np.flatnonzero(is_nc), np.flatnonzero(~is_nc)
    rng = np.random.default_rng(seed)
    nc, ca = rng.permutation(nc), rng.permutation(ca)
    share = len(nc) / len(meta)
    fixed_heldout = fixed_heldout or stratified
    strata = None
    if fixed_heldout:
        h_nc = min(int(round(n_held * share)), len(nc))
        h_ca = min(n_held - h_nc, len(ca))
        k_nc = min(int(round(n_train * nc_frac)), len(nc) - h_nc)
        k_ca = min(n_train - k_nc, len(ca) - h_ca)
        he = np.concatenate([nc[:h_nc], ca[:h_ca]])
        if stratified:
            meta["band"] = pd.cut(meta.residues, BAND_EDGES, labels=False)
            tr_nc, t_nc = stratified_take(meta, nc[h_nc:], k_nc, [c for c in ("source", "cyclic", "band") if c in meta.columns])
            tr_ca, t_ca = stratified_take(meta, ca[h_ca:], k_ca, [c for c in ("source", "band") if c in meta.columns])
            tr, k_nc = np.concatenate([tr_nc, tr_ca]), len(tr_nc)
            strata = {"noncanonical": t_nc, "canonical": t_ca}
        else:
            tr = np.concatenate([nc[h_nc:h_nc + k_nc], ca[h_ca:h_ca + k_ca]])
    else:
        k_nc = min(int(round(n_train * nc_frac)), len(nc) // 2)
        k_ca = min(n_train - k_nc, len(ca) // 2)
        h_nc = min(int(round(n_held * share)), len(nc) - k_nc)
        h_ca = min(n_held - h_nc, len(ca) - k_ca)
        tr = np.concatenate([nc[:k_nc], ca[:k_ca]])
        he = np.concatenate([nc[k_nc:k_nc + h_nc], ca[k_ca:k_ca + h_ca]])
    assert not set(tr.tolist()) & set(he.tolist())
    tr = rng.permutation(tr)
    train = read_smiles(path, tr)
    held = read_smiles(path, he)
    comp = {"pool_rows": len(meta), "pool_noncanonical_share": round(share, 4), "sampling": "stratified" if stratified else "random",
            "train": len(tr), "train_noncanonical": int(k_nc), "held_out": len(he), "held_out_noncanonical": int(h_nc)}
    kinds = np.where(is_nc[he], "noncanonical", "canonical")
    return train, held, meta.residues.to_numpy()[he], kinds, comp, strata


def benchmark_sets(names, normalise=True, workers=8):
    """{benchmark: (SMILES, residues)}. With `normalise` the SMILES are the student's input form (neutralised, RDKit canonical) and the
    residues are counted on that form (a protonated N-terminus [NH3+] is not matched by the residue pattern)."""
    import benchmarks as BM
    from eval_single import n_residues
    out = {}
    for b in names:
        sm = sorted(set(BM.load(b).smiles.dropna().astype(str)))
        if normalise:
            sm, _ = NM.normalise_many(sm, workers)
        out[b] = (sm, np.array([max(n_residues(s), 1) for s in sm]))
    return out


def measure(tok, smiles, residues, base_len=None):
    """Metrics of one tokenizer on one set; also returns the per-molecule lengths ([CLS] and [SEP] included)."""
    ids = tok.encode_batch(smiles)
    n = np.array([len(i) + 2 for i in ids])
    unk = np.array([i.count(T.UNK) for i in ids])
    row = {"n": len(smiles), "vocab": tok.vocab_size, "tok_per_res": float(np.median(n / np.maximum(residues, 1))),
           "len_mean": float(n.mean()), "len_median": float(np.median(n)), "len_p95": float(np.percentile(n, 95)),
           "len_p99": float(np.percentile(n, 99)), "len_max": int(n.max())}
    for L in LIMITS:
        row["pct_gt%d" % L] = 100 * float((n > L).mean())
    row["pct_mol_unk"] = 100 * float((unk > 0).mean())
    row["unk_per_1000_tok"] = 1000 * float(unk.sum() / max(n.sum(), 1))
    if tok.family == "atombpe":
        row["pct_mol_fallback"] = 100 * float(np.mean([bool(tok.fallback_atoms(s)) for s in smiles]))
        v = tok.vocab()
        bad = sum(1 for s, i in zip(smiles, ids) if "".join(v[t] for t in i if t >= len(T.SPECIALS)) != s)
        row["roundtrip_fail"] = int(bad)
    row["len_vs_A"] = float(n.mean() / base_len) if base_len else 1.0
    return row, n, ids


def covering(n, p):
    return int(np.ceil(np.percentile(n, 100 * p)))


def show(title, df, **kw):
    print("\n== " + title)
    print(df.to_string(index=False, float_format=lambda x: "%.2f" % x, **kw))


def peak_mb():
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0      # Linux: kB
    except Exception:
        pass
    try:
        import psutil                                                             # Windows (laptop tests)
        return psutil.Process().memory_info().peak_wset / 2 ** 20
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=os.path.join(CORPUS_DIR, "pool.parquet"))
    ap.add_argument("--vocab", nargs="*", type=int, default=[1024, 2048, 4096], help="learned vocabulary sizes (special tokens and the "
                    "95 printable ASCII characters count towards it)")
    ap.add_argument("--n-train", type=int, default=200_000, help="pool molecules the tokenizer is learned on (memory grows with it)")
    ap.add_argument("--n-heldout", type=int, default=50_000, help="held-out pool molecules for the report")
    ap.add_argument("--nc-frac", type=float, default=0.4, help="share of the training sample that is non-canonical (pool: about 19%%; "
                    "the benchmarks are mostly non-canonical). A choice, not tested.")
    ap.add_argument("--min-atom-count", type=int, default=20, help="a multi-character atom seen fewer times is spelled out letter by letter")
    ap.add_argument("--min-frequency", type=int, default=100, help="a BPE merge must occur at least this often in the training sample")
    ap.add_argument("--max-token-len", type=int, default=0, help="longest token in atoms (0 = no cap)")
    ap.add_argument("--fixed-heldout", action="store_true", help="draw the held-out molecules first, so they are the same whatever --n-train is "
                    "(use it to compare runs with different --n-train; off = the sampling of the first run)")
    ap.add_argument("--sample", choices=["random", "stratified"], default="random",
                    help="random = the first run; stratified = equal quota per source / cyclic / residue-band stratum inside the "
                         "non-canonical and the canonical half (implies --fixed-heldout)")
    ap.add_argument("--no-normalise", dest="normalise", action="store_false",
                    help="do NOT normalise the SMILES first (default: neutralise charges + RDKit canonical, the student's input form; src/student/normalise.py)")
    ap.add_argument("--random-frac", type=float, default=0.0, help="share of the TRAINING sample written as a random (non-canonical) spelling of the same "
                    "molecule, so the learned merges also cover what random-SMILES augmentation feeds the student; the held-out set and the report stay canonical")
    ap.add_argument("--workers", type=int, default=8, help="processes for the normalisation (lab rule: at most half of the CPU)")
    ap.add_argument("--max-gb", type=float, default=24.0, help="refuse to learn when the projected peak memory is above this")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bench", nargs="*", default=BENCH)
    ap.add_argument("--tok-dir", default=T.TOK_DIR)
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "student_tokenizer"))
    ap.add_argument("--report-only", action="store_true", help="do not learn; report A, B and the learned tokenizers found in --tok-dir")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t_start = time.time()

    print("pool:", a.pool)
    train, held, held_res, held_kind, comp, strata = sample_pool(a.pool, a.n_train, a.n_heldout, a.nc_frac, a.seed, a.fixed_heldout,
                                                                 a.sample == "stratified")
    print("sample: %s" % json.dumps(comp))
    if a.normalise:
        t0 = time.time()
        n_before = sum(map(len, train))
        old_train, old_held = train, held
        train, bad1 = NM.normalise_many(train, a.workers)
        held, bad2 = NM.normalise_many(held, a.workers)
        comp.update({"normalised": True, "changed_train": sum(x != y for x, y in zip(old_train, train)),
                     "changed_held_out": sum(x != y for x, y in zip(old_held, held)), "unparsable": bad1 + bad2})
        print("normalised (neutralised + RDKit canonical): %d of %d training and %d of %d held-out molecules changed, %d unparsable (kept as they are) | %.0f s"
              % (comp["changed_train"], len(train), comp["changed_held_out"], len(held), bad1 + bad2, time.time() - t0))
        del old_train, old_held
    else:
        comp["normalised"] = False
    if a.random_frac > 0:                       # the student is trained on random spellings too (2026-11-22): let BPE see them, so they do not fall to letters
        t0 = time.time()
        train, n_rand = NM.randomise_fraction(train, a.random_frac, a.workers, a.seed)
        comp["random_spelling_share"] = a.random_frac
        print("%d of %d training molecules (%.0f%%) replaced by a random spelling (held-out set stays canonical) | %.0f s" % (
            n_rand, len(train), 100 * a.random_frac, time.time() - t0))
    chars = sum(map(len, train))
    proj = chars * BYTES_PER_CHAR / 1e9
    print("training sample: %s molecules, %s characters (mean %.0f) | projected peak memory about %.1f GB (estimate: %d bytes per character)"
          % (format(len(train), ","), format(chars, ","), chars / max(len(train), 1), proj, BYTES_PER_CHAR))
    if strata is not None:
        pd.concat([t.assign(half=h) for h, t in strata.items()]).to_csv(os.path.join(a.out, "strata.csv"), index=False)
        for h, t in strata.items():
            print("  %s: %d strata, %d taken whole, smallest quota %d, largest %d" % (
                h, len(t), int((t.taken >= t.available).sum()), t.taken.min(), t.taken.max()))
    if proj > a.max_gb and not a.report_only:
        sys.exit("projected peak memory %.1f GB is above --max-gb %.0f: lower --n-train (or raise --max-gb after checking `free -g`)" % (proj, a.max_gb))

    toks = [T.Released(False), T.Released(True)]
    timing = {}
    if a.report_only:
        for d in sorted(os.listdir(a.tok_dir)) if os.path.isdir(a.tok_dir) else []:
            if d.startswith("atombpe_v") and os.path.exists(os.path.join(a.tok_dir, d, "tokenizer.json")):
                toks.append(T.AtomBPE.load(os.path.join(a.tok_dir, d)))
    else:
        for V in a.vocab:
            t0 = time.time()
            tk = T.AtomBPE.train(train, V, a.min_atom_count, a.min_frequency, a.max_token_len or None, name="C_atombpe_v%d" % V)
            timing[V] = round(time.time() - t0, 1)
            folder = os.path.join(a.tok_dir, "atombpe_v%d" % V)
            tk.save(folder)
            print("learned vocab %d in %.0f s (%d base atoms, %d atoms spelled out) -> %s%s" % (
                tk.vocab_size, timing[V], tk.meta["n_atoms"], tk.meta["atoms_below_min_count"], folder,
                "  [peak memory %.0f MB]" % peak_mb() if peak_mb() else ""))
            toks.append(tk)
    assert len(toks) > 2, "no learned tokenizer (give --vocab, or learn one first)"

    sets = {"pool_heldout": (held, held_res)}
    sets.update(benchmark_sets(a.bench, a.normalise, a.workers))
    rows, band_rows, all_len = [], [], {}
    for dname, (sm, res) in sets.items():
        base = None
        for tk in toks:
            t0 = time.time()
            row, n, ids = measure(tk, sm, res, base)
            base = base if base is not None else row["len_mean"]
            row.update({"dataset": dname, "tokenizer": tk.name, "encode_s": round(time.time() - t0, 1)})
            rows.append(row)
            all_len.setdefault(tk.name, {})[dname] = n
            if dname == "pool_heldout":
                for lo, hi in BANDS:
                    m = (res >= lo) & (res <= hi)
                    if m.sum() >= 20:
                        band_rows.append({"tokenizer": tk.name, "residues": "%d-%d" % (lo, hi), "n": int(m.sum()),
                                          "median": float(np.median(n[m])), "p95": float(np.percentile(n[m], 95)),
                                          "p99": float(np.percentile(n[m], 99)), "max": int(n[m].max())})
    df = pd.DataFrame(rows)
    first = ["dataset", "tokenizer", "n", "vocab", "tok_per_res", "len_median", "len_p99", "len_max", "pct_gt256", "pct_gt512",
             "pct_mol_unk", "len_vs_A"]
    cols = [c for c in df.columns if c not in first]
    df = df[first + cols]
    df.to_csv(os.path.join(a.out, "summary.csv"), index=False)

    for dname in sets:
        sub = df[df.dataset == dname]
        show(dname + " (%s molecules)" % format(int(sub.n.iloc[0]), ","),
             sub[["tokenizer", "vocab", "tok_per_res", "len_median", "len_p99", "len_max", "pct_gt256", "pct_gt512", "pct_mol_unk", "len_vs_A"]
                 + [c for c in ("pct_mol_fallback", "roundtrip_fail") if c in sub.columns]].dropna(axis=1, how="all"))

    bands = pd.DataFrame(band_rows)
    bands.to_csv(os.path.join(a.out, "by_band.csv"), index=False)
    show("held-out pool: tokens (with [CLS], [SEP]) by residues", bands)

    cover = []
    for tk in toks:
        nb = np.concatenate([all_len[tk.name][b] for b in sets if b != "pool_heldout"])
        npool = all_len[tk.name]["pool_heldout"]
        cover.append({"tokenizer": tk.name, **{"pool_%s" % k: covering(npool, p) for k, p in (("99", .99), ("99.9", .999), ("100", 1.0))},
                      **{"bench_%s" % k: covering(nb, p) for k, p in (("99", .99), ("99.9", .999), ("100", 1.0))}})
    cover = pd.DataFrame(cover)
    show("maximum length needed to keep this share of molecules whole (the student's max length is chosen from this)", cover)

    # which tokens are used most (held-out pool) for each learned tokenizer, and how long they are
    use_rows, ex_lines = [], []
    for tk in toks:
        if tk.family != "atombpe":
            continue
        cnt = np.bincount(np.concatenate([np.array(i) for i in tk.encode_batch(held)]), minlength=tk.vocab_size)
        v = tk.vocab()
        order = np.argsort(-cnt)[:300]
        use_rows += [{"tokenizer": tk.name, "rank": r + 1, "token": v[i], "count": int(cnt[i]), "share_pct": 100 * float(cnt[i] / cnt.sum()),
                      "atoms": len(T.MULTI.findall(v[i])) + len(T.MULTI.sub("", v[i]))} for r, i in enumerate(order)]
        used = int((cnt > 0).sum())
        lens = np.array([len(T.MULTI.findall(x)) + len(T.MULTI.sub("", x)) for x in v[len(T.SPECIALS):]])
        print("\n%s: %d of %d tokens used on the held-out pool; token length in atoms: median %d, p90 %d, max %d; top 12: %s" % (
            tk.name, used, tk.vocab_size, np.median(lens), np.percentile(lens, 90), lens.max(),
            [v[i] for i in np.argsort(-cnt)[:12]]))
    pd.DataFrame(use_rows).to_csv(os.path.join(a.out, "token_use.csv"), index=False)

    # worked examples: the same molecules under every tokenizer
    nc_i = int(np.flatnonzero(held_kind == "noncanonical")[0]) if (held_kind == "noncanonical").any() else 0
    lg = np.flatnonzero(held_res >= 31)
    ex = [("Ala-Leu-Trp", "CC(C)C[C@H](NC(=O)[C@@H](N)C)C(=O)N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O"),
          ("non-canonical, held-out pool", held[nc_i])] + ([("%d residues, held-out pool" % held_res[lg[0]], held[int(lg[0])])] if len(lg) else [])
    for title, s in ex:
        ex_lines.append("### %s\n%s" % (title, s))
        for tk in toks:
            t = tk.tokens(s)
            ex_lines.append("%-22s %3d tokens: %s%s" % (tk.name, len(t) + 2, " | ".join(t[:40]), " ..." if len(t) > 40 else ""))
        ex_lines.append("")
    with open(os.path.join(a.out, "examples.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(ex_lines))
    print("\n== examples (first 40 tokens)\n" + "\n".join(ex_lines))

    report = {"args": vars(a), "sample": comp, "learn_seconds": timing, "peak_memory_mb": peak_mb(), "total_seconds": round(time.time() - t_start, 1),
              "tokenizers_version": __import__("tokenizers").__version__, "learned": {t.name: t.meta for t in toks if t.family == "atombpe"},
              "max_length_needed": cover.to_dict("records")}
    with open(os.path.join(a.out, "report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("\nwrote %s (%.0f s)" % (a.out, time.time() - t_start))


if __name__ == "__main__":
    main()

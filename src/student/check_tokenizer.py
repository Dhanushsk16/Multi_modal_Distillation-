"""Can this tokenizer read every molecule? Checks the WHOLE pool and any other SMILES (your new benchmarks). CPU only.

    python src/student/check_tokenizer.py --tokenizer atombpe_v2048                 # the whole pool + the ten benchmarks
    python src/student/check_tokenizer.py --tokenizer atombpe_v2048 --no-pool --no-bench --csv NEW.csv --col SMILES
    python src/student/check_tokenizer.py --tokenizer atombpe_v2048 --max-molecules 500000     # first N pool rows only (quick)
    python src/student/check_tokenizer.py --tokenizer released                      # the same questions for PeptideCLM-2's own tokenizer

For every dataset: molecules with a non-ASCII character (the learned tokenizer maps those to [UNK]); molecules with an [UNK];
molecules with an atom that is spelled out letter by letter (not an error, but a sign that the chemistry is unfamiliar: it
costs tokens and the student has barely seen those pieces), with the most common such atoms; round-trip failures
(decode(encode(s)) != s, learned tokenizers only); length percentiles, share above 128 / 192 / 256 tokens, characters per token
(works for any molecule) and tokens per residue (peptide sets).
Input form: by default every SMILES is normalised first (neutralised charges, RDKit canonical: src/student/normalise.py), which is what the
student reads; --no-normalise checks the strings as given, --raw-too shows both for --csv files and benchmarks.
Writes check_<tokenizer>.csv and spelled_out_<tokenizer>.csv under --out.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import normalise as NM  # noqa: E402
import tokenizer as T  # noqa: E402
from paths import CORPUS_DIR, RESULTS_DIR  # noqa: E402

BENCH = ["CellPPD", "THPep", "AmpHGT", "PAMPA", "PepMSND", "PL_cCPP", "PL_cSol", "PL_ncCPP", "PL_cBinding", "PL_ncBinding"]


class Acc:
    """Running totals for one dataset (the pool is read in batches)."""

    def __init__(self, tok):
        self.tok, self.n, self.non_ascii, self.mol_unk, self.mol_fb, self.fail = tok, 0, 0, 0, 0, 0
        self.chars = self.toks = 0
        self.lens, self.tpr, self.fb = [], [], Counter()
        self.vocab = tok.vocab() if tok.family == "atombpe" else None

    def add(self, smiles, residues=None):
        smiles = [s for s in smiles if isinstance(s, str) and s]
        ids = self.tok.encode_batch(smiles)
        n = np.array([len(i) + 2 for i in ids])                    # with [CLS] and [SEP]
        self.n += len(smiles)
        self.non_ascii += sum(any(ord(c) < 32 or ord(c) > 126 for c in s) for s in smiles)
        self.mol_unk += sum(1 for i in ids if i.count(T.UNK))
        self.chars += sum(map(len, smiles))
        self.toks += int(n.sum()) - 2 * len(smiles)
        self.lens.append(n.astype(np.int32))
        if residues is not None:
            self.tpr.append((n / np.maximum(np.asarray(residues), 1)).astype(np.float32))
        if self.vocab is not None:
            for s, i in zip(smiles, ids):
                fb = self.tok.fallback_atoms(s)
                if fb:
                    self.mol_fb += 1
                    self.fb.update(set(fb))
                if "".join(self.vocab[t] for t in i if t >= len(T.SPECIALS)) != s:
                    self.fail += 1

    def row(self, name):
        n = np.concatenate(self.lens)
        r = {"dataset": name, "n": self.n, "pct_non_ascii": 100 * self.non_ascii / self.n, "pct_mol_unk": 100 * self.mol_unk / self.n,
             "chars_per_token": self.chars / max(self.toks, 1), "len_median": float(np.median(n)), "len_p99": float(np.percentile(n, 99)),
             "len_max": int(n.max()), "pct_gt128": 100 * float((n > 128).mean()), "pct_gt192": 100 * float((n > 192).mean()),
             "pct_gt256": 100 * float((n > 256).mean())}
        r["tok_per_res"] = float(np.median(np.concatenate(self.tpr))) if self.tpr else float("nan")
        if self.vocab is not None:
            r["pct_mol_spelled_out"], r["roundtrip_fail"] = 100 * self.mol_fb / self.n, self.fail
        return r


def pool_acc(tok, path, max_molecules, normalise=True, workers=8):
    import pyarrow.parquet as pq
    acc, t0, done = Acc(tok), time.time(), 0
    cols = ["smiles", "residues"]
    for b in pq.ParquetFile(path).iter_batches(batch_size=100_000, columns=cols):
        d = b.to_pandas()
        if max_molecules and done + len(d) > max_molecules:
            d = d.iloc[: max_molecules - done]
        sm = NM.normalise_many(d.smiles.tolist(), workers)[0] if normalise else d.smiles.tolist()
        acc.add(sm, d.residues.to_numpy())
        done += len(d)
        print("  pool: %s molecules checked (%.0f s)" % (format(done, ","), time.time() - t0), flush=True)
        if max_molecules and done >= max_molecules:
            break
    return acc


def csv_smiles(path, col):
    d = pd.read_csv(path, sep=None, engine="python")
    if col is None:
        for c in d.columns:
            if str(c).strip().lower() in ("smiles", "canonical_smiles", "isomeric_smiles", "merge_smiles", "smi"):
                col = c
                break
    assert col is not None and col in d.columns, "%s: give --col (columns: %s)" % (path, list(d.columns))
    return d[col].dropna().astype(str).drop_duplicates().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="atombpe_v2048", help="released | fixed | a folder or name under checkpoints/student_tokenizer")
    ap.add_argument("--pool", default=os.path.join(CORPUS_DIR, "pool.parquet"))
    ap.add_argument("--no-pool", action="store_true")
    ap.add_argument("--max-molecules", type=int, default=0, help="only the first N pool rows (0 = all)")
    ap.add_argument("--no-bench", action="store_true", help="skip the ten benchmarks")
    ap.add_argument("--bench", nargs="*", default=BENCH)
    ap.add_argument("--csv", nargs="*", default=[], help="other SMILES files (csv / tsv with a header)")
    ap.add_argument("--col", default=None, help="SMILES column of the --csv files (default: guessed)")
    ap.add_argument("--no-normalise", dest="normalise", action="store_false",
                    help="check the SMILES as they are (default: the student's input form, neutralised + RDKit canonical, src/student/normalise.py)")
    ap.add_argument("--raw-too", action="store_true", help="for --csv files and benchmarks add a row for the raw strings next to the normalised one")
    ap.add_argument("--workers", type=int, default=8, help="processes for the normalisation (lab rule: at most half of the CPU)")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "student_tokenizer_check"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tok = T.load_tokenizer(a.tokenizer)
    print("tokenizer: %s (vocabulary %d)" % (tok.name, tok.vocab_size))
    rows, spelled = [], []

    def finish(acc, name):
        rows.append(acc.row(name))
        if acc.fb:
            spelled.extend({"dataset": name, "atom": k, "molecules": v} for k, v in acc.fb.most_common(30))
            print("  %-22s spelled-out atoms (molecules): %s" % (name, acc.fb.most_common(6)))

    if not a.no_pool:
        assert os.path.exists(a.pool), "no pool at %s" % a.pool
        finish(pool_acc(tok, a.pool, a.max_molecules, a.normalise, a.workers), "POOL" + (" (first %d)" % a.max_molecules if a.max_molecules else " (all)"))
    sets = {}
    if not a.no_bench:
        import benchmarks as BM
        for b in a.bench:
            sets[b] = sorted(set(BM.load(b).smiles.dropna().astype(str)))
    for f in a.csv:
        sets[os.path.basename(f)] = csv_smiles(f, a.col)
    from eval_single import n_residues

    def residues(sm, peptides):
        return np.array([max(n_residues(s), 1) for s in sm]) if peptides else None

    for name, sm in sets.items():
        peptides = name in BENCH
        views = [("[normalised]", NM.normalise_many(sm, a.workers))] if a.normalise else [("", (sm, 0))]
        if a.normalise and a.raw_too:
            views.insert(0, ("[raw]", (sm, 0)))
        for tag, (x, bad) in views:
            acc = Acc(tok)
            acc.add(x, residues(x, peptides))
            finish(acc, (name + " " + tag).strip())
            if bad:
                print("  %s: %d SMILES could not be parsed by RDKit and were left as they are" % (name, bad))
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "check_%s.csv" % tok.name), index=False)
    pd.DataFrame(spelled).to_csv(os.path.join(a.out, "spelled_out_%s.csv" % tok.name), index=False)
    cols = [c for c in ("dataset", "n", "pct_non_ascii", "pct_mol_unk", "pct_mol_spelled_out", "roundtrip_fail", "chars_per_token", "tok_per_res",
                        "len_median", "len_p99", "len_max", "pct_gt128") if c in df.columns]
    print("\n" + df[cols].to_string(index=False, float_format=lambda x: "%.2f" % x))
    bad = df[(df.pct_non_ascii > 0) | (df.pct_mol_unk > 0) | (df.get("roundtrip_fail", 0) > 0)]
    print("\n%s" % ("every dataset: no non-ASCII character, no [UNK], no round-trip failure" if bad.empty else
                    "CHECK THESE (non-ASCII, [UNK] or round-trip failure): " + ", ".join(bad.dataset)))
    print("wrote", a.out)


if __name__ == "__main__":
    main()

"""Profile the cleaned corpus by source, kind and residue count (read-only).

    python src/corpus/profile_corpus.py                 # clean.parquet as saved
    python src/corpus/profile_corpus.py --thresh 0.8    # re-cut scored.parquet instead

Answers two questions before the corpus is subsampled: where to cap the peptide
length (the PepLand range is ~30 residues, the corpus window is 50) and how large the
final corpus N can be for a non-canonical share p. N_max = min(S/p, C/(1-p)) with S, C
the non-canonical and canonical molecules under the cap: an UPPER BOUND, because
near-duplicate removal has not run yet.

Writes $RESULTS_DIR/corpus_profile/: profile.json, by_residues.csv, caps.csv, tokens.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import CORPUS_DIR, PCLM_DIR, RESULTS_DIR  # noqa: E402

EDGES = [1, 4, 9, 14, 19, 24, 30, 40, 50]
LABELS = ["2-4", "5-9", "10-14", "15-19", "20-24", "25-30", "31-40", "41-50"]
CAPS = (20, 25, 30, 40, 50)
SHARES = (0.15, 0.25, 0.35)
STUDENT_PARAMS = 10.5e6


def n(x):
    return format(int(x), ",")


def show(title, t, ints=()):
    t = t.copy()
    for c in ints:
        t[c] = t[c].map(n)
    print("\n== " + title)
    print(t.to_string(index=False))


def load(thresh):
    import pyarrow.parquet as pq
    path = os.path.join(CORPUS_DIR, "scored.parquet" if thresh else "clean.parquet")
    cols = ["residues", "kind", "cyclic", "source"] + (["bench_sim"] if thresh else [])
    df = pq.read_table(path, columns=cols).to_pandas(categories=["kind", "source"])
    if thresh:
        df = df[df.bench_sim < thresh - 1e-6].drop(columns="bench_sim")
    df["bin"] = pd.cut(df.residues, bins=EDGES, labels=LABELS)
    assert df["bin"].notna().all(), "residue counts outside 2-50: not the cleaned window"
    return path, df


def by_source(df):
    g = df.assign(gt30=df.residues > 30).groupby(["source", "kind"], observed=True)
    t = pd.DataFrame({"n": g.size(), "median": g.residues.median(), "p90": g.residues.quantile(0.9),
                      "gt30": g.gt30.sum()}).reset_index()
    t["pool_pct"] = (100 * t.n / len(df)).round(1)
    t["gt30_pct"] = (100 * t.gt30 / t.n).round(1)
    return t


def by_bin(df):
    p = df.groupby(["bin", "kind"], observed=True).size().unstack(fill_value=0)
    tot = p.sum(axis=1)
    p["nc_pct"] = (100 * p.noncanonical / tot).round(1)
    p["cum_pct"] = (100 * tot.cumsum() / len(df)).round(1)
    return p.reset_index()


def caps(df):
    rows = []
    for L in CAPS:
        s = df[df.residues <= L]
        c, k = int((s.kind == "canonical").sum()), int((s.kind == "noncanonical").sum())
        row = {"cap": L, "pool": len(s), "kept_pct": round(100 * len(s) / len(df), 1), "canonical": c,
               "noncanonical": k, "nc_pct": round(100 * k / max(len(s), 1), 1),
               "nc_cyclic": int(((s.kind == "noncanonical") & s.cyclic).sum())}
        for p in SHARES:
            row["N_max@%d%%" % round(100 * p)] = int(min(k / p, c / (1 - p)))
        rows.append(row)
    return pd.DataFrame(rows)


def read_smiles(path, positions):
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


def tokens(path, df, tok_path, k):
    try:
        from tokenizers import Tokenizer
        tk = Tokenizer.from_file(tok_path)
    except Exception as e:
        print("\n(token section skipped: %s)" % e)
        return None
    rng, parts = np.random.default_rng(0), []
    for kind in ("canonical", "noncanonical"):
        sub = df[df.kind == kind]
        pos = rng.choice(sub.index.to_numpy(), size=min(k, len(sub)), replace=False)
        ids = [len(e.ids) for e in tk.encode_batch(read_smiles(path, pos))]
        parts.append(pd.DataFrame({"kind": kind, "residues": df.residues.loc[pos].to_numpy(), "tokens": ids}))
    return pd.concat(parts, ignore_index=True)


def token_tables(t):
    rows = []
    for kind, g in t.groupby("kind"):
        for cap in (30, 50):
            s = g[g.residues <= cap]
            rows.append({"kind": kind, "cap": cap, "sampled": len(s), "mean": round(s.tokens.mean(), 1),
                         "median": s.tokens.median(), "p90": s.tokens.quantile(0.9),
                         "p99": s.tokens.quantile(0.99), "max": s.tokens.max(),
                         "tok_per_res": round(s.tokens.sum() / s.residues.sum(), 2),
                         **{">%d tok %%" % L: round(100 * (s.tokens > L).mean(), 1) for L in (128, 192, 256)}})
    st = pd.DataFrame(rows)
    mean = {(r.kind, r.cap): r["mean"] for _, r in st.iterrows()}
    budget = []
    for cap in (30, 50):
        for p in (0.15, 0.25):
            for N in (2e6, 4e6):
                tok = N * (p * mean[("noncanonical", cap)] + (1 - p) * mean[("canonical", cap)])
                budget.append({"cap": cap, "nc_share": p, "N": int(N), "tokens_M": round(tok / 1e6),
                               "tok_per_param": round(tok / STUDENT_PARAMS, 1)})
    return st, pd.DataFrame(budget)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresh", type=float, default=None,
                    help="re-cut scored.parquet at this Tanimoto instead of reading clean.parquet")
    ap.add_argument("--tokenizer", default=os.path.join(PCLM_DIR, "tokenizer.json"))
    ap.add_argument("--token-sample", type=int, default=100_000, help="molecules per kind to tokenise")
    ap.add_argument("--out", default=os.path.join(RESULTS_DIR, "corpus_profile"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    path, df = load(a.thresh)
    nc = int((df.kind == "noncanonical").sum())
    print("pool: %s molecules from %s%s | canonical %s | non-canonical %s (%.1f%%) | above 30 residues: %s (%.1f%%)"
          % (n(len(df)), os.path.basename(path), " cut at %.2f" % a.thresh if a.thresh else "",
             n(len(df) - nc), n(nc), 100 * nc / len(df), n((df.residues > 30).sum()),
             100 * (df.residues > 30).mean()))

    src = by_source(df)
    show("source x kind", src, ints=("n", "gt30"))
    binned = by_bin(df)
    show("residue bins x kind", binned, ints=("canonical", "noncanonical"))
    nc_df = df[df.kind == "noncanonical"]
    g = nc_df.assign(le30=nc_df.residues <= 30).groupby(["source", "cyclic"], observed=True)
    comp = pd.DataFrame({"n": g.size(), "median": g.residues.median(), "le30": g.le30.sum()}).reset_index()
    show("non-canonical only: source x cyclic", comp, ints=("n", "le30"))
    cap_t = caps(df)
    show("length caps: what is left, and N_max for a non-canonical share (UPPER BOUND, before near-dup removal)",
         cap_t, ints=[c for c in cap_t.columns if c not in ("cap", "kept_pct", "nc_pct")])

    df.groupby(["source", "kind", "cyclic", "bin"], observed=True).size().reset_index(name="n") \
        .to_csv(os.path.join(a.out, "by_residues.csv"), index=False)
    cap_t.to_csv(os.path.join(a.out, "caps.csv"), index=False)

    tok = tokens(path, df, a.tokenizer, a.token_sample)
    if tok is not None:
        st, budget = token_tables(tok)
        show("tokens per molecule (PeptideCLM-2 tokenizer, sample of %s per kind)" % n(a.token_sample), st)
        show("tokens per epoch if the corpus were N molecules (hypothetical; see N_max above)", budget, ints=("N",))
        pd.concat([st.assign(table="per_molecule"), budget.assign(table="epoch_budget")]) \
            .to_csv(os.path.join(a.out, "tokens.csv"), index=False)

    head = {"file": path, "thresh": a.thresh, "pool": len(df), "noncanonical": nc,
            "nc_pct": round(100 * nc / len(df), 2), "above_30": int((df.residues > 30).sum()),
            "above_30_pct": round(100 * float((df.residues > 30).mean()), 2),
            "by_source_kind": src.to_dict("records"), "caps": cap_t.to_dict("records")}
    with open(os.path.join(a.out, "profile.json"), "w") as fh:
        json.dump(head, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()

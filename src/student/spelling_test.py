#!/usr/bin/env python
"""Spelling-robustness test of a trained student (2026-11-23).

One peptide is written several ways: the CANONICAL string (what the student reads at inference) and K RANDOM spellings of the same molecule (RDKit doRandom:
random atom order and branch order, same molecule; the augmentation of the st6 recipe). The 1024-d output z (and the 1280-d pooled vector `pre`) of every
string is computed and these cosines are printed:
    canonical vs random 1                 (the `spell` number of the training log)
    random 1  vs random 2                 (two different writings, neither of them canonical)
    canonical vs a DIFFERENT peptide      (the yardstick: how far apart two really different molecules are)
First the worked examples (the strings, the token counts, the cosines), then the same over N peptides of the benchmark files, written to a CSV.

NO normalisation is applied to the random spellings: they go straight to the tokenizer (exactly as in training). The benchmark pipeline would turn every spelling
back into the canonical one before the model (normalise_many), which would hide the effect this test is about. The canonical string is the normalised one
(uncharged, RDKit canonical) as in training and in the export.

Runs on the CPU in float32 (the training log's `spell` used bfloat16 on the GPU: tiny numeric differences). RDKit's random writer cannot be seeded: two runs
give different spellings; the CSV is the record.

  python src/student/spelling_test.py --ckpt runs/st6f_best.pt --tok runs/st6f_tokenizer
  python src/student/spelling_test.py --ckpt runs/st6f_best.pt --tok runs/st6f_tokenizer --seq YGGFL --n 0       (one peptide from a sequence, no batch)
  python src/student/spelling_test.py ... --smiles "CC(C)C[C@H](N)C(=O)O" --k 4
Options: --n N (peptides of the benchmark files, default 300; 0 = none)  --k K (random spellings per peptide, default 3)  --out CSV
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import normalise as NM  # noqa: E402
import tokenizer as T  # noqa: E402
from model import PAD, Student  # noqa: E402

BENCH_FILES = ["CellPPD_test.csv", "THPep_test.csv", "amp_test.csv"]


def load(ckpt, tok_dir):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg, args = ck["cfg"], ck["args"]
    model = Student(**cfg)
    model.load_state_dict(ck["model"])
    del model.train_only
    model.eval()
    folder = tok_dir or args["tok"]
    if not os.path.isdir(folder):
        sys.exit("the tokenizer folder %r does not exist here (the checkpoint says %r). Pass the tokenizer folder with --tok." % (folder, args["tok"]))
    tok = T.load_tokenizer(folder)
    return model, tok, ck.get("epoch", "?"), args.get("canon_max_len", 128), cfg["max_len"]


@torch.no_grad()
def embed(model, ids_list, bs):
    """list of token-id lists (with [CLS] / [SEP]) -> (z [N, 1024], pre [N, 1280]) float32 numpy."""
    zs, ps = [], []
    for i in range(0, len(ids_list), bs):
        b = ids_list[i:i + bs]
        L = max(len(x) for x in b)
        arr = np.full((len(b), L), PAD, np.int64)
        for j, x in enumerate(b):
            arr[j, :len(x)] = x
        x = torch.from_numpy(arr)
        o = model(x, x != PAD)
        zs.append(o["z"].float().numpy())
        ps.append(o["pre"].float().numpy())
    return np.concatenate(zs), np.concatenate(ps)


def cos(a, b):
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def same_molecule(canon, spelling):
    from rdkit import Chem
    m = Chem.MolFromSmiles(spelling)
    return m is not None and Chem.MolToSmiles(m) == canon


def prepare(smiles, tok, k, canon_max, rand_max):
    """-> dict list: canonical string, k random spellings, their token-id lists. Molecules RDKit cannot read, or whose canonical / random spellings would be cut, are dropped (counted)."""
    norm, _ = NM.normalise_many([str(s) for s in smiles], workers=1)
    keep = [i for i, s in enumerate(norm) if s]
    norm = [norm[i] for i in keep]
    rs, _ = NM.random_spellings(norm, k, workers=1)
    out, n_long, n_diff = [], 0, 0
    for c, r in zip(norm, rs):
        ic = tok.encode(c)
        ir = [tok.encode(x) for x in r]
        if len(ic) + 2 > canon_max or any(len(x) + 2 > rand_max for x in ir):
            n_long += 1
            continue
        if not all(same_molecule(c, x) for x in r):
            n_diff += 1
            continue
        out.append({"canonical": c, "random": r, "ids_c": tok.with_special(ic), "ids_r": [tok.with_special(x) for x in ir]})
    return out, len(smiles) - len(keep), n_long, n_diff


def score(model, items, bs):
    """per peptide: cosines of z and of pre. Returns a DataFrame."""
    k = len(items[0]["ids_r"])
    flat = [it["ids_c"] for it in items] + [x for it in items for x in it["ids_r"]]
    z, p = embed(model, flat, bs)
    n = len(items)
    res = {}
    for name, e in (("z", z), ("pre", p)):
        c = e[:n]
        rr = e[n:].reshape(n, k, -1)                                                  # the k random spellings of peptide i follow each other
        r = [rr[:, j] for j in range(k)]
        res["%s_canon_rand1" % name] = cos(c, r[0])
        res["%s_canon_randmean" % name] = np.mean([cos(c, r[j]) for j in range(k)], 0)
        if k >= 2:
            res["%s_rand1_rand2" % name] = cos(r[0], r[1])
            pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
            res["%s_randpairs_mean" % name] = np.mean([cos(r[a], r[b]) for a, b in pairs], 0)
        other = np.roll(np.arange(n), 1) if n > 1 else np.zeros(1, int)           # a different peptide of the list
        res["%s_canon_other" % name] = cos(c, c[other])
    return pd.DataFrame(res)


def worked(model, tok, smiles, label, k, canon_max, rand_max):
    items, bad, n_long, n_diff = prepare([smiles], tok, k, canon_max, rand_max)
    print("\n" + "=" * 100 + "\nworked example: %s" % label)
    if not items:
        print("  dropped (unreadable %d, too long %d, spelling not the same molecule %d)" % (bad, n_long, n_diff))
        return
    it = items[0]
    show = lambda s: s if len(s) <= 160 else s[:157] + "..."
    print("  canonical : %s  (%d tokens)" % (show(it["canonical"]), len(it["ids_c"]) - 2))
    for j, (r, ids) in enumerate(zip(it["random"], it["ids_r"]), 1):
        print("  random %d  : %s  (%d tokens)" % (j, show(r), len(ids) - 2))
    # the yardstick: a second, unrelated peptide
    from rdkit import Chem
    other = Chem.MolToSmiles(Chem.MolFromSequence("LFK"))
    o_items, *_ = prepare([other], tok, 1, canon_max, rand_max)
    flat = [it["ids_c"]] + it["ids_r"] + [o_items[0]["ids_c"]]
    z, p = embed(model, flat, 1)                                        # one string per forward pass: no padding at all
    for name, e in (("z (1024-d output)", z), ("pre (1280-d pooled)", p)):
        c, r, o = e[0:1], e[1:1 + k], e[1 + k:2 + k]
        print("  %s" % name)
        print("    canonical vs random 1            : %.5f" % cos(c, r[0:1])[0])
        if k >= 2:
            print("    random 1  vs random 2            : %.5f" % cos(r[0:1], r[1:2])[0])
        print("    canonical vs every random (mean) : %.5f" % np.mean([cos(c, r[j:j + 1])[0] for j in range(k)]))
        print("    canonical vs a different peptide : %.5f   (Leu-Phe-Lys, for scale)" % cos(c, o)[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=os.path.join(HERE, "..", "..", "runs", "st6f_best.pt"))
    ap.add_argument("--tok", default=None, help="the tokenizer folder (default: the path stored in the checkpoint = a server path)")
    ap.add_argument("--smiles", default=None, help="one peptide as SMILES (worked example)")
    ap.add_argument("--seq", default=None, help="one peptide as one-letter sequence (worked example), e.g. YGGFL")
    ap.add_argument("--n", type=int, default=300, help="peptides of the benchmark test files for the summary (0 = none)")
    ap.add_argument("--k", type=int, default=3, help="random spellings per peptide")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0, help="which peptides are drawn from the benchmark files (the spellings themselves cannot be seeded)")
    ap.add_argument("--out", default=None, help="CSV with one row per peptide (default runs/spelling_test.csv)")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    assert a.k >= 2, "--k must be at least 2 (random 1 vs random 2)"

    model, tok, epoch, canon_max, rand_max = load(a.ckpt, a.tok)
    print("student: %s | epoch %s | %.3fM parameters | tokenizer %s (%d tokens) | canonical cut at %d tokens, random spellings at %d | %d random spellings per peptide"
          % (os.path.basename(a.ckpt), epoch, sum(p.numel() for p in model.parameters()) / 1e6, tok.name, tok.vocab_size, canon_max, rand_max, a.k))

    from rdkit import Chem
    examples = []
    if a.smiles:
        examples.append(("given SMILES", a.smiles))
    elif a.seq:
        m = Chem.MolFromSequence(a.seq)
        assert m is not None, "RDKit cannot read the sequence %r" % a.seq
        examples.append(("sequence " + a.seq, Chem.MolToSmiles(m)))
    else:
        examples.append(("Leu-enkephalin, sequence YGGFL (5 residues)", Chem.MolToSmiles(Chem.MolFromSequence("YGGFL"))))
        examples.append(("first 12 residues of magainin 2, GIGKFLHSAKKF", Chem.MolToSmiles(Chem.MolFromSequence("GIGKFLHSAKKF"))))
    for label, s in examples:
        worked(model, tok, s, label, a.k, canon_max, rand_max)

    if a.n > 0:
        bdir = os.path.join(HERE, "..", "..", "data", "benchmarks")
        pool = pd.concat([pd.read_csv(os.path.join(bdir, f), usecols=["smiles"]).assign(bench=f.split("_")[0]) for f in BENCH_FILES if os.path.exists(os.path.join(bdir, f))],
                         ignore_index=True).dropna().drop_duplicates("smiles")
        pick = pool.sample(n=min(a.n, len(pool)), random_state=a.seed).reset_index(drop=True)
        items, bad, n_long, n_diff = prepare(pick.smiles.tolist(), tok, a.k, canon_max, rand_max)
        print("\n" + "=" * 100 + "\nbatch: %d peptides drawn from %s (seed %d) -> %d used (unreadable %d, canonical or a spelling too long for the student %d, a spelling that is not the same molecule %d)"
              % (len(pick), "+".join(f.split("_")[0] for f in BENCH_FILES), a.seed, len(items), bad, n_long, n_diff))
        df = score(model, items, a.bs)
        df.insert(0, "canonical", [it["canonical"] for it in items])
        df.insert(1, "tokens", [len(it["ids_c"]) - 2 for it in items])
        df.insert(2, "random1", [it["random"][0] for it in items])
        out = a.out or os.path.join(HERE, "..", "..", "runs", "spelling_test.csv")
        df.to_csv(out, index=False)
        print("%-44s %9s %9s %9s %9s" % ("cosine", "mean", "median", "min", "5th pct"))
        rows = [("z   canonical vs random 1", "z_canon_rand1"), ("z   random 1 vs random 2", "z_rand1_rand2"), ("z   canonical vs different peptide", "z_canon_other"),
                ("pre canonical vs random 1", "pre_canon_rand1"), ("pre random 1 vs random 2", "pre_rand1_rand2"), ("pre canonical vs different peptide", "pre_canon_other")]
        for lab, col in rows:
            v = df[col].to_numpy()
            print("%-44s %9.5f %9.5f %9.5f %9.5f" % (lab, v.mean(), np.median(v), v.min(), np.percentile(v, 5)))
        v = df["z_canon_rand1"].to_numpy()
        print("share of peptides with z cosine (canonical vs random 1) above 0.99: %.1f%% | above 0.999: %.1f%%" % (100 * (v > 0.99).mean(), 100 * (v > 0.999).mean()))
        print("per-peptide CSV: %s" % os.path.abspath(out))
        worst = df.sort_values("z_canon_rand1").head(3)
        print("the three least stable peptides (z, canonical vs random 1):")
        for _, r in worst.iterrows():
            print("  %.4f  %d tokens  %s" % (r.z_canon_rand1, r.tokens, r.canonical[:110]))


if __name__ == "__main__":
    main()

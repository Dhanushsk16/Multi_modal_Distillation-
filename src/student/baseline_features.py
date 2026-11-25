"""Two NO-TRAINING baselines for the student's benchmark numbers (a minute; GPU or CPU). They show how much of a score comes from the tokenizer and the
architecture alone, not from anything learned.

    python src/student/baseline_features.py --like stc [--name st0] [--seed 0]

Written for both suites (the files the harness reads, like export_student.py):
    features/<suite>/teacher_bot.npz          X = how often each of the 1024 tokens occurs in the (neutralised, canonical) SMILES: a bag-of-tokens, NO model.
                                              With the learned tokenizer most tokens are residue-sized pieces, so this is close to an amino-acid composition.
    features/<suite>/teacher_<name>.npz       the 1024-d output of an UNTRAINED student: the architecture, tokenizer and (random) initialisation of run --like,
    features/<suite>/teacher_<name>_pre.npz   and its pooled 1280-d vector. Random weights, seed --seed.
Then:  LEARNERS=linear bash run_student.sh harness bot      and      ... harness st0 st0_pre
(the harness adds the teacher v3_base). Nothing here sees a label or a trained weight.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import normalise as NM  # noqa: E402
import tokenizer as T  # noqa: E402
from export_student import SUITES, embed  # noqa: E402
from model import Student, count_params  # noqa: E402
from train_student import STUDENT_DIR  # noqa: E402


def save(path, smiles, X, ok):
    np.savez_compressed(path + ".tmp.npz", smiles=smiles, X=X, ok=ok)
    os.replace(path + ".tmp.npz", path)
    print("  wrote", path, X.shape, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--like", required=True, help="a trained student run whose tokenizer and architecture are copied (its weights are NOT used)")
    ap.add_argument("--name", default="st0", help="name of the untrained student")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bot-name", default="bot")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default=STUDENT_DIR)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(os.path.join(a.out_dir, a.like, "best.pt"), map_location="cpu", weights_only=False)
    cfg, args = ck["cfg"], ck["args"]
    tok = T.load_tokenizer(args["tok"])
    max_len = args.get("canon_max_len", args.get("max_len", cfg["max_len"]))
    torch.manual_seed(a.seed)
    model = Student(**cfg).to(dev).eval()                               # random initialisation: never trained
    print("untrained student '%s': %.3fM parameters, seed %d, architecture and tokenizer of '%s' (%s)" % (
        a.name, count_params(model)[0] / 1e6, a.seed, a.like, args["tok"]), flush=True)
    for key, folder in SUITES.items():
        src = os.path.join(folder, "ecfp.npz")
        if not os.path.exists(src):
            print("skipped %s: %s is missing" % (key, src))
            continue
        smiles = np.load(src, allow_pickle=True)["smiles"]
        norm, bad = NM.normalise_many([str(s) for s in smiles], a.workers)
        enc = tok.encode_batch(norm)
        V = tok.vocab_size
        X = np.zeros((len(enc), V), np.float32)
        for i, e in enumerate(enc):
            if len(e):
                X[i] = np.bincount(np.asarray(e), minlength=V)[:V]
        ok = np.array([len(e) > 0 for e in enc])
        print("%-8s %d molecules | unparsable %d | bag of tokens: mean %.1f tokens per molecule, %d of %d tokens occur" % (
            key, len(smiles), bad, X.sum(1).mean(), int((X.sum(0) > 0).sum()), V), flush=True)
        save(os.path.join(folder, "teacher_%s.npz" % a.bot_name), smiles, X, ok)
        z, pre, ok2, cut = embed(model, tok, norm, max_len, dev)
        save(os.path.join(folder, "teacher_%s.npz" % a.name), smiles, z, ok2)
        save(os.path.join(folder, "teacher_%s_pre.npz" % a.name), smiles, pre, ok2)
    print("next: harness for %s and %s %s_pre" % (a.bot_name, a.name, a.name))


if __name__ == "__main__":
    main()

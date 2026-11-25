"""The student's embeddings of the benchmark molecules, in the files the evaluation harness reads.

    python src/student/export_student.py --name st3 [--use-last] [--suites both|single|pepland]

For every benchmark molecule of both suites (the SMILES lists of features/<suite>/ecfp.npz) the molecule is neutralised and canonicalised exactly as
the pool was (src/student/normalise.py), tokenised with the student's tokenizer, and embedded with the EMA weights of best.pt. Written to

    features/<suite>/teacher_<name>.npz        X = the 1024-d output          (the harness: --teachers <name>)
    features/<suite>/teacher_<name>_pre.npz    X = the 1280-d pooled vector before the last linear layer   (--teachers <name>_pre)

(the prefix `teacher_` is only the harness's file convention). Each file has smiles (the benchmark's own spelling, same order as every other
file), X float32, ok. Molecules longer than the student's maximum length are cut (printed). Nothing here sees a label.
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
from paths import FEAT_DIR  # noqa: E402
import normalise as NM  # noqa: E402
import tokenizer as T  # noqa: E402
from model import PAD, Student, count_params  # noqa: E402
from train_student import STUDENT_DIR  # noqa: E402

SUITES = {"single": FEAT_DIR, "pepland": os.path.join(os.path.dirname(FEAT_DIR), "pepland_suite")}


@torch.no_grad()
def embed(model, tok, smiles, max_len, dev, bs=512):
    """-> z [N, 1024], pre [N, 1280], ok [N], number cut at max_len"""
    ids = [tok.with_special(e, max_len) for e in tok.encode_batch(smiles)]
    full = [len(e) + 2 for e in tok.encode_batch(smiles)]
    order = np.argsort([len(x) for x in ids], kind="stable")
    z = np.zeros((len(ids), model.cfg["out"]), np.float32)
    pre = np.zeros((len(ids), model.cfg["n_query"] * model.cfg["d"]), np.float32)
    for i in range(0, len(ids), bs):
        sel = order[i:i + bs]
        L = max(len(ids[j]) for j in sel)
        x = torch.zeros((len(sel), L), dtype=torch.long)
        for r, j in enumerate(sel):
            x[r, :len(ids[j])] = torch.tensor(ids[j])
        x = x.to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            o = model(x, x != PAD)
        z[sel] = o["z"].float().cpu().numpy()
        pre[sel] = o["pre"].float().cpu().numpy()
    ok = np.array([len(x) >= 3 for x in ids])
    return z, pre, ok, int(sum(f > max_len for f in full))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--use-last", action="store_true", help="the raw weights of last.pt instead of the EMA weights of best.pt (not recommended)")
    ap.add_argument("--epoch", type=int, default=0, help="use the numbered snapshot epoch_NN.pt (train_student.py --save-every) instead of best.pt; "
                    "written under the name <name>_eNN, so the export of best.pt is not overwritten")
    ap.add_argument("--suites", choices=["both", "single", "pepland"], default="both")
    ap.add_argument("--out-dir", default=STUDENT_DIR)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rdir = os.path.join(a.out_dir, a.name)
    tag = a.name if a.epoch <= 0 else "%s_e%02d" % (a.name, a.epoch)           # the name of the written files
    best = torch.load(os.path.join(rdir, "best.pt" if a.epoch <= 0 else "epoch_%02d.pt" % a.epoch), map_location="cpu", weights_only=False)
    cfg, args = best["cfg"], best["args"]
    if a.use_last:
        ck = torch.load(os.path.join(rdir, "last.pt"), map_location="cpu", weights_only=False)
        state = ck["model"]
    else:
        ck = best
        state = best["model"]
    model = Student(**cfg).to(dev)
    model.load_state_dict(state)
    model.eval()
    print("student %s: %.3fM parameters at inference | weights of epoch %s | tokenizer %s" % (
        tag, count_params(model)[0] / 1e6, ck.get("epoch", "?"), args["tok"]), flush=True)
    tok = T.load_tokenizer(args["tok"])
    max_len = args.get("canon_max_len", args.get("max_len", cfg["max_len"]))
    for key, folder in SUITES.items():
        if a.suites not in ("both", key):
            continue
        src = os.path.join(folder, "ecfp.npz")
        if not os.path.exists(src):
            print("skipped %s: %s is missing" % (key, src))
            continue
        smiles = np.load(src, allow_pickle=True)["smiles"]
        norm, bad = NM.normalise_many([str(s) for s in smiles], a.workers)
        z, pre, ok, cut = embed(model, tok, norm, max_len, dev)
        print("%-8s %d molecules | unparsable %d | cut at %d tokens: %d | usable %d" % (key, len(smiles), bad, max_len, cut, int(ok.sum())), flush=True)
        for suffix, X in (("", z), ("_pre", pre)):
            out = os.path.join(folder, "teacher_%s%s.npz" % (tag, suffix))
            np.savez_compressed(out + ".tmp.npz", smiles=smiles, X=X, ok=ok)
            os.replace(out + ".tmp.npz", out)
            print("  wrote", out, X.shape, flush=True)
    print("next: python src/teacher/eval_harness.py --teachers %s %s_pre --out harness_%s" % (tag, tag, tag))


if __name__ == "__main__":
    main()

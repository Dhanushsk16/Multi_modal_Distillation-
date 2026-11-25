"""The PeptideCLM-2 authors' PepMSND post-training (train_pepmsnd_kan_paperstyle.py), with OUR TEACHER as the backbone and its trunk trained.

    python src/teacher/pepmsnd_ft_kan_v3.py --name v3_mtr                          # the four-input teacher, the authors' protocol, seed 103, 10 folds
    python src/teacher/pepmsnd_ft_kan_v3.py --name v3_mtr --select inner           # the same without the test-fold leak (an inner validation split)
    python src/teacher/pepmsnd_ft_kan_v3.py --name v3_mtr --freeze-epochs 100000   # control: the teacher never trained, only the KAN head
    python src/teacher/pepmsnd_ft_kan_v3.py --name v3_mtr --init scratch           # control: the same architecture from random weights
    python src/teacher/pepmsnd_ft_kan_v3.py --name v3_mtr --folds 1 2 --epochs 5   # a quick look
    python src/teacher/pepmsnd_ft_kan_v3.py --name v3_mtr --check                  # only: does every molecule have a vector in every view? (seconds)

What is copied from the authors, and from where (external/peptideclm2_authors/.../scripts/train_pepmsnd_kan_paperstyle.py, imported, not retyped):
  * the data: their X_train<k>.csv / X_test<k>.csv (10 cluster folds, 635 peptides, SMILES + 140 descriptor columns + Species / Environment + label);
    build_paperstyle_features (descriptors clipped at the 0.999 quantile, z-scored with the training rows; Species / Environment as two numbers);
  * the network (PeptideCLMBackbonePlusPaperKAN, lines 194-288): backbone vector -> Linear 128 + ReLU; descriptors -> PaperStyleKANBranch (two KAN
    layers 128, BatchNorm); [both] -> KAN 256 - ReLU - BatchNorm - Dropout 0.1 - KAN 64 - ReLU; [that, Species, Environment] -> KAN 1; BCE with logits.
    Their own kan.py (KANLinear). The layers are re-declared here in the same order and sizes (a Lightning module cannot be reused without its backbone);
  * the training: AdamW with two groups (backbone lr 1e-5, head lr 1e-3, torch's default weight decay), batch 32, at most 120 epochs, the backbone
    frozen for the first 3 epochs and trained after that, early stopping with patience 20 on val_mcc_best = the best-threshold MCC (their
    _best_mcc_threshold, imported), the best epoch is the model, its probabilities are saved with predicted_label = probability >= 0.5.
What differs, because the backbone is another network:
  * the backbone is the teacher (the three or four frozen-encoder vectors -> the teacher's input encoders -> trunk, 1024-d latent, standardised with the
    run's latent_stats.npz as in finetune_v3.py), standing where PeptideCLM-2 plus its mean pool stands. ITS TRUNK AND INPUT ENCODERS ARE TRAINED (the
    decoders are not in the graph); the frozen encoders underneath (PeptideCLM-2, PepLand, ECFP, MTR) stay as cached vectors, they are never touched;
  * the frozen epochs use the backbone in eval mode throughout (their Lightning loop puts it back into train mode after the first validation: a quirk);
  * the teacher takes the vectors of features/single_modality/<view>.npz, looked up by SMILES (rows without all views are dropped and counted).
--select test (default, THE AUTHORS' PROTOCOL): val_mcc_best is computed on the TEST fold and picks the epoch: optimistic by construction, kept so that the
  number is comparable with the published 0.729 / 0.741 / 0.749. --select inner: 10% of the training rows are held out, val_mcc_best is computed there, the
  test fold is only predicted once at the end (the saved column thr_label uses the threshold found on the inner split).
Output: results/pepmsnd_ft_kan_<select>/<label>/seed<S>/fold<k>/preds_fold<k>.csv (true_label, predicted_prob, predicted_label, thr, label_thr), read
by src/repro/pepmsnd_summary.py (called at the end), which reports pooled and per-fold MCC and AUROC next to the published numbers.
"""
from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from paths import BENCH_DIR, CODE, FEAT_DIR, RESULTS_DIR  # noqa: E402

AUTH_DIR = os.path.join(CODE, "external", "peptideclm2_authors", "training", "03_PepMSND_training_code", "scripts")
BODY_SKIP = ("dec.", "hide")                         # decoders and the "hidden view" vectors are not in the graph of a latent forward pass


def load_authors(path):
    sys.path.insert(0, path)
    import train_pepmsnd_kan_paperstyle as A
    return A


class TeacherKAN(nn.Module):
    """PeptideCLMBackbonePlusPaperKAN with the teacher in the backbone's place (see the module docstring)."""

    def __init__(self, A, backbone, lat, kan_dim, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        d = backbone.cfg["latent"]
        use = lat is not None
        self.register_buffer("mu", torch.from_numpy(np.asarray(lat["mu"], np.float32)) if use else torch.zeros(0))
        self.register_buffer("sd", torch.from_numpy(np.asarray(lat["sd"], np.float32)) if use else torch.zeros(0))
        self.seq_proj = nn.Sequential(nn.Linear(d, 128), nn.ReLU())
        self.kan_branch = A.PaperStyleKANBranch(kan_dim)
        self.fusion_fc1 = A.KANLinear(128 + 128, 256)
        self.fusion_bn = nn.BatchNorm1d(256)
        self.fusion_fc2 = A.KANLinear(256, 64)
        self.out_fc = A.KANLinear(64 + 2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def backbone_params(self):
        return [p for n, p in self.backbone.named_parameters() if not n.startswith(BODY_SKIP)]

    def head_params(self):
        keep = ("seq_proj", "kan_branch", "fusion_fc1", "fusion_bn", "fusion_fc2", "out_fc")
        return [p for n, p in self.named_parameters() if n.startswith(keep)]

    def set_frozen(self, frozen):
        for p in self.backbone_params():
            p.requires_grad = not frozen
        self.backbone.eval() if frozen else self.backbone.train()

    def forward(self, xs, kan_features, meta_features):
        z = self.backbone.latent(xs)
        if self.mu.numel():
            z = (z - self.mu) / self.sd
        seq_feat = self.seq_proj(z)
        kan_feat = self.kan_branch(kan_features)
        x = torch.cat([seq_feat, kan_feat], dim=1)
        x = self.relu(self.fusion_fc1(x))
        x = self.fusion_bn(x)
        x = self.dropout(x)
        x = self.relu(self.fusion_fc2(x))
        x = torch.cat([x, meta_features], dim=1)
        return self.out_fc(x).squeeze(-1)


# ------------------------------------------------------------------------------------------- data
def view_rows(df, raw, inputs):
    """Row of every molecule in each view's feature file (-1 = not there) and the mask of molecules usable in ALL views."""
    ii = {}
    for v in inputs:
        d = raw[v][0]
        ii[v] = np.array([d.get(s, -1) for s in df["SMILES"].astype(str)], dtype=np.int64)
    ok = np.logical_and.reduce([(ii[v] >= 0) & raw[v][2][np.maximum(ii[v], 0)] for v in inputs])
    return ii, ok


def to_xs(ii, mask, raw, inputs, st, dev, prep):
    return [prep(v, torch.from_numpy(np.ascontiguousarray(raw[v][1][ii[v][mask]])).to(dev), st) for v in inputs]


def take(xs, idx):
    return [x[idx] for x in xs]


@torch.no_grad()
def predict(net, xs, kan, meta, bs=256):
    net.eval()
    out = []
    for i in range(0, len(kan), bs):
        out.append(net([x[i:i + bs] for x in xs], kan[i:i + bs], meta[i:i + bs]).float())
    return torch.cat(out)


def run_fold(A, V3, FV, a, fold, seed, model0, lat, st, raw, inputs, dev, out_dir):
    train_df = pd.read_csv(os.path.join(a.data_dir, "X_train%d.csv" % fold))
    test_df = pd.read_csv(os.path.join(a.data_dir, "X_test%d.csv" % fold))
    kept = []
    for name, df in (("train", train_df), ("test", test_df)):
        ii, ok = view_rows(df, raw, inputs)
        if (~ok).mean() > 0.05:
            miss = df.loc[~ok, "SMILES"].astype(str).head(3).tolist()
            raise SystemExit("fold %d %s: %d of %d molecules have no vector in all of %s under %s (the SMILES strings differ from the feature files'?), "
                             "e.g. %s" % (fold, name, int((~ok).sum()), len(df), inputs, a.feat_dir, miss))
        kept.append((df[ok].reset_index(drop=True), ii, ok))
    (train_df, ii_tr, ok_tr), (test_df, ii_te, ok_te) = kept
    if (~ok_tr).any() or (~ok_te).any():
        print("[INFO] dropped molecules without a vector in every view: train %d, test %d" % ((~ok_tr).sum(), (~ok_te).sum()), flush=True)
    feat = A.build_paperstyle_features(train_df, test_df, append_meta_to_kan=False, clip_quantile=0.999, max_abs_descriptor=1e6)
    print("[INFO] fold %d: train %d, test %d, KAN input dim %d (descriptors only), teacher inputs %s" % (fold, len(train_df), len(test_df),
          feat.kan_train.shape[1], inputs), flush=True)
    xs_tr = to_xs(ii_tr, ok_tr, raw, inputs, st, dev, V3.prep_t)
    xs_te = to_xs(ii_te, ok_te, raw, inputs, st, dev, V3.prep_t)
    kan_tr, kan_te = torch.from_numpy(feat.kan_train).to(dev), torch.from_numpy(feat.kan_test).to(dev)
    meta_tr, meta_te = torch.from_numpy(feat.meta_train).to(dev), torch.from_numpy(feat.meta_test).to(dev)
    y_tr = torch.tensor(train_df["label"].to_numpy(np.float32), device=dev)
    y_te = torch.tensor(test_df["label"].to_numpy(np.float32), device=dev)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if a.select == "inner":                                       # 10% of the training rows monitor the training; the test fold is not looked at
        perm = np.random.RandomState(seed * 1000 + fold).permutation(len(y_tr))
        nv = max(2, int(round(0.1 * len(y_tr))))
        iv, it = torch.from_numpy(perm[:nv]).to(dev), torch.from_numpy(perm[nv:]).to(dev)
        xs_v, kan_v, meta_v, y_v = take(xs_tr, iv), kan_tr[iv], meta_tr[iv], y_tr[iv]
        xs_tr, kan_tr, meta_tr, y_tr = take(xs_tr, it), kan_tr[it], meta_tr[it], y_tr[it]
    else:                                                         # the authors: the test fold is the validation set
        xs_v, kan_v, meta_v, y_v = xs_te, kan_te, meta_te, y_te

    if a.init == "scratch":                                       # a random network of the same shape: no trained latent scale to undo
        bb = V3.TeacherV3.from_cfg(model0.cfg, 0.0).to(dev)
        net_lat = None
    else:
        bb = V3.TeacherV3.from_cfg(model0.cfg, 0.0).to(dev)
        bb.load_state_dict({k: t.detach().clone() for k, t in model0.state_dict().items()})
        net_lat = lat
    for p in bb.parameters():
        p.requires_grad_(False)
    net = TeacherKAN(A, bb, net_lat, kan_tr.shape[1]).to(dev)
    for p in net.backbone_params():
        p.requires_grad_(True)
    opt = torch.optim.AdamW([{"params": net.backbone_params(), "lr": a.backbone_lr}, {"params": net.head_params(), "lr": a.head_lr}])
    best_thr_fn = A.PeptideCLMBackbonePlusPaperKAN._best_mcc_threshold

    best, best_ep, best_thr, best_state, bad, n = -float("inf"), -1, 0.5, None, 0, len(y_tr)
    for ep in range(a.epochs):
        t0 = time.time()
        frozen = ep < a.freeze_epochs
        net.train()
        net.set_frozen(frozen)
        perm = torch.randperm(n, device=dev)
        tl = 0.0
        for i in range(0, n, a.batch):
            idx = perm[i:i + a.batch]
            if len(idx) < 2:                                      # BatchNorm cannot train on one row (Lightning would stop here)
                continue
            opt.zero_grad(set_to_none=True)
            lo = net(take(xs_tr, idx), kan_tr[idx], meta_tr[idx])
            loss = F.binary_cross_entropy_with_logits(lo, y_tr[idx])
            loss.backward()
            opt.step()
            tl += loss.detach().item() * len(idx)
        lv = predict(net, xs_v, kan_v, meta_v)
        vloss = float(F.binary_cross_entropy_with_logits(lv, y_v))
        thr, mcc = best_thr_fn(y_v.cpu().numpy(), torch.sigmoid(lv).cpu().numpy())
        print("[epoch %3d] train_loss %.4f  val_loss %.4f  val_mcc_best %.4f (thr %.3f)%s  (%.1fs)"
              % (ep, tl / n, vloss, mcc, thr, "  [backbone frozen]" if frozen else "", time.time() - t0), flush=True)
        if mcc > best:
            best, best_ep, best_thr, bad = mcc, ep, thr, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                print("[INFO] early stopping after epoch %d (best epoch %d)" % (ep, best_ep), flush=True)
                break
    net.load_state_dict(best_state)
    prob = torch.sigmoid(predict(net, xs_te, kan_te, meta_te)).cpu().numpy()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "preds_fold%d.csv" % fold)
    pd.DataFrame({"true_label": y_te.cpu().numpy().astype(float), "predicted_prob": prob.astype(float), "predicted_label": (prob >= 0.5).astype(int),
                  "thr": best_thr, "label_thr": (prob >= best_thr).astype(int)}).to_csv(path, index=False)
    print("[INFO] Saved predictions: %s (best epoch %d, val_mcc_best %.4f, %s set)" % (path, best_ep, best, "test-fold" if a.select == "test" else "inner"), flush=True)
    return best_ep, len(prob)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="the teacher run (checkpoints/teacher/<name>), e.g. v3_mtr (four inputs)")
    ap.add_argument("--select", choices=("test", "inner"), default="test",
                    help="test = the authors' protocol (the test fold picks the epoch), inner = 10%% of the training rows pick it")
    ap.add_argument("--init", choices=("ckpt", "scratch"), default="ckpt", help="scratch = the teacher's architecture from random weights (control)")
    ap.add_argument("--folds", nargs="*", type=int, default=list(range(1, 11)))
    ap.add_argument("--seeds", nargs="*", type=int, default=[103])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--backbone-lr", type=float, default=None, help="default 1e-5 (the authors'); 1e-3 with --init scratch")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--freeze-epochs", type=int, default=None, help="default 3 (the authors'); 0 with --init scratch; a huge number = the teacher never trains")
    ap.add_argument("--data-dir", default=os.path.join(BENCH_DIR, "PepMSND_data"))
    ap.add_argument("--feat-dir", default=FEAT_DIR)
    ap.add_argument("--authors-dir", default=AUTH_DIR)
    ap.add_argument("--label", default=None, help="folder name under the results root (default: the teacher name, + _scratch / _frozen)")
    ap.add_argument("--root", default=None, help="results root (default results/pepmsnd_ft_kan_<select>)")
    ap.add_argument("--redo", action="store_true", help="recompute folds that already have predictions")
    ap.add_argument("--use-last", action="store_true", help="start from last.pt instead of best.pt")
    ap.add_argument("--no-summary", action="store_true")
    ap.add_argument("--check", action="store_true", help="only report how many molecules of the folds have a vector in every view, then stop (seconds)")
    a = ap.parse_args()
    if a.backbone_lr is None:
        a.backbone_lr = 1e-3 if a.init == "scratch" else 1e-5
    if a.freeze_epochs is None:
        a.freeze_epochs = 0 if a.init == "scratch" else 3
    label = a.label or (a.name + ("_scratch" if a.init == "scratch" else "") + ("_frozen" if a.freeze_epochs >= a.epochs else ""))
    root = a.root or os.path.join(RESULTS_DIR, "pepmsnd_ft_kan_%s" % a.select)

    A = load_authors(a.authors_dir)
    import finetune_v3 as FV
    import train_teacher_v3 as V3
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model0, st, lat = V3.load_model(a.name, dev, a.use_last)
    inputs = list(model0.inputs)
    raw, _ = FV.load_raw(a.feat_dir, inputs)
    print("teacher %s: inputs %s, latent %d-d, %.1fM parameters (%s) | label %s | selection: %s\n"
          "recipe (the authors'): AdamW, backbone lr %g / head lr %g, backbone frozen for the first %d epochs, batch %d, <= %d epochs, patience %d on val_mcc_best, "
          "seeds %s, folds %s\nresults: %s"
          % (a.name, inputs, model0.cfg["latent"], sum(p.numel() for p in model0.parameters()) / 1e6, "random weights" if a.init == "scratch" else "from the checkpoint",
             label, "the TEST fold picks the epoch (the authors' protocol, optimistic)" if a.select == "test" else "10% of the training rows pick the epoch (no test leak)",
             a.backbone_lr, a.head_lr, a.freeze_epochs, a.batch, a.epochs, a.patience, a.seeds, a.folds, root), flush=True)
    if a.check:                                                   # coverage only: does every molecule of the authors' folds have a vector in every view?
        for fold in a.folds:
            for name in ("train", "test"):
                df = pd.read_csv(os.path.join(a.data_dir, "X_%s%d.csv" % (name, fold)))
                _, ok = view_rows(df, raw, inputs)
                print("fold %2d %-5s: %d molecules, %d with a vector in all of %s%s" % (fold, name, len(df), int(ok.sum()), inputs,
                      "" if ok.all() else "   e.g. missing: %s" % df.loc[~ok, "SMILES"].astype(str).head(2).tolist()), flush=True)
        return
    for seed in a.seeds:
        for fold in a.folds:
            out_dir = os.path.join(root, label, "seed%d" % seed, "fold%d" % fold)
            if os.path.exists(os.path.join(out_dir, "preds_fold%d.csv" % fold)) and not a.redo:
                print("seed %d fold %d: done, skipped" % (seed, fold), flush=True)
                continue
            t0 = time.time()
            ep, n = run_fold(A, V3, FV, a, fold, seed, model0, lat, st, raw, inputs, dev, out_dir)
            print("seed %d fold %d: %d test molecules, best epoch %d, %.0fs" % (seed, fold, n, ep, time.time() - t0), flush=True)
    if not a.no_summary:
        extra = []
        for sd in sorted(os.listdir(os.path.join(root, label))) if os.path.isdir(os.path.join(root, label)) else []:
            parts = [pd.read_csv(os.path.join(root, label, sd, "fold%d" % k, "preds_fold%d.csv" % k)) for k in range(1, 11)
                     if os.path.exists(os.path.join(root, label, sd, "fold%d" % k, "preds_fold%d.csv" % k))]
            if len(parts) == 10:
                from sklearn.metrics import matthews_corrcoef
                y = np.concatenate([p.true_label.to_numpy() for p in parts]).astype(int)
                lt = np.concatenate([p.label_thr.to_numpy() for p in parts]).astype(int)
                extra.append("%s %s: pooled MCC at the monitored set's best threshold %.4f" % (label, sd, matthews_corrcoef(y, lt)))
        print("\n".join(extra))
        subprocess.run([sys.executable, os.path.join(HERE, "..", "repro", "pepmsnd_summary.py"), "--root", root])


if __name__ == "__main__":
    main()

"""PepMSND with the PeptideCLM-2 authors' KAN recipe (train_pepmsnd_kan_paperstyle.py), with OUR STUDENT standing where PeptideCLM-2 stands.

    python src/student/pepmsnd_kan_student.py --run st6f --epoch 2                       # authors' protocol (the TEST fold picks the epoch), student trained
    python src/student/pepmsnd_kan_student.py --run st6f --epoch 2 --select inner        # the same without the test-fold leak (an inner validation split)
    python src/student/pepmsnd_kan_student.py --run st6f --epoch 2 --select inner --freeze-epochs 100000    # control: the student never trains, only the KAN head
    python src/student/pepmsnd_kan_student.py --run st6f --epoch 2 --folds 1 2 --epochs 5                   # a quick look
    python src/student/pepmsnd_kan_student.py --run st6f --epoch 2 --check               # only: files present, molecules readable (seconds)

What is copied from the authors (external/peptideclm2_authors/.../scripts/train_pepmsnd_kan_paperstyle.py; it cannot be imported because it needs `lightning`,
which the server does not have, so these pieces are re-declared here in the same order and sizes; only their own kan.py, plain PyTorch, is imported):
  * the data: their X_train<k>.csv / X_test<k>.csv (10 cluster folds, 635 peptides, SMILES + 140 descriptor columns + Species / Environment + label);
    build_paperstyle_features (descriptors clipped at the 0.999 quantile, z-scored with the training rows; Species / Environment as two numbers);
  * the network (PeptideCLMBackbonePlusPaperKAN): backbone vector -> Linear 128 + ReLU; descriptors -> PaperStyleKANBranch (two KAN layers 128, BatchNorm);
    [both] -> KAN 256 - ReLU - BatchNorm - Dropout 0.1 - KAN 64 - ReLU; [that, Species, Environment] -> KAN 1; BCE with logits;
  * the training: AdamW with two groups (backbone lr 1e-5, head lr 1e-3, torch's default weight decay), batch 32, at most 120 epochs, the backbone frozen for the
    first 3 epochs and trained after that, early stopping with patience 20 on val_mcc_best = the best-threshold MCC (their _best_mcc_threshold), the best epoch is
    the model, its probabilities are saved with predicted_label = probability >= 0.5.
What differs, because the backbone is another network:
  * the backbone is the student reading the normalised SMILES with its own tokenizer; its 1024-d output `z` (--feature pre: the 1280-d pooled vector) stands
    where PeptideCLM-2's mean-pooled last hidden state stands, followed by a LayerNorm WITHOUT parameters (PeptideCLM-2's last layer is a LayerNorm, so its pooled
    vector is already of unit scale; the student's output is not). The student's weights are trained (all of them, except in the control);
  * 1e-5 is the authors' backbone learning rate for their 40M model (their larger models 7e-6 / 8e-6); the student has 10.8M parameters.
--select test (default, THE AUTHORS' PROTOCOL): val_mcc_best is computed on the TEST fold and picks the epoch: optimistic by construction, kept so that the number is
  comparable with the published 0.729 / 0.741 / 0.749. --select inner: 10% of the training rows are held out, val_mcc_best is computed there, the test fold is only
  predicted once at the end (the saved column label_thr uses the threshold found on the inner split).
Output: results/pepmsnd_ft_kan_student_<select>/<label>/seed<S>/fold<k>/preds_fold<k>.csv (true_label, predicted_prob, predicted_label, thr, label_thr) and a
printed table: MCC at 0.5 on the predictions pooled over the 10 folds (the harness's convention), its fold mean, AUROC, and the oracle MCC at the monitored set's
best threshold, mean +- sd over seeds, next to the published numbers.
"""
from __future__ import annotations

import argparse
import copy
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import finetune_student as FS  # noqa: E402
from paths import BENCH_DIR, CODE, RESULTS_DIR  # noqa: E402

AUTH_DIR = os.path.join(CODE, "external", "peptideclm2_authors", "training", "03_PepMSND_training_code", "scripts")
META_DROP_FOR_DESC = ["ID", "PMID", "SMILES", "label", "Length", "SE-3", "Species", "Environment"]      # the authors'
PUBLISHED = [("PepMSND paper, its full model with KAN (its own protocol)", 0.726), ("PeptideCLM-2 MLM + KAN (journal paper)", 0.729),
             ("PeptideCLM-2 Hybrid + KAN", 0.741), ("PeptideCLM-2 MTR + KAN", 0.749), ("PeptideCLM-2 without KAN (read off its Fig. 5B, about)", 0.60)]


def load_kan(path):
    if not os.path.exists(os.path.join(path, "kan.py")):
        raise SystemExit("the authors' kan.py is missing: %s/kan.py (copy external/peptideclm2_authors/training/03_PepMSND_training_code/scripts/kan.py there; "
                         "it is plain PyTorch)" % path)
    sys.path.insert(0, path)
    from kan import KANLinear
    where = os.path.abspath(sys.modules[KANLinear.__module__].__file__)
    print("KANLinear comes from %s" % where, flush=True)
    if not where.startswith(os.path.abspath(path)):                # another package called `kan` (e.g. pykan in site-packages) must never be used silently
        raise SystemExit("the wrong `kan` module was imported (%s instead of %s/kan.py): remove it from the environment or fix sys.path" % (where, path))
    return KANLinear


# ------------------------------------------------------------------------------------------- copied from the authors' script
def build_paperstyle_features(train_df, test_df, append_meta_to_kan, clip_quantile, max_abs_descriptor):
    desc_train = train_df.drop(columns=META_DROP_FOR_DESC).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    desc_test = test_df.drop(columns=META_DROP_FOR_DESC).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    desc_train = np.nan_to_num(desc_train, nan=0.0, posinf=0.0, neginf=0.0)
    desc_test = np.nan_to_num(desc_test, nan=0.0, posinf=0.0, neginf=0.0)
    if 0.5 < clip_quantile < 1.0:
        low = np.quantile(desc_train, 1.0 - clip_quantile, axis=0)
        high = np.quantile(desc_train, clip_quantile, axis=0)
        desc_train = np.clip(desc_train, low, high)
        desc_test = np.clip(desc_test, low, high)
    if max_abs_descriptor > 0:
        desc_train = np.clip(desc_train, -max_abs_descriptor, max_abs_descriptor)
        desc_test = np.clip(desc_test, -max_abs_descriptor, max_abs_descriptor)
    mean, std = desc_train.mean(axis=0), desc_train.std(axis=0)
    std[std < 1e-8] = 1.0
    desc_train, desc_test = (desc_train - mean) / std, (desc_test - mean) / std
    meta_train = train_df[["Species", "Environment"]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    meta_test = test_df[["Species", "Environment"]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    if append_meta_to_kan:
        return np.hstack([desc_train, meta_train]).astype(np.float32), np.hstack([desc_test, meta_test]).astype(np.float32), meta_train, meta_test
    return desc_train.astype(np.float32), desc_test.astype(np.float32), meta_train, meta_test


def best_mcc_threshold(y_true, y_prob):
    """The authors' _best_mcc_threshold: the threshold (among the predicted values and just outside them) with the best MCC; ties: closest to 0.5."""
    y_true_bin = (y_true >= 0.5).astype(int)
    u = np.unique(y_prob)
    if u.size == 0:
        return 0.5, float("nan")
    candidates = np.concatenate([[np.nextafter(u[0], -np.inf)], u, [np.nextafter(u[-1], np.inf)]])
    best_t, best_mcc = float(candidates[0]), -np.inf
    for t in candidates:
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum((y_true_bin == 1) & (y_pred == 1))
        tn = np.sum((y_true_bin == 0) & (y_pred == 0))
        fp = np.sum((y_true_bin == 0) & (y_pred == 1))
        fn = np.sum((y_true_bin == 1) & (y_pred == 0))
        denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        mcc = -1.0 if denom == 0 else ((tp * tn) - (fp * fn)) / np.sqrt(denom)
        if (mcc > best_mcc + 1e-12) or (abs(mcc - best_mcc) <= 1e-12 and abs(t - 0.5) < abs(best_t - 0.5)):
            best_mcc, best_t = float(mcc), float(t)
    return best_t, best_mcc


class PaperStyleKANBranch(nn.Module):
    def __init__(self, KANLinear, in_dim):
        super().__init__()
        self.fc1 = KANLinear(in_dim, 128)
        self.fc2 = KANLinear(128, 128)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm1d(128)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.norm(x)
        return self.relu(self.fc2(x))


class StudentKAN(nn.Module):
    """PeptideCLMBackbonePlusPaperKAN with the student in the backbone's place (see the module docstring)."""

    def __init__(self, KANLinear, student, feat, d_in, kan_dim, dropout=0.1):
        super().__init__()
        self.student, self.feat = student, feat
        self.feat_norm = nn.LayerNorm(d_in, elementwise_affine=False)
        self.seq_proj = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU())
        self.kan_branch = PaperStyleKANBranch(KANLinear, kan_dim)
        self.fusion_fc1 = KANLinear(128 + 128, 256)
        self.fusion_bn = nn.BatchNorm1d(256)
        self.fusion_fc2 = KANLinear(256, 64)
        self.out_fc = KANLinear(64 + 2, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def head_params(self):
        keep = ("feat_norm", "seq_proj", "kan_branch", "fusion_fc1", "fusion_bn", "fusion_fc2", "out_fc")
        return [p for n, p in self.named_parameters() if n.startswith(keep)]

    def set_frozen(self, frozen):
        for p in self.student.parameters():
            p.requires_grad = not frozen

    def forward(self, ids, kan_features, meta_features):
        f = self.feat_norm(FS.features(self.student, ids, self.feat))
        x = torch.cat([self.seq_proj(f), self.kan_branch(kan_features)], dim=1)
        x = self.relu(self.fusion_fc1(x))
        x = self.fusion_bn(x)
        x = self.dropout(x)
        x = self.relu(self.fusion_fc2(x))
        x = torch.cat([x, meta_features], dim=1)
        return self.out_fc(x).squeeze(-1)


@torch.no_grad()
def predict(net, ids, kan, meta, bs=256):
    net.eval()
    return torch.cat([net(ids[i:i + bs], kan[i:i + bs], meta[i:i + bs]).float() for i in range(0, len(kan), bs)])


# ------------------------------------------------------------------------------------------- one fold
def read_fold(a, fold, tok, max_len, dev):
    out = []
    for name in ("train", "test"):
        df = pd.read_csv(os.path.join(a.data_dir, "X_%s%d.csv" % (name, fold)))
        ids, ok, bad = FS.tokenise(df["SMILES"].astype(str), tok, max_len, a.workers, dev)
        out.append((df[ok].reset_index(drop=True), ids[torch.from_numpy(np.flatnonzero(ok)).to(dev)], int((~ok).sum())))
    return out


def run_fold(a, KANLinear, base, tok, max_len, fold, seed, dev, out_dir):
    (train_df, ids_tr, drop_tr), (test_df, ids_te, drop_te) = read_fold(a, fold, tok, max_len, dev)
    kan_tr_np, kan_te_np, meta_tr_np, meta_te_np = build_paperstyle_features(train_df, test_df, False, 0.999, 1e6)
    if drop_tr or drop_te:
        print("[INFO] dropped molecules the student cannot read: train %d, test %d" % (drop_tr, drop_te), flush=True)
    print("[INFO] fold %d: train %d, test %d, KAN input dim %d (descriptors only)" % (fold, len(train_df), len(test_df), kan_tr_np.shape[1]), flush=True)
    kan_tr, kan_te = torch.from_numpy(kan_tr_np).to(dev), torch.from_numpy(kan_te_np).to(dev)
    meta_tr, meta_te = torch.from_numpy(meta_tr_np).to(dev), torch.from_numpy(meta_te_np).to(dev)
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
        ids_v, kan_v, meta_v, y_v = ids_tr[iv], kan_tr[iv], meta_tr[iv], y_tr[iv]
        ids_tr, kan_tr, meta_tr, y_tr = ids_tr[it], kan_tr[it], meta_tr[it], y_tr[it]
    else:                                                         # the authors: the test fold is the validation set
        ids_v, kan_v, meta_v, y_v = ids_te, kan_te, meta_te, y_te
    net = StudentKAN(KANLinear, copy.deepcopy(base), a.feature, FS.feat_dim(base, a.feature), kan_tr.shape[1]).to(dev)
    opt = torch.optim.AdamW([{"params": list(net.student.parameters()), "lr": a.backbone_lr}, {"params": net.head_params(), "lr": a.head_lr}])
    best, best_ep, best_thr, best_state, bad, n = -float("inf"), -1, 0.5, None, 0, len(y_tr)
    for ep in range(a.epochs):
        t0 = time.time()
        frozen = ep < a.freeze_epochs
        net.set_frozen(frozen)
        net.train()
        perm = torch.randperm(n, device=dev)
        tl = 0.0
        for i in range(0, n, a.batch):
            idx = perm[i:i + a.batch]
            if len(idx) < 2:                                      # BatchNorm cannot train on one row (Lightning would stop here)
                continue
            opt.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(net(ids_tr[idx], kan_tr[idx], meta_tr[idx]), y_tr[idx])
            loss.backward()
            opt.step()
            tl += loss.detach().item() * len(idx)
        lv = predict(net, ids_v, kan_v, meta_v)
        vloss = float(F.binary_cross_entropy_with_logits(lv, y_v))
        thr, mcc = best_mcc_threshold(y_v.cpu().numpy(), torch.sigmoid(lv).cpu().numpy())
        if a.verbose:
            print("[epoch %3d] train_loss %.4f  val_loss %.4f  val_mcc_best %.4f (thr %.3f)%s  (%.1fs)"
                  % (ep, tl / n, vloss, mcc, thr, "  [student frozen]" if frozen else "", time.time() - t0), flush=True)
        if mcc > best:
            best, best_ep, best_thr, bad = mcc, ep, thr, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                break
    net.load_state_dict(best_state)
    prob = torch.sigmoid(predict(net, ids_te, kan_te, meta_te)).cpu().numpy()
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "preds_fold%d.csv" % fold)
    pd.DataFrame({"true_label": y_te.cpu().numpy().astype(float), "predicted_prob": prob.astype(float), "predicted_label": (prob >= 0.5).astype(int),
                  "thr": best_thr, "label_thr": (prob >= best_thr).astype(int)}).to_csv(path, index=False)
    print("[INFO] saved %s (best epoch %d, val_mcc_best %.4f, %s set)" % (path, best_ep, best, "test-fold" if a.select == "test" else "inner"), flush=True)
    return best_ep, len(prob)


def summarise(root, label, seeds):
    from sklearn.metrics import matthews_corrcoef, roc_auc_score
    rows = []
    for sd in seeds:
        parts = [pd.read_csv(os.path.join(root, label, "seed%d" % sd, "fold%d" % k, "preds_fold%d.csv" % k)) for k in range(1, 11)
                 if os.path.exists(os.path.join(root, label, "seed%d" % sd, "fold%d" % k, "preds_fold%d.csv" % k))]
        if len(parts) < 10:
            print("seed %d: only %d of 10 folds are done" % (sd, len(parts)))
            continue
        y = np.concatenate([p.true_label.to_numpy() for p in parts]).astype(int)
        pr = np.concatenate([p.predicted_prob.to_numpy() for p in parts])
        lt = np.concatenate([p.label_thr.to_numpy() for p in parts]).astype(int)
        rows.append({"seed": sd, "MCC_pooled": matthews_corrcoef(y, pr >= 0.5), "MCC_foldmean": float(np.mean([matthews_corrcoef(p.true_label.astype(int), p.predicted_label) for p in parts])),
                     "AUROC": roc_auc_score(y, pr), "MCC_oracle_thr": matthews_corrcoef(y, lt)})
    if not rows:
        return
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(root, label, "summary.csv"), index=False)
    print("\n=== PepMSND, 10 cluster folds, %s | mean +- sd over %d seed(s) ===" % (label, len(r)))
    for c in ("MCC_pooled", "MCC_foldmean", "AUROC", "MCC_oracle_thr"):
        print("  %-16s %.3f +- %.3f" % (c, r[c].mean(), 0.0 if len(r) < 2 else r[c].std()))
    print("published, for comparison (their protocols, PeptideCLM-2 fine-tuned; the authors' epoch choice uses the test fold):")
    for n, v in PUBLISHED:
        print("  %-62s %.3f" % (n, v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="the student run (checkpoints/student_distill/<run>)")
    ap.add_argument("--epoch", type=int, default=0, help="0 = best.pt; N = the snapshot epoch_NN.pt")
    ap.add_argument("--feature", choices=("z", "pre"), default="z")
    ap.add_argument("--select", choices=("test", "inner"), default="test", help="test = the authors' protocol (the test fold picks the epoch), inner = 10%% of the training rows")
    ap.add_argument("--folds", nargs="*", type=int, default=list(range(1, 11)))
    ap.add_argument("--seeds", nargs="*", type=int, default=[103])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--backbone-lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--freeze-epochs", type=int, default=3, help="the authors' 3; a huge number = the student never trains (control)")
    ap.add_argument("--data-dir", default=os.path.join(BENCH_DIR, "PepMSND_data"))
    ap.add_argument("--authors-dir", default=AUTH_DIR)
    ap.add_argument("--label", default=None, help="folder name under the results root (default: the run name, + _frozen for the control)")
    ap.add_argument("--root", default=None, help="results root (default results/pepmsnd_ft_kan_student_<select>)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--verbose", action="store_true", help="print every epoch")
    ap.add_argument("--redo", action="store_true", help="recompute folds that already have predictions")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    KANLinear = load_kan(a.authors_dir)
    base, tok, max_len, ep_w, path = FS.load_student(a.run, a.epoch, dev)
    name = a.run if a.epoch <= 0 else "%s_e%02d" % (a.run, a.epoch)
    label = a.label or (name + ("_frozen" if a.freeze_epochs >= a.epochs else ""))
    root = a.root or os.path.join(RESULTS_DIR, "pepmsnd_ft_kan_student_%s" % a.select)
    print("student %s: weights of epoch %s, %.3fM parameters, feature %s | label %s | selection: %s\n"
          "recipe (the authors'): AdamW, student lr %g / head lr %g, student frozen for the first %d epochs, batch %d, <= %d epochs, patience %d on val_mcc_best, "
          "seeds %s, folds %s\nresults: %s"
          % (name, ep_w, sum(p.numel() for p in base.parameters()) / 1e6, a.feature, label,
             "the TEST fold picks the epoch (the authors' protocol, optimistic)" if a.select == "test" else "10% of the training rows pick the epoch (no test leak)",
             a.backbone_lr, a.head_lr, a.freeze_epochs, a.batch, a.epochs, a.patience, a.seeds, a.folds, root), flush=True)
    if a.check:
        for fold in a.folds:
            (tr, _, d1), (te, _, d2) = read_fold(a, fold, tok, max_len, dev)
            print("fold %2d: train %d rows (%d unreadable), test %d rows (%d unreadable)" % (fold, len(tr) + d1, d1, len(te) + d2, d2), flush=True)
        return
    for seed in a.seeds:
        for fold in a.folds:
            out_dir = os.path.join(root, label, "seed%d" % seed, "fold%d" % fold)
            if os.path.exists(os.path.join(out_dir, "preds_fold%d.csv" % fold)) and not a.redo:
                print("seed %d fold %d: done, skipped" % (seed, fold), flush=True)
                continue
            t0 = time.time()
            ep, n = run_fold(a, KANLinear, base, tok, max_len, fold, seed, dev, out_dir)
            print("seed %d fold %d: %d test molecules, best epoch %d, %.0fs" % (seed, fold, n, ep, time.time() - t0), flush=True)
    summarise(root, label, a.seeds)


if __name__ == "__main__":
    main()

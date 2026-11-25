"""Plain-PyTorch port of the authors' train_pepmsnd_species_env.py, for a server without `lightning`.

    python src/repro/pepmsnd_port.py --model_name aaronfeller/peptideclm-2-mtr-large --fold 3 --save_path <dir> --seed 103

Same arguments and the same output (preds_fold<k>.csv: true_label, predicted_prob, predicted_label) as the authors' script
(external/peptideclm2_authors/training/03_PepMSND_training_code/scripts/train_pepmsnd_species_env.py, read line by line), same recipe:
  * data: X_train<k>.csv / X_test<k>.csv of the 10 cluster folds; label = column `label`; SMILES = column `SMILES`;
  * Species / Environment turned into one-hot columns with the TRAINING part's columns (missing -> "UNK"; the test part is aligned);
  * every SMILES tokenised at once, truncation at 2048 tokens, padding to the longest molecule of the part (padding=True);
  * the full PeptideCLM-2 backbone is TRAINABLE; masked mean of the last hidden state, joined with the one-hot vector, then
    Linear(d + meta, d) -> SiLU -> Dropout(0.1) -> Linear(d, 1); BCE with logits;
  * AdamW, lr 1e-5 (torch defaults otherwise), batch 32, at most 10 epochs, fp32;
  * as in the authors' script THE TEST FOLD IS THE VALIDATION SET: val_loss is computed on it after every epoch, training stops after
    `patience` (3) epochs without a lower val_loss, the epoch with the lowest val_loss is the model that predicts the test fold. That
    is a leak (the test fold chooses the epoch); it is kept because this port exists to reproduce the published numbers;
  * predicted_label = probability >= 0.5.
Differences (practical only): no Lightning (a hand-written loop), the shuffling order comes from torch.randperm instead of a DataLoader,
the best weights are kept in CPU memory instead of a checkpoint file, and --micro_batch n splits each batch of 32 into pieces of n (gradients
accumulated, one optimiser step per 32 molecules) when 32 does not fit in memory. The model is loaded through featurize.fix_rotary
(rotary buffers rebuilt) and the SDPA mask fix is installed.
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
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import featurize  # noqa: E402
from paths import BENCH_DIR, PCLM_DIR, PCLM_MTR_DIR, _hf_model_dir  # noqa: E402

ALIAS = {
    "aaronfeller/peptideclm-2-mlm-large": PCLM_DIR,
    "aaronfeller/peptideclm-2-mtr-large": PCLM_MTR_DIR,
    "aaronfeller/peptideclm-2-hybrid-large": _hf_model_dir("peptideclm-2-hybrid-large", "PCLM_HYBRID_DIR"),
}


def masked_mean_pool(h, mask):                         # authors' function
    m = mask.to(h.dtype).unsqueeze(-1)
    return (h * m).sum(1) / m.sum(1).clamp(min=1e-6)


def get_last_hidden_state(outputs):                    # authors' function
    if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
        return outputs.last_hidden_state
    if isinstance(outputs, dict):
        if "last_hidden_state" in outputs:
            return outputs["last_hidden_state"]
        if "last_layer" in outputs:
            v = outputs["last_layer"]
            return v["last_hidden_state"] if isinstance(v, dict) and "last_hidden_state" in v else v
    if isinstance(outputs, (list, tuple)):
        return outputs[0]
    return outputs


def infer_backbone_dim(config):                        # authors' function
    d = (getattr(config, "hidden_size", None) or getattr(config, "embed_dim", None)
         or getattr(config, "d_model", None) or getattr(config, "dim", None))
    if d is None:
        raise ValueError("cannot infer the embedding dimension from %s" % config)
    return int(d)


def species_env_onehot(train_df, test_df):             # authors' function
    tm = train_df[["Species", "Environment"]].copy().fillna("UNK").astype(str)
    em = test_df[["Species", "Environment"]].copy().fillna("UNK").astype(str)
    tr = pd.get_dummies(tm, columns=["Species", "Environment"], prefix=["Species", "Environment"])
    te = pd.get_dummies(em, columns=["Species", "Environment"], prefix=["Species", "Environment"]).reindex(columns=tr.columns, fill_value=0)
    return tr.to_numpy(dtype=np.float32), te.to_numpy(dtype=np.float32), list(tr.columns)


class Net(nn.Module):
    def __init__(self, model_name, meta_dim, dropout=0.1):
        super().__init__()
        from transformers import AutoModel
        self.backbone = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        featurize.fix_rotary(self.backbone)
        d = infer_backbone_dim(self.backbone.config)
        self.classifier = nn.Sequential(nn.Linear(d + meta_dim, d), nn.SiLU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def forward(self, input_ids, attention_mask, meta):
        h = get_last_hidden_state(self.backbone(input_ids=input_ids, attention_mask=attention_mask))
        return self.classifier(torch.cat([masked_mean_pool(h, attention_mask), meta], 1)).squeeze(-1)


def encode(tok, smiles, max_length):
    return tok(smiles, truncation=True, padding=True, max_length=max_length, return_tensors="pt")


@torch.no_grad()
def predict_logits(net, enc, meta, dev, bs=32):
    net.eval()
    out = []
    for i in range(0, len(meta), bs):
        sl = slice(i, i + bs)
        out.append(net(enc["input_ids"][sl].to(dev), enc["attention_mask"][sl].to(dev), meta[sl].to(dev)).float().cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--fold", type=int, required=True, choices=list(range(1, 11)))
    ap.add_argument("--data_dir", default=os.path.join(BENCH_DIR, "PepMSND_data"))
    ap.add_argument("--save_path", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--micro_batch", type=int, default=0, help="pieces of this size inside one batch (0 = no splitting)")
    ap.add_argument("--max_epochs", type=int, default=10)
    ap.add_argument("--learning_rate", type=float, default=1e-5)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--num_workers", type=int, default=4)        # accepted for compatibility, unused
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    featurize._patch_sdpa()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = ALIAS.get(a.model_name, a.model_name)

    train_df = pd.read_csv(os.path.join(a.data_dir, "X_train%d.csv" % a.fold))
    test_df = pd.read_csv(os.path.join(a.data_dir, "X_test%d.csv" % a.fold))
    m_tr, m_te, cols = species_env_onehot(train_df, test_df)
    print("[INFO] Species/Environment one-hot dimension: %d | train %d, test %d" % (len(cols), len(train_df), len(test_df)), flush=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    enc_tr, enc_te = encode(tok, train_df["SMILES"].astype(str).tolist(), a.max_length), encode(tok, test_df["SMILES"].astype(str).tolist(), a.max_length)
    y_tr = torch.tensor(train_df["label"].to_numpy(dtype=np.float32))
    y_te = torch.tensor(test_df["label"].to_numpy(dtype=np.float32))
    m_tr, m_te = torch.tensor(m_tr), torch.tensor(m_te)
    print("[INFO] padded length: train %d, test %d tokens" % (enc_tr["input_ids"].shape[1], enc_te["input_ids"].shape[1]), flush=True)

    net = Net(path, m_tr.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.learning_rate)
    mb = a.micro_batch or a.batch_size
    best, best_state, best_ep, bad = float("inf"), None, -1, 0
    n = len(y_tr)
    for ep in range(a.max_epochs):
        t0 = time.time()
        net.train()
        perm = torch.randperm(n)
        tl = 0.0
        for i in range(0, n, a.batch_size):
            idx = perm[i:i + a.batch_size]
            opt.zero_grad(set_to_none=True)
            for j in range(0, len(idx), mb):
                p = idx[j:j + mb]
                lo = net(enc_tr["input_ids"][p].to(dev), enc_tr["attention_mask"][p].to(dev), m_tr[p].to(dev))
                loss = F.binary_cross_entropy_with_logits(lo.float(), y_tr[p].to(dev)) * (len(p) / len(idx))
                loss.backward()
                tl += loss.detach().item() * len(idx)
            opt.step()
        val = float(F.binary_cross_entropy_with_logits(predict_logits(net, enc_te, m_te, dev), y_te))
        print("[epoch %2d] train_loss %.4f  val_loss(test fold) %.4f  (%.0fs)" % (ep, tl / n, val, time.time() - t0), flush=True)
        if val < best:
            best, best_ep, bad = val, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= a.patience:
                print("[INFO] early stopping after epoch %d (best epoch %d)" % (ep, best_ep), flush=True)
                break
    net.load_state_dict(best_state)
    prob = torch.sigmoid(predict_logits(net, enc_te, m_te, dev)).numpy()
    os.makedirs(a.save_path, exist_ok=True)
    out = os.path.join(a.save_path, "preds_fold%d.csv" % a.fold)
    pd.DataFrame({"true_label": y_te.numpy().astype(float), "predicted_prob": prob.astype(float),
                  "predicted_label": (prob >= 0.5).astype(int)}).to_csv(out, index=False)
    print("[INFO] Saved predictions: %s (best epoch %d, val_loss %.4f)" % (out, best_ep, best), flush=True)


if __name__ == "__main__":
    main()

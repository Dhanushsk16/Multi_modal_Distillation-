"""The peptide benchmarks in one shape, each with the split protocol it ships with.

Every loader returns a DataFrame with:
    smiles   molecule
    y        label (0/1) or value
    split    'train' / 'test' for fixed splits; fold id (int) for cluster CV
    extra_*  per-row context features that are part of the task, not the molecule
             (PepMSND: Species, Environment -- the same peptide can carry different
             labels in different contexts, so every representation gets these too)

Protocols:
    CellPPD   fixed train/test                 classification
    THPep     fixed train/test                 classification
    PepMSND   10-fold cluster CV (64 per fold) classification
    PAMPA     6-fold leave-one-cluster-out     regression (log Papp)
    AmpHGT    train+val -> test                classification; length stress test

PepLand-paper suite (PL_*). PepLand ships the raw data only (data/eval), no split files, so the
protocol below is OURS, fixed and documented, and the numbers are not comparable with the paper's:
5-fold CV, folds fixed by seed 0, rows with the same molecule always share a fold (so a repeated
peptide is never in train and test), classification folds stratified by label.
    PL_cCPP     c-CPP.txt     canonical cell-penetrating, 1162/1162   classification (sequence -> SMILES)
    PL_cSol     c-Sol.txt     canonical solubility, 877/634           classification (sequence -> SMILES)
    PL_ncCPP    nc-CPP.csv    CycPeptMPDB permeability, 7334 cyclic   regression (Permeability)
    PL_cBinding c-binding.csv protein-peptide affinity, 1806 pairs    regression (affinity), peptide only
    PL_ncBinding nc-binding.csv same, peptides with non-canonical AAs regression (affinity), peptide only
The binding tasks give the model only the PEPTIDE (Merge_SMILES); the protein partner is not an input.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from paths import BENCH_DIR as B

TASK = {"CellPPD": "clf", "THPep": "clf", "PepMSND": "clf", "PAMPA": "reg", "AmpHGT": "clf"}
MAIN = ["CellPPD", "THPep", "PepMSND", "PAMPA"]
STRESS = ["AmpHGT"]
PEPLAND = ["PL_cCPP", "PL_cSol", "PL_cBinding", "PL_ncCPP", "PL_ncBinding"]
PL_SUITE = ["PL_cCPP", "PL_cSol", "PL_ncCPP", "PL_cBinding", "PL_ncBinding"]
TASK.update({"PL_cCPP": "clf", "PL_cSol": "clf", "PL_ncCPP": "reg", "PL_cBinding": "reg", "PL_ncBinding": "reg"})
# THPep as the JOURNAL PeptideCLM-2 paper (J. Chem. Inf. Model. 2026, 66, 8110) scores it: "random 5-fold cross validation", three repetitions.
# THPep_cv0 / _cv1 / _cv2 = ALL 609 molecules (the shipped train + test files together), one random fold assignment each (seed 0 / 1 / 2).
THPEP_CV = ["THPep_cv0", "THPep_cv1", "THPep_cv2"]
TASK.update({n: "clf" for n in THPEP_CV})
# THPep with the splits the PeptideCLM-2 authors ACTUALLY used (2026-11-24; found in the result files of their repository, see src/build_thpep_official_splits.py):
# THPep_o101 / _o202 / _o303 = their seeds 101 / 202 / 303, each ONE split of the 609 molecules: train 389 + val 98 (both used as training rows here, 487) and
# a test set of 122 (35 positives); the paper reports the mean over the three seeds. The ids come from src/benchmark_splits/thpep_official_splits.csv, the
# molecules from THPep_main90_smiles_classes.csv (the authors' file, on the server in data/benchmarks).
THPEP_OFF = ["THPep_o101", "THPep_o202", "THPep_o303"]
TASK.update({n: "clf" for n in THPEP_OFF})
# PepLand's two binding tasks AS INTENDED: the peptide AND its protein partner (column seq1). The protein is an ESM-2 3B vector
# (featurize_esm.py --proteins -> features/pepland_suite/protein_esm.npz), given to every representation as extra_prot_* columns (the way
# PepMSND's context columns are), so each learner sees [peptide representation | protein vector].
#   ...Prot        the same rows and the same folds as PL_cBinding / PL_ncBinding (rows with the same peptide share a fold; a protein can
#                  be in train and test with different peptides)
#   ...ProtUnseen  folds by PROTEIN (rows with the same protein share a fold): the protein of a test pair is never in the training data
PL_PROT = ["PL_cBindingProt", "PL_ncBindingProt", "PL_cBindingProtUnseen", "PL_ncBindingProtUnseen"]
TASK.update({n: "reg" for n in PL_PROT})


def _fixed(train, test):
    a = pd.read_csv(train).assign(split="train")
    b = pd.read_csv(test).assign(split="test")
    return pd.concat([a, b], ignore_index=True).rename(columns={"label": "y"})[["smiles", "y", "split"]]


def _folds(smiles, strat=None, k=5, seed=0):
    """Fold id (0..k-1) per row. Rows with the same molecule share a fold; with `strat` (labels) the
    folds are stratified by each molecule's first label."""
    smiles = pd.Series(list(smiles)).reset_index(drop=True)
    first = smiles.drop_duplicates()
    rng = np.random.default_rng(seed)
    fold = {}
    if strat is None:
        for pos, i in enumerate(rng.permutation(len(first))):
            fold[first.iloc[i]] = pos % k
    else:
        lab = np.asarray(list(strat))[first.index.to_numpy()]
        for c in np.unique(lab):
            for pos, i in enumerate(rng.permutation(np.flatnonzero(lab == c))):
                fold[first.iloc[i]] = pos % k
    return smiles.map(fold).to_numpy().astype(int)


def _seq_smiles(seq):
    from rdkit import Chem
    m = Chem.MolFromSequence(seq)
    return Chem.MolToSmiles(m) if m is not None else None


def _pepland(name):
    P = f"{B}/pepland"
    if name in ("PL_cCPP", "PL_cSol"):
        d = pd.read_csv(f"{P}/{'c-CPP' if name == 'PL_cCPP' else 'c-Sol'}.txt", header=None, names=["seq", "y"])
        d["smiles"] = d.seq.astype(str).str.strip().map(_seq_smiles)
        d = d.dropna(subset=["smiles", "y"])
        d = d[d.groupby("smiles").y.transform("nunique") == 1].drop_duplicates("smiles")   # drop label conflicts
        d = d.assign(y=d.y.astype(int)).reset_index(drop=True)
        return d.assign(split=_folds(d.smiles, strat=d.y))[["smiles", "y", "split"]]
    if name == "PL_ncCPP":
        d = pd.read_csv(f"{P}/nc-CPP.csv", usecols=["SMILES", "Permeability"]).dropna().reset_index(drop=True)
        d = d.rename(columns={"SMILES": "smiles", "Permeability": "y"})
        return d.assign(split=_folds(d.smiles))[["smiles", "y", "split"]]
    f = {"PL_cBinding": "c-binding.csv", "PL_ncBinding": "nc-binding.csv"}[name]
    d = pd.read_csv(f"{P}/{f}", low_memory=False, index_col=0)
    d = d.rename(columns={"Merge_SMILES": "smiles", "affinity": "y"})[["smiles", "y"]].dropna().reset_index(drop=True)
    return d.assign(split=_folds(d.smiles))[["smiles", "y", "split"]]


def _binding_prot(name):
    """PL_cBindingProt etc.: smiles, y, split, plus extra_prot_0000.. (the ESM-2 vector of the protein partner)."""
    from paths import FEAT_DIR
    unseen = name.endswith("Unseen")
    f = {"PL_cBinding": "c-binding.csv", "PL_ncBinding": "nc-binding.csv"}[name.replace("Unseen", "").replace("Prot", "")]
    d = pd.read_csv(f"{B}/pepland/{f}", low_memory=False, index_col=0)
    d = d.rename(columns={"Merge_SMILES": "smiles", "affinity": "y"})[["smiles", "y", "seq1"]].dropna().reset_index(drop=True)
    d["seq1"] = d.seq1.astype(str).str.strip()
    p = os.path.join(os.path.dirname(FEAT_DIR), "pepland_suite", "protein_esm.npz")
    if not os.path.exists(p):
        raise FileNotFoundError("%s is missing: python src/featurize_esm.py --proteins (GPU) makes it" % p)
    z = np.load(p)
    idx = {s: i for i, s in enumerate(z["seqs"].tolist())}
    X = z["X"].astype(np.float32)[[idx[s] for s in d.seq1]]
    cols = pd.DataFrame(X, columns=["extra_prot_%04d" % i for i in range(X.shape[1])])
    split = _folds(d.seq1 if unseen else d.smiles)               # unseen: the protein is the unit that never crosses a fold
    return pd.concat([d[["smiles", "y"]].assign(split=split), cols], axis=1)


def thpep_official(name, keep_val=False):
    """THPep_o101 / _o202 / _o303 as smiles, y, split. keep_val=False (the frozen probes, the harness): the authors' train + val rows are all 'train' (487) and the
    test set is 'test' (122). keep_val=True (fine-tuning): the authors' 98 validation rows stay 'val' (389 train / 98 val / 122 test), so the fine-tuning scripts
    train ONE model on train and stop it early on val, as the paper did; the test molecules are never used for any choice."""
    sp = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_splits", "thpep_official_splits.csv"))
    full = pd.read_csv(f"{B}/THPep_main90_smiles_classes.csv")
    d = full.merge(sp, on="id", how="inner", suffixes=("", "_split"))
    assert len(full) == len(sp) == len(d) == 609, "THPep official split: the ids do not match THPep_main90_smiles_classes.csv (%d / %d / %d rows)" % (len(full), len(sp), len(d))
    assert (d["class"] == d["class_split"]).all(), "THPep official split: labels differ from THPep_main90_smiles_classes.csv"
    part = d["s" + name[-3:]]
    split = np.where(part == "test", "test", np.where(part == "val", "val", "train")) if keep_val else np.where(part == "test", "test", "train")
    return pd.DataFrame({"smiles": d.smiles, "y": d["class"], "split": split})


def load(name):
    if name in THPEP_OFF:
        return thpep_official(name)
    if name in PL_SUITE:
        return _pepland(name)
    if name in PL_PROT:
        return _binding_prot(name)
    if name in THPEP_CV:
        d = pd.concat([pd.read_csv(f"{B}/THPep_train.csv"), pd.read_csv(f"{B}/THPep_test.csv")], ignore_index=True)
        d = d.rename(columns={"label": "y"})[["smiles", "y"]]
        return d.assign(split=_folds(d.smiles, strat=None, k=5, seed=int(name[-1])))[["smiles", "y", "split"]]
    if name == "CellPPD":
        return _fixed(f"{B}/CellPPD_train.csv", f"{B}/CellPPD_test.csv")
    if name == "THPep":
        return _fixed(f"{B}/THPep_train.csv", f"{B}/THPep_test.csv")
    if name == "AmpHGT":
        tr = pd.concat([pd.read_csv(f"{B}/amp_train.csv"), pd.read_csv(f"{B}/amp_val.csv")])
        te = pd.read_csv(f"{B}/amp_test.csv")
        df = pd.concat([tr.assign(split="train"), te.assign(split="test")], ignore_index=True)
        return df.rename(columns={"label": "y"})[["smiles", "y", "split"]]
    if name == "PepMSND":
        d = pd.read_csv(f"{B}/PepMSND_clustered_data.csv")
        return pd.DataFrame({"smiles": d.SMILES, "y": d.label, "split": d.cluster.astype(int),
                             "extra_species": d.Species, "extra_environment": d.Environment})
    if name == "PAMPA":
        d = pd.read_csv(f"{B}/PAMPA_clusters.csv")
        return pd.DataFrame({"smiles": d.SMILES, "y": d.PAMPA, "split": d.cluster.astype(int)})
    raise KeyError(name)


def selected():
    """BENCHMARKS=CellPPD,THPep restricts a run (smoke tests); default is all."""
    env = os.environ.get("BENCHMARKS")
    return env.split(",") if env else MAIN + STRESS


def all_smiles(names=None):
    """Every distinct SMILES across the benchmarks -- featurised once, shared."""
    names = names or selected()
    s = pd.concat([load(n).smiles for n in names]).dropna().astype(str)
    return sorted(set(s))


def pepland_items():
    """Molecules of the PepLand-paper benchmarks as {item: benchmark}. Used for corpus
    leakage removal only. Sequence-only sets give "seq:ACDE..." items (turned into a
    molecule by Chem.MolFromSequence inside the fingerprint worker); the rest give SMILES."""
    P = f"{B}/pepland"
    out = {}
    for name, f in (("PL_cCPP", "c-CPP.txt"), ("PL_cSol", "c-Sol.txt")):
        d = pd.read_csv(f"{P}/{f}", header=None)
        for q in d[0].dropna().astype(str).str.strip():
            out.setdefault("seq:" + q, name)
    for name, f, cols in (("PL_ncCPP", "nc-CPP.csv", ["SMILES"]),
                          ("PL_cBinding", "c-binding.csv", ["ligand_SMILES", "Merge_SMILES"]),
                          ("PL_ncBinding", "nc-binding.csv", ["ligand_SMILES", "Merge_SMILES"])):
        d = pd.read_csv(f"{P}/{f}", low_memory=False)
        for c in cols:
            for q in d[c].dropna().astype(str).str.strip():
                out.setdefault(q, name)
    return out

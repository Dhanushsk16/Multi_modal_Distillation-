#!/usr/bin/env python
"""Server-side check before the THPep run on the authors' splits (benchmarks THPep_o101 / _o202 / _o303, src/benchmarks.py).

The benchmark features of every representation are keyed by the SMILES of THPep_train.csv + THPep_test.csv. The official splits take their molecules from
THPep_main90_smiles_classes.csv (the authors' file), so the two sets of SMILES must be identical, and the labels must agree. Prints OK or stops with an error.

    python ~/code/peptidefusion/src/teacher/check_thpep_official.py
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import benchmarks as BM  # noqa: E402

ours = pd.concat([pd.read_csv(f"{BM.B}/THPep_train.csv"), pd.read_csv(f"{BM.B}/THPep_test.csv")], ignore_index=True)
lab_ours = dict(zip(ours.smiles, ours.label))
print("THPep_train + THPep_test: %d rows, %d unique SMILES" % (len(ours), ours.smiles.nunique()))
bad = 0
for name in BM.THPEP_OFF:
    d = BM.load(name)
    tr, te = d[d.split == "train"], d[d.split == "test"]
    missing = set(d.smiles) - set(ours.smiles)
    wrong = sum(lab_ours.get(s) != y for s, y in zip(d.smiles, d.y) if s in lab_ours)
    print("%s: %d rows (train %d, test %d with %d positives) | SMILES not in our feature set: %d | label disagreements: %d" % (
        name, len(d), len(tr), len(te), int(te.y.sum()), len(missing), wrong))
    bad += len(missing) + wrong
if bad:
    sys.exit("CHECK FAILED: %d problems (a SMILES string or label differs between the authors' file and ours): do NOT run the harness; send this output back" % bad)
print("OK: the three official splits use exactly the molecules (and labels) of the existing THPep features")

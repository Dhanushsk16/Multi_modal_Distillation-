"""RDKit 2D descriptors: the fourth view of the teacher (replaces PepLand in teacher v2).

One function, used by every writer of the view (featurize.py --rep rdkit for the benchmarks,
corpus/cache_views.py --view rdkit for the pool), so the cache and the benchmark features are the same
function of the SMILES.

The set is RDKit's full descriptor list (Descriptors._descList, ~210 values) minus the ones that are
slow or numerically unstable on peptides: Ipc / AvgIpc (information content of the path counts, values up
to 1e30), BertzCT (the slowest one, minutes on long peptides) and the BCUT2D family (they need a valid
Gasteiger charge on every atom and fail on unusual ones). The local PAMPA test (2026-11-07, leave-cluster-out,
GBT) gave R2 0.350 for this trimmed set against 0.314 for the full list, 22 ms per molecule.

Values come back as float32. A single descriptor that fails, is not finite or is above 1e9 in magnitude
becomes NaN (the row stays usable); a molecule that does not parse, or has more than 10% of its
descriptors missing, is ok=False. The names are saved next to every file (key "names") so a different
RDKit version, which can change the list, is caught instead of silently shifting columns.

Preprocessing for models (preprocess_np here, the same formula in train_teacher.prep): signed log,
sign(x) * log1p(|x|), which tames the heavy tails (MolWt, Kappa, Chi, VSA sums span 3-4 orders of
magnitude on 5-30 residue peptides), then per-feature standardisation with training statistics; NaN -> 0
after standardisation, i.e. the mean.
"""
from __future__ import annotations

import numpy as np

EXCLUDE_EXACT = {"Ipc", "AvgIpc", "BertzCT"}
EXCLUDE_PREFIX = ("BCUT2D_",)
MAX_ABS = 1e9
MAX_MISSING = 0.10

_FUNCS = None


def _funcs():
    """[(name, function)] in a fixed order (RDKit's order, trimmed). Built once per process."""
    global _FUNCS
    if _FUNCS is None:
        from rdkit.Chem import Descriptors
        _FUNCS = [(n, f) for n, f in Descriptors._descList
                  if n not in EXCLUDE_EXACT and not n.startswith(EXCLUDE_PREFIX)]
    return _FUNCS


def names():
    return [n for n, _ in _funcs()]


def n_desc():
    return len(_funcs())


# The four EState Max/Min descriptors each recompute the per-atom EState indices (20-80 ms on a 30-mer, 3/4 of
# the cost of the whole list); one computation gives all four (identical values, checked in the tests).
_ESTATE = {"MaxEStateIndex": lambda e: e.max(), "MinEStateIndex": lambda e: e.min(),
           "MaxAbsEStateIndex": lambda e: np.abs(e).max(), "MinAbsEStateIndex": lambda e: np.abs(e).min()}


def describe(smi):
    """SMILES -> float32 [n_desc] with NaN for failed descriptors, or None if the molecule is unusable."""
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import EState
    RDLogger.DisableLog("rdApp.*")
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    fs = _funcs()
    out = np.full(len(fs), np.nan, np.float32)
    try:
        e = np.asarray(EState.EStateIndices(m), np.float64)
    except Exception:
        e = None
    for i, (n, f) in enumerate(fs):
        try:
            v = float(_ESTATE[n](e)) if (n in _ESTATE and e is not None) else float(f(m))
        except Exception:
            continue
        if np.isfinite(v) and abs(v) <= MAX_ABS:
            out[i] = v
    if np.isnan(out).mean() > MAX_MISSING:
        return None
    return out


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))


def preprocess_np(X, ok=None):
    """Unsupervised clean-up for the sklearn probes of eval_single.py: signed log, NaN -> the column median
    of the usable rows of the matrix it is applied to (no labels involved). X [n, d] -> float32 [n, d]."""
    Z = signed_log(np.asarray(X, np.float64))
    good = np.ones(len(Z), bool) if ok is None else np.asarray(ok, bool)
    ref = np.where(np.isfinite(Z[good]), Z[good], np.nan)
    with np.errstate(all="ignore"):
        med = np.nanmedian(ref, axis=0) if len(ref) else np.zeros(Z.shape[1])
    med = np.where(np.isfinite(med), med, 0.0)
    Z = np.where(np.isfinite(Z), Z, med[None, :])
    return Z.astype(np.float32)

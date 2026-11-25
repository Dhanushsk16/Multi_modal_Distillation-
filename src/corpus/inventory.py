"""Estimate how many usable peptides each source holds, without downloading it all.

For each source, classify a uniform sample with peptide_filters.classify and scale
the fractions up to the source's full size:

    pclm_esmatlas   PeptideCLM-2 ESMAtlas peptides  (HF train/small_molecules.parquet --
                    the upstream file names are swapped)
    pclm_pubchem    PeptideCLM-2 PubChem            (HF train/peptides.parquet)
    pepland_stage1  PepLand stage-1 PeptideAtlas + UniParc (local zip, train.csv)

Sampling: parquet sources read evenly spaced row groups remotely (only the smiles
column); the PepLand CSV is streamed and every k-th row kept.

    python src/corpus/inventory.py --per-source 40000
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from peptide_filters import classify  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "results", "corpus_inventory")
HF = "datasets/aaronfeller/peptideclm-2-pretraining-data/train/"


def sample_parquet(name, n_target, n_groups=24):
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    with fs.open(HF + name) as fh:
        pf = pq.ParquetFile(fh)
        total = pf.metadata.num_rows
        groups = np.linspace(0, pf.metadata.num_row_groups - 1, n_groups).astype(int)
        parts = [pf.read_row_group(int(g), columns=["smiles"]).to_pandas().smiles for g in groups]
    s = pd.concat(parts, ignore_index=True)
    return s.sample(min(n_target, len(s)), random_state=0).tolist(), total


def sample_pepland(n_target):
    z = zipfile.ZipFile(os.path.join(ROOT, "data", "sources", "pepland", "pep_atlas_uniparc_smiles.zip"))
    names = [n for n in z.namelist() if n.endswith(".csv") and "__MACOSX" not in n]
    total, keep = 0, []
    # first pass: count rows; second pass: keep every k-th
    for n in names:
        with z.open(n) as fh:
            total += sum(1 for _ in fh) - 1
    k = max(1, total // n_target)
    i = 0
    for n in names:
        with z.open(n) as fh:
            head = fh.readline().decode().strip().split(",")
            col = head.index("smiles") if "smiles" in head else 0
            for line in io.TextIOWrapper(fh, encoding="utf-8"):
                if i % k == 0:
                    keep.append(line.rstrip("\n").split(",")[col])
                i += 1
    return keep, total


def profile(name, smiles, total, workers):
    t0 = time.time()
    with Pool(workers) as p:
        rows = p.map(classify, smiles, chunksize=200)
    df = pd.DataFrame(rows)
    n = len(df)
    pep = df[df.kind != "not_peptide"]
    res = {
        "source": name, "total_rows": int(total), "sampled": n,
        "frac_parse_fail": round((df["std"].isna()).mean(), 4),
        "frac_peptide_le50": round(len(pep) / n, 4),
        "frac_canonical": round((df.kind == "canonical").mean(), 4),
        "frac_noncanonical": round((df.kind == "noncanonical").mean(), 4),
        "frac_cyclic_of_noncanonical": round(pep[pep.kind == "noncanonical"].cyclic.mean(), 3) if len(pep) else None,
        "dup_rate_in_sample": round(1 - df["std"].dropna().nunique() / max(1, df["std"].notna().sum()), 4),
        "residues_median": float(pep.residues.median()) if len(pep) else None,
        "share_res_le30_of_peptides": round((pep.residues <= 30).mean(), 3) if len(pep) else None,
        "minutes": round((time.time() - t0) / 60, 2),
    }
    res["est_canonical"] = int(res["frac_canonical"] * total)
    res["est_noncanonical"] = int(res["frac_noncanonical"] * total)
    # also the >50-residue share, so the window's cost is visible
    allpep = df[(df.residues >= 2)]
    res["frac_peptide_gt50"] = round((allpep.residues > 50).mean() * len(allpep) / n, 4)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-source", type=int, default=40000)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    jobs = {
        "pclm_esmatlas": lambda: sample_parquet("small_molecules.parquet", a.per_source),
        "pclm_pubchem": lambda: sample_parquet("peptides.parquet", a.per_source * 5),   # peptides are rare here
        "pepland_stage1": lambda: sample_pepland(a.per_source),
    }
    for name, fn in jobs.items():
        if a.only and name not in a.only:
            continue
        t0 = time.time()
        smiles, total = fn()
        print('%s: sampled %d of %s rows (%.1f min)' % (name, len(smiles), format(total, ','), (time.time() - t0) / 60), flush=True)
        r = profile(name, smiles, total, a.workers)
        print(json.dumps(r, indent=1), flush=True)
        json.dump(r, open(os.path.join(OUT, name + '.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()

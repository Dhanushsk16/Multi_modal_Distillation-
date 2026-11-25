"""Download ONLY the smiles column of a PeptideCLM-2 pretraining file from Hugging Face.

The upstream file names are swapped:
    small_molecules.parquet  holds ESMAtlas peptides  (9.5M rows)
    peptides.parquet         holds PubChem            (108M rows)

Resumable: each parquet row group is saved as its own part file, so an interrupted
download continues where it stopped. A final check compares the row count on disk
with the source's metadata.

    python src/corpus/download_hf.py --file small_molecules.parquet --name pclm_esmatlas
    python src/corpus/download_hf.py --file peptides.parquet        --name pclm_pubchem
    python src/corpus/download_hf.py --file ... --name ... --size-only
"""
from __future__ import annotations

import argparse
import glob
import os
import time

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPO = "datasets/aaronfeller/peptideclm-2-pretraining-data/train/"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--size-only", action="store_true")
    a = ap.parse_args()

    fs = HfFileSystem()
    with fs.open(REPO + a.file) as fh:
        pf = pq.ParquetFile(fh)
        md = pf.metadata
        col = [md.schema.column(i).name for i in range(md.num_columns)].index("smiles")
        size = sum(md.row_group(g).column(col).total_compressed_size for g in range(md.num_row_groups))
        print("%s: %s rows, %d row groups, smiles column %.2f GB"
              % (a.file, format(md.num_rows, ","), md.num_row_groups, size / 1e9), flush=True)
        if a.size_only:
            return

        out = os.path.join(ROOT, "data", "sources", a.name)
        os.makedirs(out, exist_ok=True)
        t0, done = time.time(), 0
        for g in range(md.num_row_groups):
            part = os.path.join(out, "part_%05d.parquet" % g)
            if os.path.exists(part):
                done += 1
                continue
            tbl = pf.read_row_group(g, columns=["smiles"])
            pq.write_table(tbl, part + ".tmp")
            os.replace(part + ".tmp", part)
            done += 1
            if done % 50 == 0 or done == md.num_row_groups:
                print("  %d/%d row groups | %.1f min" % (done, md.num_row_groups, (time.time() - t0) / 60), flush=True)

    parts = sorted(glob.glob(os.path.join(out, 'part_*.parquet')))
    rows = sum(pq.ParquetFile(x).metadata.num_rows for x in parts)
    disk = sum(os.path.getsize(x) for x in parts)
    ok = rows == md.num_rows and len(parts) == md.num_row_groups
    print('%s: %d parts, %s rows on disk vs %s expected, %.2f GB -> %s'
          % (a.name, len(parts), format(rows, ','), format(md.num_rows, ','), disk / 1e9,
             'COMPLETE' if ok else 'INCOMPLETE'))


if __name__ == '__main__':
    main()

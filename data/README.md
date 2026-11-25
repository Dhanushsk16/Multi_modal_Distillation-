# data/corpus

The training corpus of the teacher and the student, built from public sources (`src/corpus/`, scripts in `scripts/run_extract.sh`). The files are not stored in git (size) and are not distributed, because they are derived from sources that belong to their authors; `SHA256SUMS.txt` lists the checksums of the files used here. The benchmark data (`data/benchmarks/`) belong to their authors and are not included; `src/benchmarks.py` documents the layout.

| File | Rows | Columns | Size | What it is |
|---|---|---|---|---|
| `scored.parquet` | 12,028,166 | smiles, residues, kind, cyclic, source, bench_sim, bench_name | 1124 MB | Every extracted corpus molecule with its highest ECFP4 Tanimoto similarity to a benchmark molecule (bench_sim, bench_name): re-cut at another threshold with `clean_corpus.py --thresh`. |
| `clean.parquet` | 10,245,107 | smiles, residues, kind, cyclic, source, bench_sim, bench_name | 868 MB | The corpus after removing molecules within Tanimoto 0.9 of any benchmark molecule (`src/corpus/clean_corpus.py`). |
| `pool.parquet` | 3,400,449 | smiles, residues, kind, cyclic, source, shard | 337 MB | The training pool of the teacher and the student (3.40 M molecules), selected from clean.parquet by `src/corpus/build_pool.py` (extended to 50 residues with `--extend --cap 50`). Rows are shuffled and numbered into 51 shards; the validation rows are those with row % 100 == 0. |
| `pool_v1.parquet` | 2,640,418 | smiles, residues, kind, cyclic, source, shard | 219 MB | Backup of the first pool (peptides of at most 30 residues, 2.64 M molecules) made before the extension. |
| `pool_norm.parquet` | 3,400,449 | smiles_norm | 333 MB | The pool after the student's input normalisation (charges neutralised, RDKit canonical SMILES; `src/student/normalise.py`), row-aligned with pool.parquet. |
| `sources/pepland_stage1.parquet` | 7,612,415 | smiles, residues, kind, cyclic, source | 561 MB | Extracted source: PepLand's stage-1 peptides (PeptideAtlas and UniParc). |
| `sources/pclm_esmatlas.parquet` | 3,746,162 | smiles, residues, kind, cyclic, source | 500 MB | Extracted source: the ESMAtlas peptides of the PeptideCLM-2 corpus. |
| `sources/pclm_pubchem.parquet` | 692,174 | smiles, residues, kind, cyclic, source | 40 MB | Extracted source: the PubChem molecules of the PeptideCLM-2 corpus that are peptides (the source of non-canonical and cyclic peptides). |

Small files: `bench_fps.pkl`, `clean.json`, `leak_report.csv`, `pool.json`, `scored.json` (counts and reports written by the scripts above).

Not included: the per-molecule embedding caches of the pool (about 45 GB; rebuild with `scripts/run_cache.sh`, see the main README), the text export `pool_smiles/` (rewritten by `cache_views.py --export`) and the raw source dumps.

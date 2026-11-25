"""Filter one source down to the peptides we keep; write it with full bookkeeping.

    python src/corpus/extract.py --source pepland_stage1 --workers 32
    python src/corpus/extract.py --source pclm_pubchem --workers 32

Every molecule is standardised and classified (peptide_filters.classify). Kept:
peptides with MIN_RES..50 residues (MIN_RES = 2; PubChem uses 4 to drop drug-like
molecules that merely contain two amino-acid units). Exact duplicates are dropped
on the standardised SMILES.

Work is split into shards so a killed job resumes: each shard writes its own
parquet + json under <source>_shards/ and finished shards are skipped on re-run.
The merge pass (automatic after the last shard, or --merge-only) concatenates
them, de-duplicates across shards and writes:

    $DATA/data/corpus/sources/<source>.parquet   smiles, residues, kind, cyclic, source
    $DATA/data/corpus/sources/<source>.json      counts at every filter step

Every source gets a cheap carbonyl-count pre-filter on the raw string before
RDKit is allowed to parse anything; on PubChem (108M rows, ~0.6% peptides) that
is the difference between hours and days. prefilter_check.py proves the filter
drops no real peptide -- a non-canonical one lost here is lost for good, since
PubChem is their only source.

Smoke test on a subset without touching the real shards:

    BTP_DATA=/tmp/smoke PARTS_PER_SHARD=2 PEPLAND_SHARD_ROWS=2000 \
        python src/corpus/extract.py --source pclm_pubchem --limit-shards 2
"""
from __future__ import annotations

import argparse
import glob
import io
import json
import os
import sys
import time
import zipfile
import multiprocessing as mp
from collections import deque
from multiprocessing.connection import wait as wait_conns

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from peptide_filters import classify  # noqa: E402
from paths import CORPUS_SRC, SOURCES_DIR  # noqa: E402

CHUNK = 50_000          # SMILES handed to the worker pool at a time
BATCH = 50              # SMILES per worker task
# The 2026-11-03 PubChem run hung for 10 h on one molecule RDKit never returned
# from, and multiprocessing.Pool waited forever. Now a batch over BATCH_TIMEOUT
# gets its worker killed and is retried one molecule at a time; a molecule over
# MOL_TIMEOUT (or one that crashes its worker) is counted as timeout_or_crash.
BATCH_TIMEOUT = float(os.environ.get("BATCH_TIMEOUT", 300))
MOL_TIMEOUT = float(os.environ.get("MOL_TIMEOUT", 60))
HEARTBEAT_S = 120
# Shard sizing. Smaller shards cost less when a job is killed, but a shard is
# also the resume granularity, so keep them in the tens of minutes. Both can be
# overridden from the environment (smoke tests use tiny values).
PEPLAND_SHARD_ROWS = int(os.environ.get("PEPLAND_SHARD_ROWS", 500_000))
PARTS_PER_SHARD = int(os.environ.get("PARTS_PER_SHARD", 0))
COLS = ["smiles", "residues", "kind", "cyclic"]
STEPS = ("parse_fail", "prefiltered", "not_peptide_or_out_of_window",
         "below_min_res", "timeout_or_crash", "duplicate", "kept")

# Cheap pre-filter on the raw SMILES string, ~1000x cheaper than parsing it.
# A peptide with R residues has >= R-1 backbone amides, each with a carbonyl, and
# every carbonyl is written "=O" or "O=" -- so counting those substrings is a
# sound LOWER BOUND on the amide count and cannot drop a peptide long enough to
# keep.
#
# Matching amide patterns directly ("C(=O)N", "NC(=O)") is NOT sound, and this
# was measured, not guessed: ring-closure digits split the pattern ("N1C(=O)")
# and it silently dropped 18% of PAMPA. Counting nitrogens is unsound for the
# same class of reason -- "N" before an aromatic atom ("Nc1ccccc1") is a real
# nitrogen that a naive character rule throws away.
MIN_CARBONYLS = {2: 1, 4: 3}            # by min_res: a peptide needs min_res-1 amides


def carbonyls(smiles):
    return smiles.count("=O") + smiles.count("O=")


def peptidic(smiles, min_hits):
    """Could this string hold min_hits+1 amino-acid units? Must never reject a
    molecule that classify() would keep."""
    return carbonyls(smiles) >= min_hits


# --- readers: yield (shard_key, [smiles, ...]); a shard ends when the key changes --

def parquet_reader(subdir, default_parts_per_shard):
    def reader(done):
        import pyarrow.parquet as pq
        per = PARTS_PER_SHARD or default_parts_per_shard
        parts = sorted(glob.glob(os.path.join(SOURCES_DIR, subdir, "part_*.parquet")))
        assert parts, "no parts in %s/%s" % (SOURCES_DIR, subdir)
        for i in range(0, len(parts), per):
            group = parts[i:i + per]
            key = os.path.basename(group[0])[: -len(".parquet")]
            if done(key):
                continue
            buf = []
            for p in group:
                for s in pq.read_table(p, columns=["smiles"]).column("smiles").to_pylist():
                    if s:
                        buf.append(s)
                    if len(buf) >= CHUNK:
                        yield key, buf
                        buf = []
            yield key, buf          # closes the shard even when empty
    return reader


def pepland_reader(rows_per_shard=PEPLAND_SHARD_ROWS):
    def reader(done):
        z = zipfile.ZipFile(os.path.join(SOURCES_DIR, "pepland",
                                         "pep_atlas_uniparc_smiles.zip"))
        for part in ("train", "valid", "test"):
            with z.open("pep_atlas_uniparc_smiles/%s.csv" % part) as fh:
                text = io.TextIOWrapper(fh, encoding="utf-8")
                assert text.readline().strip() == "smiles", "unexpected header"
                cur, buf, i = None, [], 0
                for line in text:
                    key = "%s_%03d" % (part, i // rows_per_shard)
                    i += 1
                    if key != cur:
                        if cur is not None:
                            yield cur, buf
                        cur, buf = key, []
                    if done(key):
                        continue        # still counted, just never classified
                    s = line.strip()
                    if s:
                        buf.append(s)
                    if len(buf) >= CHUNK:
                        yield cur, buf
                        buf = []
                if cur is not None:
                    yield cur, buf
    return reader


SOURCES = {
    "pepland_stage1": {"reader": pepland_reader(), "min_res": 2},
    "pclm_esmatlas": {"reader": parquet_reader("pclm_esmatlas", 40), "min_res": 2},
    # 108M rows, ~0.6% peptides: small shards so a kill costs little
    "pclm_pubchem": {"reader": parquet_reader("pclm_pubchem", 10), "min_res": 4},
}
for _s in SOURCES.values():
    _s["prefilter"] = MIN_CARBONYLS[_s["min_res"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, choices=list(SOURCES))
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--merge-only", action="store_true")
    ap.add_argument("--limit-shards", type=int, default=0,
                    help="stop after N shards and skip the merge (smoke test)")
    a = ap.parse_args()
    spec = SOURCES[a.source]
    shard_dir = os.path.join(CORPUS_SRC, a.source + "_shards")
    os.makedirs(shard_dir, exist_ok=True)
    out_pq = os.path.join(CORPUS_SRC, a.source + ".parquet")
    if os.path.exists(out_pq):
        print("already done:", out_pq)
        return

    def shard_paths(key):
        return (os.path.join(shard_dir, key + ".parquet"),
                os.path.join(shard_dir, key + ".json"))

    def done(key):
        p, j = shard_paths(key)
        return os.path.exists(p) and os.path.exists(j)

    if not a.merge_only:
        extract(a.source, spec, a.workers, shard_paths, done, a.limit_shards)
    if a.limit_shards:
        print("--limit-shards: stopping before the merge")
        return
    merge(a.source, shard_dir, out_pq)


def _worker(conn, fn, init=None):
    if init:
        init()
    conn.send("ready")                  # imports and init are done: the parent starts its clock now
    while True:
        try:
            batch = conn.recv()
        except EOFError:
            return
        conn.send([fn(s) for s in batch])


class Workers:
    """Process pool where each worker is watched: one RDKit never returns from
    costs a timeout, not the night. spawn, not fork, so no worker inherits a lock
    held by the parent's pyarrow threads."""

    def __init__(self, n, fn, init=None):
        self.ctx, self.fn, self.init = mp.get_context("spawn"), fn, init
        self.procs = [self._start() for _ in range(n)]

    def _start(self):
        parent, child = self.ctx.Pipe()
        p = self.ctx.Process(target=_worker, args=(child, self.fn, self.init), daemon=True)
        p.start()
        child.close()
        return p, parent

    def _restart(self, w):
        p, c = self.procs[w]
        p.kill()
        p.join()
        c.close()
        self.procs[w] = self._start()

    def close(self):
        for p, c in self.procs:
            p.kill()
            c.close()

    def run(self, smiles):
        """Yield (smiles, result) in any order; result is None for a molecule that
        timed out or crashed its worker."""
        todo = deque((smiles[i:i + BATCH], False) for i in range(0, len(smiles), BATCH))
        busy = {}                                   # worker -> (batch, single, deadline)
        while todo or busy:
            for w in range(len(self.procs)):
                if w not in busy and todo:
                    batch, single = todo.popleft()
                    self.procs[w][1].send(batch)
                    busy[w] = (batch, single,
                               time.time() + (MOL_TIMEOUT if single else BATCH_TIMEOUT))
            by_conn = {self.procs[w][1]: w for w in busy}
            for c in wait_conns(list(by_conn), timeout=5):
                w = by_conn[c]
                try:
                    res = c.recv()
                except (EOFError, OSError):
                    res = None                      # the worker died
                if isinstance(res, str):            # "ready": a fresh worker finished importing
                    batch, single, _ = busy[w]
                    busy[w] = (batch, single,
                               time.time() + (MOL_TIMEOUT if single else BATCH_TIMEOUT))
                    continue
                if res is not None:
                    yield from zip(busy.pop(w)[0], res)
                    continue
                self._restart(w)
                yield from self._failed(busy.pop(w), todo)
            now = time.time()
            for w in [w for w, (_, _, dl) in busy.items() if now > dl]:
                self._restart(w)
                yield from self._failed(busy.pop(w), todo)

    @staticmethod
    def _failed(job, todo):
        batch, single, _ = job
        if single:
            yield batch[0], None
        else:                                       # find the culprit one by one
            todo.extendleft(([s], True) for s in batch)


def extract(source, spec, workers, shard_paths, done, limit=0):
    pool = Workers(workers, classify)
    try:
        _extract(pool, source, spec, shard_paths, done, limit)
    finally:
        pool.close()


def _extract(pool, source, spec, shard_paths, done, limit):
    min_res, min_hits = spec["min_res"], spec["prefilter"]
    t0, total_raw, n_shards, nchunk = time.time(), 0, 0, 0
    last_beat = time.time()
    cur, rows, seen, stats = None, [], set(), None
    bad_log = os.path.join(os.path.dirname(shard_paths("x")[0]), "timeout_or_crash.txt")

    def mins():
        return (time.time() - t0) / 60

    def flush():
        if cur is None:
            return
        p, j = shard_paths(cur)
        if os.path.exists(p) and os.path.exists(j):
            return                       # finished in an earlier run
        got = sum(stats[k] for k in STEPS if k != "kept")
        assert got + len(rows) == stats["raw"], \
            "bookkeeping mismatch in %s: %d vs %d" % (cur, got + len(rows), stats["raw"])
        stats["kept"] = len(rows)
        df = pd.DataFrame(rows, columns=COLS)
        df["source"] = source
        df.to_parquet(p + ".tmp", index=False)
        os.replace(p + ".tmp", p)
        with open(j + ".tmp", "w") as fh:
            json.dump(stats, fh, indent=1)
        os.replace(j + ".tmp", j)
        print("%s %s done: %s read, %s kept | %s read overall | %.1f min"
              % (source, cur, format(stats["raw"], ","), format(len(rows), ","),
                 format(total_raw, ","), mins()), flush=True)

    for key, chunk in spec["reader"](done):
        if key != cur:
            flush()
            if limit and n_shards >= limit:
                cur = None              # nothing half-done to write
                break
            n_shards += 1
            cur, rows, seen = key, [], set()
            stats = dict({"shard": key, "raw": 0}, **{k: 0 for k in STEPS})
        stats["raw"] += len(chunk)
        total_raw += len(chunk)
        if min_hits:
            keep = [s for s in chunk if peptidic(s, min_hits)]
            stats["prefiltered"] += len(chunk) - len(keep)
        else:
            keep = chunk
        for s, r in pool.run(keep):
            if r is None:
                stats["timeout_or_crash"] += 1
                with open(bad_log, "a") as fh:
                    fh.write("%s\t%s\n" % (key, s))
                print("%s %s: skipped a molecule that timed out or crashed its worker"
                      " (len %d), see %s" % (source, key, len(s), bad_log), flush=True)
            elif r["std"] is None:
                stats["parse_fail"] += 1
            elif r["kind"] == "not_peptide":
                stats["not_peptide_or_out_of_window"] += 1
            elif r["residues"] < min_res:
                stats["below_min_res"] += 1
            elif r["std"] in seen:
                stats["duplicate"] += 1
            else:
                seen.add(r["std"])
                rows.append((r["std"], r["residues"], r["kind"], r["cyclic"]))
        nchunk += 1
        # every ~250k rows, and at least every HEARTBEAT_S so a stall is visible
        if nchunk % 5 == 0 or time.time() - last_beat > HEARTBEAT_S:
            last_beat = time.time()
            print("%s %s: %s read in shard, %s kept | %s read overall | %.1f min"
                  % (source, key, format(stats["raw"], ","), format(len(rows), ","),
                     format(total_raw, ","), mins()), flush=True)
    flush()
    print("%s: shards done, %s rows read in %.1f min"
          % (source, format(total_raw, ","), mins()), flush=True)


def merge(source, shard_dir, out_pq):
    shards = sorted(glob.glob(os.path.join(shard_dir, "*.parquet")))
    assert shards, "nothing to merge in " + shard_dir
    stats = {"source": source, "shards": len(shards), "raw": 0}
    stats.update({k: 0 for k in STEPS})
    for s in shards:
        with open(s[: -len(".parquet")] + ".json") as fh:
            d = json.load(fh)
        for k in ("raw",) + STEPS:
            stats[k] += d.get(k, 0)             # timeout_or_crash is newer than some shards
    df = pd.concat((pd.read_parquet(s) for s in shards), ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset="smiles", ignore_index=True)
    stats["duplicate"] += before - len(df)          # same peptide in two shards
    stats["kept"] = len(df)
    stats["kept_canonical"] = int((df.kind == "canonical").sum())
    stats["kept_noncanonical"] = int((df.kind == "noncanonical").sum())
    stats["kept_cyclic"] = int(df.cyclic.sum())
    stats["residues_median"] = float(df.residues.median()) if len(df) else None
    got = sum(stats[k] for k in STEPS)
    assert got == stats["raw"], "bookkeeping mismatch: %d vs %d" % (got, stats["raw"])
    df.to_parquet(out_pq + ".tmp", index=False)
    os.replace(out_pq + ".tmp", out_pq)
    with open(os.path.join(os.path.dirname(out_pq), source + ".json"), "w") as fh:
        json.dump(stats, fh, indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()

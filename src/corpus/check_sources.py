"""Read-only check of the three extracted sources: files present, bookkeeping adds up.

    python ~/code/peptidefusion/src/corpus/check_sources.py
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import CORPUS_SRC  # noqa: E402

STEPS = ("parse_fail", "prefiltered", "not_peptide_or_out_of_window",
         "below_min_res", "timeout_or_crash", "duplicate", "kept")


def main():
    total, bad = 0, False
    for s in ("pepland_stage1", "pclm_esmatlas", "pclm_pubchem"):
        pq, js = (os.path.join(CORPUS_SRC, s + e) for e in (".parquet", ".json"))
        if not (os.path.exists(pq) and os.path.exists(js)):
            print("\n%s: MISSING (%s)" % (s, "parquet" if not os.path.exists(pq) else "json"))
            bad = True
            continue
        j = json.load(open(js))
        df = pd.read_parquet(pq)
        ok = sum(j.get(k, 0) for k in STEPS) == j["raw"] and j["kept"] == len(df)
        uniq = df.smiles.nunique() == len(df)
        bad |= not (ok and uniq)
        total += len(df)
        print("\n%s: raw %s, kept %s | canonical %s, noncanonical %s, cyclic %s"
              % (s, format(j["raw"], ","), format(len(df), ","),
                 format(j["kept_canonical"], ","), format(j["kept_noncanonical"], ","),
                 format(j["kept_cyclic"], ",")))
        print("  median residues %s | timeout_or_crash %s | bookkeeping %s | unique smiles %s"
              % (j["residues_median"], j.get("timeout_or_crash", 0),
                 "OK" if ok else "MISMATCH", uniq))
        print("  columns", list(df.columns))
    skipped = os.path.join(CORPUS_SRC, "pclm_pubchem_shards", "timeout_or_crash.txt")
    if os.path.exists(skipped):
        print("\nskipped-molecule log: %d lines" % sum(1 for _ in open(skipped)))
    print("\nTOTAL kept across sources: %s | %s" % (format(total, ","), "PROBLEM" if bad else "all OK"))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()

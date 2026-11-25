"""Table of eval_mlm.py results: every model on identical molecules and masks.

    python ~/code/shape_pilot/distill/summarize_mlm_eval.py
"""
from __future__ import annotations

import csv
import glob
import json
import os

DATA = os.environ.get("BTP_DATA", "/mnt/wwn-0x5000c500e64c5cc0/dhanushsk/BTPData")
EVAL = f"{DATA}/results/mlm_eval"
GROUPS = ["all", "peptide", "small_molecule", "lipid"]


def main():
    rows = [json.load(open(p)) for p in glob.glob(f"{EVAL}/*.json")]
    assert rows, "no results in %s" % EVAL
    rows.sort(key=lambda r: r["all"]["loss"])

    head = "%-16s %8s" % ("model", "params") + "".join(
        " | %-15s" % g[:15] for g in GROUPS)
    print(head + "\n" + " " * 25 + "".join(" | %-15s" % "loss   acc" for _ in GROUPS))
    print("-" * len(head))
    for r in rows:
        cells = "".join(" | %.4f %.3f  " % (r[g]["loss"], r[g]["acc"]) if g in r
                        else " | %-15s" % "-" for g in GROUPS)
        print("%-16s %7.1fM%s" % (r["name"], r["params_M"], cells))
    n = rows[0]["all"]["masked_tokens"]
    print("\nmasked tokens scored per model: %s (identical for every model)" % "{:,}".format(n))
    assert all(r["all"]["masked_tokens"] == n for r in rows), \
        "models were scored on different masks -- max_tokens/seed differ"

    out = f"{EVAL}/summary.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["model", "params_M"] + [f"{g}_{m}" for g in GROUPS for m in ("loss", "acc")])
        for r in rows:
            w.writerow([r["name"], r["params_M"]] + [
                r[g][m] if g in r else "" for g in GROUPS for m in ("loss", "acc")])
    print("wrote", out)


if __name__ == "__main__":
    main()

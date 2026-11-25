"""One table for the 10M shape pilot: which depth/width wins at a fixed budget.

For every shape that has been run, reads (all on the data drive):
    {CKPT_DIR}/shape_pilot/<run>/history.json, final_val.json   from train_kd.py
    {CKPT_DIR}/students/student-<run>/config.json               the shape
    {LOG_DIR}/shape_pilot/<run>.log                             wall time
    {RESULTS_DIR}/shape_pilot/latency/<run>_cuda_bf16.json      from bench_latency.py

Decision rule (fixed before the runs): lowest final val MLM loss; if the top
two are within 0.02 nats, prefer the one with fewer blocks -- latency is
sequential in depth, and a 10M model exists to be cheap.

    python ~/code/shape_pilot/distill/summarize_pilot.py
"""
from __future__ import annotations

import csv
import json
import os
import re

DATA = os.environ.get("BTP_DATA", "/mnt/wwn-0x5000c500e64c5cc0/dhanushsk/BTPData")
CKPT_DIR = f"{DATA}/checkpoints"
RESULTS_DIR = f"{DATA}/results"
LOG_DIR = f"{DATA}/logs"

RUNS = f"{CKPT_DIR}/shape_pilot"
OUT = f"{RESULTS_DIR}/shape_pilot"
TIE = 0.02


def run_row(run):
    d = os.path.join(RUNS, run)
    hist = json.load(open(os.path.join(d, "history.json")))
    evals = [h["eval"] for h in hist if "eval" in h]
    shape = run.split("_")[0]                     # "s320x10_lr5e-4" -> "s320x10"
    cfg = json.load(open(f"{CKPT_DIR}/students/student-{shape}/config.json"))
    row = {"run": run, "embed_dim": cfg["embed_dim"], "blocks": cfg["num_blocks"],
           "heads": cfg["num_heads"], "ffn": cfg["ffn_hidden_dim"],
           "finished": os.path.exists(os.path.join(d, "final_val.json"))}

    final = os.path.join(d, "final_val.json")
    last = json.load(open(final)) if row["finished"] else (evals[-1] if evals else {})
    row["val_loss"] = round(last.get("hard", float("nan")), 4)
    row["val_acc"] = round(last.get("acc_student", float("nan")), 4)
    row["steps"] = max((h.get("step", -1) for h in hist if "step" in h), default=-1) + 1

    row["wall_min"] = row["params_M"] = None
    log = f"{LOG_DIR}/shape_pilot/{run}.log"
    if os.path.exists(log):
        txt = open(log, errors="ignore").read()
        m = re.findall(r"done: \d+ steps in ([\d.]+) min", txt)
        row["wall_min"] = float(m[-1]) if m else None
        p = re.findall(r"student ([\d.]+)M", txt)
        row["params_M"] = float(p[0]) if p else None

    lat = f"{OUT}/latency/{shape}_cuda_bf16.json"
    row["ms_per_mol"] = json.load(open(lat))["ms_per_molecule"] if os.path.exists(lat) else None
    return row


def main():
    runs = sorted(r for r in os.listdir(RUNS)
                  if os.path.exists(os.path.join(RUNS, r, "history.json")))
    assert runs, "no runs under %s" % RUNS
    rows = sorted((run_row(r) for r in runs), key=lambda r: r["val_loss"])

    cols = ["run", "params_M", "embed_dim", "blocks", "val_loss", "val_acc",
            "steps", "wall_min", "ms_per_mol", "finished"]
    print(" | ".join("%-10s" % c for c in cols))
    for r in rows:
        print(" | ".join("%-10s" % ("-" if r.get(c) is None else r.get(c)) for c in cols))

    done = [r for r in rows if r["finished"]]
    if len(done) >= 2:
        best = done[0]
        if done[1]["val_loss"] - best["val_loss"] < TIE:
            pick = min(done[:2], key=lambda r: r["blocks"])
            print("\ntop two within %.2f nats (%.4f vs %.4f): tie -> fewer blocks: %s"
                  % (TIE, best["val_loss"], done[1]["val_loss"], pick["run"]))
            print("rerun these two at LR=5e-4 before committing.")
        else:
            print("\nwinner: %s (val loss %.4f)" % (best["run"], best["val_loss"]))

    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)
    print("wrote", f"{OUT}/summary.csv")


if __name__ == "__main__":
    main()

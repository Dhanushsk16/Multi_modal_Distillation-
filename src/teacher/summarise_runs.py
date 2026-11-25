"""One table that says whether teacher-v3 training runs finished and how healthy they look (reads results/teacher/<run>/history.csv).

    python src/teacher/summarise_runs.py                 # the nine runs of `run_teacher_v3.sh seeds` + `abl`
    python src/teacher/summarise_runs.py --all           # every run found
    python src/teacher/summarise_runs.py abl_base_s1 v3  # chosen runs

Columns: epochs run, best epoch, how it stopped (early = patience ran out), minutes, done (the run reached its end: latent_stats.npz
exists), validation loss on standard / modified molecules at the best epoch, and the R2 with which each view is rebuilt at the best
epoch. Do NOT rank variants by the loss: they rebuild different things (no ECFP has no ECFP term, ECFP-weight 2 counts it twice);
the benchmarks (eval_harness.py) are the judge.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from paths import RESULTS_DIR, TEACHER_DIR  # noqa: E402

NINE = "abl_base_s1 abl_base_s2 abl_base_s3 abl_noh_s1 abl_vd0_s1 abl_vd3_s1 abl_wh2_s1 abl_noe_s1 abl_bal19_s1".split()
D = os.path.join(RESULTS_DIR, "teacher")


def num(r, k):
    return ("%.3f" % float(r[k])) if r.get(k) not in (None, "", "nan") else "  -  "


args = sys.argv[1:]
if "--all" in args:
    names = sorted(x for x in os.listdir(D) if os.path.exists(os.path.join(D, x, "history.csv"))) if os.path.isdir(D) else []
else:
    names = args or NINE
print("%-14s %6s %5s %7s %6s %5s | %7s %7s | R2 standard: PepLand PeptideCLM ECFP ESM-2 | R2 modified: PepLand PeptideCLM ECFP"
      % ("run", "epochs", "best", "stop", "min", "done", "val_std", "val_mod"))
for n in names:
    p = os.path.join(D, n, "history.csv")
    if not os.path.exists(p):
        print("%-14s no history.csv: not started, or crashed before the first epoch" % n)
        continue
    rows = list(csv.DictReader(open(p)))
    best = [r for r in rows if r["best"] == "1"][-1]
    done = "yes" if os.path.exists(os.path.join(TEACHER_DIR, n, "latent_stats.npz")) else "NO"
    stop = "early" if int(rows[-1]["bad"]) >= 8 else "cap/run"
    mins = sum(float(r["seconds"]) for r in rows) / 60
    print("%-14s %6d %5s %7s %6.1f %5s | %7s %7s | %s %s %s %s | %s %s %s" % (
        n, len(rows), best["epoch"], stop, mins, done, num(best, "loss_std"), num(best, "loss_mod"),
        num(best, "r2_P_std"), num(best, "r2_L_std"), num(best, "r2_H_std"), num(best, "r2_E_std"),
        num(best, "r2_P_mod"), num(best, "r2_L_mod"), num(best, "r2_H_mod")))

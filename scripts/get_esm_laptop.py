"""Download of ESM-2 weights on a machine with internet (the lab server has none).

    python scripts\\get_esm_laptop.py                         # esm2_t36_3B_UR50D (3B, 2560-d, ~11 GB) -> Downloads\\esm2_t36_3B_UR50D
    python scripts\\get_esm_laptop.py --size 650M             # esm2_t33_650M_UR50D (~2.6 GB)
    python scripts\\get_esm_laptop.py --repo facebook/esm2_t6_8M_UR50D --dest C:\\path\\esm8M    # any ESM-2 repo (small ones for tests)

Downloads safetensors weights (falls back to the PyTorch .bin files when a repo has none), then prints a checklist.
Needs `pip install huggingface_hub` (already there when `transformers` is installed). Resumable: run it again.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil

SIZES = {"3B": "facebook/esm2_t36_3B_UR50D", "650M": "facebook/esm2_t33_650M_UR50D"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=sorted(SIZES), default="3B")
    ap.add_argument("--repo", default=None, help="Hugging Face repo id (overrides --size)")
    ap.add_argument("--dest", default=None, help="local folder (default: ~/Downloads/<repo name>)")
    a = ap.parse_args()
    from huggingface_hub import snapshot_download
    repo = a.repo or SIZES[a.size]
    name = repo.split("/")[-1]
    dest = a.dest or os.path.join(os.path.expanduser("~"), "Downloads", name)
    small = ["*.json", "*.txt", "*.model"]
    print("downloading %s -> %s" % (repo, dest))
    snapshot_download(repo_id=repo, local_dir=dest, allow_patterns=small + ["*.safetensors"])
    if not glob.glob(os.path.join(dest, "*.safetensors")):
        print("no safetensors in the repo: downloading the PyTorch .bin weights")
        snapshot_download(repo_id=repo, local_dir=dest, allow_patterns=small + ["*.bin"])
    shutil.rmtree(os.path.join(dest, ".cache"), ignore_errors=True)          # Hugging Face's bookkeeping: not needed on the server
    files = sorted(os.listdir(dest))
    weights = [f for f in files if f.endswith((".safetensors", ".bin"))]
    need = ["config.json", "vocab.txt"]
    missing = [f for f in need if f not in files] + ([] if weights else ["weights (*.safetensors or *.bin)"])
    size_gb = sum(os.path.getsize(os.path.join(dest, f)) for f in files) / 1e9
    print("\nfiles: %s\ntotal %.2f GB" % (files, size_gb))
    if missing:
        raise SystemExit("INCOMPLETE, missing: %s" % missing)
    print("OK. The weights are in %s." % dest)


if __name__ == "__main__":
    main()

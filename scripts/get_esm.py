"""SERVER-side: test whether this machine can reach the internet, and download ESM-2 only if it can.

    python ~/code/peptidefusion/scripts/get_esm.py --check            # just the connectivity test (takes < 30 s)
    python ~/code/peptidefusion/scripts/get_esm.py                    # check, then download esm2_t36_3B_UR50D (3B, 2560-d, ~11 GB)
    python ~/code/peptidefusion/scripts/get_esm.py --size 650M        # esm2_t33_650M_UR50D (~2.6 GB)

It downloads into $CKPT_DIR/<repo name> on the data drive (never into home), safetensors first (PyTorch .bin files as a fallback),
and can be run again to resume. When the server is offline it says so plainly and stops: then download the weights on a machine with internet
(scripts/get_esm_laptop.py). The test resolves the name, then asks Hugging Face for ESM's config.json
(any HTTP answer, even an error code, proves the server is online; only DNS failures, refused connections and timeouts mean offline).
Proxy settings (http_proxy / https_proxy) are honoured, as they would be by the real download.
"""
from __future__ import annotations

import argparse
import glob
import os
import socket
import sys
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
from paths import CKPT_DIR, DATA  # noqa: E402

SIZES = {"3B": "facebook/esm2_t36_3B_UR50D", "650M": "facebook/esm2_t33_650M_UR50D"}
HOST = "huggingface.co"


def internet(timeout=10):
    """-> (online: bool, lines: list of what was seen)."""
    lines = []
    prox = {k: v for k, v in os.environ.items() if k.lower() in ("http_proxy", "https_proxy", "all_proxy", "no_proxy")}
    lines.append("proxy settings: %s" % (prox if prox else "none"))
    try:
        ip = socket.gethostbyname(HOST)
        lines.append("DNS: %s -> %s" % (HOST, ip))
    except OSError as e:
        if not prox:
            lines.append("DNS: cannot resolve %s (%s)" % (HOST, e))
            return False, lines
        lines.append("DNS: cannot resolve %s directly (%s); a proxy may still handle it" % (HOST, e))
    url = "https://%s/facebook/esm2_t6_8M_UR50D/resolve/main/config.json" % HOST
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=timeout)
        lines.append("HTTPS to %s: HTTP %s" % (HOST, r.status))
        return True, lines
    except urllib.error.HTTPError as e:            # the server answered: we are online
        lines.append("HTTPS to %s: HTTP %s (answered, so the internet is reachable)" % (HOST, e.code))
        return True, lines
    except Exception as e:                         # URLError, timeout, refused, SSL, ...
        lines.append("HTTPS to %s failed: %s: %s" % (HOST, type(e).__name__, e))
        return False, lines


def download(repo, dest):
    os.environ.setdefault("HF_HOME", os.path.join(DATA, "cache", "huggingface"))      # never ~/.cache
    from huggingface_hub import snapshot_download
    small = ["*.json", "*.txt", "*.model"]
    print("downloading %s -> %s (resumable)" % (repo, dest), flush=True)
    snapshot_download(repo_id=repo, local_dir=dest, allow_patterns=small + ["*.safetensors"])
    if not glob.glob(os.path.join(dest, "*.safetensors")):
        print("no safetensors in the repo: downloading the PyTorch .bin weights", flush=True)
        snapshot_download(repo_id=repo, local_dir=dest, allow_patterns=small + ["*.bin"])
    files = sorted(os.listdir(dest))
    weights = [f for f in files if f.endswith((".safetensors", ".bin"))]
    gb = sum(os.path.getsize(os.path.join(dest, f)) for f in files if os.path.isfile(os.path.join(dest, f))) / 1e9
    print("files: %s\ntotal %.2f GB" % (files, gb))
    missing = [f for f in ("config.json", "vocab.txt") if f not in files] + ([] if weights else ["weights"])
    if missing:
        raise SystemExit("INCOMPLETE, missing: %s (run the same command again to resume)" % missing)
    print("OK: %s is complete" % dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only test the internet connection")
    ap.add_argument("--size", choices=sorted(SIZES), default="3B")
    ap.add_argument("--repo", default=None, help="Hugging Face repo id (overrides --size)")
    ap.add_argument("--dest", default=None, help="folder (default: $CKPT_DIR/<repo name>)")
    ap.add_argument("--timeout", type=int, default=10)
    a = ap.parse_args()
    online, lines = internet(a.timeout)
    for l in lines:
        print("  " + l)
    if not online:
        print("\nRESULT: this machine can NOT reach the internet (or Hugging Face).\n"
              "Download the weights on a machine with internet instead:  python scripts\\get_esm_laptop.py")
        sys.exit(2)
    print("\nRESULT: this machine CAN reach Hugging Face.")
    if a.check:
        return
    repo = a.repo or SIZES[a.size]
    download(repo, a.dest or os.path.join(CKPT_DIR, repo.split("/")[-1]))


if __name__ == "__main__":
    main()

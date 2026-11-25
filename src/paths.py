"""Every path in one place.

Lab server rules: code lives under ~/code, everything read or written at runtime
lives on the data drive, and all paths are absolute. BTP_DATA overrides the data
root (e.g. for a local test); nothing here ever resolves relative to the cwd.
"""
import os

DATA = os.environ.get("BTP_DATA", "/mnt/wwn-0x5000c500e64c5cc0/dhanushsk/BTPData")
CKPT_DIR = f"{DATA}/checkpoints"
RESULTS_DIR = f"{DATA}/results"
LOG_DIR = f"{DATA}/logs"

BENCH_DIR = f"{DATA}/data/benchmarks"
SOURCES_DIR = f"{DATA}/data/sources"          # raw downloads (pepland/, pclm_*/)
CORPUS_DIR = f"{DATA}/data/corpus"            # everything built from them
CORPUS_SRC = f"{CORPUS_DIR}/sources"          # one parquet per extracted source
FEAT_DIR = f"{DATA}/features/single_modality"
SINGLE_OUT = f"{RESULTS_DIR}/single_modality"
CACHE_DIR = f"{DATA}/features/corpus"         # per-molecule view caches of the pool (cache_views.py)
TEACHER_DIR = f"{CKPT_DIR}/teacher"           # teacher checkpoints, one folder per run name

def _hf_model_dir(name, env=None):
    """Model folder checkpoints/<name> if it exists (env var `env` overrides it); otherwise the newest snapshot of the Hugging Face
    cache copy checkpoints/models--aaronfeller--<name>/snapshots/<hash>/ (how the MTR models were downloaded on the server)."""
    import glob
    flat = (os.environ.get(env, "") if env else "") or f"{CKPT_DIR}/{name}"
    if os.path.exists(f"{flat}/model.safetensors"):
        return flat
    snaps = glob.glob(f"{CKPT_DIR}/models--aaronfeller--{name}/snapshots/*/model.safetensors")
    return os.path.dirname(max(snaps, key=os.path.getmtime)) if snaps else flat


PCLM_DIR = f"{CKPT_DIR}/peptideclm-2-mlm-large"
PCLM_MTR_DIR = _hf_model_dir("peptideclm-2-mtr-large", "PCLM_MTR_DIR")    # same architecture and tokenizer as the MLM model, other weights
PEPLAND_MODEL = f"{CKPT_DIR}/pepland_model"
ESM_DIR = os.environ.get("ESM_DIR", f"{CKPT_DIR}/esm2_t36_3B_UR50D")     # ESM-2 3B weights (scripts/get_esm_laptop.py downloads them); ESM_DIR=... for another size

CODE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))   # the code checkout
EXTERNAL = f"{CODE}/external"                                            # pepland/, mapchiral/

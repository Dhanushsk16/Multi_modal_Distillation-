"""Run one of the PeptideCLM-2 authors' training scripts UNEDITED, under the fixes this server needs.

    python src/repro/run_authors_script.py <script.py> [the script's own arguments]

The scripts are verbatim copies in external/peptideclm2_authors/ (SOURCE.txt: repository, commit). Three things are added around
them, none of which changes what they compute:
  1. model names: the scripts ask for "aaronfeller/peptideclm-2-mlm-large" etc.; the server has no internet, so the names are
     mapped to the local folders (paths.py). Anything else is passed through untouched.
  2. the rotary buffers: the checkpoint does not hold them and they come out as garbage after from_pretrained (README.md, "Known
     pitfalls"); featurize.fix_rotary rebuilds and verifies them after EVERY AutoModel.from_pretrained (the scripts call it again
     inside load_from_checkpoint). The SDPA mask fix of featurize._patch_sdpa is installed too.
  3. ACCUM=<n> (environment): the Lightning Trainer gets accumulate_grad_batches=n, so a run that does not fit in memory at batch 32
     can use `--batch_size 32/n` and still take one optimiser step per 32 molecules (the effective batch of the authors' recipe).
     Default 1 = nothing changes.
Offline mode is switched on (HF_HUB_OFFLINE=1).
"""
from __future__ import annotations

import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import featurize  # noqa: E402
from paths import PCLM_DIR, PCLM_MTR_DIR, _hf_model_dir  # noqa: E402

ALIAS = {
    "aaronfeller/peptideclm-2-mlm-large": PCLM_DIR,
    "aaronfeller/peptideclm-2-mtr-large": PCLM_MTR_DIR,
    "aaronfeller/peptideclm-2-hybrid-large": _hf_model_dir("peptideclm-2-hybrid-large", "PCLM_HYBRID_DIR"),
}


def install():
    import transformers
    featurize._patch_sdpa()
    for cls in (transformers.AutoModel, transformers.AutoTokenizer):
        orig = cls.from_pretrained

        def patched(name, *a, __orig=orig, __is_model=(cls is transformers.AutoModel), **k):
            name = ALIAS.get(str(name), name)
            out = __orig(name, *a, **k)
            if __is_model:
                n = featurize.fix_rotary(out)
                print("[run_authors_script] %s loaded, rotary buffers rebuilt and verified in %d modules" % (name, n), flush=True)
            return out

        cls.from_pretrained = staticmethod(patched)
    acc = int(os.environ.get("ACCUM", "1"))
    if acc > 1:
        import lightning as L
        init = L.Trainer.__init__

        def trainer_init(self, *a, **k):
            k["accumulate_grad_batches"] = acc
            init(self, *a, **k)

        L.Trainer.__init__ = trainer_init
        print("[run_authors_script] accumulate_grad_batches=%d" % acc, flush=True)


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    script = os.path.abspath(sys.argv[1])
    install()
    sys.argv = [script] + sys.argv[2:]
    sys.path.insert(0, os.path.dirname(script))               # `from kan import KANLinear` (kan.py lies next to the script)
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main()

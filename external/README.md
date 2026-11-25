# external/

Third-party code that the scripts read from this folder (`src/paths.py`: `EXTERNAL = <repo>/external`). It is copied here unchanged, without its git history, at the commits that were used; each folder keeps its own licence file.

| Folder | Source and commit | Licence | Used for |
|---|---|---|---|
| `pepland/` | https://github.com/zhangruochi/pepland @ `875dca817a0be6bc1f6a3322d42142a5f4438706` | Apache-2.0 | PepLand embeddings (`src/featurize_pepland.py`, `src/corpus/cache_views.py --view pepland`); its trained weights are in `pepland/cpkt/` (26 MB) |
| `mapchiral/` | https://github.com/reymond-group/mapchiral @ `f7b1032e501d3f26ac1f8bd211cab6278a524957` | MIT (stated in its README; its `LICENSE` file is empty) | the MAPC fingerprint (`src/featurize.py --rep mapc`) |
| `peptideclm2_authors/` | https://github.com/AaronFeller/PeptideCLM-2 @ `6b9708d4cb05717307d310daaf1c0f88c71ff084`, folder `training/03_PepMSND_training_code/` (verbatim, see `SOURCE.txt`) | MIT | `src/repro/run_authors_script.py` (reproduction of the PepMSND training) |

The PeptideCLM-2 repository also holds the benchmark data and the result files from which the THPep splits were rebuilt (`src/build_thpep_official_splits.py`); clone it separately if you want to repeat that step:

```
git clone https://github.com/AaronFeller/PeptideCLM-2.git
git -C PeptideCLM-2 checkout 6b9708d4cb05717307d310daaf1c0f88c71ff084
```

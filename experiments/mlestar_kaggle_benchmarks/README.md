# MLE-STAR Kaggle benchmark experiments

This directory is Jiaozi's isolated, reproducible experimentation package for
[MLE-STAR: Machine Learning Engineering Agent via Search and Targeted
Refinement](https://arxiv.org/abs/2506.15692). It keeps its Python environment
and runtime dependencies separate from the root application while remaining a
versioned part of the course project.

The agent follows the paper's experimental sequence:

1. Retrieve task-specific candidate models and evaluate them on fixed folds.
2. Merge only validation-improving initial solutions.
3. Ablate individual pipeline blocks, refine the highest-impact block, and
   retain only improvements.
4. Select ensembles from out-of-fold (OOF) predictions only.

The package includes skrub DataOps utilities for serializable metadata and
artifact-path graphs. The current `compare` runner calls the dataset adapters
directly; adapters own tensors and models and persist their outputs inside the
run directory.

## Install

```bash
cd experiments/mlestar_kaggle_benchmarks
python -m pip install -e '.[vision,llm,kaggle,dev]'
python -m pytest -q
python -m mlestar.cli compare --benchmark leaf_classification \
  --data-root examples/synthetic_leaf --run-root /tmp/mlestar-smoke \
  --seeds 13 --no-submit
```

## Scope of benchmark results

Seven of the ten catalogued tasks have executable training adapters (one
tabular, six image-classification, sharing a common timm fine-tuning
pipeline). The remaining three -- object detection, segmentation, and
image denoising -- are registered in the catalog but not yet implemented,
and `mlestar compare` fails loudly rather than fabricating a result for
them.

The current runner compares baseline, initial, targeted-refinement, and ensemble
arms with deterministic fold construction and identical seeds. It does not
enforce equal wall-clock budgets across arms. The historical Kaggle
competitions may be closed, and this package reports offline OOF metrics only;
any external leaderboard score needs a separate, verifiable receipt.

See `docs/EVALUATION_PROTOCOL.md` for the study protocol and
`docs/BENCHMARK_STATUS.md` for the current adapter coverage.

## Reproducible smoke experiment

Run the scripted synthetic-data check with Python 3.11 or newer:

```bash
python scripts/run_smoke_experiment.py --output-dir /tmp/mlestar-smoke
```

The runner fixes the seed at 13, uses only the committed synthetic Leaf
Classification fixture, and never writes or uploads a Kaggle submission.
It writes `comparison.csv`, `result.json`, and `manifest.json`. The manifest
records the configuration, per-file data hashes, a dependency-safe environment
summary, Git revision state, project version, source hash, and output hashes.
This provenance manifest belongs specifically to the scripted smoke entry
point; the general `mlestar compare` command does not create it.
Use a fresh output directory for every run and keep generated artifacts outside
the repository.

For Colab, use `notebooks/mlestar_kaggle_experiments.ipynb` and follow
`docs/COLAB.md`.

The experiment package is licensed under Apache-2.0; see `LICENSE` in this
directory.

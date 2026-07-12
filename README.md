# Standalone MLE-STAR reproduction

This repository is an independent reproduction of [MLE-STAR: Machine Learning
Engineering Agent via Search and Targeted Refinement](https://arxiv.org/abs/2506.15692).
It is not part of Jiaozi and has no Jiaozi runtime dependency.

The agent follows the paper's experimental sequence:

1. Retrieve task-specific candidate models and evaluate them on fixed folds.
2. Merge only validation-improving initial solutions.
3. Ablate individual pipeline blocks, refine the highest-impact block, and
   retain only improvements.
4. Select ensembles from out-of-fold (OOF) predictions only.

Every run is a skrub DataOps graph whose state contains immutable metadata and
artifact paths. Dataset adapters own tensors/models and persist their outputs
inside the run directory.

## Install

```bash
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

The ten catalogued tasks are evaluated with fixed folds, identical seeds and
matched time budgets across baseline, initial, targeted-refinement and ensemble
arms. The historical Kaggle competitions may be closed: a current public score
is only reported if Kaggle accepts a submission. Otherwise the report records
the API error and compares offline OOF metrics only.

See `docs/superpowers/plans/2026-07-11-standalone-mlestar-reproduction.md` for
the implementation plan and `docs/EVALUATION_PROTOCOL.md` for the final study
protocol.

For Colab, use `notebooks/mlestar_kaggle_experiments.ipynb` and follow
`docs/COLAB.md`.

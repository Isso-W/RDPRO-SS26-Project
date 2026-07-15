# Experimental Results

This report distinguishes checked-in measurements from pending GPU/Kaggle work. No missing score is estimated or filled in.

## Completed: Cassava paired loss comparison

The repository contains ten fold-level records: five folds for focal loss and the same five folds for cross entropy. Both arms use seed 42, EfficientNet-B0, 224-pixel inputs, five epochs, ordinary shuffled sampling, and the same fold-file SHA-256 value.

Primary metric: macro-F1. The paired difference is `cross entropy - focal`.

| Fold | Focal | Cross entropy | Difference |
| ---: | ---: | ---: | ---: |
| 0 | 0.7035 | 0.7180 | +0.0145 |
| 1 | 0.7190 | 0.7231 | +0.0041 |
| 2 | 0.7211 | 0.7242 | +0.0031 |
| 3 | 0.6972 | 0.7013 | +0.0041 |
| 4 | 0.7107 | 0.7224 | +0.0117 |

The collector reconstructs a mean paired difference of approximately `+0.0075`, standard error `0.0023`, and tie band `±0.0050`, producing the testbed verdict `CE_WINS`. All five paired directions favor cross entropy.

Source records: `experiments/ab_loss_imbalance/results/outcomes.jsonl`.

Interpretation is deliberately limited to this Cassava setup. It is evidence for moderately imbalanced, natural-image multiclass classification. It is not evidence that cross entropy is universally preferable for severe imbalance or medical data, so the global Jiaozi class-imbalance default remains focal loss.

## Completed: offline reproducibility checks

The following checks exercise software and experiment contracts but are not model-quality benchmarks:

- root unit tests and deterministic Module 4 synthetic smoke generation;
- reconstruction of the Cassava verdict from the ten checked-in JSONL records;
- standalone benchmark unit tests and a synthetic Leaf Classification comparison with submission disabled.

Exact commands are in `EXPERIMENTS.md` and the continuous-integration workflow. A passing synthetic run proves that the scripted path works; it does not constitute a public leaderboard result.

## Pending

| Study | Status | Reason no score is reported |
| --- | --- | --- |
| SIIM-ISIC CE versus focal paired run | Pending | Dataset download, accepted competition terms, GPU time, and the full paired matrix are required. |
| Remaining real-data loss-comparison testbeds | Pending | No checked-in fold records exist. |
| Standalone real Kaggle benchmark suite | Pending | The repository does not contain verified real-data OOF runs for all catalog entries. |
| Public Kaggle scores | Pending | A score is reported only after an explicit submission is accepted and processed by Kaggle. |
| Detection, segmentation, and denoising standalone adapters | Not implemented | These catalog entries fail loudly rather than fabricating a result. |

Seven standalone benchmark entries have executable tabular or image-classification adapters; three entries remain catalog-only. See `experiments/mlestar_kaggle_benchmarks/docs/BENCHMARK_STATUS.md` for the maintained status table.

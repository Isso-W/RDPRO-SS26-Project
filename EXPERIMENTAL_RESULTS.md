# Experimental Results

This report separates local validation, packaged inference, accepted Kaggle submissions, blocked submissions, and unexecuted templates. No missing score is estimated. Every reported value is traceable to a stored notebook output and its per-cell log listed in [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Jiaozi pipeline runs

| Competition | Stored validation result | Kaggle result | Evidence and limitation |
| --- | --- | --- | --- |
| APTOS 2019 Blindness Detection | QWK `0.891612`, accuracy `0.818554`, epoch 14 | No score | Submission creation succeeded, but both submit attempts returned HTTP 400 and the submissions query returned no records. [`log`](experiments/notebook_runs/logs/jiaozi/aptos_2019.log) |
| Global Wheat Detection | Local `mAP@0.50:0.75` **proxy** `0.543473` | No score | The ten-row submission passed the notebook's format check, but Kaggle returned HTTP 400. The local value is explicitly a proxy, not the official Kaggle metric. [`log`](experiments/notebook_runs/logs/jiaozi/global_wheat_detection.log) |
| Dogs vs. Cats Redux | Log loss `0.019187`, accuracy `0.997000`, epoch 14 | Public `0.06594`, private `0.06594` | Submission status is complete in the stored receipt. [`log`](experiments/notebook_runs/logs/jiaozi/dogs_vs_cats_redux.log) |
| Histopathologic Cancer Detection | ROC-AUC `0.980494`, accuracy `0.936416`, epoch 15 | Public `0.9599`, private `0.9623` | The training cell raised a post-run guard because the log contained `fallback`. The scores are real stored outputs, but this run must **not** be described as a verified DINOv3 result. [`log`](experiments/notebook_runs/logs/jiaozi/histopathologic_cancer.log) |
| Plant Pathology 2020 | ROC-AUC `0.976635`, accuracy `0.914601`, epoch 40 | No score | A submission file was written; the submit cell has no execution count or output. [`log`](experiments/notebook_runs/logs/jiaozi/plant_pathology_2020.log) |
| Aerial Cactus Identification | ROC-AUC `0.999922`, accuracy `0.997714`, epoch 20 | No score | Training and asset packaging completed. A separate inference notebook produced a `(4000, 2)` submission file, but no processed Kaggle receipt is stored. [`training log`](experiments/notebook_runs/logs/jiaozi/aerial_cactus.log), [`inference log`](experiments/notebook_runs/logs/jiaozi/aerial_cactus_inference.log) |
| Dog Breed Identification | Log loss `0.437159`, accuracy `0.864016`, epoch 7 | No score | The notebook contains a `KeyboardInterrupt` in the training cell; a later cell loaded and reported an existing best checkpoint. This is checkpoint evidence, not a clean uninterrupted end-to-end run. [`log`](experiments/notebook_runs/logs/jiaozi/dog_breed.log) |
| RANZCR CLiP | Mean ROC-AUC `0.857726`, epoch 16 | No score | Training and packaging completed. The separate inference notebook passed strict checkpoint validation and produced a `(3582, 12)` file, but no processed score is stored. [`training log`](experiments/notebook_runs/logs/jiaozi/ranzcr_clip.log), [`inference log`](experiments/notebook_runs/logs/jiaozi/ranzcr_clip_inference.log) |
| TGS Salt Identification Challenge | Official validation metric `0.707875`, epoch 18 | Public `0.69426`, private `0.72590` | Submission status is complete in the stored receipt. [`log`](experiments/notebook_runs/logs/jiaozi/tgs_salt.log) |
| Ultrasound Nerve Segmentation | Dice `0.403563`, epoch 14 | Public `0.47818`, private `0.45699` | The `(5508, 2)` submission was accepted and processed. [`log`](experiments/notebook_runs/logs/jiaozi/ultrasound_nerve_segmentation.log) |

Validation values and leaderboard values are not directly interchangeable: they may use different data splits, aggregation, post-processing, or evaluation implementations. The table therefore reports both without treating one as a reproduction of the other.

## MLE-STAR three-seed runs

The supplied MLE runs use seeds 13, 29, and 47. The table reports the mean and sample standard deviation of the stored `mlestar_ensemble` arm. Higher is better for QWK and ROC-AUC; lower is better for log loss.

| Competition | Validation metric, mean ± sample SD | Kaggle result | Evidence and limitation |
| --- | ---: | --- | --- |
| APTOS 2019 Blindness Detection | QWK `0.816731 ± 0.004392` | No score | The stored ensemble values are `0.811921`, `0.820527`, and `0.817745`. Kaggle returned HTTP 400. [`log`](experiments/notebook_runs/logs/mlestar/aptos_2019.log) |
| Dog Breed Identification | Log loss `2.464719 ± 0.006293` | Public `1.45005`, private `1.45005` | Submission status is complete. [`log`](experiments/notebook_runs/logs/mlestar/dog_breed.log) |
| Aerial Cactus Identification | ROC-AUC `0.999070 ± 0.000200` | No score | Kaggle returned HTTP 400. [`log`](experiments/notebook_runs/logs/mlestar/aerial_cactus.log) |
| Dogs vs. Cats Redux | Log loss `0.127188 ± 0.003363` | Public `0.10219`, private `0.10219` | Submission status is complete. [`log`](experiments/notebook_runs/logs/mlestar/dogs_vs_cats_redux.log) |
| Histopathologic Cancer Detection | ROC-AUC `0.922755 ± 0.003545` | Public `0.9604`, private `0.9521` | Submission status is complete. [`log`](experiments/notebook_runs/logs/mlestar/histopathologic_cancer.log) |
| Plant Pathology 2020 | ROC-AUC `0.928495 ± 0.003266` | Public `0.91061`, private `0.89453` | The notebook records stage recovery before completing the run and submission. [`log`](experiments/notebook_runs/logs/mlestar/plant_pathology_2020.log) |
| RANZCR CLiP | Mean ROC-AUC `0.714074 ± 0.003146` | No score | The three stored selected-model values are `0.716431`, `0.715289`, and `0.710502`; asset packaging and submission validation completed. [`log`](experiments/notebook_runs/logs/mlestar/ranzcr_clip.log) |

The MLE logs also retain failed automated merge attempts. `initial_merge` emitted `AssertionError` records, so those stages are not counted as successful model improvements. In most stored comparison tables the final ensemble arm equals the selected ResNet18 arm; APTOS records only a small seed-13 change. These results demonstrate completed three-seed execution and submission handling, not a general ensemble advantage.

The RANZCR inference and TGS Salt MLE notebooks contain no stored outputs. They are listed as templates in `EXPERIMENTS.md` and are excluded from the results table.

## Completed Cassava paired loss comparison

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

Source records: [`experiments/ab_loss_imbalance/results/outcomes.jsonl`](experiments/ab_loss_imbalance/results/outcomes.jsonl).

Interpretation is deliberately limited to this Cassava setup. It is not evidence that cross entropy is universally preferable for severe imbalance or medical data, so the global Jiaozi class-imbalance default remains focal loss.

## Offline software checks

The following checks exercise software and experiment contracts but are not model-quality benchmarks:

- root unit tests and deterministic Module 4 synthetic smoke generation;
- reconstruction of the Cassava verdict from the ten checked-in JSONL records;
- standalone MLE benchmark unit tests and a synthetic Leaf Classification run with submission disabled;
- JSON validation of all reviewer notebooks and manifest verification of every code/output cell.

Exact commands are in `EXPERIMENTS.md` and the continuous-integration workflow. A passing synthetic run proves that the scripted path works; it does not constitute a public leaderboard result.

## Still pending

| Study | Status | Reason no result is reported |
| --- | --- | --- |
| SIIM-ISIC CE versus focal paired run | Pending | Dataset access, accepted rules, GPU time, and the complete paired fold matrix are required. |
| Remaining real-data loss-comparison testbeds | Pending | No checked-in paired fold records exist. |
| Detection, segmentation, and denoising standalone MLE adapters | Not implemented | These catalog entries fail explicitly rather than fabricate a result. |
| Fresh rerun of every reviewer notebook on current `main` | Pending | Stored outputs originate from the supplied archives; current-environment reruns require Kaggle data access, accepted rules, credentials, and GPU time. |

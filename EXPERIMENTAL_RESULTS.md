# Experimental Results

This report separates local validation, packaged inference, accepted Kaggle submissions, blocked submissions, and unexecuted templates. No missing score is estimated. Notebook-era values are traceable to the stored output and per-cell log listed in [`EXPERIMENTS.md`](EXPERIMENTS.md). Later leaderboard values and ranks supplied in `RDPRO_Experiment - V2.csv` are preserved separately in the normalized [`score table`](experiments/notebook_runs/results/rdpro_experiment_v2_scores.csv), with provenance recorded in its [`source manifest`](experiments/notebook_runs/results/source_manifest.json).

The supplemental table appears to describe submissions made after some archived notebook runs. When an archived run shows HTTP 400, no submit-cell execution, or no receipt, the later score is reported as supplemental evidence rather than rewritten into the historical cell log.

The evaluation asks four questions: whether the generated pipelines obtain credible external scores, whether they cover several vision task types, whether they produce usable training/inference/submission artifacts, and whether the independent MLE-STAR-style search stages add consistent value over their selected baselines. The evidence below answers those questions without averaging incompatible metrics such as ROC-AUC, QWK, log loss, Dice, and thresholded IoU precision.

## Jiaozi pipeline runs

| Competition | Stored validation result | Kaggle result | Evidence and limitation |
| --- | --- | --- | --- |
| APTOS 2019 Blindness Detection | QWK `0.891612`, accuracy `0.818554`, epoch 14 | Public `0.698035`, private `0.865298`; rank `2022/2929` (69%) | The archived attempts returned HTTP 400; the score and rank come only from the later supplemental table. [`log`](experiments/notebook_runs/logs/jiaozi/aptos_2019.log), [`score row`](experiments/notebook_runs/results/rdpro_experiment_v2_scores.csv) |
| Global Wheat Detection | Local `mAP@0.50:0.75` **proxy** `0.543473` | Public `0.6624`, private `0.5546`; rank `1364/1742` (78%) | The archived attempt returned HTTP 400. The leaderboard result is supplemental, and the local value remains a proxy rather than the official metric. [`log`](experiments/notebook_runs/logs/jiaozi/global_wheat_detection.log), [`score row`](experiments/notebook_runs/results/rdpro_experiment_v2_scores.csv) |
| Dogs vs. Cats Redux | Log loss `0.019187`, accuracy `0.997000`, epoch 14 | Public `0.06594`, private `0.06594`; rank `152/1315` (12%) | The scores match the stored complete receipt; the supplemental table adds rank. [`log`](experiments/notebook_runs/logs/jiaozi/dogs_vs_cats_redux.log) |
| Histopathologic Cancer Detection | ROC-AUC `0.980494`, accuracy `0.936416`, epoch 15 | Public `0.9599`, private `0.9623`; rank `259/1149` (23%) | The scores match the stored output; the supplemental table adds rank. The run contains a backbone fallback warning and must **not** be described as a verified DINOv3 result. [`log`](experiments/notebook_runs/logs/jiaozi/histopathologic_cancer.log) |
| Plant Pathology 2020 | ROC-AUC `0.976635`, accuracy `0.914601`, epoch 40 | Public `0.94719`, private `0.94069`; rank `780/1318` (59%) | The archived submit cell was not executed; the leaderboard result comes only from the later supplemental table. [`log`](experiments/notebook_runs/logs/jiaozi/plant_pathology_2020.log), [`score row`](experiments/notebook_runs/results/rdpro_experiment_v2_scores.csv) |
| Aerial Cactus Identification | ROC-AUC `0.999922`, accuracy `0.997714`, epoch 20 | Public `0.9998`, private `0.9998`; rank `455/1221` (37%) | The archived notebooks stop at a submission file without a receipt; the leaderboard result comes only from the later supplemental table. [`training log`](experiments/notebook_runs/logs/jiaozi/aerial_cactus.log), [`inference log`](experiments/notebook_runs/logs/jiaozi/aerial_cactus_inference.log) |
| Dog Breed Identification | Log loss `0.437159`, accuracy `0.864016`, epoch 7 | Public `0.42151`, private `0.42151`; rank `591/1280` (46%) | The training cell was interrupted and a later cell used an existing checkpoint. The leaderboard result comes only from the supplemental table. [`log`](experiments/notebook_runs/logs/jiaozi/dog_breed.log), [`score row`](experiments/notebook_runs/results/rdpro_experiment_v2_scores.csv) |
| RANZCR CLiP | Mean ROC-AUC `0.857726`, epoch 16 | Public `0.85209`, private `0.86120`; rank `1337/1547` (86%) | The archived inference notebook produced a submission file but no receipt; the leaderboard result comes only from the later supplemental table. [`training log`](experiments/notebook_runs/logs/jiaozi/ranzcr_clip.log), [`inference log`](experiments/notebook_runs/logs/jiaozi/ranzcr_clip_inference.log) |
| TGS Salt Identification Challenge | Official validation metric `0.707875`, epoch 18 | Public `0.69426`, private `0.72590`; rank `2499/3219` (77%) | The scores match the stored complete receipt; the supplemental table adds rank. [`log`](experiments/notebook_runs/logs/jiaozi/tgs_salt.log) |
| Ultrasound Nerve Segmentation | Dice `0.403563`, epoch 14 | Public `0.47818`, private `0.45699`; rank `798/1595` (50%) | The scores match the stored processed receipt; the supplemental table adds rank. [`log`](experiments/notebook_runs/logs/jiaozi/ultrasound_nerve_segmentation.log) |

Validation values and leaderboard values are not directly interchangeable: they may use different data splits, aggregation, post-processing, or evaluation implementations. The table therefore reports both without treating one as a reproduction of the other.

## MLE-STAR three-seed runs

The supplied MLE runs use seeds 13, 29, and 47. The table reports the mean and sample standard deviation of the stored `mlestar_ensemble` arm. Higher is better for QWK and ROC-AUC; lower is better for log loss.

| Competition | Validation metric, mean ± sample SD | Kaggle result | Evidence and limitation |
| --- | ---: | --- | --- |
| APTOS 2019 Blindness Detection | QWK `0.816731 ± 0.004392` | Public `0.518347`, private `0.783345`; rank `2390/2929` (82%) | The archived attempt returned HTTP 400; the score and rank come only from the later supplemental table. [`log`](experiments/notebook_runs/logs/mlestar/aptos_2019.log), [`score row`](experiments/notebook_runs/results/rdpro_experiment_v2_scores.csv) |
| Dog Breed Identification | Log loss `2.464719 ± 0.006293` | Public `1.45005`, private `1.45005`; rank `884/1280` (69%) | The scores match the stored complete receipt; the supplemental table adds rank. [`log`](experiments/notebook_runs/logs/mlestar/dog_breed.log) |
| Aerial Cactus Identification | ROC-AUC `0.999070 ± 0.000200` | Public `0.9997`, private `0.9997`; rank `504/1221` (41%) | The archived attempt returned HTTP 400; the score and rank come only from the later supplemental table. [`log`](experiments/notebook_runs/logs/mlestar/aerial_cactus.log), [`score row`](experiments/notebook_runs/results/rdpro_experiment_v2_scores.csv) |
| Dogs vs. Cats Redux | Log loss `0.127188 ± 0.003363` | Public `0.10219`, private `0.10219`; rank `457/1315` (35%) | The scores match the stored complete receipt; the supplemental table adds rank. [`log`](experiments/notebook_runs/logs/mlestar/dogs_vs_cats_redux.log) |
| Histopathologic Cancer Detection | ROC-AUC `0.922755 ± 0.003545` | Public `0.9604`, private `0.9521`; rank `507/1149` (44%) | The scores match the stored complete receipt; the supplemental table adds rank. [`log`](experiments/notebook_runs/logs/mlestar/histopathologic_cancer.log) |
| Plant Pathology 2020 | ROC-AUC `0.928495 ± 0.003266` | Public `0.91061`, private `0.89453`; rank `1022/1318` (78%) | The scores match the recovered stored run; the supplemental table adds rank. [`log`](experiments/notebook_runs/logs/mlestar/plant_pathology_2020.log) |
| RANZCR CLiP | Mean ROC-AUC `0.714074 ± 0.003146` | Public `0.82516`, private `0.82809`; rank `1373/1547` (89%) | The archived notebook stops after asset validation without a receipt; the leaderboard result comes only from the later supplemental table. [`log`](experiments/notebook_runs/logs/mlestar/ranzcr_clip.log), [`score row`](experiments/notebook_runs/results/rdpro_experiment_v2_scores.csv) |

The MLE logs also retain failed automated merge attempts. `initial_merge` emitted `AssertionError` records, so those stages are not counted as successful model improvements. In most stored comparison tables the final ensemble arm equals the selected ResNet18 arm; APTOS records only a small seed-13 change. These results demonstrate completed three-seed execution and submission handling, not a general ensemble advantage.

The RANZCR inference and TGS Salt MLE notebooks contain no stored outputs. They remain templates in `EXPERIMENTS.md`; the RANZCR leaderboard value above is therefore supplemental rather than notebook output.

## Supplemental MLE-STAR leaderboard-only results

The source table includes two results with no corresponding executed notebook evidence in this repository. They are reported for completeness, not as reproduced runs.

| Competition | Kaggle result | Limitation |
| --- | --- | --- |
| TGS Salt Identification Challenge | Public `0.73788`, private `0.77157`; rank `2141/3219` (66%) | The checked-in MLE notebook is an unexecuted template. The source cell containing `2018` under the local-score heading is treated as a shifted year, not a validation result. |
| Leaf Classification | Public `0.67531`, private `0.67531`; rank `1145/1595` (72%) | No corresponding notebook or cell log was supplied. |

Source: normalized [`score table`](experiments/notebook_runs/results/rdpro_experiment_v2_scores.csv) and [`source manifest`](experiments/notebook_runs/results/source_manifest.json).

## Paired leaderboard comparison

Eight competitions have a Jiaozi and MLE-STAR public/private score in the supplemental table. The table is useful for within-competition comparison, but its provenance boundary still applies: some values match archived Kaggle receipts, while others are later leaderboard records without a receipt in the stored Notebook.

| Competition | Metric direction | Jiaozi private | MLE-STAR private | Reported winner |
| --- | --- | ---: | ---: | --- |
| Plant Pathology 2020 | Higher ROC-AUC is better | `0.94069` | `0.89453` | Jiaozi |
| APTOS 2019 | Higher QWK is better | `0.865298` | `0.783345` | Jiaozi |
| Dog Breed Identification | Lower log loss is better | `0.42151` | `1.45005` | Jiaozi |
| Aerial Cactus Identification | Higher ROC-AUC is better | `0.9998` | `0.9997` | Jiaozi |
| Dogs vs. Cats Redux | Lower log loss is better | `0.06594` | `0.10219` | Jiaozi |
| Histopathologic Cancer Detection | Higher ROC-AUC is better | `0.9623` | `0.9521` | Jiaozi |
| RANZCR CLiP | Higher mean ROC-AUC is better | `0.86120` | `0.82809` | Jiaozi |
| TGS Salt Identification | Higher thresholded IoU precision is better | `0.72590` | `0.77157` | MLE-STAR |

On these reported private scores, Jiaozi is higher or lower in the correct metric direction on seven of eight competitions; MLE-STAR leads on TGS Salt. The largest differences are Dog Breed log loss (`1.02854` lower for Jiaozi), APTOS private QWK (`0.081953` higher for Jiaozi), Plant private ROC-AUC (`0.04616` higher for Jiaozi), and TGS private score (`0.04567` higher for MLE-STAR). These are descriptive comparisons of the supplied leaderboard records, not statistical tests and not evidence that either system is universally better.

Global Wheat Detection and Ultrasound Nerve Segmentation are Jiaozi-only coverage results in the supplied table. They demonstrate that the stored workflow reached detection and segmentation submission paths; they are not paired wins because no corresponding MLE-STAR score is present.

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

## Interpretation and threats to validity

- The strongest reported Jiaozi differences occur on classification and multi-label tasks, while MLE-STAR leads the supplied TGS Salt segmentation result. Aerial Cactus is nearly saturated for both systems, so its `0.0001` difference has little practical meaning.
- Jiaozi's Global Wheat and Ultrasound results broaden task coverage, but their moderate private scores show that adapter availability is not the same as competition-level specialization.
- Internal validation protocols differ between the two workflows. External scores are the clearest common surface, but several are supplemental records rather than receipts captured in the archived execution.
- Public and private splits can disagree substantially, especially for APTOS and Global Wheat. No conclusion should rely on only one split.
- Leaderboard ranks are retrospective. They do not imply medal, prize, or eligibility status.
- Hardware, package versions, dataset availability, and checkpoint provenance were not controlled uniformly across all supplied runs. Runtime or cost superiority is therefore not claimed.
- The Histopathologic Jiaozi log contains a backbone fallback warning. Its numeric result is retained, but it is not evidence of a verified DINOv3 execution.
- Raw scores from different competitions are not averaged because their metrics and scales are incompatible.

## Still pending

| Study | Status | Reason no result is reported |
| --- | --- | --- |
| SIIM-ISIC CE versus focal paired run | Pending | Dataset access, accepted rules, GPU time, and the complete paired fold matrix are required. |
| Remaining real-data loss-comparison testbeds | Pending | No checked-in paired fold records exist. |
| Detection, segmentation, and denoising standalone MLE adapters | Not implemented | These catalog entries fail explicitly rather than fabricate a result. |
| Fresh rerun of every reviewer notebook on current `main` | Pending | Stored outputs originate from the supplied archives; current-environment reruns require Kaggle data access, accepted rules, credentials, and GPU time. |

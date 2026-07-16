# Experimental Results

This section evaluates three questions:

1. **Effectiveness:** can the GraphRAG-based pipeline produce leaderboard-competitive computer-vision solutions across different Kaggle tasks?
2. **Comparative performance:** how does the GraphRAG-based pipeline compare with the supplied MLE-STAR-style baseline implementation on tasks where both systems have leaderboard records?
3. **Extension capability:** can the same agent workflow extend beyond standard image classification to multi-label classification, object detection, and image segmentation?

The commands, notebooks, and logs needed to inspect or rerun the experiments are listed in [`EXPERIMENTS.md`](EXPERIMENTS.md).

Different competitions use different metrics, so raw ROC-AUC, QWK, log loss, Dice, mAP, and thresholded-IoU scores are **not averaged across tasks**. Cross-task conclusions are instead based on within-competition comparisons, recorded leaderboard ranks, and coverage across task types. Missing scores are not estimated.

> **Comparison boundary.** In this report, *MLE-STAR-style baseline* refers to the supplied local benchmark implementation used in these experiments. The results should not be interpreted as an official reproduction or official leaderboard result of the full published MLE-STAR system.

## 1. Main Findings

Our results tell a mixed but useful story.

- The GraphRAG-based pipeline has stored execution evidence for **ten image tasks**, covering ordinary classification, multi-label classification, object detection, and semantic segmentation.
- **Eight tasks have leaderboard records for both the GraphRAG-based pipeline and the MLE-STAR-style baseline.** On the reported private scores, the GraphRAG-based pipeline performs better in the correct metric direction on **seven of eight** paired tasks. The baseline is stronger on **TGS Salt Identification**.
- The comparative advantage is most consistent on classification-oriented tasks. Using the recorded rank fraction `rank / total` as a descriptive cross-task indicator, the GraphRAG-based pipeline improves over the baseline by a **median of about 15.5 percentage points** across the eight paired tasks; lower rank fraction is better. The largest improvements are on Dogs vs. Cats, Dog Breed, Histopathologic Cancer, and Plant Pathology.
- The absolute leaderboard positions are nevertheless **mixed rather than uniformly strong**. The GraphRAG-based pipeline's best recorded rank fractions are 12% on Dogs vs. Cats and 23% on Histopathologic Cancer, while several tasks remain in the middle or lower part of the leaderboard. Therefore, the evidence supports a claim of **consistent improvement over the supplied baseline**, not a claim of state-of-the-art Kaggle performance.
- The two unpaired extension runs, Global Wheat Detection and Ultrasound Nerve Segmentation, show that the workflow can reach detection and segmentation submission paths. Their moderate leaderboard positions also show that **task coverage is ahead of task specialization**.
- TGS Salt provides an important negative result: the MLE-STAR-style baseline is stronger on this segmentation benchmark. Together with the moderate Ultrasound result, this indicates that segmentation remains less mature than the GraphRAG-based pipeline's classification path.
- The controlled Cassava CE-versus-focal experiment shows that the knowledge base should **not encode an unconditional preference for focal loss whenever class imbalance is present**. In this setup, cross entropy wins all five paired folds.

The central conclusion is therefore:

> **The GraphRAG-based pipeline is not yet a universally strong Kaggle solver, but under the supplied evaluation setup it shows a consistent comparative advantage on classification-heavy tasks and broader end-to-end task coverage. Its main remaining weakness is benchmark-specific adaptation for harder extension tasks, especially segmentation and detection.**

## 2. Evaluation Scope

The ten tasks evaluated with the GraphRAG-based pipeline are grouped by their role in the evaluation.

| Group | Tasks | Purpose |
| --- | --- | --- |
| Core paired image benchmarks | Plant Pathology 2020, APTOS 2019, Dog Breed, Aerial Cactus, Dogs vs. Cats, Histopathologic Cancer, RANZCR CLiP | Compare the GraphRAG-based pipeline with the supplied MLE-STAR-style baseline on common image benchmarks. |
| Segmentation benchmark | TGS Salt Identification | Test whether the comparative pattern extends beyond classification. |
| GraphRAG-based pipeline extensions | Global Wheat Detection, Ultrasound Nerve Segmentation | Test whether the agent can extend to detection and an additional segmentation domain. |

TGS Salt is an official MLE-bench task but is **not part of the Lite/Low-complexity split**; it belongs to the Medium split. The seven core classification-oriented paired tasks above are part of the Lite set used in this evaluation.

## 3. Paired Kaggle Leaderboard Comparison

Eight competitions have public/private leaderboard records for both systems in the normalized score table. Private score is used as the primary within-competition comparison surface below. Because several records are supplemental rather than stored Kaggle receipts, the table is descriptive rather than a controlled statistical comparison.

`Rank fraction = rank / total teams`; **lower is better**.

| Competition | Metric direction | GraphRAG private | Baseline private | GraphRAG rank fraction | Baseline rank fraction | Winner |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Plant Pathology 2020 | Higher ROC-AUC is better | `0.94069` | `0.89453` | `59.2%` | `77.5%` | GraphRAG |
| APTOS 2019 | Higher QWK is better | `0.865298` | `0.783345` | `69.0%` | `81.6%` | GraphRAG |
| Dog Breed Identification | Lower log loss is better | `0.42151` | `1.45005` | `46.2%` | `69.1%` | GraphRAG |
| Aerial Cactus Identification | Higher ROC-AUC is better | `0.9998` | `0.9997` | `37.3%` | `41.3%` | GraphRAG |
| Dogs vs. Cats Redux | Lower log loss is better | `0.06594` | `0.10219` | `11.6%` | `34.8%` | GraphRAG |
| Histopathologic Cancer Detection | Higher ROC-AUC is better | `0.9623` | `0.9521` | `22.5%` | `44.1%` | GraphRAG |
| RANZCR CLiP | Higher mean ROC-AUC is better | `0.86120` | `0.82809` | `86.4%` | `88.8%` | GraphRAG |
| TGS Salt Identification | Higher thresholded-IoU precision is better | `0.72590` | `0.77157` | `77.6%` | `66.5%` | Baseline |

The GraphRAG-based pipeline wins seven of the eight paired private-score comparisons. However, the magnitude of improvement should be interpreted through **within-task** scores or ranks rather than raw score differences across different metrics.

The rank comparison gives a clearer cross-task view:

- Dogs vs. Cats: the GraphRAG-based pipeline improves the recorded rank fraction by about **23.2 percentage points**.
- Dog Breed: improvement of about **22.9 points**.
- Histopathologic Cancer: improvement of about **21.6 points**.
- Plant Pathology: improvement of about **18.4 points**.
- APTOS: improvement of about **12.6 points**.
- Aerial Cactus and RANZCR: only small rank improvements, so these should not be presented as large practical wins.
- TGS Salt: the baseline is better by about **11.1 rank-fraction points** and `0.04567` private score.

This pattern supports a narrower and more defensible claim than "GraphRAG is better than MLE-STAR":

> **On the supplied paired benchmark runs, the GraphRAG-based pipeline is more consistently effective on classification-heavy tasks, while the segmentation result shows that this advantage does not yet generalize uniformly across task types.**

## 4. Absolute Performance and Current Capability Boundary

The paired win rate alone can overstate the maturity of the system. The absolute leaderboard positions show that the GraphRAG-based pipeline is competitive on some tasks but still has substantial room for improvement.

| Competition | GraphRAG recorded rank | Rank fraction | Interpretation |
| --- | ---: | ---: | --- |
| Dogs vs. Cats Redux | `152/1315` | `11.6%` | Strongest recorded relative result. |
| Histopathologic Cancer Detection | `259/1149` | `22.5%` | Competitive classification result. |
| Aerial Cactus Identification | `455/1221` | `37.3%` | Good score, but the task is close to saturation for both systems. |
| Dog Breed Identification | `591/1280` | `46.2%` | Mid-leaderboard despite a large gain over the baseline. |
| Ultrasound Nerve Segmentation | `798/1595` | `50.0%` | Demonstrates extension coverage, not strong specialization. |
| Plant Pathology 2020 | `780/1318` | `59.2%` | Better than the baseline but not a high leaderboard placement. |
| APTOS 2019 | `2022/2929` | `69.0%` | Private score is much stronger than public score, but overall placement remains moderate. |
| TGS Salt Identification | `2499/3219` | `77.6%` | Segmentation remains a weakness; baseline is stronger. |
| Global Wheat Detection | `1364/1742` | `78.3%` | End-to-end detection path works, but competition-level adaptation is limited. |
| RANZCR CLiP | `1337/1547` | `86.4%` | The weakest recorded relative placement despite a private-score advantage over the baseline. |

This distinction is important for the overall story:

- **Relative result:** the GraphRAG-based pipeline is usually better than the supplied baseline.
- **Absolute result:** the GraphRAG-based pipeline is not yet consistently near the top of historical Kaggle leaderboards.
- **Research value:** the system's current strength is robust automated solution generation and comparative improvement across heterogeneous tasks, rather than winning individual competitions.

## 5. GraphRAG-Based Pipeline Results

| Competition | Stored validation result | Kaggle result | Evidence and limitation |
| --- | --- | --- | --- |
| APTOS 2019 Blindness Detection | QWK `0.891612`, accuracy `0.818554`, epoch 14 | Public `0.698035`, private `0.865298`; rank `2022/2929` (rank fraction `69%`) | The archived attempts returned HTTP 400; the score and rank come only from the later supplemental table. [`log`](experiments/notebook_runs/logs/jiaozi/aptos_2019.log), [`score row`](experiments/notebook_runs/results/rdpro_experiment_v2_scores.csv) |
| Global Wheat Detection | Local `mAP@0.50:0.75` **proxy** `0.543473` | Public `0.6624`, private `0.5546`; rank `1364/1742` (rank fraction `78%`) | The archived attempt returned HTTP 400. The leaderboard result is supplemental, and the local value remains a proxy rather than the official metric. [`log`](experiments/notebook_runs/logs/jiaozi/global_wheat_detection.log), [`score row`](experiments/notebook_runs/results/rdpro_experiment_v2_scores.csv) |
| Dogs vs. Cats Redux | Log loss `0.019187`, accuracy `0.997000`, epoch 14 | Public `0.06594`, private `0.06594`; rank `152/1315` (rank fraction `12%`) | The scores match the stored complete receipt; the supplemental table adds rank. [`log`](experiments/notebook_runs/logs/jiaozi/dogs_vs_cats_redux.log) |
| Histopathologic Cancer Detection | ROC-AUC `0.980494`, accuracy `0.936416`, epoch 15 | Public `0.9599`, private `0.9623`; rank `259/1149` (rank fraction `23%`) | The scores match the stored output; the supplemental table adds rank. The run contains a backbone fallback warning and must not be described as a verified DINOv3 result. [`log`](experiments/notebook_runs/logs/jiaozi/histopathologic_cancer.log) |
| Plant Pathology 2020 | ROC-AUC `0.976635`, accuracy `0.914601`, epoch 40 | Public `0.94719`, private `0.94069`; rank `780/1318` (rank fraction `59%`) | The archived submit cell was not executed; the leaderboard result comes only from the later supplemental table. [`log`](experiments/notebook_runs/logs/jiaozi/plant_pathology_2020.log), [`score row`](experiments/notebook_runs/results/rdpro_experiment_v2_scores.csv) |
| Aerial Cactus Identification | ROC-AUC `0.999922`, accuracy `0.997714`, epoch 20 | Public `0.9998`, private `0.9998`; rank `455/1221` (rank fraction `37%`) | The archived notebooks stop at a submission file without a receipt; the leaderboard result comes only from the later supplemental table. [`training log`](experiments/notebook_runs/logs/jiaozi/aerial_cactus.log), [`inference log`](experiments/notebook_runs/logs/jiaozi/aerial_cactus_inference.log) |
| Dog Breed Identification | Log loss `0.437159`, accuracy `0.864016`, epoch 7 | Public `0.42151`, private `0.42151`; rank `591/1280` (rank fraction `46%`) | The training cell was interrupted and a later cell used an existing checkpoint. The leaderboard result comes only from the supplemental table. [`log`](experiments/notebook_runs/logs/jiaozi/dog_breed.log), [`score row`](experiments/notebook_runs/results/rdpro_experiment_v2_scores.csv) |
| RANZCR CLiP | Mean ROC-AUC `0.857726`, epoch 16 | Public `0.85209`, private `0.86120`; rank `1337/1547` (rank fraction `86%`) | The archived inference notebook produced a submission file but no receipt; the leaderboard result comes only from the later supplemental table. [`training log`](experiments/notebook_runs/logs/jiaozi/ranzcr_clip.log), [`inference log`](experiments/notebook_runs/logs/jiaozi/ranzcr_clip_inference.log) |
| TGS Salt Identification Challenge | Official validation metric `0.707875`, epoch 18 | Public `0.69426`, private `0.72590`; rank `2499/3219` (rank fraction `77%`) | The scores match the stored complete receipt; the supplemental table adds rank. [`log`](experiments/notebook_runs/logs/jiaozi/tgs_salt.log) |
| Ultrasound Nerve Segmentation | Dice `0.403563`, epoch 14 | Public `0.47818`, private `0.45699`; rank `798/1595` (rank fraction `50%`) | The scores match the stored processed receipt; the supplemental table adds rank. [`log`](experiments/notebook_runs/logs/jiaozi/ultrasound_nerve_segmentation.log) |

Validation and leaderboard values are not directly interchangeable. They may use different splits, aggregation rules, post-processing, or metric implementations. The table therefore reports both without treating one as a reproduction of the other.

## 6. MLE-STAR-Style Baseline Runs

The supplied baseline runs use seeds 13, 29, and 47. For stored multi-seed experiments, the table reports the mean and sample standard deviation of the recorded `mlestar_ensemble` arm. Higher is better for QWK and ROC-AUC; lower is better for log loss.

| Competition | Validation metric, mean ± sample SD | Kaggle result | Evidence and limitation |
| --- | ---: | --- | --- |
| APTOS 2019 Blindness Detection | QWK `0.816731 ± 0.004392` | Public `0.518347`, private `0.783345`; rank `2390/2929` (rank fraction `82%`) | The archived attempt returned HTTP 400; the score and rank come only from the later supplemental table. [`log`](experiments/notebook_runs/logs/mlestar/aptos_2019.log), [`score row`](experiments/notebook_runs/results/rdpro_experiment_v2_scores.csv) |
| Dog Breed Identification | Log loss `2.464719 ± 0.006293` | Public `1.45005`, private `1.45005`; rank `884/1280` (rank fraction `69%`) | The scores match the stored complete receipt; the supplemental table adds rank. [`log`](experiments/notebook_runs/logs/mlestar/dog_breed.log) |
| Aerial Cactus Identification | ROC-AUC `0.999070 ± 0.000200` | Public `0.9997`, private `0.9997`; rank `504/1221` (rank fraction `41%`) | The archived attempt returned HTTP 400; the score and rank come only from the later supplemental table. [`log`](experiments/notebook_runs/logs/mlestar/aerial_cactus.log), [`score row`](experiments/notebook_runs/results/rdpro_experiment_v2_scores.csv) |
| Dogs vs. Cats Redux | Log loss `0.127188 ± 0.003363` | Public `0.10219`, private `0.10219`; rank `457/1315` (rank fraction `35%`) | The scores match the stored complete receipt; the supplemental table adds rank. [`log`](experiments/notebook_runs/logs/mlestar/dogs_vs_cats_redux.log) |
| Histopathologic Cancer Detection | ROC-AUC `0.922755 ± 0.003545` | Public `0.9604`, private `0.9521`; rank `507/1149` (rank fraction `44%`) | The scores match the stored complete receipt; the supplemental table adds rank. [`log`](experiments/notebook_runs/logs/mlestar/histopathologic_cancer.log) |
| Plant Pathology 2020 | ROC-AUC `0.928495 ± 0.003266` | Public `0.91061`, private `0.89453`; rank `1022/1318` (rank fraction `78%`) | The scores match the recovered stored run; the supplemental table adds rank. [`log`](experiments/notebook_runs/logs/mlestar/plant_pathology_2020.log) |
| RANZCR CLiP | Mean ROC-AUC `0.714074 ± 0.003146` | Public `0.82516`, private `0.82809`; rank `1373/1547` (rank fraction `89%`) | The archived notebook stops after asset validation without a receipt; the leaderboard result comes only from the later supplemental table. [`log`](experiments/notebook_runs/logs/mlestar/ranzcr_clip.log), [`score row`](experiments/notebook_runs/results/rdpro_experiment_v2_scores.csv) |

The logs also retain failed automated merge attempts. `initial_merge` emitted `AssertionError` records, so those stages are not counted as successful improvements. In most stored comparison tables, the final ensemble arm equals the selected ResNet18 arm; APTOS records only a small seed-13 change. This evidence does not demonstrate a general ensemble advantage.

The RANZCR and TGS Salt baseline notebooks do not contain stored completed submission output. Their later leaderboard values should therefore be treated as supplemental evidence rather than reproduced notebook results.

### Supplemental baseline leaderboard result

| Competition | Kaggle result | Limitation |
| --- | --- | --- |
| TGS Salt Identification Challenge | Public `0.73788`, private `0.77157`; rank `2141/3219` (rank fraction `66%`) | The checked-in baseline notebook is an unexecuted template; this value comes from the later normalized score table rather than a stored notebook receipt. |

The previous Leaf Classification supplemental row is not part of the current ten-task evaluation and is therefore omitted from the main results narrative.

## 7. Extension Results: What Works and What Does Not Yet

The extension experiments should be presented as a capability-boundary analysis rather than as additional wins.

### Global Wheat Detection

The agent successfully reached an object-detection training and submission path, but the private score `0.5546` and recorded rank `1364/1742` show that the generated solution was not highly competitive. This result demonstrates **functional extension**, not strong detector specialization.

### Ultrasound Nerve Segmentation

The agent produced a valid segmentation workflow and Kaggle submission, but the private Dice `0.45699` and rank `798/1595` are moderate. This again demonstrates end-to-end adaptability more clearly than benchmark dominance.

### TGS Salt Identification

TGS is the most important counterexample in the paired comparison. The GraphRAG-based pipeline reaches the segmentation task end-to-end, but the MLE-STAR-style baseline records a higher private score (`0.77157` vs. `0.72590`) and a better rank fraction (`66.5%` vs. `77.6%`). This indicates that the GraphRAG-based pipeline's extension mechanism still needs stronger task-specific reasoning about segmentation architecture, loss design, threshold tuning, post-processing, and submission encoding.

Taken together, the extension results support the following claim:

> **The GraphRAG-based pipeline can extend beyond classification at the pipeline level, but extension quality is not yet uniformly competitive. The next research step is not merely adding more adapters; it is improving how the agent specializes those adapters to the metric and structure of each new task.**

## 8. Controlled Knowledge-Base Validation: Cross Entropy vs. Focal Loss

This experiment evaluates one knowledge-base rule conflict: for a class-imbalanced Cassava setup, should the system prefer focal loss or ordinary cross entropy?

The repository contains ten fold-level records: five folds for focal loss and the same five folds for cross entropy. Both arms use seed 42, EfficientNet-B0, 224-pixel inputs, five epochs, ordinary shuffled sampling, and the same fold-file SHA-256 value.

Primary metric: macro-F1. The paired difference is `cross entropy - focal`.

| Fold | Focal | Cross entropy | Difference |
| ---: | ---: | ---: | ---: |
| 0 | `0.7035` | `0.7180` | `+0.0145` |
| 1 | `0.7190` | `0.7231` | `+0.0041` |
| 2 | `0.7211` | `0.7242` | `+0.0031` |
| 3 | `0.6972` | `0.7013` | `+0.0041` |
| 4 | `0.7107` | `0.7224` | `+0.0117` |

The collector reconstructs a mean paired difference of approximately `+0.0075`, standard error `0.0023`, and tie band `±0.0050`, producing the testbed verdict `CE_WINS`. All five paired directions favor cross entropy.

Source records: [`experiments/ab_loss_imbalance/results/outcomes.jsonl`](experiments/ab_loss_imbalance/results/outcomes.jsonl).

The correct interpretation is **local but actionable**:

- This experiment does not prove that cross entropy is universally better for imbalanced or medical data.
- It does show that a blanket rule such as "class imbalance -> prefer focal loss" is too strong.
- The knowledge base should therefore treat focal loss as a candidate strategy conditioned on task characteristics and validation evidence, rather than as an unconditional default.

This controlled experiment is evidence that the knowledge base can be refined by falsifying overly broad rules, not evidence that the knowledge base is already optimal.

## 9. Evidence Quality

The result tables use the following evidence categories.

| Evidence category | Meaning |
| --- | --- |
| Stored notebook output and per-cell log | The public notebook copy keeps textual outputs, and the matching `.log` file records every code cell. |
| Complete Kaggle receipt | The archived run contains a completed submission receipt or processed score output. |
| Supplemental leaderboard record | The score appears in the later normalized score table, but not as a completed receipt in the archived notebook. |
| Local validation or proxy metric | The value comes from local validation or a proxy metric and is not the official Kaggle score. |
| Template or no-output notebook | The notebook is kept for completeness, but it is not counted as executed result evidence. |

Values from executed notebooks are traceable to the stored output and per-cell logs listed in [`EXPERIMENTS.md`](EXPERIMENTS.md). Later leaderboard values and ranks from `RDPRO_Experiment - V2.csv` are preserved in the normalized [`score table`](experiments/notebook_runs/results/rdpro_experiment_v2_scores.csv), with source details in the [`source manifest`](experiments/notebook_runs/results/source_manifest.json).

When an archived notebook shows HTTP 400, no submit-cell execution, or no receipt, the later score is treated as supplemental evidence rather than edited back into the historical execution log.

## 10. Software and Evidence Checks

These checks verify the code and evidence pipeline, not model quality:

- root unit tests and deterministic Module 4 synthetic smoke generation;
- reconstruction of the Cassava verdict from the ten checked-in JSONL records;
- standalone baseline benchmark unit tests and synthetic adapter smoke runs with submission disabled;
- JSON validation of reviewer notebooks and manifest verification of code/output cells.

Exact commands are in [`EXPERIMENTS.md`](EXPERIMENTS.md). A passing synthetic run proves that the scripted path works; it does not constitute a public leaderboard result.

## 11. What These Results Do Not Show

These results should not be read as a universal ranking of the GraphRAG-based pipeline and the full MLE-STAR system. In particular, they do not prove that:

- the GraphRAG-based pipeline is better on every vision task;
- the GraphRAG-based pipeline is state of the art on the evaluated Kaggle competitions;
- the GraphRAG-based pipeline is faster or cheaper to run;
- detection and segmentation are as mature as the classification path;
- the observed differences are statistically significant across the task distribution;
- supplemental leaderboard rows are equivalent to archived notebook receipts.

The strongest supported claim is narrower:

> **Across the eight supplied paired leaderboard records, the GraphRAG-based pipeline is more consistent than the supplied MLE-STAR-style baseline on classification-heavy tasks, while the TGS result and the unpaired extension tasks reveal clear remaining weaknesses outside that core regime.**

## 12. Limitations

- Several leaderboard values are supplemental records rather than receipts captured in the archived notebook execution.
- The baseline comparison is against the supplied local MLE-STAR-style implementation and should not be presented as a definitive comparison with the full official MLE-STAR system.
- Internal validation protocols differ between the GraphRAG-based pipeline and baseline workflows. External scores are the clearest common surface, but they still come from different runs and sometimes different evidence types.
- Public and private splits can disagree substantially, especially for APTOS and Global Wheat. No conclusion should rely on only one split.
- Recorded late-submission leaderboard ranks are retrospective and do not imply medal, prize, or historical eligibility status.
- Aerial Cactus is nearly saturated for both systems, so its `0.0001` private-score difference has little practical meaning.
- Global Wheat and Ultrasound broaden task coverage, but their moderate private scores show that adapter availability is not the same as competition-level specialization.
- Hardware, package versions, dataset availability, and checkpoint provenance were not controlled uniformly across all supplied runs. Runtime or cost superiority is therefore not claimed.
- The Histopathologic GraphRAG log contains a backbone fallback warning. Its numeric result is retained, but it is not evidence of a verified DINOv3 execution.
- Raw scores from different competitions are not averaged because their metrics and scales are incompatible.

# Reproducible Experiments and Stored Run Logs

This file lists the experiments included in this repository and points to the
notebooks, scripts, commands, logs, and checked-in records used to inspect the
run evidence or rerun the parts that are reproducible from this repo.

Detailed score interpretation and pending-result discussion are kept in
[`EXPERIMENTAL_RESULTS.md`](EXPERIMENTAL_RESULTS.md).

The executed notebooks keep their textual cell outputs, so a reviewer can open a notebook on GitHub and inspect the output directly. The matching `.log` file repeats every code cell in order, including cells with no output, warnings, tracebacks, submission receipts, and submission failures.

Raw Kaggle data, credentials, submissions, checkpoints, and embedded competition images are not committed. Binary rich-media output was removed from the public evidence copies; textual output was retained. Source archive and notebook SHA-256 values are recorded in [`experiments/notebook_runs/manifest.json`](experiments/notebook_runs/manifest.json).

The later leaderboard records supplied in `RDPRO_Experiment - V2.csv` are normalized at [`experiments/notebook_runs/results/rdpro_experiment_v2_scores.csv`](experiments/notebook_runs/results/rdpro_experiment_v2_scores.csv). Their source hash and normalization rules are in [`source_manifest.json`](experiments/notebook_runs/results/source_manifest.json). These rows supplement the cell logs; they do not replace them or claim that a later score appeared in an earlier notebook execution.

## Running a notebook for review

The notebooks under [`experiments/notebook_runs/notebooks/`](experiments/notebook_runs/notebooks/) are the main review copies. To rerun one in Colab:

1. Upload the single notebook matching the competition in the tables below.
2. Select a GPU runtime when the notebook trains or performs image inference.
3. Add `KAGGLE_API_TOKEN` and, when required, `HF_TOKEN` through Colab Secrets. Do not paste either value into a cell.
4. Accept the competition rules in Kaggle before downloading data or submitting.
5. Choose **Runtime → Run all**.

The setup cells now clone `Isso-W/RDPRO-SS26-Project` at `main`. The visible outputs are historical evidence from the supplied archives, not a claim that `main` was rerun after export. A fresh rerun can differ because packages, hardware, data availability, and the current `main` revision may have changed.

## GraphRAG-based AutoML pipeline notebooks

| Competition | Notebook to run | Per-cell log | Stored run state |
| --- | --- | --- | --- |
| APTOS 2019 Blindness Detection | [`aptos_2019.ipynb`](experiments/notebook_runs/notebooks/jiaozi/aptos_2019.ipynb) | [`aptos_2019.log`](experiments/notebook_runs/logs/jiaozi/aptos_2019.log) | Validation complete; Kaggle API rejected the submission |
| Global Wheat Detection | [`global_wheat_detection.ipynb`](experiments/notebook_runs/notebooks/jiaozi/global_wheat_detection.ipynb) | [`global_wheat_detection.log`](experiments/notebook_runs/logs/jiaozi/global_wheat_detection.log) | Local proxy validation complete; Kaggle API rejected the submission |
| Dogs vs. Cats Redux | [`dogs_vs_cats_redux.ipynb`](experiments/notebook_runs/notebooks/jiaozi/dogs_vs_cats_redux.ipynb) | [`dogs_vs_cats_redux.log`](experiments/notebook_runs/logs/jiaozi/dogs_vs_cats_redux.log) | Scored submission |
| Histopathologic Cancer Detection | [`histopathologic_cancer.ipynb`](experiments/notebook_runs/notebooks/jiaozi/histopathologic_cancer.ipynb) | [`histopathologic_cancer.log`](experiments/notebook_runs/logs/jiaozi/histopathologic_cancer.log) | Scored submission; backbone fallback warning retained |
| Plant Pathology 2020 | [`plant_pathology_2020.ipynb`](experiments/notebook_runs/notebooks/jiaozi/plant_pathology_2020.ipynb) | [`plant_pathology_2020.log`](experiments/notebook_runs/logs/jiaozi/plant_pathology_2020.log) | Validation and submission file complete; submit cell not executed |
| Aerial Cactus Identification | [`aerial_cactus.ipynb`](experiments/notebook_runs/notebooks/jiaozi/aerial_cactus.ipynb) | [`aerial_cactus.log`](experiments/notebook_runs/logs/jiaozi/aerial_cactus.log) | Validation and asset package complete |
| Dog Breed Identification | [`dog_breed.ipynb`](experiments/notebook_runs/notebooks/jiaozi/dog_breed.ipynb) | [`dog_breed.log`](experiments/notebook_runs/logs/jiaozi/dog_breed.log) | Best-checkpoint result present after an interrupted training cell |
| RANZCR CLiP | [`ranzcr_clip.ipynb`](experiments/notebook_runs/notebooks/jiaozi/ranzcr_clip.ipynb) | [`ranzcr_clip.log`](experiments/notebook_runs/logs/jiaozi/ranzcr_clip.log) | Validation and asset package complete |
| TGS Salt Identification Challenge | [`tgs_salt.ipynb`](experiments/notebook_runs/notebooks/jiaozi/tgs_salt.ipynb) | [`tgs_salt.log`](experiments/notebook_runs/logs/jiaozi/tgs_salt.log) | Scored submission |
| Ultrasound Nerve Segmentation | [`ultrasound_nerve_segmentation.ipynb`](experiments/notebook_runs/notebooks/jiaozi/ultrasound_nerve_segmentation.ipynb) | [`ultrasound_nerve_segmentation.log`](experiments/notebook_runs/logs/jiaozi/ultrasound_nerve_segmentation.log) | Scored submission |

Two small consumer notebooks exercise packaged inference assets rather than training:

| Competition | Inference notebook | Per-cell log | Stored run state |
| --- | --- | --- | --- |
| Aerial Cactus Identification | [`aerial_cactus_inference.ipynb`](experiments/notebook_runs/notebooks/jiaozi/aerial_cactus_inference.ipynb) | [`aerial_cactus_inference.log`](experiments/notebook_runs/logs/jiaozi/aerial_cactus_inference.log) | Submission file ready; no stored score |
| RANZCR CLiP | [`ranzcr_clip_inference.ipynb`](experiments/notebook_runs/notebooks/jiaozi/ranzcr_clip_inference.ipynb) | [`ranzcr_clip_inference.log`](experiments/notebook_runs/logs/jiaozi/ranzcr_clip_inference.log) | Checkpoint validation passed and submission file ready; no stored score |

## MLE-STAR notebooks

The large combined archive notebook was split into one notebook per competition for easier review. Shared setup and adapter cells, the selected competition cell, and all of that cell's stored outputs are retained.

| Competition | Notebook to run | Per-cell log | Stored run state |
| --- | --- | --- | --- |
| APTOS 2019 Blindness Detection | [`aptos_2019.ipynb`](experiments/notebook_runs/notebooks/mlestar/aptos_2019.ipynb) | [`aptos_2019.log`](experiments/notebook_runs/logs/mlestar/aptos_2019.log) | Three-seed validation complete; Kaggle API rejected the submission |
| Dog Breed Identification | [`dog_breed.ipynb`](experiments/notebook_runs/notebooks/mlestar/dog_breed.ipynb) | [`dog_breed.log`](experiments/notebook_runs/logs/mlestar/dog_breed.log) | Three-seed validation and scored submission |
| Aerial Cactus Identification | [`aerial_cactus.ipynb`](experiments/notebook_runs/notebooks/mlestar/aerial_cactus.ipynb) | [`aerial_cactus.log`](experiments/notebook_runs/logs/mlestar/aerial_cactus.log) | Three-seed validation complete; Kaggle API rejected the submission |
| Dogs vs. Cats Redux | [`dogs_vs_cats_redux.ipynb`](experiments/notebook_runs/notebooks/mlestar/dogs_vs_cats_redux.ipynb) | [`dogs_vs_cats_redux.log`](experiments/notebook_runs/logs/mlestar/dogs_vs_cats_redux.log) | Three-seed validation and scored submission |
| Histopathologic Cancer Detection | [`histopathologic_cancer.ipynb`](experiments/notebook_runs/notebooks/mlestar/histopathologic_cancer.ipynb) | [`histopathologic_cancer.log`](experiments/notebook_runs/logs/mlestar/histopathologic_cancer.log) | Three-seed validation and scored submission |
| Plant Pathology 2020 | [`plant_pathology_2020.ipynb`](experiments/notebook_runs/notebooks/mlestar/plant_pathology_2020.ipynb) | [`plant_pathology_2020.log`](experiments/notebook_runs/logs/mlestar/plant_pathology_2020.log) | Recovered three-seed run and scored submission |
| RANZCR CLiP | [`ranzcr_clip.ipynb`](experiments/notebook_runs/notebooks/mlestar/ranzcr_clip.ipynb) | [`ranzcr_clip.log`](experiments/notebook_runs/logs/mlestar/ranzcr_clip.log) | Three-seed validation and asset package complete; no stored score |

The supplied MLE archive also contains two unexecuted templates. They are included for completeness, but they are not result evidence:

| Competition | Template | Per-cell log |
| --- | --- | --- |
| RANZCR CLiP inference | [`ranzcr_clip_inference_template.ipynb`](experiments/notebook_runs/notebooks/mlestar/ranzcr_clip_inference_template.ipynb) | [`ranzcr_clip_inference_template.log`](experiments/notebook_runs/logs/mlestar/ranzcr_clip_inference_template.log) |
| TGS Salt | [`tgs_salt_template.ipynb`](experiments/notebook_runs/notebooks/mlestar/tgs_salt_template.ipynb) | [`tgs_salt_template.log`](experiments/notebook_runs/logs/mlestar/tgs_salt_template.log) |

The supplemental score table also reports a later MLE-STAR leaderboard result for TGS Salt. It has no corresponding executed notebook output in this repository, so it is labeled `supplemental_leaderboard_only` in the normalized file and discussed separately in `EXPERIMENTAL_RESULTS.md`.

## CE versus focal loss on class-imbalanced data

The paired-fold harness is in `experiments/ab_loss_imbalance/`. Its checked-in Cassava records are at [`experiments/ab_loss_imbalance/results/outcomes.jsonl`](experiments/ab_loss_imbalance/results/outcomes.jsonl).

Executable files and checked-in records:

- [`experiments/ab_loss_imbalance/run_ab.py`](experiments/ab_loss_imbalance/run_ab.py)
- [`experiments/ab_loss_imbalance/collect.py`](experiments/ab_loss_imbalance/collect.py)
- [`experiments/ab_loss_imbalance/results/outcomes.jsonl`](experiments/ab_loss_imbalance/results/outcomes.jsonl)

Rebuild the summary from the ten checked-in fold records:

```bash
python -m experiments.ab_loss_imbalance.collect
```

Run the offline logic tests:

```bash
python -m pytest experiments/ab_loss_imbalance/tests \
  module4_agent/tests/test_fold_injection.py -q
```

Run new training only on a machine with accepted Kaggle competition terms, local credentials, the dataset, and a GPU:

```bash
python -m experiments.ab_loss_imbalance.run_ab \
  --testbed cassava \
  --data-root /path/to/kaggle_data \
  --output /path/to/ab_runs
```

The experiment uses seed 42 and a shared fold file for both loss arms. The collector reports paired `CE - focal` differences. A result from one dataset does not change the project-wide focal-loss default.

## Standalone offline reproduction

The independent Python 3.11 project is isolated at `experiments/mlestar_kaggle_benchmarks/`. Install and run it from that directory so its dependencies do not alter the root environment:

The main scripted entry point is [`experiments/mlestar_kaggle_benchmarks/scripts/run_smoke_experiment.py`](experiments/mlestar_kaggle_benchmarks/scripts/run_smoke_experiment.py).

```bash
cd experiments/mlestar_kaggle_benchmarks
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[vision,dev]'
python -m pytest -q
python scripts/run_smoke_experiment.py --output-dir /tmp/mlestar-smoke
```

The scripted smoke fixes seed 13, uses synthetic image-classification data, writes a configuration/environment/data-hash manifest, and asserts that no submission file is produced. It is an offline software check, not a Kaggle score.

## Root pipeline and Module 4 smoke run

The relevant entry points are [`pipeline.py`](pipeline.py) and [`module4_agent/__main__.py`](module4_agent/__main__.py).

```bash
python -m pip install -e '.[dev]'
M4_LLM_PROVIDER=none python -m module4_agent \
  --input module4_agent/examples/sample_m3_output.json \
  --output /tmp/graphrag-automl-module4-smoke
```

This validates generated code on synthetic inputs. It is not evidence of real-data model quality.

## Evidence export utility

[`experiments/notebook_runs/export_evidence.py`](experiments/notebook_runs/export_evidence.py) reproduces the public notebook copies, per-cell logs, and manifest from the four supplied archives. It strips embedded binary media, updates rerun setup cells, and retains textual output. The archives themselves are intentionally not committed.

The matching evidence test is [`experiments/notebook_runs/test_evidence.py`](experiments/notebook_runs/test_evidence.py).

After a text-only edit to an already exported public notebook, refresh its derived cell logs and manifest metadata without needing the private source archives:

```bash
python experiments/notebook_runs/export_evidence.py \
  --refresh-current \
  --output experiments/notebook_runs
python -m pytest experiments/notebook_runs/test_evidence.py -q
```

The evidence test reconstructs every log from the current Notebook JSON and requires an exact match, in addition to checking notebook hashes, cell counts, execution counts, output counts, and removal of embedded media.

Do not commit `.env`, Kaggle tokens, raw competition data, checkpoints, generated projects, submission files, OOF/test predictions, or private Kaggle artifact handles.

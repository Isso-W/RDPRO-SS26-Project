# MLE-STAR Vision Benchmark and Colab Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible task contracts and Colab runners for the ten screenshot competitions, use the new MLE-STAR agent for real fold-based experiments, and attempt Kaggle scoring only when the competition still accepts submissions and the account is authorized.

**Architecture:** A benchmark catalog will define immutable, testable contracts for data layout, targets, fold strategy, metric direction, submission schema, and modality-specific adapter. A generic Colab bootstrap downloads permitted Kaggle data, creates versioned Drive artifacts, runs one local MLE-STAR producer per candidate/fold, and creates a final OOF-safe consumer submission. The existing Plant Pathology script becomes a thin compatibility wrapper over the generic runner instead of a separate one-off workflow.

**Tech Stack:** Python 3.10+, Kaggle API, Google Colab GPU, Google Drive, skrub, PyTorch/timm, albumentations, scikit-learn, optional segmentation/detection dependencies selected per adapter.

---

## Competition inventory

The catalog must freeze these task shapes before training. The runner will snapshot each competition's current Kaggle rules/data page and sample submission in `competition_snapshot.json`; Kaggle pages can change and most of these historical competitions may no longer accept new leaderboard submissions.

| Catalog key | Kaggle slug | modality | local selection metric | final format |
| --- | --- | --- | --- | --- |
| `plant_pathology_2020` | `plant-pathology-2020-fgvc7` | multi-label image classification | mean ROC AUC | four probability columns |
| `aptos_2019` | `aptos2019-blindness-detection` | ordinal image classification | quadratic weighted kappa | integer severity 0--4 |
| `dog_breed` | `dog-breed-identification` | 120-class image classification | multiclass log loss | one probability per breed |
| `global_wheat` | `global-wheat-detection` | object detection | official IoU-threshold mAP | image ID plus prediction string |
| `ultrasound_nerve` | `ultrasound-nerve-segmentation` | binary image segmentation | official mask-overlap metric | run-length encoded mask |
| `leaf_classification` | `leaf-classification` | tabular multiclass classification | official classification metric | ID plus class label |
| `aerial_cactus` | `aerial-cactus-identification` | binary image classification | ROC AUC | ID plus probability |
| `dogs_vs_cats` | `dogs-vs-cats-redux-kernels-edition` | binary image classification | log loss | ID plus probability |
| `histopathologic_cancer` | `histopathologic-cancer-detection` | binary patch classification | ROC AUC | ID plus probability |
| `denoising_dirty_documents` | `denoising-dirty-documents` | image-to-image denoising | RMSE | pixel-level image outputs |

Do not use a generic `categorical_accuracy` setting for Plant Pathology, do not treat APTOS as ordinary five-class accuracy, and do not turn the detection/segmentation/denoising outputs into class labels.

## Target file map

**Create**

- `benchmarks/__init__.py`
- `benchmarks/contracts.py` -- competition contract dataclasses and validation.
- `benchmarks/catalog.py` -- the ten canonical entries and aliases.
- `benchmarks/ingest.py` -- Kaggle download, data-layout discovery, sample-submission verification, and rule snapshot.
- `benchmarks/adapters/__init__.py`
- `benchmarks/adapters/classification.py`
- `benchmarks/adapters/ordinal.py`
- `benchmarks/adapters/detection.py`
- `benchmarks/adapters/segmentation.py`
- `benchmarks/adapters/denoising.py`
- `benchmarks/submit.py` -- submission validation, submit/poll receipt, and blocked status recording.
- `colab/__init__.py` -- local Colab-runner package marker; it is unrelated to `google.colab`.
- `colab/bootstrap.py` -- install, secret validation, Drive paths, and pinned run metadata.
- `colab/run_benchmark.py` -- prepare, train producer, consume, submit, and resume CLI.
- `colab/mlestar_benchmarks.ipynb` -- a parameterised Colab notebook committed without outputs.
- `tests/benchmarks/conftest.py`
- `tests/benchmarks/test_contracts.py`
- `tests/benchmarks/test_catalog.py`
- `tests/benchmarks/test_ingest.py`
- `tests/benchmarks/test_adapters.py`
- `tests/benchmarks/test_submit.py`
- `docs/COLAB_MLESTAR.md`
- `docs/BENCHMARK_STATUS.md`

**Modify**

- `vision_benchmark_catalog.py` -- re-export `benchmarks.catalog` for existing callers during migration.
- `ingestion/kaggle_loader.py` -- call the catalog/ingestion adapter rather than hard-coded image-only assumptions.
- `run_kaggle_benchmark.py` -- use `mlestar` output and the generic adapter.
- `kaggle_submit.py` -- retain its public function names while delegating validation/submission receipts to `benchmarks.submit`.
- `experiments/plant_pathology_2020_fgvc7_agentic/colab_prepare_train.py` -- delegate to `colab.run_benchmark` and preserve CLI flags.
- `experiments/plant_pathology_2020_fgvc7_agentic/COLAB_RUNBOOK.md` -- point to the generic notebook and compatibility command.
- `requirements.txt`, `pyproject.toml`, `.gitignore`, `README.md`.

### Task 1: Add explicit benchmark contracts and correct the catalog

**Files:**
- Create: `benchmarks/contracts.py`
- Create: `benchmarks/catalog.py`
- Create: `tests/benchmarks/test_contracts.py`
- Create: `tests/benchmarks/test_catalog.py`
- Modify: `vision_benchmark_catalog.py`

- [ ] **Step 1: Write failing catalog tests**

```python
from benchmarks.catalog import get_benchmark, list_benchmarks


def test_screenshot_competitions_have_distinct_modality_metric_and_submission_contracts():
    catalog = {item.key: item for item in list_benchmarks()}
    assert set(catalog) >= {
        "plant_pathology_2020", "aptos_2019", "dog_breed", "global_wheat",
        "ultrasound_nerve", "leaf_classification", "aerial_cactus", "dogs_vs_cats",
        "histopathologic_cancer", "denoising_dirty_documents",
    }
    assert catalog["aptos_2019"].metric.name == "qwk"
    assert catalog["global_wheat"].modality == "object_detection"
    assert catalog["denoising_dirty_documents"].submission.kind == "image_directory"


def test_alias_resolves_to_canonical_contract():
    assert get_benchmark("plant-pathology-2020-fgvc7").key == "plant_pathology_2020"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python3 -m pytest tests/benchmarks/test_contracts.py tests/benchmarks/test_catalog.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'benchmarks'`.

- [ ] **Step 3: Implement the contract classes and ten entries**

Implement `SubmissionContract(kind, id_column, prediction_columns, filename, rle_order)`, `FoldContract(strategy, n_splits, group_column, time_column)`, and `BenchmarkContract(key, competition, modality, metric, labels, data_hints, submission, folds, query)`. Validate that classification probability columns have explicit class order; detection requires box encoding; segmentation requires RLE metadata; denoising requires an output directory/filename transform.

Move the existing Cassava, State Farm, SIIM-ISIC, Hugging Face entries into a clearly separate `EXTRA_BENCHMARKS` map. `vision_benchmark_catalog.py` must re-export `BENCHMARKS`, `ALIASES`, and `get_benchmark` from the new module so existing tests/callers continue to work.

- [ ] **Step 4: Run catalog tests**

Run: `python3 -m pytest tests/benchmarks/test_contracts.py tests/benchmarks/test_catalog.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks vision_benchmark_catalog.py tests/benchmarks/test_contracts.py tests/benchmarks/test_catalog.py
git commit -m "feat: add ten competition benchmark contracts"
```

### Task 2: Ingest data safely and snapshot competition constraints

**Files:**
- Create: `benchmarks/ingest.py`
- Create: `tests/benchmarks/test_ingest.py`
- Modify: `ingestion/kaggle_loader.py`

- [ ] **Step 1: Write failing layout-discovery tests**

```python
from benchmarks.ingest import discover_layout, validate_sample_submission


def test_multilabel_layout_discovers_images_labels_and_submission(plant_fixture, plant_contract):
    layout = discover_layout(plant_fixture, plant_contract)
    assert layout.train_csv.name == "train.csv"
    assert layout.image_dir.name == "images"
    assert layout.sample_submission.name == "sample_submission.csv"
    validate_sample_submission(layout.sample_submission, plant_contract)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/benchmarks/test_ingest.py -q`

Expected: FAIL with `ImportError` for `discover_layout`.

- [ ] **Step 3: Implement download, layout, and snapshot logic**

`download_competition()` must authenticate through Kaggle's environment variables or `~/.kaggle/kaggle.json`, download only to `<data-root>/<slug>/raw`, extract there, and return a typed layout. It must never copy raw data into the repository. `discover_layout()` resolves only the glob hints from the contract and fails if zero or multiple incompatible paths are found. `validate_sample_submission()` verifies ID order/uniqueness and exact expected columns or the specified denoising file mapping.

`write_competition_snapshot()` captures slug, retrieval time, local Kaggle SDK version, sample schema, local data fingerprint, and command result from `kaggle competitions files <slug>`. Put raw command error text into `availability_error` and mark `submission_availability` as `unknown`, `available`, or `blocked`; do not claim an old competition is submittable without a successful API response.

- [ ] **Step 4: Run ingestion tests**

Run: `python3 -m pytest tests/benchmarks/test_ingest.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/ingest.py ingestion/kaggle_loader.py tests/benchmarks/test_ingest.py
git commit -m "feat: add safe benchmark ingestion and snapshots"
```

### Task 3: Implement modality adapters and submission validators

**Files:**
- Create: `benchmarks/adapters/classification.py`
- Create: `benchmarks/adapters/ordinal.py`
- Create: `benchmarks/adapters/detection.py`
- Create: `benchmarks/adapters/segmentation.py`
- Create: `benchmarks/adapters/denoising.py`
- Create: `tests/benchmarks/test_adapters.py`

- [ ] **Step 1: Write failing output-format tests**

```python
def test_aptos_thresholds_convert_continuous_oof_predictions_to_legal_classes():
    labels = apply_ordinal_thresholds([0.2, 1.1, 2.8, 3.9], [0.5, 1.5, 2.5, 3.5])
    assert labels == [0, 1, 3, 4]


def test_segmentation_rle_round_trip(mask):
    assert decode_rle(encode_rle(mask), mask.shape).tolist() == mask.tolist()


def test_detection_submission_uses_the_exact_contract_columns(wheat_contract, boxes):
    frame = build_detection_submission(boxes, wheat_contract)
    assert list(frame.columns) == ["image_id", "PredictionString"]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/benchmarks/test_adapters.py -q`

Expected: FAIL with missing adapter functions.

- [ ] **Step 3: Implement adapters with OOF-only post-processing**

Classification adapter: class order, binary/multilabel probability clipping, exact ID alignment, multiclass log loss, ROC AUC, and log loss.

Ordinal adapter: continuous expected-severity prediction, OOF QWK threshold search, persisted thresholds, and integer submission labels.

Detection adapter: annotation parsing, per-image box prediction schema, official-contract prediction-string encoder, and IoU metric dispatcher. Implement only the exact string separator and coordinates declared in the catalog contract.

Segmentation adapter: mask loading, RLE encode/decode using the specified flatten order, overlap scorer, and one-row-per-image submission.

Denoising adapter: input/target pairing, RMSE, clipping to `[0, 255]` or `[0, 1]` as declared, and preservation of expected output filenames/directories.

All adapters must refuse train/test ID overlap and must write metrics/thresholds/postprocessing parameters into an artifact JSON.

- [ ] **Step 4: Run adapter tests**

Run: `python3 -m pytest tests/benchmarks/test_adapters.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add benchmarks/adapters tests/benchmarks/test_adapters.py
git commit -m "feat: add modality-specific benchmark adapters"
```

### Task 4: Build a generic Colab bootstrap and Drive artifact layout

**Files:**
- Create: `colab/bootstrap.py`
- Create: `colab/run_benchmark.py`
- Create: `colab/mlestar_benchmarks.ipynb`
- Create: `tests/benchmarks/test_submit.py`
- Modify: `requirements.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing environment validation tests**

```python
from colab.bootstrap import validate_colab_environment


def test_colab_validation_reports_missing_kaggle_credentials_without_printing_secret(monkeypatch):
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    result = validate_colab_environment(require_kaggle=True)
    assert result.ready is False
    assert "KAGGLE_KEY" in result.missing
    assert "secret" not in result.to_dict()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/benchmarks/test_submit.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'colab'`.

- [ ] **Step 3: Implement bootstrap and parameterised notebook**

`validate_colab_environment()` detects Colab, GPU, Drive mount, Kaggle SDK, credentials, and optional LLM credentials without outputting values. `bootstrap_run()` creates:

```text
/content/drive/MyDrive/jiaozi-runs/<benchmark>/<run-id>/
  data/  runs/  remote_outputs/  submissions/  environment.json
```

The notebook must have parameter cell variables `BENCHMARK`, `RUN_ID`, `INITIAL_CANDIDATES`, `OUTER_ROUNDS`, `INNER_ROUNDS`, `EPOCHS`, `FOLD`, and `SUBMIT`; it must call the same Python CLI as a non-notebook run. Do not embed a Kaggle token, LLM key, Drive ID, data, training outputs, or notebook outputs in the committed notebook. Install the pinned root requirements plus `albumentations`, `timm`, `opencv-python-headless`, and only the adapter-specific package chosen by the benchmark.

Add `"benchmarks*"` and `"colab*"` to `[tool.setuptools.packages.find].include` in `pyproject.toml` so `python -m colab.run_benchmark` resolves when the project is installed.

- [ ] **Step 4: Run bootstrap tests**

Run: `python3 -m pytest tests/benchmarks/test_submit.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add colab tests/benchmarks/test_submit.py requirements.txt .gitignore
git commit -m "feat: add reproducible Colab benchmark bootstrap"
```

### Task 5: Connect benchmark execution to MLE-STAR producers and consumer

**Files:**
- Modify: `colab/run_benchmark.py`
- Modify: `run_kaggle_benchmark.py`
- Modify: `kaggle_submit.py`
- Modify: `experiments/plant_pathology_2020_fgvc7_agentic/colab_prepare_train.py`
- Modify: `experiments/plant_pathology_2020_fgvc7_agentic/COLAB_RUNBOOK.md`

- [ ] **Step 1: Write failing orchestration test**

```python
from colab.run_benchmark import build_mlestar_command


def test_colab_command_passes_immutable_data_and_fold_paths(tmp_path):
    command = build_mlestar_command("plant_pathology_2020", tmp_path / "data", tmp_path / "run", fold=2)
    assert "--task" in command and "--data-root" in command and "--run-dir" in command
    assert command[-1] == "2"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/benchmarks/test_submit.py -q`

Expected: FAIL with `ImportError` for `build_mlestar_command`.

- [ ] **Step 3: Implement producer/consumer execution commands**

`colab.run_benchmark prepare` authenticates, downloads, snapshots, creates task JSON, and writes folds. `train` calls `python -m mlestar run` with the selected fold/run limits. `consume` validates producer receipts and asks `mlestar.ensemble` to produce the final artifact. `submit` calls `benchmarks.submit.submit_and_poll` only after `validate_submission()` passes.

`run_kaggle_benchmark.py` becomes a compatibility command that calls `prepare` then prints the exact Colab/local train command. `kaggle_submit.py` preserves `predict_and_submit()` but delegates all ID/schema checks, submission receipt structure, polling, and `blocked` status handling. The Plant Pathology helper forwards its existing CLI flags to the generic runner, preserving existing documented commands.

- [ ] **Step 4: Run orchestration tests**

Run: `python3 -m pytest tests/benchmarks/test_submit.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add colab run_kaggle_benchmark.py kaggle_submit.py experiments/plant_pathology_2020_fgvc7_agentic tests/benchmarks/test_submit.py
git commit -m "feat: run MLE-STAR benchmark producers from Colab"
```

### Task 6: Run a representative modality matrix before all ten competitions

**Files:**
- Create: `docs/BENCHMARK_STATUS.md`
- Modify: `docs/COLAB_MLESTAR.md`

- [ ] **Step 1: Add a machine-readable status table template**

Create `docs/BENCHMARK_STATUS.md` with one row per catalog key and columns: contract checked, data downloaded, fold file path/fingerprint, baseline OOF metric, MLE-STAR best OOF metric, ensemble OOF metric, submitted, Kaggle status, public score, run directory, and blocker. Initial values must be `not_run`, not invented scores.

- [ ] **Step 2: Run four representative smoke experiments in this order**

Run after credentials and GPU access exist:

```bash
python -m colab.run_benchmark prepare plant_pathology_2020 --data-root "$RUN_ROOT/data"
python -m colab.run_benchmark train plant_pathology_2020 --run-root "$RUN_ROOT/runs" --initial-candidates 2 --outer-rounds 1 --inner-rounds 1 --epochs 1
python -m colab.run_benchmark prepare global_wheat --data-root "$RUN_ROOT/data"
python -m colab.run_benchmark train global_wheat --run-root "$RUN_ROOT/runs" --initial-candidates 1 --outer-rounds 0 --inner-rounds 0 --epochs 1
python -m colab.run_benchmark prepare ultrasound_nerve --data-root "$RUN_ROOT/data"
python -m colab.run_benchmark train ultrasound_nerve --run-root "$RUN_ROOT/runs" --initial-candidates 1 --outer-rounds 0 --inner-rounds 0 --epochs 1
python -m colab.run_benchmark prepare denoising_dirty_documents --data-root "$RUN_ROOT/data"
python -m colab.run_benchmark train denoising_dirty_documents --run-root "$RUN_ROOT/runs" --initial-candidates 1 --outer-rounds 0 --inner-rounds 0 --epochs 1
```

Expected: each command writes a valid inventory, fold artifact where applicable, DataOps report, real local metric, and explicit `not_submitted`, `scored`, or `blocked` submission state.

- [ ] **Step 3: Expand to the remaining six after their representative adapter passes**

For `aptos_2019`, `dog_breed`, `leaf_classification`, `aerial_cactus`, `dogs_vs_cats`, and `histopathologic_cancer`, run `prepare`, a one-epoch/one-fold `train`, validate the adapter submission, and record the OOF score. Increase to the planned folds/candidates only after the one-fold run passes data, metric, and submission-schema checks.

- [ ] **Step 4: Attempt submission only where the Kaggle API permits it**

Run: `python -m colab.run_benchmark submit <benchmark> --run-dir <completed-run-dir>`

Expected: a `submission_receipt.json` with `status: scored`, or `status: blocked` plus API error text, timestamp, retry count, exact next action, and the local artifact path. Do not describe historical competition results as new Kaggle scores when submissions are closed.

- [ ] **Step 5: Commit only code and sanitized status data**

```bash
git add docs/BENCHMARK_STATUS.md docs/COLAB_MLESTAR.md
git commit -m "docs: add benchmark execution matrix"
```

### Task 7: Document the exact user-operated Colab steps and final verification

**Files:**
- Create: `docs/COLAB_MLESTAR.md`
- Modify: `README.md`

- [ ] **Step 1: Document account setup without secrets**

The guide must instruct the user to: create/log in to a Kaggle account; open and accept each competition's rules; create an API token; store `KAGGLE_USERNAME` and `KAGGLE_KEY` in Colab Secrets (not in notebook code or Git); optionally store the LLM provider key as a Colab Secret; select a GPU runtime; and mount Drive. Include the exact verification command:

```bash
python -m colab.run_benchmark doctor --require-gpu --require-kaggle
```

- [ ] **Step 2: Document the first full Plant Pathology run**

```bash
export RUN_ROOT=/content/drive/MyDrive/jiaozi-runs/plant-pathology-2020/$(date +%Y%m%d-%H%M%S)
python -m colab.run_benchmark prepare plant_pathology_2020 --data-root "$RUN_ROOT/data"
python -m colab.run_benchmark train plant_pathology_2020 --run-root "$RUN_ROOT/runs" --fold 0 --initial-candidates 2 --outer-rounds 1 --inner-rounds 2 --epochs 5
python -m colab.run_benchmark consume plant_pathology_2020 --run-dir "$RUN_ROOT/runs"
python -m colab.run_benchmark validate-submission plant_pathology_2020 --run-dir "$RUN_ROOT/runs"
```

Document that a proper blend needs all folds with identical `folds.parquet`; one-fold runs are smoke tests, not an ensemble selection result.

- [ ] **Step 3: Run test and static checks**

Run: `python3 -m pytest tests/benchmarks -q && python3 -m pytest module4_agent/tests test_kaggle_orchestrator.py -q && git diff --check`

Expected: PASS or an explicitly recorded missing-package failure; no raw data, secrets, predictions, checkpoints, or submissions are staged.

- [ ] **Step 4: Commit**

```bash
git add docs/COLAB_MLESTAR.md README.md
git commit -m "docs: add secure Colab execution guide"
```

## Acceptance criteria

- Each screenshot competition has a typed data/metric/fold/submission contract, with the correct modality-specific adapter.
- Every Colab run is reproducible from a Drive run directory and emits data/fold/code/environment/artifact metadata.
- The MLE-STAR agent is selected by local OOF metrics, not public leaderboard luck or proxy scores.
- Kaggle credentials and competition data never enter Git or notebook outputs.
- Submission is attempted only after schema validation; historical/closed competitions produce a concrete blocked receipt rather than a fabricated score.
- Plant Pathology remains runnable through its existing entry point, but uses the generic reproducible flow.

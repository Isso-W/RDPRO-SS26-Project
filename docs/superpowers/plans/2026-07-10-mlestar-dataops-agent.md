# MLE-STAR DataOps Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a faithful, reproducible MLE-STAR-style agent which searches for task-specific approaches, generates Python ML projects expressed as skrub DataOps DAGs, evaluates them with real local validation, ablates one pipeline component at a time, refines the highest-impact component, and produces an OOF-safe ensemble.

**Architecture:** Add an `mlestar` package without replacing the existing Module 1--4 prototype. The orchestrator will persist a task contract, a data inventory, immutable fold assignments, retrieved evidence, generated candidate projects, real experiment receipts, audits, ablations, and ensemble artifacts in one run directory. Generated projects will expose a `build_dataops_plan()` function; DataOps will own loading, manifests, split/fold selection, feature/data preparation, training invocation, prediction, metric calculation, and submission assembly, while PyTorch/timm remains the image-model implementation inside deferred nodes.

**Tech Stack:** Python 3.10+, skrub DataOps, pandas, scikit-learn, PyTorch/timm, OpenAI-compatible LLM provider, JSONL/JSON artifacts, pytest, Kaggle API only at the submission boundary.

---

## Scope and non-negotiable rules

- This replaces the current `module4_agent.refinement.proxy_evaluate` decision path only for the new `mlestar` command. Existing Module 4 remains backwards compatible.
- Do not edit the already-dirty `retrieval/rag_retrieval.py`.
- A score may guide selection only when it was calculated from a fixed fold, OOF prediction, or an explicit holdout. `proxy_*` scores must never appear in an MLE-STAR run receipt.
- Generated programs may modify only their run directory. The executor invokes one allowlisted command, `python -m mlestar.generated_runner --project <project-dir> --mode <mode>`, with a sanitized environment and timeout.
- All generated code must contain the five markers `data_loading`, `data_preparation`, `model`, `training`, and `prediction`; no model-selection or preprocessing code is permitted outside its marker range.
- `skrub` is mandatory for the agent and for every generated project. It is not an optional visualization dependency.
- Search is evidence retrieval, not code copying. Persist title, URL, short evidence summary, license note, and retrieval time; never write raw web pages or Kaggle data into Git.

## Target file map

**Create**

- `mlestar/__init__.py` -- public package exports.
- `mlestar/contracts.py` -- typed task, component, evidence, experiment, audit, and ensemble contracts.
- `mlestar/dataops.py` -- generic DataOps node factories and generated-project protocol.
- `mlestar/dataset.py` -- inventory, persistent folds, metric dispatch, and dataset fingerprinting.
- `mlestar/search.py` -- injected/live search provider boundary and evidence normalisation.
- `mlestar/generation.py` -- LLM prompts, candidate project generation, AST/component validation, and deterministic fallback project.
- `mlestar/executor.py` -- constrained subprocess execution and receipt parsing.
- `mlestar/audits.py` -- leakage, data-usage, dependency, and output-schema checks.
- `mlestar/ablation.py` -- no-op component variants and impact ranking from real validation.
- `mlestar/refinement.py` -- inner-loop plan/proposal parsing and component-only patching.
- `mlestar/ensemble.py` -- OOF alignment, non-negative simplex blend search, and final consumer writer.
- `mlestar/workflow.py` -- initial candidate, outer-loop, inner-loop, and ensemble orchestration.
- `mlestar/__main__.py` -- `python -m mlestar` CLI.
- `mlestar/prompts.py` -- JSON-only prompts for evidence ranking, project generation, refinement, and ensemble planning.
- `tests/mlestar/conftest.py` -- synthetic tabular/image-manifest fixtures and fake LLM/search providers.
- `tests/mlestar/test_contracts.py`
- `tests/mlestar/test_dataops.py`
- `tests/mlestar/test_dataset.py`
- `tests/mlestar/test_search.py`
- `tests/mlestar/test_generation.py`
- `tests/mlestar/test_executor.py`
- `tests/mlestar/test_audits.py`
- `tests/mlestar/test_ablation.py`
- `tests/mlestar/test_ensemble.py`
- `tests/mlestar/test_workflow.py`
- `docs/MLESTAR_DATAOPS.md` -- architecture, artifact contract, security model, and local quick start.

**Modify**

- `pyproject.toml` -- runtime dependencies, package discovery, and `jiaozi-mlestar` command.
- `requirements.txt` -- pin the runtime packages used by generated DataOps projects.
- `.gitignore` -- ignore `runs/`, Kaggle credentials, data, checkpoints, OOF predictions, and submissions.
- `README.md` -- link to the new agent and state its difference from the legacy Module 4 proxy loop.
- `module4_agent/refinement.py` -- label its public proxy API as legacy in the docstring; keep behavior unchanged.

## Artifact contract

Every run is stored at `runs/<run_id>/` and has this minimum tree:

```text
task.json                 inventory.json             folds.parquet
search_evidence.json      candidates.json            audit.jsonl
experiments.jsonl         oof/<candidate>.parquet    predictions/<candidate>.parquet
projects/<candidate>/     ensemble/                  final_report.json
```

`experiments.jsonl` records `experiment_id`, parent ID, candidate ID, component, stage, fold, metric name/direction/value, elapsed seconds, status, code SHA-256, data fingerprint, OOF path, prediction path, and error text. A run is successful only when `final_report.json` points to valid OOF predictions and, when test data exists, a schema-checked submission.

### Task 1: Add reproducible contracts and package configuration

**Files:**
- Create: `mlestar/__init__.py`
- Create: `mlestar/contracts.py`
- Create: `tests/mlestar/test_contracts.py`
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing contract round-trip tests**

```python
from mlestar.contracts import Component, MetricSpec, TaskContract


def test_task_contract_round_trip_and_component_set():
    task = TaskContract(
        task_id="tiny_binary",
        modality="image_classification",
        target_columns=["target"],
        id_column="image_id",
        metric=MetricSpec(name="roc_auc", greater_is_better=True),
        components=[Component(name="data_loading"), Component(name="data_preparation"),
                    Component(name="model"), Component(name="training"),
                    Component(name="prediction")],
    )
    assert TaskContract.from_dict(task.to_dict()) == task


def test_task_contract_rejects_missing_or_duplicate_component_markers():
    with pytest.raises(ValueError, match="exactly"):
        TaskContract(task_id="x", modality="tabular", target_columns=["y"], id_column="id",
                     metric=MetricSpec("accuracy", True), components=[Component("model")])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/mlestar/test_contracts.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'mlestar'`.

- [ ] **Step 3: Implement serializable, validated contracts**

Implement frozen dataclasses `MetricSpec`, `Component`, `TaskContract`, `DatasetInventory`, `SearchEvidence`, `CandidateProject`, `ExperimentReceipt`, `AuditFinding`, and `EnsembleReceipt`. Use `to_dict()` and explicit `from_dict()` methods, ISO-8601 UTC timestamps, and these constants:

```python
COMPONENT_NAMES = (
    "data_loading", "data_preparation", "model", "training", "prediction",
)
MODALITIES = {
    "tabular", "image_classification", "object_detection",
    "image_segmentation", "image_to_image",
}
```

`TaskContract.__post_init__` must require each component exactly once, a nonempty task/ID/target field, and a nonempty metric. `ExperimentReceipt.success` is true only when `status == "success"`, `metric_value is not None`, and `oof_path` exists in its serialized payload.

- [ ] **Step 4: Add package dependencies and command entry point**

Add `"skrub>=0.6"`, `"scikit-learn"`, `"timm"`, and `"kaggle>=2.2.2"` to both dependency declarations. Add package patterns `"mlestar*"` and a console command:

```toml
[project.scripts]
jiaozi-mlestar = "mlestar.__main__:main"
```

Ignore `/runs/`, `/kaggle_data/`, `/remote_outputs/`, `kaggle.json`, `*.pt`, `*.pth`, `*.ckpt`, `*.safetensors`, `*.parquet`, and `submission*.csv` without ignoring source fixtures.

- [ ] **Step 5: Run the contract tests**

Run: `python3 -m pytest tests/mlestar/test_contracts.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add mlestar tests/mlestar/test_contracts.py pyproject.toml requirements.txt .gitignore
git commit -m "feat: add MLE-STAR run contracts"
```

### Task 2: Make DataOps the generated-pipeline boundary

**Files:**
- Create: `mlestar/dataops.py`
- Create: `tests/mlestar/test_dataops.py`

- [ ] **Step 1: Write failing DataOps protocol tests**

```python
from mlestar.dataops import build_dataops_plan


def test_generated_plan_is_lazy_and_executes_all_five_components(tmp_path, tiny_contract):
    plan = build_dataops_plan(tiny_contract, tmp_path / "project", mode="validate")
    description = plan.skb.describe_steps()
    for component in tiny_contract.component_names:
        assert component in description
    output = plan.skb.eval()
    assert output["mode"] == "validate"
    assert output["component_trace"] == list(tiny_contract.component_names)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/mlestar/test_dataops.py -q`

Expected: FAIL with `ImportError` for `build_dataops_plan`.

- [ ] **Step 3: Implement the stable DataOps protocol**

Implement `build_dataops_plan(contract, project_dir, mode)` with `skrub.var` inputs and five named `skrub.deferred` nodes. The nodes must call these project-local functions in order: `load_data`, `prepare_data`, `build_model`, `train_model`, and `predict_or_submit`. Each node receives and returns one JSON-compatible dictionary containing `run_context`, `component_trace`, artifact paths, and the previous node result. Name nodes with `.skb.set_name(component)` when that API is available, otherwise wrap each node with a `named_step` function whose name is the component.

The plan must not evaluate during construction. `run_dataops_project()` calls `.skb.eval()`, writes `dataops_report.json` with `describe_steps()` and `full_report()` output when available, and returns the terminal dictionary. Use `Path.resolve()` and reject a project path outside `run_context["workspace_root"]`.

- [ ] **Step 4: Run the DataOps test**

Run: `python3 -m pytest tests/mlestar/test_dataops.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mlestar/dataops.py tests/mlestar/test_dataops.py
git commit -m "feat: add DataOps project execution boundary"
```

### Task 3: Create immutable inventories, folds, and metric-correct OOF evaluation

**Files:**
- Create: `mlestar/dataset.py`
- Create: `tests/mlestar/test_dataset.py`

- [ ] **Step 1: Write failing fold and metric tests**

```python
from mlestar.dataset import make_folds, score_oof


def test_stratified_folds_are_persistent_and_cover_each_row_once(binary_frame, tmp_path):
    folds = make_folds(binary_frame, target="target", strategy="stratified", n_splits=3, seed=7,
                       output_path=tmp_path / "folds.parquet")
    assert sorted(folds["fold"].tolist()) == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert (tmp_path / "folds.parquet").exists()


def test_roc_auc_score_uses_only_oof_rows():
    assert score_oof("roc_auc", [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8]) == 1.0
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/mlestar/test_dataset.py -q`

Expected: FAIL with `ImportError` for `make_folds`.

- [ ] **Step 3: Implement inventory, folds, and scorers**

`inspect_dataset()` must recursively inventory every file under the supplied data root, recording relative path, byte size, SHA-256 for files up to 64 MiB, CSV columns, image count/extensions, and the sample-submission columns when present. It writes `inventory.json` and a deterministic fingerprint derived from relative paths plus sizes/hashes.

Implement `make_folds()` with `StratifiedKFold`, `StratifiedGroupKFold`, `GroupKFold`, `KFold`, and a chronological split selected by `strategy`. It must reject a requested split count larger than the smallest class count and must write the chosen fold plus row identifier into Parquet. `score_oof()` must dispatch `roc_auc`, `multiclass_log_loss`, `log_loss`, `accuracy`, `qwk`, `rmse`, `dice`, and `map_iou`; lower-is-better metrics return their natural error value and rely on `MetricSpec.greater_is_better` for comparison.

- [ ] **Step 4: Run the dataset tests**

Run: `python3 -m pytest tests/mlestar/test_dataset.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mlestar/dataset.py tests/mlestar/test_dataset.py
git commit -m "feat: add immutable folds and OOF scoring"
```

### Task 4: Retrieve and record task-specific model evidence

**Files:**
- Create: `mlestar/search.py`
- Create: `tests/mlestar/test_search.py`
- Create: `mlestar/prompts.py`

- [ ] **Step 1: Write failing provider-normalisation tests**

```python
from mlestar.search import StaticSearchProvider, retrieve_model_evidence


def test_retrieval_deduplicates_urls_and_keeps_model_plus_example_code(tmp_path, tiny_contract):
    provider = StaticSearchProvider([
        {"title": "Efficient model", "url": "https://example.test/a", "snippet": "Use TinyNet.", "code": "TinyNet()"},
        {"title": "Duplicate", "url": "https://example.test/a", "snippet": "Duplicate", "code": ""},
    ])
    evidence = retrieve_model_evidence(tiny_contract, provider, limit=4, output_path=tmp_path / "search_evidence.json")
    assert len(evidence) == 1
    assert evidence[0].model_hint == "TinyNet"
    assert evidence[0].example_code == "TinyNet()"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/mlestar/test_search.py -q`

Expected: FAIL with `ImportError` for `StaticSearchProvider`.

- [ ] **Step 3: Implement an injectable search boundary**

Define a `SearchProvider` protocol with `search(query: str, limit: int) -> list[dict]`, a `StaticSearchProvider` for tests/manual evidence, and an `OpenAIWebSearchProvider` that requests search results through the configured OpenAI-compatible provider only when it advertises web-search tool support. The fallback must raise `SearchUnavailable` with an actionable message; it must not invent citations.

`retrieve_model_evidence()` builds one query from modality, metric, task description, and resource constraints; normalizes result title/URL/snippet/code/license; deduplicates canonical URLs; extracts a one-line `model_hint`; writes sorted evidence to JSON; and returns at most `limit` entries. `prompts.py` must request a source URL and compact model/example-code summary in JSON, not copied prose.

- [ ] **Step 4: Run the search tests**

Run: `python3 -m pytest tests/mlestar/test_search.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mlestar/search.py mlestar/prompts.py tests/mlestar/test_search.py
git commit -m "feat: add evidence-backed model search"
```

### Task 5: Generate component-scoped DataOps projects and validate their source

**Files:**
- Create: `mlestar/generation.py`
- Create: `tests/mlestar/test_generation.py`

- [ ] **Step 1: Write failing source-contract tests**

```python
from mlestar.generation import validate_generated_project


def test_validator_accepts_all_component_markers_and_dataops_entrypoint(tmp_path, tiny_contract):
    project = write_valid_project(tmp_path)
    result = validate_generated_project(project, tiny_contract)
    assert result.is_valid


def test_validator_rejects_training_code_outside_training_marker(tmp_path, tiny_contract):
    project = write_project_with_top_level_fit(tmp_path)
    result = validate_generated_project(project, tiny_contract)
    assert "outside component marker" in result.errors[0]
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/mlestar/test_generation.py -q`

Expected: FAIL with `ImportError` for `validate_generated_project`.

- [ ] **Step 3: Implement generation and deterministic fallback**

Generate `project.py`, `config.json`, `requirements.txt`, and `README.md` per candidate. `project.py` must import `build_dataops_plan` from `mlestar.dataops`, expose `build_dataops_plan(config_path, mode)`, and contain these exact marker pairs:

```python
# MLESTAR_COMPONENT:data_loading:START
# MLESTAR_COMPONENT:data_loading:END
# MLESTAR_COMPONENT:data_preparation:START
# MLESTAR_COMPONENT:data_preparation:END
# MLESTAR_COMPONENT:model:START
# MLESTAR_COMPONENT:model:END
# MLESTAR_COMPONENT:training:START
# MLESTAR_COMPONENT:training:END
# MLESTAR_COMPONENT:prediction:START
# MLESTAR_COMPONENT:prediction:END
```

Use the LLM only to fill the marker bodies and require a JSON envelope with `files`, `rationale`, and `assumptions`. If the provider is unavailable or returns invalid code, emit the deterministic sklearn baseline for tabular tasks and a minimal `timm` image classifier for image classification; never silently omit an evidence-selected model. `validate_generated_project()` must parse AST, reject `eval`, `exec`, shell/network subprocesses, absolute writes outside the project directory, reads of test targets, imports missing from requirements, duplicate/missing markers, and top-level model fitting.

- [ ] **Step 4: Run generation tests**

Run: `python3 -m pytest tests/mlestar/test_generation.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mlestar/generation.py tests/mlestar/test_generation.py
git commit -m "feat: generate validated component-scoped DataOps projects"
```

### Task 6: Execute projects safely and persist real receipts

**Files:**
- Create: `mlestar/executor.py`
- Create: `tests/mlestar/test_executor.py`

- [ ] **Step 1: Write failing execution tests**

```python
from mlestar.executor import execute_project


def test_executor_parses_real_metric_and_rejects_wrong_output_schema(tmp_path, tiny_contract):
    project = write_executable_project(tmp_path, metric=0.875)
    receipt = execute_project(project, tiny_contract, workspace_root=tmp_path, timeout_seconds=10)
    assert receipt.status == "success"
    assert receipt.metric_value == 0.875
    assert receipt.oof_path.endswith("oof.parquet")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/mlestar/test_executor.py -q`

Expected: FAIL with `ImportError` for `execute_project`.

- [ ] **Step 3: Implement the runner contract**

Add an internal `mlestar.generated_runner` module which loads only `project.py`, builds its DataOps plan, calls `.skb.eval()`, and prints exactly one JSON object to stdout. `execute_project()` must run that module with `cwd=project`, `PYTHONNOUSERSITE=1`, a minimal `PATH`, no credential environment variables, a timeout, and `workspace_root` containment checks. It must require terminal keys `metric_name`, `metric_value`, `oof_path`, `prediction_path`, `submission_path`, and `component_trace`; write stdout/stderr plus a receipt JSON; and return a failed receipt rather than raise on training errors.

- [ ] **Step 4: Run execution tests**

Run: `python3 -m pytest tests/mlestar/test_executor.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mlestar/executor.py tests/mlestar/test_executor.py
git commit -m "feat: execute DataOps projects with real receipts"
```

### Task 7: Enforce data-use and leakage audits before execution

**Files:**
- Create: `mlestar/audits.py`
- Create: `tests/mlestar/test_audits.py`

- [ ] **Step 1: Write failing audit tests**

```python
from mlestar.audits import audit_project


def test_audit_flags_test_statistics_used_for_imputation(tmp_path, tiny_contract):
    project = write_project(tmp_path, "all_rows = pd.concat([train, test]); mean = all_rows.x.mean()")
    findings = audit_project(project, tiny_contract, inventory={"files": ["train.csv", "test.csv"]})
    assert any(item.code == "test_statistics" and item.severity == "error" for item in findings)


def test_audit_flags_unreferenced_required_mask_directory(tmp_path, tiny_contract):
    project = write_project(tmp_path, "train = pd.read_csv('train.csv')")
    findings = audit_project(project, tiny_contract, inventory={"files": ["train.csv", "masks/001.png"]})
    assert any(item.code == "unused_data_source" for item in findings)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/mlestar/test_audits.py -q`

Expected: FAIL with `ImportError` for `audit_project`.

- [ ] **Step 3: Implement static and manifest audits**

Parse AST plus normalized source strings. Emit error findings for concatenating train/test before a fitted transform, fitting transformers on test rows, target access from test tables, and validation labels used during inference. Emit warning findings for inventory files or declared image/mask directories never referenced, missing fold path use, nondeterministic random seeds, and unpinned third-party imports. `audit_project()` writes JSONL findings. `workflow.py` must block execution on error findings and preserve warnings in `final_report.json`.

- [ ] **Step 4: Run audit tests**

Run: `python3 -m pytest tests/mlestar/test_audits.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mlestar/audits.py tests/mlestar/test_audits.py
git commit -m "feat: audit generated projects for leakage and data use"
```

### Task 8: Replace heuristic refinement with real ablation and component-only patches

**Files:**
- Create: `mlestar/ablation.py`
- Create: `mlestar/refinement.py`
- Create: `tests/mlestar/test_ablation.py`

- [ ] **Step 1: Write failing ablation tests**

```python
from mlestar.ablation import rank_component_impact


def test_largest_metric_drop_selects_component_with_metric_direction():
    scores = {"data_loading": 0.80, "data_preparation": 0.71, "model": 0.78,
              "training": 0.76, "prediction": 0.79}
    assert rank_component_impact(0.82, scores, greater_is_better=True) == "data_preparation"


def test_refinement_replaces_only_selected_marker_body(tmp_path):
    before = write_marked_project(tmp_path)
    after = apply_component_patch(before, "training", "scheduler = cosine_scheduler(optimizer)")
    assert component_body(after, "model") == component_body(before, "model")
    assert "cosine_scheduler" in component_body(after, "training")
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/mlestar/test_ablation.py -q`

Expected: FAIL with `ImportError` for `rank_component_impact`.

- [ ] **Step 3: Implement ablation and refinement**

`make_ablation_project()` must copy a validated project and replace exactly one marker body with its declared no-op baseline: raw manifest read, identity preparation, baseline model, one-epoch/default training, or unaugmented prediction. Execute every ablation against the same folds and seeds as its parent. `rank_component_impact()` compares baseline to ablation using metric direction and selects the largest harmful delta; ties resolve by component order.

`request_refinement_plans()` asks for at most four JSON plans for the selected component, includes earlier plan/metric feedback, and rejects plans modifying another component. `apply_component_patch()` replaces only bytes inside the selected marker pair, revalidates AST/audits, and creates a child candidate ID. Stop an inner loop after no improvement or after `inner_rounds`; stop the outer loop after `outer_rounds` or no unrefined impactful component remains.

- [ ] **Step 4: Run the ablation tests**

Run: `python3 -m pytest tests/mlestar/test_ablation.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mlestar/ablation.py mlestar/refinement.py tests/mlestar/test_ablation.py
git commit -m "feat: add real targeted component refinement"
```

### Task 9: Create OOF-safe ensemble selection and consumer predictions

**Files:**
- Create: `mlestar/ensemble.py`
- Create: `tests/mlestar/test_ensemble.py`

- [ ] **Step 1: Write failing ensemble tests**

```python
from mlestar.ensemble import fit_simplex_blend


def test_blend_uses_oof_rows_and_returns_non_negative_sum_one_weights():
    result = fit_simplex_blend(
        y_true=[0, 1, 0, 1],
        oof_by_candidate={"a": [0.1, 0.8, 0.2, 0.7], "b": [0.2, 0.9, 0.1, 0.8]},
        metric_name="roc_auc",
    )
    assert set(result.weights) == {"a", "b"}
    assert all(weight >= 0 for weight in result.weights.values())
    assert sum(result.weights.values()) == pytest.approx(1.0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/mlestar/test_ensemble.py -q`

Expected: FAIL with `ImportError` for `fit_simplex_blend`.

- [ ] **Step 3: Implement constrained ensembling**

Align candidates by immutable row ID and reject missing/duplicate OOF rows, different class orders, or predictions produced in-fold. For binary/multiclass probabilities, grid-search or optimize non-negative weights that sum to one using only OOF metric values. For ordinal labels, blend continuous expected severity then tune legal thresholds on OOF only. For detection/segmentation/image-to-image, start with one validated model unless every candidate exports a same-schema OOF artifact; only then invoke an LLM for a JSON strategy that is translated into allowlisted merge operators.

Write `ensemble/blend_report.json`, `oof_ensemble.parquet`, `test_ensemble.parquet`, `submission.csv`, and `ensemble_receipt.json`. The receipt must include base candidate metrics, OOF correlation matrix, weights/operator, final OOF score, and schema validation result.

- [ ] **Step 4: Run ensemble tests**

Run: `python3 -m pytest tests/mlestar/test_ensemble.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mlestar/ensemble.py tests/mlestar/test_ensemble.py
git commit -m "feat: add OOF-safe MLE-STAR ensembling"
```

### Task 10: Orchestrate candidate search, real execution, refinement, and reporting

**Files:**
- Create: `mlestar/workflow.py`
- Create: `mlestar/__main__.py`
- Create: `tests/mlestar/test_workflow.py`
- Modify: `module4_agent/refinement.py`

- [ ] **Step 1: Write failing end-to-end fake-provider test**

```python
from mlestar.workflow import run_mlestar


def test_workflow_runs_search_initial_candidates_ablation_and_ensemble(tmp_path, tiny_task_bundle, fake_llm, fake_search):
    report = run_mlestar(
        task_path=tiny_task_bundle.task_path, data_root=tiny_task_bundle.data_root,
        run_dir=tmp_path / "run", llm=fake_llm, search=fake_search,
        initial_candidates=2, outer_rounds=1, inner_rounds=1,
    )
    assert report["status"] == "success"
    assert report["best_experiment"]["metric_value"] >= 0.0
    assert (tmp_path / "run" / "final_report.json").exists()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python3 -m pytest tests/mlestar/test_workflow.py -q`

Expected: FAIL with `ImportError` for `run_mlestar`.

- [ ] **Step 3: Implement the workflow and CLI**

`run_mlestar()` must: load/validate task JSON; inventory data; make folds; retrieve four-or-fewer evidence-backed initial candidates; generate/audit/execute candidates; select the best real metric; run configured outer/inner refinement; blend the best distinct candidates; and write `final_report.json`. It must retain failed receipts and never select them. It must only compare results with identical data fingerprint, fold file, metric, and seed.

Implement:

```bash
python -m mlestar run \
  --task tasks/plant-pathology-2020/task.json \
  --data-root /content/kaggle_data/plant-pathology-2020-fgvc7 \
  --run-dir runs/plant-pathology-2020/seed-123 \
  --initial-candidates 4 --outer-rounds 4 --inner-rounds 4 --seed 123
```

Add `--plan-only`, `--no-search`, `--resume`, `--submit`, `--timeout-seconds`, and `--max-wall-clock-minutes`. `--submit` must delegate to the benchmark submission adapter from the second plan and must refuse when the artifact contract is incomplete. Change the legacy proxy loop docstring to say it is a synthetic smoke-only loop and is not used by `mlestar`.

- [ ] **Step 4: Run workflow tests**

Run: `python3 -m pytest tests/mlestar/test_workflow.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add mlestar tests/mlestar/test_workflow.py module4_agent/refinement.py
git commit -m "feat: orchestrate MLE-STAR DataOps workflow"
```

### Task 11: Document, smoke-test, and protect legacy behavior

**Files:**
- Create: `docs/MLESTAR_DATAOPS.md`
- Modify: `README.md`

- [ ] **Step 1: Write the local quick-start and artifact documentation**

Document the task JSON schema, the five marker contract, LLM/search environment variables, strict workspace behavior, how to inspect `dataops_report.json`, how OOF selection differs from a public leaderboard, how to resume a run, and the Colab handoff. State clearly that MLE-STAR in this repository is a reproduction of the method, not a claim of identical model/provider results to Google.

- [ ] **Step 2: Run focused tests**

Run: `python3 -m pytest tests/mlestar -q`

Expected: PASS.

- [ ] **Step 3: Run existing regression suites**

Run: `python3 -m pytest module4_agent/tests test_kaggle_orchestrator.py test_pipeline.py -q`

Expected: PASS; record environment-only package failures without weakening tests.

- [ ] **Step 4: Run a plan-only local smoke**

Run: `python3 -m mlestar run --task tests/fixtures/mlestar/tiny_task.json --data-root tests/fixtures/mlestar/tiny_data --run-dir /private/tmp/jiaozi-mlestar-smoke --plan-only --no-search`

Expected: `final_report.json` exists, has `status: "planned"`, and no subprocess training is invoked.

- [ ] **Step 5: Check source and commit**

Run: `python3 -m compileall mlestar && git diff --check && git status --short`

Expected: compilation succeeds, no whitespace errors, and no data/checkpoints/submissions are staged.

```bash
git add docs/MLESTAR_DATAOPS.md README.md
git commit -m "docs: add MLE-STAR DataOps quick start"
```

## Acceptance criteria

- A generated candidate is an executable skrub DataOps plan with all five named ML components.
- Search evidence, data inventory, data fingerprint, folds, receipts, OOF predictions, audit findings, and ensemble artifact paths are persisted for every run.
- Candidate selection, ablation selection, refinement selection, thresholding, and blending use real validation/OOF metrics only.
- Generated code cannot access test targets, fit transforms on test data, execute arbitrary shell commands, or write outside the run workspace.
- Initial retrieval, outer targeted-refinement loop, inner plan loop, debugger/audit gates, and OOF-safe ensemble map directly to the MLE-STAR paper's behavior.
- Existing Module 4 tests continue to work; its proxy loop is clearly separated from the new agent.

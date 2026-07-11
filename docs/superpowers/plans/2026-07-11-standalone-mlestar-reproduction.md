# Standalone MLE-STAR Reproduction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build an independently installable MLE-STAR reproduction that uses skrub DataOps and produces metric-correct, fixed-fold baseline-versus-MLE-STAR comparisons for the ten specified Kaggle tasks.

**Architecture:** The standalone root contains only the mlestar package, benchmark contracts, tests, documentation and notebooks. Dataset adapters own data discovery, fixed folds, model training, metrics and submission schema; MLE-STAR owns evidence-backed initialization, ablation-selected block refinement, debug/leakage/data-use gates and OOF-only ensembling. DataOps passes only immutable JSON metadata and artifact paths, never models or tensors.

**Tech Stack:** Python 3.11+, skrub, pandas/numpy/scikit-learn, PyTorch/timm/torchvision/Pillow, Kaggle CLI, optional OpenAI-compatible LLM, pytest.

---

## Target repository layout

~~~
LICENSE  README.md  pyproject.toml  requirements*.txt  .gitignore
mlestar/  benchmarks/  notebooks/  docs/  tests/  examples/
~~~

The final branch must not contain cv_autodl_agent/, retrieval/, module*_agent/, pipeline.py, Chroma data, raw competition data, checkpoints, OOF/test predictions, submissions or credentials. Preserve the prototype as codex/mlestar-dataops-prototype, then make codex/mlestar-kaggle-benchmarks an orphan standalone branch. Its current PR must be closed or marked non-mergeable before the force-push; an orphan branch cannot be merged into Jiaozi main without deleting Jiaozi files.

### Task 1: Isolate the repository and package

**Files:**
- Create: pyproject.toml
- Create: mlestar/__init__.py
- Create: tests/test_standalone_layout.py
- Create: README.md
- Create: .gitignore

- [ ] **Step 1: Write the failing isolation test**

~~~python
from pathlib import Path


def test_standalone_tree_has_no_jiaozi_modules() -> None:
    root = Path(__file__).parents[1]
    assert (root / "mlestar").is_dir()
    assert (root / "benchmarks").is_dir()
    assert not any((root / name).exists() for name in (
        "cv_autodl_agent", "retrieval", "module1_agent", "module4_agent", "pipeline.py",
    ))
~~~

- [ ] **Step 2: Verify the test fails on the current Jiaozi-derived tree**

Run: python -m pytest tests/test_standalone_layout.py -q

Expected: FAIL because the current branch inherits Jiaozi modules.

- [ ] **Step 3: Make the branch orphan and write package metadata**

~~~bash
git branch codex/mlestar-dataops-prototype codex/mlestar-kaggle-benchmarks
git checkout --orphan codex/mlestar-kaggle-benchmarks
git rm -rf .
~~~

~~~toml
[project]
name = "mlestar-dataops"
version = "0.2.0"
requires-python = ">=3.11"
dependencies = ["numpy", "pandas", "scikit-learn", "skrub>=0.9,<0.10"]

[project.optional-dependencies]
vision = ["torch", "torchvision", "timm", "pillow"]
llm = ["openai>=2.0.0"]
kaggle = ["kaggle>=2.0.0"]
dev = ["pytest"]

[project.scripts]
mlestar = "mlestar.cli:main"
~~~

- [ ] **Step 4: Add the package marker and verify the isolation test**

~~~python
"""Standalone reproduction of MLE-STAR with skrub DataOps."""

__version__ = "0.2.0"
~~~

Run: python -m pytest tests/test_standalone_layout.py -q

Expected: 1 passed.

- [ ] **Step 5: Commit the standalone skeleton**

~~~bash
git add .
git commit -m "chore: create standalone MLE-STAR repository"
~~~

### Task 2: Add reproducible task, metric, fold and artifact contracts

**Files:**
- Create: mlestar/contracts.py
- Create: mlestar/metrics.py
- Create: mlestar/artifacts.py
- Create: tests/test_contracts.py
- Create: tests/test_metrics.py

- [ ] **Step 1: Write failing contract tests**

~~~python
import pytest

from mlestar.metrics import score_metric


def test_metric_directions_are_explicit() -> None:
    assert score_metric("roc_auc", [0, 1], [0.1, 0.9]).greater_is_better is True
    assert score_metric("rmse", [0.0, 1.0], [0.0, 0.5]).greater_is_better is False


def test_artifact_cannot_escape_run_directory(tmp_path) -> None:
    from mlestar.artifacts import RunArtifacts
    with pytest.raises(ValueError, match="run directory"):
        RunArtifacts(tmp_path).resolve("../submission.csv")
~~~

- [ ] **Step 2: Run tests to confirm they fail**

Run: python -m pytest tests/test_contracts.py tests/test_metrics.py -q

Expected: import errors.

- [ ] **Step 3: Implement the contracts used by every execution phase**

~~~python
@dataclass(frozen=True)
class TaskSpec:
    key: str
    competition: str
    modality: str
    metric: MetricSpec
    fold: FoldSpec
    submission: SubmissionSpec
    target_columns: tuple[str, ...]

@dataclass(frozen=True)
class ExperimentReceipt:
    experiment_id: str
    parent_experiment_id: str | None
    phase: str
    candidate_id: str
    metric_value: float | None
    fold_scores: tuple[float, ...]
    seed: int
    oof_path: str | None
    test_path: str | None
    error: str | None
~~~

Implement ROC-AUC, multiclass log loss, QWK, RMSE, Dice and the competition detection metric with exact lower/higher direction. RunArtifacts.resolve() calls Path.resolve() and rejects a result outside the run directory.

- [ ] **Step 4: Run contract tests**

Run: python -m pytest tests/test_contracts.py tests/test_metrics.py -q

Expected: valid JSON round trips plus invalid path and metric cases pass.

- [ ] **Step 5: Commit the contracts**

~~~bash
git add mlestar tests
git commit -m "feat: add reproducible task and artifact contracts"
~~~

### Task 3: Execute adapters through skrub DataOps

**Files:**
- Create: mlestar/dataops.py
- Create: mlestar/executor.py
- Create: tests/test_dataops.py
- Create: tests/test_executor.py

- [ ] **Step 1: Write the failing DataOps test**

~~~python
def test_graph_evaluates_each_component_once_and_keeps_only_paths(tmp_path) -> None:
    from mlestar.dataops import build_run_graph
    graph = build_run_graph(tmp_path, ("load", "folds", "train", "evaluate"))
    result = graph.skb.eval({"run_context": {"run_dir": str(tmp_path)}})
    assert result["component_trace"] == ["load", "folds", "train", "evaluate"]
    assert all(isinstance(value, str) for value in result["artifacts"].values())
~~~

- [ ] **Step 2: Confirm it fails**

Run: python -m pytest tests/test_dataops.py -q

Expected: import error.

- [ ] **Step 3: Implement the immutable DataOps graph and child executor**

~~~python
def build_run_graph(run_dir: Path, phases: tuple[str, ...]):
    state = skrub.var("run_context")
    for phase in phases:
        state = skrub.deferred(run_phase)(state, phase, str(run_dir)).skb.set_name(phase)
    return state
~~~

run_phase() deep-copies metadata, appends exactly one phase and validates all artifact paths. execute_adapter() uses a clean child environment without KAGGLE_API_TOKEN, OPENAI_API_KEY or provider secrets.

- [ ] **Step 4: Run DataOps and executor tests**

Run: python -m pytest tests/test_dataops.py tests/test_executor.py -q

Expected: component trace is stable and child-process tests observe no secrets.

- [ ] **Step 5: Commit DataOps integration**

~~~bash
git add mlestar tests
git commit -m "feat: run MLE-STAR artifacts through DataOps"
~~~

### Task 4: Implement initial search, evaluation and merge

**Files:**
- Create: mlestar/search.py
- Create: mlestar/generation.py
- Create: mlestar/initialization.py
- Create: tests/test_initialization.py

- [ ] **Step 1: Write a failing candidate-selection test**

~~~python
def test_initialization_records_all_candidates_and_only_an_improving_merge(tmp_path, task_spec, fake_adapter) -> None:
    from mlestar.initialization import initialize_solution
    result = initialize_solution(task_spec, fake_adapter, ["linear", "tree", "bad"], tmp_path, seed=7)
    assert result.best.candidate_id == "tree"
    assert [item.candidate_id for item in result.receipts] == ["linear", "tree", "bad"]
    assert result.merge_receipts[-1].metric_value == result.best.metric_value
~~~

- [ ] **Step 2: Confirm it fails**

Run: python -m pytest tests/test_initialization.py -q

Expected: import error.

- [ ] **Step 3: Implement the first paper stage**

~~~python
def initialize_solution(task, adapter, evidence, run_dir, seed):
    candidates = generate_candidates(task, evidence)
    receipts = [adapter.evaluate(item, phase="initial", seed=seed) for item in candidates]
    best = choose_best([item for item in receipts if item.metric_value is not None], task.metric)
    merged = merge_candidates_incrementally(adapter, best, receipts, seed)
    return InitializationResult(best=choose_best([best, *merged], task.metric), receipts=receipts, merge_receipts=merged)
~~~

Persist every query, URL, excerpt, license note, generated source hash and receipt. The offline provider returns deterministic adapter templates rather than inventing a score. The production provider retrieves four model candidates by default and evaluates each before a metric-improving merge is accepted.

- [ ] **Step 4: Run initialization tests and commit**

Run: python -m pytest tests/test_initialization.py -q

Expected: failed candidate evaluations are kept, but never selected.

~~~bash
git add mlestar tests
git commit -m "feat: evaluate search-informed initial solutions"
~~~

### Task 5: Connect ablation, targeted refinement, debugging and audits

**Files:**
- Create: mlestar/refinement.py
- Create: mlestar/audits.py
- Create: mlestar/debug.py
- Create: tests/test_refinement.py
- Create: tests/test_audits.py

- [ ] **Step 1: Write the failing targeted-refinement tests**

~~~python
def test_refinement_changes_the_highest_impact_block_and_rolls_back(tmp_path, task_spec, fake_adapter) -> None:
    from mlestar.refinement import refine_solution
    result = refine_solution(task_spec, fake_adapter, baseline="base", outer_rounds=1, inner_rounds=2, seed=11)
    assert result.target_blocks == ("model",)
    assert result.accepted_receipt.parent_experiment_id == result.baseline_receipt.experiment_id
    assert result.rejected_receipts[0].metric_value <= result.baseline_receipt.metric_value


def test_leakage_gate_blocks_test_statistics() -> None:
    from mlestar.audits import audit_source
    assert "test_statistics" in {item.code for item in audit_source("scaler.fit(test_x)")}
~~~

- [ ] **Step 2: Confirm tests fail**

Run: python -m pytest tests/test_refinement.py tests/test_audits.py -q

Expected: import errors.

- [ ] **Step 3: Implement outer and inner loops**

~~~python
for _ in range(config.outer_rounds):
    target = max_impact_block(run_one_block_ablations(current, adapter, task, seed), task.metric)
    for plan in planner.propose(target, history, limit=config.inner_rounds):
        proposed = patch_exactly_one_block(current, target, plan)
        checked = audit_and_repair(proposed, task, inventory)
        receipt = evaluate_with_debug_retries(checked, adapter, seed, parent=current_receipt.experiment_id)
        current, current_receipt = accept_only_improvement(current, current_receipt, proposed, receipt, task.metric)
~~~

Audit blocks fitted preprocessing on test data, train/test concatenation, test-label usage, dynamic execution, network/subprocess imports, writes outside the run directory and unused mandatory data sources. Debug retries receive only traceback, task contract and the last executable candidate; record every retry and rollback in receipt lineage.

- [ ] **Step 4: Run refinement tests and commit**

Run: python -m pytest tests/test_refinement.py tests/test_audits.py -q

Expected: a metric-worse attempt is written to history but cannot replace the incumbent.

~~~bash
git add mlestar tests
git commit -m "feat: add ablation-guided targeted refinement"
~~~

### Task 6: Add OOF-only agent-planned ensembling

**Files:**
- Create: mlestar/ensemble.py
- Create: tests/test_ensemble.py

- [ ] **Step 1: Write an OOF alignment test**

~~~python
import pytest


def test_ensemble_rejects_missing_oof_rows() -> None:
    from mlestar.ensemble import select_ensemble
    with pytest.raises(ValueError, match="same row ids"):
        select_ensemble({"a": ([1, 2], [0.2, 0.8]), "b": ([1], [0.1])}, [0, 1], "roc_auc")
~~~

- [ ] **Step 2: Confirm it fails, then implement selection**

Run: python -m pytest tests/test_ensemble.py -q

Expected: import error.

~~~python
def select_ensemble(oof_by_candidate, y_true, metric, plans=(), max_rounds=5):
    aligned = align_oof_rows(oof_by_candidate)
    candidates = [simplex_grid_plan(aligned)] + [validate_plan(plan, aligned) for plan in plans[:max_rounds]]
    return choose_best([evaluate_plan(plan, aligned, y_true, metric) for plan in candidates], metric)
~~~

Persist candidate receipts, aligned row ids, weights/operation, OOF score and test-prediction path. A stacker, threshold or postprocess may fit only on OOF.

- [ ] **Step 3: Run ensemble tests and commit**

Run: python -m pytest tests/test_ensemble.py -q

Expected: only a metric-improving, OOF-aligned plan is selected.

~~~bash
git add mlestar tests
git commit -m "feat: select ensembles from OOF artifacts"
~~~

### Task 7: Build all ten metric-correct adapters

**Files:**
- Create: benchmarks/catalog.py
- Create: mlestar/adapters/base.py
- Create: mlestar/adapters/tabular.py
- Create: mlestar/adapters/image_classification.py
- Create: mlestar/adapters/detection.py
- Create: mlestar/adapters/segmentation.py
- Create: mlestar/adapters/denoising.py
- Create: tests/adapters/test_tabular.py
- Create: tests/adapters/test_image_classification.py
- Create: tests/adapters/test_dense_prediction.py

- [ ] **Step 1: Write synthetic adapter tests**

~~~python
def test_leaf_adapter_writes_fixed_fold_oof_and_submission(synthetic_leaf, tmp_path) -> None:
    from mlestar.adapters.tabular import LeafClassificationAdapter
    receipt = LeafClassificationAdapter().evaluate(synthetic_leaf, tmp_path, seed=3)
    assert len(receipt.fold_scores) == 5
    assert (tmp_path / receipt.oof_path).is_file()
    assert (tmp_path / receipt.test_path).is_file()


def test_ultrasound_adapter_uses_column_major_rle(synthetic_mask) -> None:
    from mlestar.adapters.segmentation import encode_rle
    assert encode_rle(synthetic_mask, order="column_major") == "2 2"
~~~

- [ ] **Step 2: Confirm adapter tests fail**

Run: python -m pytest tests/adapters -q

Expected: import errors.

- [ ] **Step 3: Implement one adapter interface and ten exact contracts**

~~~python
class TaskAdapter(Protocol):
    def prepare(self, data_root: Path, task: TaskSpec, run_dir: Path, seed: int) -> DatasetManifest: ...
    def evaluate(self, candidate: CandidateSpec, phase: str, seed: int, parent_experiment_id: str | None = None) -> ExperimentReceipt: ...
    def predict(self, candidate: CandidateSpec, receipt: ExperimentReceipt) -> Path: ...
~~~

Implement Leaf as skrub.TableVectorizer plus tree baseline; Plant Pathology, APTOS ordinal QWK, Dog Breed, Aerial Cactus, Dogs-vs-Cats and Histopathologic Cancer as timm image classification; Global Wheat as detection/AP with source-group folds; Ultrasound Nerve as Dice plus column-major RLE segmentation; Dirty Documents as image-to-image RMSE. Read class/column order from the sample submission where required. Every adapter writes fixed folds.parquet, OOF, test predictions and a schema-validated local submission.

- [ ] **Step 4: Run synthetic modality tests and commit**

Run: python -m pytest tests/adapters -q

Expected: all five modality families create valid artifacts without a network or GPU; tests monkeypatch one epoch and 8-12 synthetic records.

~~~bash
git add benchmarks mlestar/adapters tests/adapters
git commit -m "feat: add metric-correct benchmark adapters"
~~~

### Task 8: Compare baseline and full MLE-STAR under matched budgets

**Files:**
- Create: mlestar/experiment.py
- Create: mlestar/cli.py
- Create: docs/EVALUATION_PROTOCOL.md
- Create: docs/BENCHMARK_STATUS.md
- Create: tests/test_experiment.py

- [ ] **Step 1: Write the comparison test**

~~~python
def test_compare_uses_same_seeds_folds_and_budget_for_each_arm(tmp_path, fake_adapter) -> None:
    from mlestar.experiment import compare
    report = compare(fake_adapter, seeds=(13, 29, 47), budget_minutes=10, run_dir=tmp_path)
    assert report["arms"] == ["baseline", "mlestar_initial", "mlestar_refined", "mlestar_ensemble"]
    assert report["paired_folds"] is True
    assert set(report["summary"]) == {"mean", "sem", "wins", "failures"}
~~~

- [ ] **Step 2: Confirm it fails, then implement the comparison CLI**

Run: python -m pytest tests/test_experiment.py -q

Expected: import error.

~~~bash
mlestar compare --benchmark leaf_classification \
  --data-root /content/kaggle/leaf-classification \
  --run-root /content/drive/MyDrive/mlestar-runs \
  --seeds 13 29 47 --budget-minutes 60 --no-submit
~~~

The report includes all arms, exact fold ids, seeds, wall-clock budget, OOF mean/SEM, per-fold scores, valid-run rate, search/refinement/ensemble trial counts, run manifest and comparison.csv. Closed, rule-unaccepted or missing competitions record blocked with the Kaggle command/error, never a made-up public score. The ten historical competitions are expected closed, so public-LB results are separate from current offline OOF comparisons.

- [ ] **Step 3: Run comparison test and commit**

Run: python -m pytest tests/test_experiment.py -q

Expected: deterministic report and no submission call.

~~~bash
git add mlestar tests docs
git commit -m "feat: compare baseline and MLE-STAR experiments"
~~~

### Task 9: Provide the Colab entry point and final verification

**Files:**
- Create: notebooks/mlestar_kaggle_experiments.ipynb
- Create: docs/COLAB.md
- Create: tests/test_notebook.py
- Modify: README.md

- [ ] **Step 1: Write the notebook safety test**

~~~python
import json


def test_notebook_uses_one_token_secret_and_defaults_no_submit() -> None:
    notebook = json.loads(open("notebooks/mlestar_kaggle_experiments.ipynb", encoding="utf-8").read())
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "KAGGLE_API_TOKEN" in source
    assert "KAGGLE_USERNAME" not in source
    assert "KAGGLE_KEY" not in source
    assert "SUBMIT = False" in source
~~~

- [ ] **Step 2: Confirm it fails**

Run: python -m pytest tests/test_notebook.py -q

Expected: missing notebook.

- [ ] **Step 3: Create the Colab workflow**

The notebook installs .[vision,llm,kaggle], reads only KAGGLE_API_TOKEN from Colab Secrets, lists all ten task keys, requires manual rules acceptance before download, calls mlestar compare, renders comparison.csv, and requires the separate explicit assignment SUBMIT = True before it exposes a submission command. It must never put tokens, raw data or artifacts into Git.

- [ ] **Step 4: Run all offline checks**

~~~bash
python -m pytest -q
python -m json.tool notebooks/mlestar_kaggle_experiments.ipynb >/dev/null
python -m mlestar --help
python -m mlestar compare --benchmark leaf_classification \
  --data-root examples/synthetic_leaf --run-root /tmp/mlestar-smoke \
  --seeds 13 --budget-minutes 1 --no-submit
~~~

Expected: tests pass, the notebook parses, the CLI responds and the smoke run writes comparison.json without making a Kaggle submission.

- [ ] **Step 5: Commit and publish the standalone branch**

~~~bash
git add README.md docs notebooks tests
git commit -m "docs: add standalone Colab comparison workflow"
git push --force-with-lease origin codex/mlestar-kaggle-benchmarks
~~~

## Self-review

- **Spec coverage:** Tasks 1 and 9 make the repository independent and runnable. Tasks 2-6 implement search, initial evaluation/merge, ablation-targeted refinement, debugging, leakage/data-use checks and OOF ensembling. Task 7 covers every requested competition modality; Task 8 enforces paired comparison and records availability blockers.
- **Placeholder scan:** The plan specifies paths, unit tests, commands, artifact layouts and runtime outcomes; no result is assumed when a Kaggle task is closed or inaccessible.
- **Type consistency:** TaskSpec, ExperimentReceipt, TaskAdapter, fixed folds, OOF artifacts and MetricSpec are the shared interfaces in every task.


# Image Classification Adapters Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add six new trainable image-classification benchmark adapters (`plant_pathology_2020`, `aptos_2019`, `dog_breed`, `aerial_cactus`, `dogs_vs_cats`, `histopathologic_cancer`) sharing one timm-based training pipeline, wire them into `experiment.py`'s `compare()`, and extend the Colab notebook to download and run all seven now-implemented tasks.

**Architecture:** One `ImageClassificationAdapter` base class in `mlestar/adapters/vision.py` owns fold splitting, timm model construction/fine-tuning, modality-specific loss/activation/scoring, and `ExperimentReceipt` packaging (mirroring `LeafClassificationAdapter`'s `evaluate`/`merge`/`run` contract exactly, so `initialization.py`/`refinement.py`/`ensemble.py` need zero changes). Six subclasses each implement one method, `_load_dataset`, returning that competition's `(image_paths, labels, ids)` from its real (non-uniform) on-disk layout.

**Tech Stack:** Python 3.11+, torch/torchvision, timm, pillow (all already declared as the `vision` extra in `pyproject.toml`), pandas/numpy/scikit-learn (already core deps), pytest.

## Global Constraints

- Adapters must implement the existing `CandidateEvaluator` protocol (`evaluate`, `merge`) from `mlestar/initialization.py` unchanged. The protocol itself, `initialize_solution`, and `improves` are not modified. `choose_best`'s error *message* is improved in Task 8 to include per-candidate `receipt.error` text (learned from the leaf_classification debugging session on this branch, where that message's genericness cost real debugging time) -- this is additive and does not change any function's signature or the conditions under which it raises.
- Every adapter's `run()` must wrap execution in `try/except Exception`, returning a receipt with `metric_value=None` and a populated `error` string on failure (never raise out of `run()`), matching `LeafClassificationAdapter`.
- Tests must run with `pretrained=False` (no network access, no weight download) and complete in a few seconds each, keeping the full suite fast and offline (current baseline: 28 tests in ~6.5s).
- This plan is **OOF-only**: adapters compute out-of-fold predictions for metric scoring but do not load a Kaggle test split or write `submission.csv`. `ExperimentReceipt.test_path=None` on success, which the contract already permits (`contracts.py` only validates path fields when they are not `None`). Justification: `compare()`'s baseline/initial/refined/ensemble pipeline and `select_ensemble` only ever consume `.oof` and `.y_true` (verified by reading `experiment.py` and `ensemble.py`); submission requires `SUBMIT=True`, which is out of scope per the design doc.
- Object detection (`global_wheat`), segmentation (`ultrasound_nerve`), and denoising (`denoising_dirty_documents`) stay `NotImplementedError` -- explicitly out of scope.
- No new top-level directories (keeps `tests/test_standalone_layout.py` passing unchanged).

---

## Task 1: Synthetic fixtures for the six new benchmarks

**Files:**
- Create: `examples/synthetic_plant_pathology/train.csv`, `examples/synthetic_plant_pathology/images/*.jpg`
- Create: `examples/synthetic_aptos/train.csv`, `examples/synthetic_aptos/train_images/*.png`
- Create: `examples/synthetic_dog_breed/labels.csv`, `examples/synthetic_dog_breed/sample_submission.csv`, `examples/synthetic_dog_breed/train/*.jpg`
- Create: `examples/synthetic_aerial_cactus/train.csv`, `examples/synthetic_aerial_cactus/train/*.jpg`
- Create: `examples/synthetic_dogs_vs_cats/train/*.jpg`
- Create: `examples/synthetic_histopathologic_cancer/train_labels.csv`, `examples/synthetic_histopathologic_cancer/train/*.tif`

**Interfaces:**
- Produces: six fixture directories under `examples/`, one per new benchmark, each with 6 labeled rows (works with `FoldSpec(n_splits=5)` from `benchmarks/catalog.py`: each fold gets 1-2 validation rows). Later tasks' tests read these paths directly.

- [ ] **Step 1: Write the one-off fixture generator**

Create `/tmp/generate_vision_fixtures.py` (not committed -- deleted after running, matching how the notebook-patch scripts were used earlier on this branch):

```python
"""One-off script: generate tiny synthetic image fixtures for the six new
image-classification benchmarks, matching each competition's real directory
and label-file conventions."""
import csv
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("examples")


def _tiny_image(seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    array = (rng.random((8, 8, 3)) * 255).astype("uint8")
    return Image.fromarray(array, mode="RGB")


def plant_pathology() -> None:
    root = ROOT / "synthetic_plant_pathology"
    (root / "images").mkdir(parents=True, exist_ok=True)
    rows = [
        ("Train_0", 1, 0, 0, 0),
        ("Train_1", 0, 1, 0, 0),
        ("Train_2", 0, 0, 1, 0),
        ("Train_3", 0, 0, 0, 1),
        ("Train_4", 1, 0, 0, 0),
        ("Train_5", 0, 1, 1, 0),
    ]
    for i, row in enumerate(rows):
        _tiny_image(i).save(root / "images" / f"{row[0]}.jpg")
    with (root / "train.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_id", "healthy", "multiple_diseases", "rust", "scab"])
        writer.writerows(rows)


def aptos() -> None:
    root = ROOT / "synthetic_aptos"
    (root / "train_images").mkdir(parents=True, exist_ok=True)
    rows = [("id_0", 0), ("id_1", 1), ("id_2", 2), ("id_3", 3), ("id_4", 4), ("id_5", 2)]
    for i, row in enumerate(rows):
        _tiny_image(i).save(root / "train_images" / f"{row[0]}.png")
    with (root / "train.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id_code", "diagnosis"])
        writer.writerows(rows)


def dog_breed() -> None:
    root = ROOT / "synthetic_dog_breed"
    (root / "train").mkdir(parents=True, exist_ok=True)
    rows = [
        ("d0", "beagle"), ("d1", "pug"), ("d2", "beagle"),
        ("d3", "poodle"), ("d4", "pug"), ("d5", "poodle"),
    ]
    for i, row in enumerate(rows):
        _tiny_image(i).save(root / "train" / f"{row[0]}.jpg")
    with (root / "labels.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "breed"])
        writer.writerows(rows)
    with (root / "sample_submission.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "beagle", "poodle", "pug"])
        writer.writerow(["d0", 0, 0, 0])


def aerial_cactus() -> None:
    root = ROOT / "synthetic_aerial_cactus"
    (root / "train").mkdir(parents=True, exist_ok=True)
    rows = [
        ("c0.jpg", 1), ("c1.jpg", 0), ("c2.jpg", 1),
        ("c3.jpg", 0), ("c4.jpg", 1), ("c5.jpg", 0),
    ]
    for i, row in enumerate(rows):
        _tiny_image(i).save(root / "train" / row[0])
    with (root / "train.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "has_cactus"])
        writer.writerows(rows)


def dogs_vs_cats() -> None:
    root = ROOT / "synthetic_dogs_vs_cats"
    (root / "train").mkdir(parents=True, exist_ok=True)
    names = ["cat.0.jpg", "cat.1.jpg", "dog.0.jpg", "dog.1.jpg", "cat.2.jpg", "dog.2.jpg"]
    for i, name in enumerate(names):
        _tiny_image(i).save(root / "train" / name)


def histopathologic_cancer() -> None:
    root = ROOT / "synthetic_histopathologic_cancer"
    (root / "train").mkdir(parents=True, exist_ok=True)
    rows = [("h0", 1), ("h1", 0), ("h2", 1), ("h3", 0), ("h4", 1), ("h5", 0)]
    for i, row in enumerate(rows):
        _tiny_image(i).save(root / "train" / f"{row[0]}.tif")
    with (root / "train_labels.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "label"])
        writer.writerows(rows)


if __name__ == "__main__":
    plant_pathology()
    aptos()
    dog_breed()
    aerial_cactus()
    dogs_vs_cats()
    histopathologic_cancer()
    print("fixtures generated")
```

- [ ] **Step 2: Run it from the repo root**

Run: `python3 /tmp/generate_vision_fixtures.py` (run from `/Users/wang/Documents/Jiaozi`)
Expected: prints `fixtures generated`; `examples/synthetic_*` directories now exist with images + CSVs.

- [ ] **Step 3: Verify each fixture loads with PIL and has the expected row count**

Run:
```bash
python3 -c "
import pandas as pd
from pathlib import Path
for name, csv_name in [
    ('synthetic_plant_pathology', 'train.csv'),
    ('synthetic_aptos', 'train.csv'),
    ('synthetic_dog_breed', 'labels.csv'),
    ('synthetic_aerial_cactus', 'train.csv'),
    ('synthetic_histopathologic_cancer', 'train_labels.csv'),
]:
    df = pd.read_csv(Path('examples') / name / csv_name)
    assert len(df) == 6, (name, len(df))
    print(name, 'OK', len(df), 'rows')
import os
cat_dog_files = os.listdir('examples/synthetic_dogs_vs_cats/train')
assert len(cat_dog_files) == 6, cat_dog_files
print('synthetic_dogs_vs_cats OK', len(cat_dog_files), 'files')
"
```
Expected: six `OK` lines, no assertion errors.

- [ ] **Step 4: Delete the one-off script and commit the fixtures**

```bash
cd /Users/wang/Documents/Jiaozi
rm /tmp/generate_vision_fixtures.py
git add examples/synthetic_plant_pathology examples/synthetic_aptos examples/synthetic_dog_breed examples/synthetic_aerial_cactus examples/synthetic_dogs_vs_cats examples/synthetic_histopathologic_cancer
git commit -m "test: add synthetic fixtures for six image-classification benchmarks"
```

---

## Task 2: `ImageClassificationAdapter` base class + `PlantPathologyAdapter`

This task builds and proves the entire shared pipeline (fold splitting, timm model, training loop, modality dispatch, scoring, receipt packaging) against the first concrete subclass. Every later subclass task only adds a `_load_dataset` override plus one test, because this task's machinery already works.

**Files:**
- Create: `mlestar/adapters/vision.py`
- Test: `tests/test_vision_adapter.py`

**Interfaces:**
- Consumes: `mlestar.artifacts.RunArtifacts`, `mlestar.contracts.ExperimentReceipt`/`TaskSpec`, `mlestar.initialization.CandidateSpec`, `mlestar.metrics.score_metric` (all unchanged, existing).
- Produces:
  - `VisionRun` dataclass: fields `receipt: ExperimentReceipt`, `y_true: np.ndarray`, `oof: np.ndarray`.
  - `ImageClassificationAdapter` base class: constructor `(data_root, run_dir, task, *, pretrained=True, image_size=128, epochs=3, batch_size=32, max_train_samples=2000)`; methods `evaluate(candidate, *, phase, seed, parent_experiment_id=None) -> ExperimentReceipt`, `merge(incumbent, addition) -> CandidateSpec`, `run(candidate, *, phase, seed, parent_experiment_id=None) -> VisionRun`.
  - `PlantPathologyAdapter(ImageClassificationAdapter)`: same constructor (inherited), overrides `_load_dataset`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_vision_adapter.py`:

```python
from pathlib import Path

import numpy as np

from benchmarks.catalog import get_task
from mlestar.adapters.vision import PlantPathologyAdapter
from mlestar.initialization import CandidateSpec

FIXTURE = Path("examples/synthetic_plant_pathology")


def test_plant_pathology_load_dataset_reads_multilabel_csv():
    task = get_task("plant_pathology_2020")
    adapter = PlantPathologyAdapter(FIXTURE, "/tmp/mlestar-vision-test-load", task, pretrained=False)
    paths, labels, ids = adapter._load_dataset(adapter.data_root)
    assert len(paths) == 6
    assert labels.shape == (6, 4)
    assert ids == ["Train_0", "Train_1", "Train_2", "Train_3", "Train_4", "Train_5"]
    assert paths[0].name == "Train_0.jpg"
    assert paths[0].exists()


def test_plant_pathology_run_produces_a_metric(tmp_path):
    task = get_task("plant_pathology_2020")
    adapter = PlantPathologyAdapter(
        FIXTURE, tmp_path, task, pretrained=False, epochs=1, max_train_samples=None
    )
    candidate = CandidateSpec("resnet18", (("model", "resnet18"),))
    result = adapter.run(candidate, phase="test", seed=13)
    assert result.receipt.error is None, result.receipt.error
    assert result.receipt.metric_value is not None
    assert 0.0 <= result.receipt.metric_value <= 1.0
    assert result.receipt.oof_path is not None
    assert result.receipt.test_path is None
    assert result.oof.shape == (6, 4)
    oof_csv = tmp_path / result.receipt.oof_path
    assert oof_csv.exists()


def test_evaluate_wraps_a_failure_instead_of_raising(tmp_path):
    task = get_task("plant_pathology_2020")
    adapter = PlantPathologyAdapter(tmp_path / "does-not-exist", tmp_path, task, pretrained=False)
    candidate = CandidateSpec("resnet18", (("model", "resnet18"),))
    receipt = adapter.evaluate(candidate, phase="test", seed=13)
    assert receipt.metric_value is None
    assert receipt.error is not None
    assert "FileNotFoundError" in receipt.error or "train.csv" in receipt.error


def test_merge_names_an_ensemble_candidate():
    task = get_task("plant_pathology_2020")
    adapter = PlantPathologyAdapter(FIXTURE, "/tmp/mlestar-vision-test-merge", task, pretrained=False)
    incumbent = CandidateSpec("resnet18", (("model", "resnet18"),))
    addition = CandidateSpec("efficientnet_b0", (("model", "efficientnet_b0"),))
    merged = adapter.merge(incumbent, addition)
    assert merged.candidate_id == "resnet18+efficientnet_b0"
    assert merged.block("model") == "ensemble:resnet18+efficientnet_b0"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_vision_adapter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'mlestar.adapters.vision'`.

- [ ] **Step 3: Install the vision extra into the project virtualenv**

Run: `.venv/bin/python -m ensurepip --upgrade && .venv/bin/python -m pip install -e '.[vision,dev]'`
Expected: installs torch, torchvision, timm, pillow into `.venv` alongside the existing deps. (If `ensurepip` is unavailable in this venv, fall back to `python3 -m pip install --user torch timm pillow` globally and run tests with the system `python3` instead of `.venv/bin/python` for this task onward -- confirm which works in this environment before continuing.)

- [ ] **Step 4: Write `mlestar/adapters/vision.py`**

```python
"""Shared timm fine-tuning pipeline for image-classification benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import KFold
from timm import create_model

from ..artifacts import RunArtifacts
from ..contracts import ExperimentReceipt, TaskSpec
from ..initialization import CandidateSpec
from ..metrics import score_metric


@dataclass(frozen=True)
class VisionRun:
    receipt: ExperimentReceipt
    y_true: np.ndarray
    oof: np.ndarray


class _ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths: list[Path], targets: torch.Tensor, image_size: int) -> None:
        self.paths = paths
        self.targets = targets
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.paths[index]).convert("RGB")
        image = image.resize((self.image_size, self.image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor, self.targets[index]


class ImageClassificationAdapter:
    """Fine-tune a timm model per fold and score out-of-fold predictions.

    Subclasses implement `_load_dataset` to return this task's
    `(image_paths, labels, ids)` from `data_root`; everything else -- fold
    splitting, model construction, training, scoring, and artifact writing
    -- is identical across tasks. This is OOF-only: it does not load a
    Kaggle test split or write a submission file (see the plan's Global
    Constraints for why that is safe for this pipeline).
    """

    def __init__(
        self,
        data_root: str | Path,
        run_dir: str | Path,
        task: TaskSpec,
        *,
        pretrained: bool = True,
        image_size: int = 128,
        epochs: int = 3,
        batch_size: int = 32,
        max_train_samples: int | None = 2000,
    ) -> None:
        self.data_root = Path(data_root).resolve()
        self.artifacts = RunArtifacts(run_dir)
        self.task = task
        self.pretrained = pretrained
        self.image_size = image_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_train_samples = max_train_samples

    def _load_dataset(self, data_root: Path) -> tuple[list[Path], np.ndarray, list[str]]:
        raise NotImplementedError

    def evaluate(
        self,
        candidate: CandidateSpec,
        *,
        phase: str,
        seed: int,
        parent_experiment_id: str | None = None,
    ) -> ExperimentReceipt:
        return self.run(candidate, phase=phase, seed=seed, parent_experiment_id=parent_experiment_id).receipt

    def merge(self, incumbent: CandidateSpec, addition: CandidateSpec) -> CandidateSpec:
        models = f"ensemble:{incumbent.block('model')}+{addition.block('model')}"
        blocks = dict(incumbent.blocks)
        blocks["model"] = models
        return CandidateSpec(
            candidate_id=f"{incumbent.candidate_id}+{addition.candidate_id}",
            blocks=tuple(blocks.items()),
            evidence_urls=tuple(sorted(set(incumbent.evidence_urls + addition.evidence_urls))),
        )

    def run(
        self,
        candidate: CandidateSpec,
        *,
        phase: str,
        seed: int,
        parent_experiment_id: str | None = None,
    ) -> VisionRun:
        started = perf_counter()
        try:
            image_paths, labels, ids = self._load_dataset(self.data_root)
            oof, folds = self._cross_validate(image_paths, labels, candidate.block("model"), seed)
            scoring_labels, scoring_oof = self._score_inputs(labels, oof)
            metric = score_metric(self.task.metric, scoring_labels, scoring_oof)
            receipt_id = f"{phase}-{candidate.candidate_id}-{uuid4().hex[:12]}"
            stem = f"{phase}_{candidate.candidate_id}_{seed}".replace("/", "_")
            oof_frame = pd.DataFrame(self._prediction_columns(oof))
            oof_frame.insert(0, self.task.submission.id_columns[0], ids)
            oof_path = self.artifacts.write_csv(f"{stem}/oof.csv", oof_frame)
            self.artifacts.write_csv(f"{stem}/folds.csv", folds)
            return VisionRun(
                receipt=ExperimentReceipt(
                    experiment_id=receipt_id,
                    parent_experiment_id=parent_experiment_id,
                    phase=phase,
                    candidate_id=candidate.candidate_id,
                    metric_value=metric.value,
                    fold_scores=tuple(self._fold_scores(folds, labels, oof)),
                    seed=seed,
                    oof_path=self.artifacts.relative(oof_path),
                    test_path=None,
                    error=None,
                ),
                y_true=labels,
                oof=oof,
            )
        except Exception as error:
            return VisionRun(
                receipt=ExperimentReceipt(
                    experiment_id=f"{phase}-{candidate.candidate_id}-{uuid4().hex[:12]}",
                    parent_experiment_id=parent_experiment_id,
                    phase=phase,
                    candidate_id=candidate.candidate_id,
                    metric_value=None,
                    fold_scores=(),
                    seed=seed,
                    oof_path=None,
                    test_path=None,
                    error=f"{type(error).__name__}: {error}",
                ),
                y_true=np.array([]),
                oof=np.empty((0,)),
            )
        finally:
            self.artifacts.write_json("runtime.json", {"elapsed_seconds": perf_counter() - started})

    # -- modality dispatch --------------------------------------------------

    def _num_classes(self, labels: np.ndarray) -> int:
        modality = self.task.modality
        if modality == "image_multilabel":
            return len(self.task.target_columns)
        if modality in {"image_binary", "image_ordinal"}:
            return 1
        if modality == "image_multiclass":
            return int(labels.max()) + 1
        raise ValueError(f"Unsupported modality: {modality!r}")

    def _prepare_targets(self, labels: np.ndarray) -> torch.Tensor:
        modality = self.task.modality
        if modality == "image_binary":
            return torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        if modality in {"image_multilabel", "image_ordinal"}:
            tensor = torch.tensor(labels, dtype=torch.float32)
            return tensor if modality == "image_multilabel" else tensor.unsqueeze(1)
        if modality == "image_multiclass":
            return torch.tensor(labels, dtype=torch.long)
        raise ValueError(f"Unsupported modality: {modality!r}")

    def _loss_fn(self) -> torch.nn.Module:
        modality = self.task.modality
        if modality in {"image_binary", "image_multilabel"}:
            return torch.nn.BCEWithLogitsLoss()
        if modality == "image_ordinal":
            return torch.nn.MSELoss()
        if modality == "image_multiclass":
            return torch.nn.CrossEntropyLoss()
        raise ValueError(f"Unsupported modality: {modality!r}")

    def _predict_probs(self, logits: torch.Tensor) -> np.ndarray:
        modality = self.task.modality
        if modality == "image_binary":
            return torch.sigmoid(logits).squeeze(1).detach().numpy()
        if modality == "image_multilabel":
            return torch.sigmoid(logits).detach().numpy()
        if modality == "image_ordinal":
            return logits.squeeze(1).detach().numpy()
        if modality == "image_multiclass":
            return torch.softmax(logits, dim=1).detach().numpy()
        raise ValueError(f"Unsupported modality: {modality!r}")

    def _score_inputs(self, labels: np.ndarray, oof: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.task.modality == "image_ordinal":
            num_classes = self._num_classes(labels)
            rounded = np.clip(np.round(oof), 0, num_classes - 1)
            return labels, rounded
        return labels, oof

    def _prediction_columns(self, oof: np.ndarray) -> dict[str, Any]:
        if oof.ndim == 1:
            return {self.task.submission.prediction_columns[0] if self.task.submission.prediction_columns else "prediction": oof}
        columns = self.task.target_columns if len(self.task.target_columns) == oof.shape[1] else tuple(
            f"class_{i}" for i in range(oof.shape[1])
        )
        return {name: oof[:, i] for i, name in enumerate(columns)}

    # -- cross validation -----------------------------------------------------

    def _build_model(self, model_name: str, num_classes: int, seed: int) -> torch.nn.Module:
        torch.manual_seed(seed)
        return create_model(model_name, pretrained=self.pretrained, num_classes=num_classes)

    def _subsample(self, train_idx: np.ndarray, seed: int) -> np.ndarray:
        if self.max_train_samples is None or len(train_idx) <= self.max_train_samples:
            return train_idx
        rng = np.random.default_rng(seed)
        return rng.choice(train_idx, size=self.max_train_samples, replace=False)

    def _fit(self, model: torch.nn.Module, dataset: _ImageDataset, seed: int) -> None:
        generator = torch.Generator().manual_seed(seed)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, generator=generator
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        loss_fn = self._loss_fn()
        model.train()
        for _ in range(self.epochs):
            for images, targets in loader:
                optimizer.zero_grad()
                logits = model(images)
                loss = loss_fn(logits, targets)
                loss.backward()
                optimizer.step()

    def _predict(self, model: torch.nn.Module, dataset: _ImageDataset) -> np.ndarray:
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        model.eval()
        outputs = []
        with torch.no_grad():
            for images, _ in loader:
                outputs.append(self._predict_probs(model(images)))
        return np.concatenate(outputs, axis=0)

    def _cross_validate(
        self, image_paths: list[Path], labels: np.ndarray, model_name: str, seed: int
    ) -> tuple[np.ndarray, pd.DataFrame]:
        num_classes = self._num_classes(labels)
        targets = self._prepare_targets(labels)
        n = len(image_paths)
        splitter = KFold(n_splits=self.task.fold.n_splits, shuffle=True, random_state=seed)
        oof_shape = (n, num_classes) if num_classes > 1 else (n,)
        oof = np.zeros(oof_shape, dtype=float)
        fold_ids = np.empty(n, dtype=int)
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(np.zeros(n))):
            train_idx = self._subsample(train_idx, seed + fold)
            model = self._build_model(model_name, num_classes, seed + fold)
            train_ds = _ImageDataset([image_paths[i] for i in train_idx], targets[train_idx], self.image_size)
            valid_ds = _ImageDataset([image_paths[i] for i in valid_idx], targets[valid_idx], self.image_size)
            self._fit(model, train_ds, seed + fold)
            oof[valid_idx] = self._predict(model, valid_ds)
            fold_ids[valid_idx] = fold
        folds = pd.DataFrame({"row_index": np.arange(n), "fold": fold_ids})
        return oof, folds

    def _fold_scores(self, folds: pd.DataFrame, labels: np.ndarray, oof: np.ndarray) -> list[float]:
        scoring_labels, scoring_oof = self._score_inputs(labels, oof)
        fold_column = folds["fold"].to_numpy()
        scores = []
        for fold in range(self.task.fold.n_splits):
            mask = fold_column == fold
            scores.append(score_metric(self.task.metric, scoring_labels[mask], scoring_oof[mask]).value)
        return scores


class PlantPathologyAdapter(ImageClassificationAdapter):
    """Four-label apple-leaf disease classification (Plant Pathology 2020)."""

    def _load_dataset(self, data_root: Path) -> tuple[list[Path], np.ndarray, list[str]]:
        csv_path = data_root / "train.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Plant Pathology needs train.csv in {data_root}.")
        frame = pd.read_csv(csv_path)
        label_columns = list(self.task.target_columns)
        image_paths = [data_root / "images" / f"{image_id}.jpg" for image_id in frame["image_id"]]
        labels = frame[label_columns].to_numpy(dtype=float)
        return image_paths, labels, frame["image_id"].astype(str).tolist()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_vision_adapter.py -v`
Expected: 4 passed.

- [ ] **Step 6: Run the full suite to confirm nothing else broke**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all previous tests plus these 4 pass (32 total).

- [ ] **Step 7: Commit**

```bash
git add mlestar/adapters/vision.py tests/test_vision_adapter.py pyproject.toml
git commit -m "feat: add ImageClassificationAdapter base class + PlantPathologyAdapter"
```

---

## Task 3: `AptosAdapter` (image_ordinal)

**Files:**
- Modify: `mlestar/adapters/vision.py` (append class)
- Modify: `tests/test_vision_adapter.py` (append tests)

**Interfaces:**
- Consumes: `ImageClassificationAdapter` from Task 2 (unchanged).
- Produces: `AptosAdapter(ImageClassificationAdapter)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vision_adapter.py`:

```python
from mlestar.adapters.vision import AptosAdapter

APTOS_FIXTURE = Path("examples/synthetic_aptos")


def test_aptos_load_dataset_reads_ordinal_csv():
    task = get_task("aptos_2019")
    adapter = AptosAdapter(APTOS_FIXTURE, "/tmp/mlestar-vision-test-aptos-load", task, pretrained=False)
    paths, labels, ids = adapter._load_dataset(adapter.data_root)
    assert len(paths) == 6
    assert labels.tolist() == [0, 1, 2, 3, 4, 2]
    assert ids == ["id_0", "id_1", "id_2", "id_3", "id_4", "id_5"]
    assert paths[0].name == "id_0.png"


def test_aptos_run_produces_a_qwk_metric(tmp_path):
    task = get_task("aptos_2019")
    adapter = AptosAdapter(
        APTOS_FIXTURE, tmp_path, task, pretrained=False, epochs=1, max_train_samples=None
    )
    candidate = CandidateSpec("resnet18", (("model", "resnet18"),))
    result = adapter.run(candidate, phase="test", seed=13)
    assert result.receipt.error is None, result.receipt.error
    assert result.receipt.metric_value is not None
    assert -1.0 <= result.receipt.metric_value <= 1.0
    assert result.oof.shape == (6,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_vision_adapter.py -v -k aptos`
Expected: FAIL with `ImportError: cannot import name 'AptosAdapter'`.

- [ ] **Step 3: Append `AptosAdapter` to `mlestar/adapters/vision.py`**

```python
class AptosAdapter(ImageClassificationAdapter):
    """Ordinal diabetic-retinopathy grading (APTOS 2019 Blindness Detection)."""

    def _load_dataset(self, data_root: Path) -> tuple[list[Path], np.ndarray, list[str]]:
        csv_path = data_root / "train.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"APTOS needs train.csv in {data_root}.")
        frame = pd.read_csv(csv_path)
        image_paths = [data_root / "train_images" / f"{id_code}.png" for id_code in frame["id_code"]]
        labels = frame["diagnosis"].to_numpy(dtype=float)
        return image_paths, labels, frame["id_code"].astype(str).tolist()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_vision_adapter.py -v -k aptos`
Expected: 2 passed.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass (34 total).

- [ ] **Step 6: Commit**

```bash
git add mlestar/adapters/vision.py tests/test_vision_adapter.py
git commit -m "feat: add AptosAdapter (image_ordinal)"
```

---

## Task 4: `DogBreedAdapter` (image_multiclass, class names from `sample_submission.csv`)

**Files:**
- Modify: `mlestar/adapters/vision.py` (append class)
- Modify: `tests/test_vision_adapter.py` (append tests)

**Interfaces:**
- Produces: `DogBreedAdapter(ImageClassificationAdapter)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vision_adapter.py`:

```python
from mlestar.adapters.vision import DogBreedAdapter

DOG_BREED_FIXTURE = Path("examples/synthetic_dog_breed")


def test_dog_breed_load_dataset_encodes_breed_names_from_labels_csv():
    task = get_task("dog_breed")
    adapter = DogBreedAdapter(DOG_BREED_FIXTURE, "/tmp/mlestar-vision-test-breed-load", task, pretrained=False)
    paths, labels, ids = adapter._load_dataset(adapter.data_root)
    assert len(paths) == 6
    assert ids == ["d0", "d1", "d2", "d3", "d4", "d5"]
    assert paths[0].name == "d0.jpg"
    # class names come from sample_submission.csv's header, alphabetically:
    # beagle, poodle, pug -- so d0/d2 (beagle) -> 0, d3/d5 (poodle) -> 1, d1/d4 (pug) -> 2
    assert labels.tolist() == [0, 2, 0, 1, 2, 1]


def test_dog_breed_run_produces_a_log_loss_metric(tmp_path):
    task = get_task("dog_breed")
    adapter = DogBreedAdapter(
        DOG_BREED_FIXTURE, tmp_path, task, pretrained=False, epochs=1, max_train_samples=None
    )
    candidate = CandidateSpec("resnet18", (("model", "resnet18"),))
    result = adapter.run(candidate, phase="test", seed=13)
    assert result.receipt.error is None, result.receipt.error
    assert result.receipt.metric_value is not None
    assert result.receipt.metric_value >= 0.0
    assert result.oof.shape == (6, 3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_vision_adapter.py -v -k dog_breed`
Expected: FAIL with `ImportError: cannot import name 'DogBreedAdapter'`.

- [ ] **Step 3: Append `DogBreedAdapter` to `mlestar/adapters/vision.py`**

```python
class DogBreedAdapter(ImageClassificationAdapter):
    """Fine-grained dog-breed classification, class names from sample_submission.csv."""

    def _load_dataset(self, data_root: Path) -> tuple[list[Path], np.ndarray, list[str]]:
        labels_path = data_root / "labels.csv"
        sample_path = data_root / "sample_submission.csv"
        if not labels_path.is_file():
            raise FileNotFoundError(f"Dog Breed needs labels.csv (not train.csv) in {data_root}.")
        if not sample_path.is_file():
            raise FileNotFoundError(f"Dog Breed needs sample_submission.csv in {data_root}.")
        frame = pd.read_csv(labels_path)
        class_names = list(pd.read_csv(sample_path).columns[1:])
        breed_to_index = {name: index for index, name in enumerate(class_names)}
        unknown = sorted(set(frame["breed"]) - set(breed_to_index))
        if unknown:
            raise ValueError(f"labels.csv has breeds not in sample_submission.csv: {unknown}")
        image_paths = [data_root / "train" / f"{id_}.jpg" for id_ in frame["id"]]
        labels = frame["breed"].map(breed_to_index).to_numpy(dtype=int)
        return image_paths, labels, frame["id"].astype(str).tolist()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_vision_adapter.py -v -k dog_breed`
Expected: 2 passed.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass (36 total).

- [ ] **Step 6: Commit**

```bash
git add mlestar/adapters/vision.py tests/test_vision_adapter.py
git commit -m "feat: add DogBreedAdapter (image_multiclass)"
```

---

## Task 5: `AerialCactusAdapter` (image_binary, CSV-based)

**Files:**
- Modify: `mlestar/adapters/vision.py` (append class)
- Modify: `tests/test_vision_adapter.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vision_adapter.py`:

```python
from mlestar.adapters.vision import AerialCactusAdapter

AERIAL_CACTUS_FIXTURE = Path("examples/synthetic_aerial_cactus")


def test_aerial_cactus_load_dataset_reads_binary_csv():
    task = get_task("aerial_cactus")
    adapter = AerialCactusAdapter(
        AERIAL_CACTUS_FIXTURE, "/tmp/mlestar-vision-test-cactus-load", task, pretrained=False
    )
    paths, labels, ids = adapter._load_dataset(adapter.data_root)
    assert len(paths) == 6
    assert labels.tolist() == [1, 0, 1, 0, 1, 0]
    assert ids == ["c0.jpg", "c1.jpg", "c2.jpg", "c3.jpg", "c4.jpg", "c5.jpg"]
    assert paths[0].name == "c0.jpg"


def test_aerial_cactus_run_produces_a_roc_auc_metric(tmp_path):
    task = get_task("aerial_cactus")
    adapter = AerialCactusAdapter(
        AERIAL_CACTUS_FIXTURE, tmp_path, task, pretrained=False, epochs=1, max_train_samples=None
    )
    candidate = CandidateSpec("resnet18", (("model", "resnet18"),))
    result = adapter.run(candidate, phase="test", seed=13)
    assert result.receipt.error is None, result.receipt.error
    assert result.receipt.metric_value is not None
    assert 0.0 <= result.receipt.metric_value <= 1.0
    assert result.oof.shape == (6,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_vision_adapter.py -v -k cactus`
Expected: FAIL with `ImportError: cannot import name 'AerialCactusAdapter'`.

- [ ] **Step 3: Append `AerialCactusAdapter` to `mlestar/adapters/vision.py`**

```python
class AerialCactusAdapter(ImageClassificationAdapter):
    """Binary cactus detection from aerial image tiles."""

    def _load_dataset(self, data_root: Path) -> tuple[list[Path], np.ndarray, list[str]]:
        csv_path = data_root / "train.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Aerial Cactus needs train.csv in {data_root}.")
        frame = pd.read_csv(csv_path)
        image_paths = [data_root / "train" / id_ for id_ in frame["id"]]
        labels = frame["has_cactus"].to_numpy(dtype=float)
        return image_paths, labels, frame["id"].astype(str).tolist()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_vision_adapter.py -v -k cactus`
Expected: 2 passed.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass (38 total).

- [ ] **Step 6: Commit**

```bash
git add mlestar/adapters/vision.py tests/test_vision_adapter.py
git commit -m "feat: add AerialCactusAdapter (image_binary)"
```

---

## Task 6: `DogsVsCatsAdapter` (image_binary, label encoded in filename, no CSV)

**Files:**
- Modify: `mlestar/adapters/vision.py` (append class)
- Modify: `tests/test_vision_adapter.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vision_adapter.py`:

```python
from mlestar.adapters.vision import DogsVsCatsAdapter

DOGS_VS_CATS_FIXTURE = Path("examples/synthetic_dogs_vs_cats")


def test_dogs_vs_cats_load_dataset_parses_label_from_filename():
    task = get_task("dogs_vs_cats")
    adapter = DogsVsCatsAdapter(
        DOGS_VS_CATS_FIXTURE, "/tmp/mlestar-vision-test-dvc-load", task, pretrained=False
    )
    paths, labels, ids = adapter._load_dataset(adapter.data_root)
    assert len(paths) == 6
    # dog=1, cat=0, sorted by filename: cat.0,cat.1,cat.2,dog.0,dog.1,dog.2
    assert labels.tolist() == [0, 0, 0, 1, 1, 1]
    assert ids == ["cat.0", "cat.1", "cat.2", "dog.0", "dog.1", "dog.2"]


def test_dogs_vs_cats_run_produces_a_log_loss_metric(tmp_path):
    task = get_task("dogs_vs_cats")
    adapter = DogsVsCatsAdapter(
        DOGS_VS_CATS_FIXTURE, tmp_path, task, pretrained=False, epochs=1, max_train_samples=None
    )
    candidate = CandidateSpec("resnet18", (("model", "resnet18"),))
    result = adapter.run(candidate, phase="test", seed=13)
    assert result.receipt.error is None, result.receipt.error
    assert result.receipt.metric_value is not None
    assert result.receipt.metric_value >= 0.0
    assert result.oof.shape == (6,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_vision_adapter.py -v -k dogs_vs_cats`
Expected: FAIL with `ImportError: cannot import name 'DogsVsCatsAdapter'`.

- [ ] **Step 3: Append `DogsVsCatsAdapter` to `mlestar/adapters/vision.py`**

```python
class DogsVsCatsAdapter(ImageClassificationAdapter):
    """Dog-versus-cat probability prediction, label encoded in the filename."""

    def _load_dataset(self, data_root: Path) -> tuple[list[Path], np.ndarray, list[str]]:
        train_dir = data_root / "train"
        if not train_dir.is_dir():
            raise FileNotFoundError(f"Dogs vs Cats needs a train/ directory in {data_root}.")
        files = sorted(train_dir.iterdir())
        if not files:
            raise FileNotFoundError(f"Dogs vs Cats train/ directory in {data_root} is empty.")
        image_paths: list[Path] = []
        labels: list[float] = []
        ids: list[str] = []
        for path in files:
            prefix = path.name.split(".")[0]
            if prefix not in {"cat", "dog"}:
                raise ValueError(f"Unexpected Dogs vs Cats filename (want cat.N.jpg/dog.N.jpg): {path.name}")
            image_paths.append(path)
            labels.append(1.0 if prefix == "dog" else 0.0)
            ids.append(path.stem)
        return image_paths, np.array(labels, dtype=float), ids
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_vision_adapter.py -v -k dogs_vs_cats`
Expected: 2 passed.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass (40 total).

- [ ] **Step 6: Commit**

```bash
git add mlestar/adapters/vision.py tests/test_vision_adapter.py
git commit -m "feat: add DogsVsCatsAdapter (image_binary, filename-encoded labels)"
```

---

## Task 7: `HistopathologicCancerAdapter` (image_binary, `.tif` + separate labels CSV)

**Files:**
- Modify: `mlestar/adapters/vision.py` (append class)
- Modify: `tests/test_vision_adapter.py` (append tests)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vision_adapter.py`:

```python
from mlestar.adapters.vision import HistopathologicCancerAdapter

HISTOPATHOLOGIC_FIXTURE = Path("examples/synthetic_histopathologic_cancer")


def test_histopathologic_cancer_load_dataset_reads_separate_labels_csv():
    task = get_task("histopathologic_cancer")
    adapter = HistopathologicCancerAdapter(
        HISTOPATHOLOGIC_FIXTURE, "/tmp/mlestar-vision-test-hcancer-load", task, pretrained=False
    )
    paths, labels, ids = adapter._load_dataset(adapter.data_root)
    assert len(paths) == 6
    assert labels.tolist() == [1, 0, 1, 0, 1, 0]
    assert ids == ["h0", "h1", "h2", "h3", "h4", "h5"]
    assert paths[0].name == "h0.tif"


def test_histopathologic_cancer_run_produces_a_roc_auc_metric(tmp_path):
    task = get_task("histopathologic_cancer")
    adapter = HistopathologicCancerAdapter(
        HISTOPATHOLOGIC_FIXTURE, tmp_path, task, pretrained=False, epochs=1, max_train_samples=None
    )
    candidate = CandidateSpec("resnet18", (("model", "resnet18"),))
    result = adapter.run(candidate, phase="test", seed=13)
    assert result.receipt.error is None, result.receipt.error
    assert result.receipt.metric_value is not None
    assert 0.0 <= result.receipt.metric_value <= 1.0
    assert result.oof.shape == (6,)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_vision_adapter.py -v -k histopathologic`
Expected: FAIL with `ImportError: cannot import name 'HistopathologicCancerAdapter'`.

- [ ] **Step 3: Append `HistopathologicCancerAdapter` to `mlestar/adapters/vision.py`**

```python
class HistopathologicCancerAdapter(ImageClassificationAdapter):
    """Metastatic cancer detection in .tif image patches, labels in a separate CSV."""

    def _load_dataset(self, data_root: Path) -> tuple[list[Path], np.ndarray, list[str]]:
        csv_path = data_root / "train_labels.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"Histopathologic Cancer needs train_labels.csv in {data_root}.")
        frame = pd.read_csv(csv_path)
        image_paths = [data_root / "train" / f"{id_}.tif" for id_ in frame["id"]]
        labels = frame["label"].to_numpy(dtype=float)
        return image_paths, labels, frame["id"].astype(str).tolist()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_vision_adapter.py -v -k histopathologic`
Expected: 2 passed.

- [ ] **Step 5: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass (42 total).

- [ ] **Step 6: Commit**

```bash
git add mlestar/adapters/vision.py tests/test_vision_adapter.py
git commit -m "feat: add HistopathologicCancerAdapter (image_binary)"
```

---

## Task 8: Wire all seven adapters into `experiment.py`, surface `receipt.error`

**Files:**
- Modify: `mlestar/experiment.py`
- Modify: `tests/test_experiment.py`

**Interfaces:**
- Consumes: all seven adapter classes (`LeafClassificationAdapter` unchanged; six new ones from Tasks 2-7).
- Produces: `compare()` accepts any of the seven benchmark keys. `choose_best` in `mlestar/initialization.py` raises the same `RuntimeError` under the same condition (no candidate produced a metric), but its message now includes each failing candidate's `receipt.error` text instead of only the generic sentence.

### Part A: surface `receipt.error` in `choose_best`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_initialization.py` (append; this file already exists and tests `choose_best`/`initialize_solution`):

```python
def test_choose_best_error_message_includes_candidate_errors():
    task = get_task("leaf_classification")  # already imported at the top of this test file
    candidate_a = CandidateSpec("a", (("model", "a"),))
    candidate_b = CandidateSpec("b", (("model", "b"),))
    receipt_a = ExperimentReceipt(
        experiment_id="a-1", parent_experiment_id=None, phase="initial", candidate_id="a",
        metric_value=None, fold_scores=(), seed=13, oof_path=None, test_path=None,
        error="FileNotFoundError: train.csv missing",
    )
    receipt_b = ExperimentReceipt(
        experiment_id="b-1", parent_experiment_id=None, phase="initial", candidate_id="b",
        metric_value=None, fold_scores=(), seed=13, oof_path=None, test_path=None,
        error="ValueError: bad label",
    )
    with pytest.raises(RuntimeError) as excinfo:
        choose_best([(candidate_a, receipt_a), (candidate_b, receipt_b)], task)
    message = str(excinfo.value)
    assert "FileNotFoundError: train.csv missing" in message
    assert "ValueError: bad label" in message
```

Check the top of `tests/test_initialization.py` already imports `pytest`, `ExperimentReceipt`, `CandidateSpec`, `choose_best`, and `get_task` (or a fixture `TaskSpec`) -- add whichever of these imports/fixtures are missing so the new test can run standalone.

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_initialization.py -v -k error_message`
Expected: FAIL -- current message is only `"No candidate produced a validation metric."`, doesn't contain either error string.

- [ ] **Step 3: Modify `choose_best` in `mlestar/initialization.py`**

Replace the existing `choose_best` function body:

```python
def choose_best(
    candidates: Iterable[tuple[CandidateSpec, ExperimentReceipt]], task: TaskSpec
) -> tuple[CandidateSpec, ExperimentReceipt]:
    """Choose the best successful receipt without treating a failure as zero."""

    pairs = list(candidates)
    valid = [(candidate, receipt) for candidate, receipt in pairs if receipt.metric_value is not None]
    if not valid:
        details = "; ".join(
            f"{candidate.candidate_id}: {receipt.error}"
            for candidate, receipt in pairs
            if receipt.error is not None
        )
        suffix = f" Candidate errors -- {details}" if details else ""
        raise RuntimeError(f"No candidate produced a validation metric.{suffix}")
    direction = bool(task.metric.greater_is_better)
    return max(valid, key=lambda item: float(item[1].metric_value)) if direction else min(
        valid, key=lambda item: float(item[1].metric_value)
    )
```

(The only change: `candidates` is materialized into `pairs` once so it can be iterated twice, and the `RuntimeError` message gains an optional `" Candidate errors -- ..."` suffix built from any receipts that have a non-`None` `error`.)

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_initialization.py -v -k error_message`
Expected: PASS.

- [ ] **Step 5: Run the full suite to confirm no other `initialization.py` test broke**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all previously-passing tests pass, plus the new one from Step 4 (43 total at this point -- Task 8 Part B's own new tests are added next).

- [ ] **Step 6: Commit**

```bash
cd /Users/wang/Documents/Jiaozi
git add mlestar/initialization.py tests/test_initialization.py
git commit -m "fix: include per-candidate errors in choose_best's RuntimeError"
```

### Part B: register all seven adapters in `experiment.py`

- [ ] **Step 1: Read current `experiment.py` to confirm line numbers before editing**

Run: `sed -n '1,30p' mlestar/experiment.py`
(Confirms the current imports and `_candidate`/`_LeafPlanner` helpers before this task's edit -- the exact lines are already shown in this plan's context from the design phase.)

- [ ] **Step 2: Write the failing test for the benchmark registry**

Append to `tests/test_experiment.py` (create the file with this content if it does not already exist; if it exists, add these two functions and their imports to the top):

```python
from pathlib import Path

from mlestar.adapters.vision import PlantPathologyAdapter
from mlestar.experiment import ADAPTER_CLASSES, compare


def test_adapter_classes_registry_covers_all_seven_implemented_benchmarks():
    assert set(ADAPTER_CLASSES) == {
        "leaf_classification",
        "plant_pathology_2020",
        "aptos_2019",
        "dog_breed",
        "aerial_cactus",
        "dogs_vs_cats",
        "histopathologic_cancer",
    }
    assert ADAPTER_CLASSES["plant_pathology_2020"] is PlantPathologyAdapter


def test_compare_still_rejects_unimplemented_benchmarks(tmp_path):
    import pytest

    with pytest.raises(NotImplementedError, match="global_wheat"):
        compare(benchmark="global_wheat", data_root=tmp_path, run_root=tmp_path)


def test_compare_runs_plant_pathology_end_to_end(tmp_path):
    report = compare(
        benchmark="plant_pathology_2020",
        data_root=Path("examples/synthetic_plant_pathology"),
        run_root=tmp_path,
        seeds=(13,),
    )
    assert report["benchmark"] == "plant_pathology_2020"
    assert report["summary"]["baseline"]["failures"] == 0
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_experiment.py -v -k "registry or unimplemented or plant_pathology"`
Expected: FAIL -- `ImportError: cannot import name 'ADAPTER_CLASSES'` (and `compare()` currently raises for every non-leaf benchmark unconditionally, including `plant_pathology_2020`).

- [ ] **Step 4: Modify `mlestar/experiment.py`**

Replace the imports and the hardcoded benchmark check. Show the full new top-of-file and `compare()` body:

```python
"""Paired baseline-versus-MLE-STAR comparisons for runnable adapters."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Sequence

import pandas as pd

from benchmarks.catalog import get_task

from .adapters.tabular import LeafClassificationAdapter
from .adapters.vision import (
    AerialCactusAdapter,
    AptosAdapter,
    DogBreedAdapter,
    DogsVsCatsAdapter,
    HistopathologicCancerAdapter,
    PlantPathologyAdapter,
)
from .ensemble import select_ensemble
from .initialization import CandidateSpec, initialize_solution
from .refinement import RefinementPlanner, refine_solution

ADAPTER_CLASSES: dict[str, type] = {
    "leaf_classification": LeafClassificationAdapter,
    "plant_pathology_2020": PlantPathologyAdapter,
    "aptos_2019": AptosAdapter,
    "dog_breed": DogBreedAdapter,
    "aerial_cactus": AerialCactusAdapter,
    "dogs_vs_cats": DogsVsCatsAdapter,
    "histopathologic_cancer": HistopathologicCancerAdapter,
}

_CANDIDATE_MODELS: dict[str, tuple[str, str]] = {
    "leaf_classification": ("extra_trees", "random_forest"),
    "plant_pathology_2020": ("resnet18", "efficientnet_b0"),
    "aptos_2019": ("resnet18", "efficientnet_b0"),
    "dog_breed": ("resnet18", "efficientnet_b0"),
    "aerial_cactus": ("resnet18", "efficientnet_b0"),
    "dogs_vs_cats": ("resnet18", "efficientnet_b0"),
    "histopathologic_cancer": ("resnet18", "efficientnet_b0"),
}


def _candidate(candidate_id: str, model: str) -> CandidateSpec:
    return CandidateSpec(candidate_id, (("model", model),))


class _AlternatingPlanner:
    """Propose the other of a fixed pair of model names, alternating each call."""

    def __init__(self, model_names: tuple[str, str]) -> None:
        self._model_names = model_names

    def propose(self, *, component, candidate, history):
        del component, history
        current = candidate.block("model")
        other = self._model_names[1] if current == self._model_names[0] else self._model_names[0]
        return (other,)


def _summary(rows: list[dict[str, object]]) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    seed_count = len({int(row["seed"]) for row in rows})
    for arm in ("baseline", "mlestar_initial", "mlestar_refined", "mlestar_ensemble"):
        values = [float(row["metric_value"]) for row in rows if row["arm"] == arm and row["metric_value"] is not None]
        output[arm] = {
            "mean": mean(values) if values else float("nan"),
            "sem": stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0,
            "wins": 0,
            "failures": seed_count - len(values),
        }
    baseline = {int(row["seed"]): float(row["metric_value"]) for row in rows if row["arm"] == "baseline" and row["metric_value"] is not None}
    for arm in ("mlestar_initial", "mlestar_refined", "mlestar_ensemble"):
        output[arm]["wins"] = sum(
            float(row["metric_value"]) < baseline[int(row["seed"])]
            for row in rows
            if row["arm"] == arm and row["metric_value"] is not None and int(row["seed"]) in baseline
        )
    return output


def compare(
    *, benchmark: str, data_root: str | Path, run_root: str | Path, seeds: Sequence[int] = (13, 29, 47),
    outer_rounds: int = 1, inner_rounds: int = 1,
) -> dict[str, object]:
    """Run paired real baseline/initial/refinement/OOF-ensemble arms for one benchmark.

    Catalog entries without a registered adapter class deliberately fail
    loudly until their modality adapter is installed; no synthetic metric is
    reported for an unavailable task.
    """

    task = get_task(benchmark)
    adapter_class = ADAPTER_CLASSES.get(benchmark)
    if adapter_class is None:
        raise NotImplementedError(
            f"{benchmark} requires its {task.modality} adapter; use the catalog for a schema-only preflight."
        )
    model_a, model_b = _CANDIDATE_MODELS[benchmark]
    root = Path(run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    receipts: list[dict[str, object]] = []
    for seed in seeds:
        seed_root = root / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        adapter = adapter_class(data_root, seed_root, task)
        baseline_candidate = _candidate(model_a, model_a)
        baseline_run = adapter.run(baseline_candidate, phase="baseline", seed=seed)
        initial = initialize_solution(
            task, adapter, (baseline_candidate, _candidate(model_b, model_b)), seed=seed
        )
        initial_run = adapter.run(initial.best, phase="initial_selected", seed=seed)
        refined = refine_solution(
            task, adapter, _AlternatingPlanner((model_a, model_b)), initial.best, initial.best_receipt,
            outer_rounds=outer_rounds, inner_rounds=inner_rounds, seed=seed,
        )
        refined_run = adapter.run(refined.candidate, phase="refined_selected", seed=seed)
        ensemble = select_ensemble(
            {
                "baseline": (range(len(baseline_run.y_true)), baseline_run.oof),
                "refined": (range(len(refined_run.y_true)), refined_run.oof),
            },
            baseline_run.y_true,
            task.metric,
        )
        arm_values = {
            "baseline": baseline_run.receipt.metric_value,
            "mlestar_initial": initial_run.receipt.metric_value,
            "mlestar_refined": refined_run.receipt.metric_value,
            "mlestar_ensemble": ensemble.score.value,
        }
        rows.extend({"seed": seed, "arm": arm, "metric_value": value} for arm, value in arm_values.items())
        receipts.extend(
            asdict(receipt)
            for receipt in (
                baseline_run.receipt, *initial.receipts, *initial.merge_receipts,
                initial_run.receipt, *refined.ablations, *refined.rejected_receipts,
                refined.receipt, refined_run.receipt,
            )
        )
    report: dict[str, object] = {
        "benchmark": benchmark,
        "metric": task.metric.name,
        "paired_folds": True,
        "seeds": list(seeds),
        "arms": ["baseline", "mlestar_initial", "mlestar_refined", "mlestar_ensemble"],
        "summary": _summary(rows),
        "status": "offline_oof_complete",
    }
    pd.DataFrame(rows).to_csv(root / "comparison.csv", index=False)
    (root / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    (root / "receipts.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in receipts), encoding="utf-8")
    return report
```

Note this replaces `_LeafPlanner` with a general `_AlternatingPlanner` usable by every benchmark (Leaf's old planner alternated `random_forest`/`extra_trees` by name; the new one alternates between whichever `(model_a, model_b)` pair `_CANDIDATE_MODELS` registers for that benchmark, which for `leaf_classification` is still exactly `("extra_trees", "random_forest")` -- same behavior, generalized).

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_experiment.py -v`
Expected: all pass, including the pre-existing Leaf Classification test(s) in that file (confirms the `_AlternatingPlanner` refactor didn't change Leaf's behavior).

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass (46 total: 43 after Part A, plus the 3 new tests from Step 2).

- [ ] **Step 7: Commit**

```bash
git add mlestar/experiment.py tests/test_experiment.py
git commit -m "feat: register all seven benchmark adapters in compare()"
```

---

## Task 9: Generalize the notebook's download cell, add six new task cells

**Files:**
- Modify: `notebooks/mlestar_kaggle_experiments.ipynb`
- Modify: `tests/test_notebook.py`

**Interfaces:**
- Consumes: the `KAGGLE_API_TOKEN`-from-environment download+extract pattern already proven working in cell 3 (leaf_classification).
- Produces: a reusable `fetch_kaggle_competition(slug, data_root)` function cell, plus one download+compare cell pair per new benchmark.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_notebook.py`:

```python
def test_notebook_has_a_reusable_download_helper_used_by_all_seven_tasks() -> None:
    notebook = json.loads(Path("notebooks/mlestar_kaggle_experiments.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "def fetch_kaggle_competition" in source
    for slug in (
        "leaf-classification",
        "plant-pathology-2020-fgvc7",
        "aptos2019-blindness-detection",
        "dog-breed-identification",
        "aerial-cactus-identification",
        "dogs-vs-cats-redux-kernels-edition",
        "histopathologic-cancer-detection",
    ):
        assert slug in source, f"missing competition slug: {slug}"
    for benchmark in (
        "leaf_classification",
        "plant_pathology_2020",
        "aptos_2019",
        "dog_breed",
        "aerial_cactus",
        "dogs_vs_cats",
        "histopathologic_cancer",
    ):
        assert f"--benchmark {benchmark}" in source, f"missing compare cell for: {benchmark}"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_notebook.py -v -k reusable`
Expected: FAIL (helper and most slugs/benchmarks don't exist yet).

- [ ] **Step 3: Rewrite the notebook's cells 3+ with a reusable helper and seven task cell-pairs**

Use this exact rewrite script (run once, then delete it):

```python
# /tmp/rewrite_notebook_all_tasks.py
import json
from pathlib import Path

path = Path("/Users/wang/Documents/Jiaozi/notebooks/mlestar_kaggle_experiments.ipynb")
nb = json.loads(path.read_text(encoding="utf-8"))

helper_source = [
    "# Reusable Kaggle competition fetcher, used by every task cell below.\n",
    "# Uses the KAGGLE_API_TOKEN secret from the previous cell -- the kaggle CLI\n",
    "# reads it directly from the environment, no credentials file needed.\n",
    "# Accept each competition's rules on kaggle.com before running its cell, or\n",
    "# upload that task's files to its DATA_ROOT yourself and skip the fetch.\n",
    "import os\n",
    "import pathlib\n",
    "import zipfile\n",
    "\n",
    "\n",
    "def fetch_kaggle_competition(slug: str, data_root: str, marker_file: str) -> None:\n",
    "    root = pathlib.Path(data_root)\n",
    "    root.mkdir(parents=True, exist_ok=True)\n",
    "    if (root / marker_file).exists():\n",
    "        return\n",
    "    if not os.environ.get('KAGGLE_API_TOKEN'):\n",
    "        raise RuntimeError(\n",
    "            f'No {marker_file} in {data_root} and no KAGGLE_API_TOKEN secret set. '\n",
    "            'Either set the secret in the previous cell or upload the data yourself.'\n",
    "        )\n",
    "    !kaggle competitions download -c {slug} -p {data_root}\n",
    "    outer_zip = root / f'{slug}.zip'\n",
    "    if outer_zip.exists():\n",
    "        with zipfile.ZipFile(outer_zip) as archive:\n",
    "            archive.extractall(root)\n",
    "    # Some competitions (e.g. leaf-classification) nest train/test/\n",
    "    # sample_submission as their own *.csv.zip archives one level deeper.\n",
    "    for nested_zip in root.glob('*.csv.zip'):\n",
    "        with zipfile.ZipFile(nested_zip) as archive:\n",
    "            archive.extractall(root)\n",
    "    if not (root / marker_file).exists():\n",
    "        raise RuntimeError(\n",
    "            f'Download did not produce {marker_file} in {data_root}. Scroll up to the '\n",
    "            'kaggle output above for the real error -- commonly the competition rules '\n",
    "            f'are not accepted yet at https://www.kaggle.com/c/{slug}/rules, or '\n",
    "            'KAGGLE_API_TOKEN is invalid/expired.'\n",
    "        )\n",
]

leaf_source = [
    "# Leaf Classification.\n",
    "DATA_ROOT = '/content/leaf-classification'\n",
    "fetch_kaggle_competition('leaf-classification', DATA_ROOT, 'train.csv')\n",
    "RUN_ROOT = '/content/mlestar-runs/leaf-classification'\n",
    "!python -m mlestar.cli compare --benchmark leaf_classification --data-root {DATA_ROOT} --run-root {RUN_ROOT} --seeds 13 29 47 --no-submit\n",
    "\n",
    "import pandas as pd\n",
    "pd.read_csv(f'{RUN_ROOT}/comparison.csv')\n",
]

TASKS = [
    ("Plant Pathology 2020", "plant_pathology_2020", "plant-pathology-2020-fgvc7", "train.csv"),
    ("APTOS 2019 Blindness Detection", "aptos_2019", "aptos2019-blindness-detection", "train.csv"),
    ("Dog Breed Identification", "dog_breed", "dog-breed-identification", "labels.csv"),
    ("Aerial Cactus Identification", "aerial_cactus", "aerial-cactus-identification", "train.csv"),
    ("Dogs vs. Cats Redux", "dogs_vs_cats", "dogs-vs-cats-redux-kernels-edition", "train"),
    ("Histopathologic Cancer Detection", "histopathologic_cancer", "histopathologic-cancer-detection", "train_labels.csv"),
]

new_cells = [
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": helper_source},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": leaf_source},
]
for title, key, slug, marker in TASKS:
    data_root = f"/content/{slug}"
    run_root = f"/content/mlestar-runs/{key}"
    task_source = [
        f"# {title}.\n",
        f"DATA_ROOT = '{data_root}'\n",
        f"fetch_kaggle_competition('{slug}', DATA_ROOT, '{marker}')\n",
        f"RUN_ROOT = '{run_root}'\n",
        f"!python -m mlestar.cli compare --benchmark {key} --data-root {{DATA_ROOT}} --run-root {{RUN_ROOT}} --seeds 13 29 47 --no-submit\n",
        "\n",
        "import pandas as pd\n",
        f"pd.read_csv(f'{run_root}/comparison.csv')\n",
    ]
    new_cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": task_source})

# Cells: 0=markdown intro, 1=clone/install, 2=KAGGLE_API_TOKEN secret,
# 3=old download cell, 4=old leaf compare cell, 5=closing markdown.
assert nb["cells"][0]["cell_type"] == "markdown"
assert nb["cells"][1]["cell_type"] == "code"
assert nb["cells"][2]["cell_type"] == "code"
closing_markdown = nb["cells"][5]
nb["cells"] = nb["cells"][:3] + new_cells + [closing_markdown]

path.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print("rewritten:", len(nb["cells"]), "cells")
```

Run: `python3 /tmp/rewrite_notebook_all_tasks.py && rm /tmp/rewrite_notebook_all_tasks.py`
Expected: prints `rewritten: 11 cells`.

- [ ] **Step 4: Validate every new code cell compiles via IPython's transformer**

```bash
python3 -c "
import json
from IPython.core.inputtransformer2 import TransformerManager
nb = json.load(open('/Users/wang/Documents/Jiaozi/notebooks/mlestar_kaggle_experiments.ipynb'))
tm = TransformerManager()
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    compile(tm.transform_cell(src), f'<cell{i}>', 'exec')
print('all code cells compile OK')
"
```
Expected: `all code cells compile OK`.

- [ ] **Step 5: Run tests to verify they pass -- two pre-existing tests will fail and must be fixed**

Run: `.venv/bin/python -m pytest tests/test_notebook.py -v`

Expected: two pre-existing tests FAIL, because the download logic moved from a
per-task cell into the generic `fetch_kaggle_competition` helper, which takes
`marker_file` as a parameter instead of hardcoding the literal string
`"train.csv"`:

- `test_notebook_download_cell_fails_loudly_if_data_still_missing` --
  `next(src for src in cells_source if "kaggle competitions download" in src)`
  now matches the helper cell, not a per-task cell, and the helper's
  post-download check reads `if not (root / marker_file).exists():`, so the
  literal substring `"train.csv"` this test looks for is gone from that cell.
- `test_notebook_extracts_nested_leaf_csv_archives` -- same helper-cell
  lookup, still valid (it only checks for `"zipfile"` and `"csv.zip"`, both
  still present), so this one should already pass; if it doesn't, the
  fixture in the next sub-step covers it too.

Fix `tests/test_notebook.py` by replacing these two functions:

```python
def test_notebook_download_cell_fails_loudly_if_data_still_missing() -> None:
    notebook = json.loads(Path("notebooks/mlestar_kaggle_experiments.ipynb").read_text(encoding="utf-8"))
    cells_source = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    helper_cell = next(src for src in cells_source if "def fetch_kaggle_competition" in src)
    # Shell magics (!cmd) don't raise on non-zero exit, so a failed Kaggle
    # download/unzip would otherwise continue silently. The helper must check
    # the outcome itself and raise before returning.
    extract_index = helper_cell.index("extractall")
    postcheck_index = helper_cell.index("marker_file", extract_index)
    raise_index = helper_cell.index("raise", postcheck_index)
    assert postcheck_index < raise_index


def test_notebook_extracts_nested_leaf_csv_archives() -> None:
    notebook = json.loads(Path("notebooks/mlestar_kaggle_experiments.ipynb").read_text(encoding="utf-8"))
    cells_source = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    helper_cell = next(src for src in cells_source if "def fetch_kaggle_competition" in src)
    # The leaf-classification competition zip nests train/test/
    # sample_submission one level deeper as their own *.csv.zip archives, so
    # the outer extraction alone would leave e.g. train.csv.zip on disk, not
    # train.csv -- this only needs to be correct once, in the shared helper.
    assert "zipfile" in helper_cell
    assert "*.csv.zip" in helper_cell or "csv.zip" in helper_cell
```

Run: `.venv/bin/python -m pytest tests/test_notebook.py -v`
Expected: all pass now (the other four pre-existing tests -- token-secret, downloads-when-missing, and the new Step 1 test -- were unaffected by the helper-cell change and already passed).

- [ ] **Step 6: Run the full suite**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass (47 total: 46 after Task 8, plus the 1 new test from Step 1).

- [ ] **Step 7: Commit**

```bash
git add notebooks/mlestar_kaggle_experiments.ipynb tests/test_notebook.py
git commit -m "feat: generalize Kaggle download helper, add cells for six new tasks"
```

---

## Task 10: Update docs, refresh `~/Downloads` copy, final verification

**Files:**
- Modify: `docs/BENCHMARK_STATUS.md`
- Modify: `README.md`

- [ ] **Step 1: Update `docs/BENCHMARK_STATUS.md`**

Add an "Adapter status" column distinguishing implemented from catalog-only tasks:

```markdown
# Benchmark status

| Key | Competition | Offline metric | Adapter status |
| --- | --- | --- | --- |
| leaf_classification | Leaf Classification | multiclass log loss | implemented |
| plant_pathology_2020 | Plant Pathology 2020 - FGVC7 | mean ROC-AUC | implemented |
| aptos_2019 | APTOS 2019 Blindness Detection | QWK | implemented |
| dog_breed | Dog Breed Identification | multiclass log loss | implemented |
| aerial_cactus | Aerial Cactus Identification | ROC-AUC | implemented |
| dogs_vs_cats | Dogs vs. Cats Redux | log loss | implemented |
| histopathologic_cancer | Histopathologic Cancer Detection | ROC-AUC | implemented |
| global_wheat | Global Wheat Detection | competition AP | not implemented (object detection) |
| ultrasound_nerve | Ultrasound Nerve Segmentation | Dice | not implemented (segmentation) |
| denoising_dirty_documents | Denoising Dirty Documents | RMSE | not implemented (image denoising) |

The runner first produces local fixed-fold OOF comparisons. The seven
implemented tasks are OOF-only: they do not write a submission file. If a
user explicitly requests a submission for `leaf_classification` (the one
task with tabular test-set support), it records the exact Kaggle acceptance
or rejection response.
```

- [ ] **Step 2: Update `README.md`'s scope description**

Find the paragraph starting "It is not part of Jiaozi..." and the Install section's benchmark-listing sentence. Replace the sentence "the current executable training adapter is Leaf Classification, while the other modality adapters remain explicit implementation work rather than fabricated results" (in the notebook markdown cell, already updated in Task 9 if present there -- check `README.md` for the same or similar sentence and update it) with:

```markdown
Seven of the ten catalogued tasks have executable training adapters (one
tabular, six image-classification, sharing a common timm fine-tuning
pipeline). The remaining three -- object detection, segmentation, and
image denoising -- are registered in the catalog but not yet implemented,
and `mlestar compare` fails loudly rather than fabricating a result for
them.
```

- [ ] **Step 3: Run the full test suite one final time**

Run: `.venv/bin/python -m pytest tests/ -q`
Expected: all pass (47 total), no warnings beyond the pre-existing sklearn `UserWarning`s about unknown categories.

- [ ] **Step 4: Refresh the `~/Downloads` notebook copy**

```bash
rm -f "/Users/wang/Downloads/mlestar_kaggle_experiments.ipynb"
cp "/Users/wang/Documents/Jiaozi/notebooks/mlestar_kaggle_experiments.ipynb" "/Users/wang/Downloads/mlestar_kaggle_experiments.ipynb"
```

- [ ] **Step 5: Commit and push**

```bash
cd /Users/wang/Documents/Jiaozi
git add docs/BENCHMARK_STATUS.md README.md
git commit -m "docs: mark six image-classification benchmarks as implemented"
git push origin codex/mlestar-kaggle-benchmarks
```

**Do not push automatically -- confirm with the user first, per this repo's established workflow on this branch (every prior push on this branch was done only after an explicit "push到github" from the user).**

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


class _PriorModel(torch.nn.Module):
    """No-op ablation baseline for refine_solution's ``pass`` component swap.

    ``refine_solution`` (mlestar/refinement.py) ablates each candidate block
    by replacing it with the literal source ``"pass"`` and checking whether
    that measurably hurts the metric. The tabular adapter handles this via
    ``DummyClassifier(strategy="prior")``; timm has no model literally named
    "pass", so `create_model("pass", ...)` fails, every ablation gets
    `metric_value=None`, and `refine_solution`'s `select_target_block` raises
    an uncaught `RuntimeError` -- crashing `compare()` for every vision
    benchmark, not just this one. This class ignores the input entirely and
    always predicts a fixed zero-logit vector (a uniform prediction after
    softmax/sigmoid), which is deliberately uninformative -- exactly what an
    ablated "no model" component should produce.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self._num_classes = num_classes

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return torch.zeros(images.shape[0], self._num_classes, dtype=torch.float32)


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
            # NOTE: do not reuse _num_classes(labels) here -- for image_ordinal
            # it returns 1 (the regression head's output width), not the count
            # of valid ordinal grade levels. Using it as the clip bound forces
            # every rounded prediction to 0. The valid range must come from
            # the labels themselves.
            max_label = int(labels.max())
            rounded = np.clip(np.round(oof), 0, max_label)
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
        if model_name == "pass":
            return _PriorModel(num_classes)
        return create_model(model_name, pretrained=self.pretrained, num_classes=num_classes)

    def _subsample(self, train_idx: np.ndarray, seed: int) -> np.ndarray:
        if self.max_train_samples is None or len(train_idx) <= self.max_train_samples:
            return train_idx
        rng = np.random.default_rng(seed)
        return rng.choice(train_idx, size=self.max_train_samples, replace=False)

    def _fit(self, model: torch.nn.Module, dataset: _ImageDataset, seed: int) -> None:
        parameters = list(model.parameters())
        if not parameters:
            # _PriorModel (the "pass" ablation baseline) has no trainable
            # weights -- Adam raises on an empty parameter list, and there is
            # nothing to fit anyway.
            return
        generator = torch.Generator().manual_seed(seed)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True, generator=generator
        )
        optimizer = torch.optim.Adam(parameters, lr=1e-4)
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
        scores: list[float] = []
        for fold in range(self.task.fold.n_splits):
            mask = fold_column == fold
            try:
                scores.append(score_metric(self.task.metric, scoring_labels[mask], scoring_oof[mask]).value)
            except ValueError:
                # A validation fold this small can end up missing a class
                # entirely (e.g. multiclass log_loss's class-count mismatch,
                # or roc_auc's "only one class present" case) -- both are
                # sklearn ValueErrors raised because the per-fold subset
                # lacks the diversity the metric needs, not a real failure.
                # NaN records "this fold couldn't be scored" rather than a
                # fabricated number; the aggregate `metric_value` (scored
                # over all rows at once, computed in `run()`, not here) is
                # unaffected since the full dataset has every class.
                scores.append(float("nan"))
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

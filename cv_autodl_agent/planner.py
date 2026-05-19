from __future__ import annotations

from copy import deepcopy
from typing import Any

from .schemas import DatasetManifest, RetrievedModelCandidate, TrainingSpec


class HeuristicTrainingSpecPlanner:
    """Deterministic fallback planner used when no LLM integration is configured."""

    def plan(self, manifest: DatasetManifest, candidate: RetrievedModelCandidate) -> TrainingSpec:
        image_size = manifest.image_size_hint or candidate.default_input_size or 224
        metric = manifest.recommended_metric or self._default_metric(manifest.task_family)
        transforms = self._default_transforms(manifest.task_family, image_size)
        loss_fn = self._default_loss(manifest.task_family)
        batch_size = self._default_batch_size(manifest.task_family, image_size)
        optimizer = self._default_optimizer(manifest.task_family)
        scheduler = self._default_scheduler(manifest.task_family)
        freeze_strategy = "train_all"
        edit_regions = {
            "transforms": ["transforms"],
            "freeze_strategy": ["freeze_strategy"],
            "optimizer/scheduler": ["optimizer", "scheduler"],
            "batch_size": ["batch_size"],
            "learning_rate": ["optimizer.learning_rate"],
            "loss_fn": ["loss_fn"],
            "image_size": ["transforms"],
            "score_threshold": ["optimizer.score_threshold"],
        }
        return TrainingSpec(
            selected_model_id=candidate.model_id,
            template_id=f"{manifest.task_family}_template",
            task_family=manifest.task_family,
            dataset_loader_strategy=self._dataset_loader_strategy(manifest),
            transforms=transforms,
            loss_fn=loss_fn,
            metric=metric,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=8 if manifest.task_family == "classification" else 12,
            batch_size=batch_size,
            freeze_strategy=freeze_strategy,
            checkpoint_policy="best",
            early_stopping={"enabled": True, "patience": 3, "monitor": metric},
            edit_regions=edit_regions,
        )

    @staticmethod
    def _default_metric(task_family: str) -> str:
        return {
            "classification": "accuracy",
            "segmentation": "miou",
            "detection": "map50",
        }[task_family]

    @staticmethod
    def _default_loss(task_family: str) -> str:
        return {
            "classification": "cross_entropy",
            "segmentation": "cross_entropy",
            "detection": "detection_default",
        }[task_family]

    @staticmethod
    def _default_transforms(task_family: str, image_size: int) -> list[str]:
        resize = f"resize:{image_size}"
        if task_family == "classification":
            return [resize, "center_crop", "normalize"]
        if task_family == "segmentation":
            return [resize, "normalize"]
        return [resize, "normalize"]

    @staticmethod
    def _default_batch_size(task_family: str, image_size: int) -> int:
        if task_family == "classification":
            return 16 if image_size <= 256 else 8
        if task_family == "segmentation":
            return 4 if image_size <= 512 else 2
        return 2 if image_size <= 640 else 1

    @staticmethod
    def _default_optimizer(task_family: str) -> dict[str, Any]:
        optimizer = {"name": "adamw", "learning_rate": 1e-4, "weight_decay": 1e-4}
        if task_family == "detection":
            optimizer["score_threshold"] = 0.15
        return optimizer

    @staticmethod
    def _default_scheduler(task_family: str) -> dict[str, Any]:
        if task_family == "classification":
            return {"name": "step", "step_size": 2, "gamma": 0.5}
        if task_family == "segmentation":
            return {"name": "step", "step_size": 3, "gamma": 0.5}
        return {"name": "cosine", "warmup_epochs": 1}

    @staticmethod
    def _dataset_loader_strategy(manifest: DatasetManifest) -> str:
        if manifest.task_family == "classification":
            if manifest.hf_dataset_id:
                return "huggingface_dataset"
            return "imagefolder" if manifest.annotation_format.lower() == "imagefolder" else "csv_labels"
        if manifest.task_family == "segmentation":
            return "paired_images_masks"
        return "coco_detection"


class LangChainTrainingSpecPlanner(HeuristicTrainingSpecPlanner):
    """Optional integration point for future LangChain structured output."""

    def __init__(self, model: Any | None = None) -> None:
        self._model = model

    def plan(self, manifest: DatasetManifest, candidate: RetrievedModelCandidate) -> TrainingSpec:
        if self._model is None:
            return super().plan(manifest, candidate)
        raise NotImplementedError(
            "LangChain-based planning hook is reserved for environments with model credentials configured."
        )



def clone_spec_with_updates(spec: TrainingSpec, updates: dict[str, Any]) -> TrainingSpec:
    payload = deepcopy(spec.to_dict())
    for key, value in updates.items():
        if "." in key:
            current = payload
            pieces = key.split(".")
            for piece in pieces[:-1]:
                current = current.setdefault(piece, {})
            current[pieces[-1]] = value
        else:
            payload[key] = value
    return TrainingSpec.from_dict(payload)

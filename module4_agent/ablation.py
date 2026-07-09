"""Controlled ablation variant generation for Module 4 refinement."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from .schemas import AblationVariant, TrainingSpec


CONTROLLED_FIELDS = (
    "optimizer",
    "learning_rate",
    "augmentation",
    "finetune_strategy",
    "unfreeze_last_n_blocks",
    "tta",
    "loss",
    "backbone",
    "checkpoint",
)

FORBIDDEN_REFINEMENT_FIELDS = (
    "task_type",
    "metric",
    "logging_format",
    "data_split_policy",
    "experiment_loop_structure",
)


def spec_id(spec: TrainingSpec) -> str:
    """Return a stable short identifier for a TrainingSpec."""

    return f"rank{spec.rank}_{spec.task_type}_{_slug(spec.backbone)}"


def training_spec_summary(spec: TrainingSpec) -> dict[str, Any]:
    """Return the fields that matter for experiment tracking and review."""

    return {
        "rank": spec.rank,
        "score": spec.score,
        "task_type": spec.task_type,
        "backbone": spec.backbone,
        "checkpoint": spec.checkpoint,
        "pretrained_hf_id": spec.pretrained_hf_id,
        "head": spec.head,
        "loss": spec.loss,
        "optimizer": spec.optimizer,
        "finetune_strategy": spec.finetune_strategy,
        "freeze_backbone": spec.freeze_backbone,
        "unfreeze_last_n_blocks": spec.unfreeze_last_n_blocks,
        "train_norm_layers": spec.train_norm_layers,
        "strategy_ablation_group": spec.strategy_ablation_group,
        "strategy_ablation_variant": spec.strategy_ablation_variant,
        "learning_rate": spec.learning_rate,
        "augmentation": spec.augmentation,
        "tta": spec.tta,
        "data_size": spec.data_size,
        "class_imbalance": spec.class_imbalance,
    }


def diff_controlled_fields(
    before: TrainingSpec | dict[str, Any],
    after: TrainingSpec | dict[str, Any],
) -> dict[str, tuple[Any, Any]]:
    """Return changed allowed fields between two specs or summaries.

    ``freeze_backbone`` is folded into ``finetune_strategy`` only when the
    strategy itself changes. A standalone freeze change is reported separately
    so the reviewer can reject it as outside the allowed refinement surface.
    """

    before_summary = _summary(before)
    after_summary = _summary(after)
    changes: dict[str, tuple[Any, Any]] = {}
    for field in CONTROLLED_FIELDS:
        if before_summary.get(field) != after_summary.get(field):
            changes[field] = (before_summary.get(field), after_summary.get(field))

    if before_summary.get("freeze_backbone") != after_summary.get("freeze_backbone"):
        if "finetune_strategy" not in changes:
            changes["freeze_backbone"] = (
                before_summary.get("freeze_backbone"),
                after_summary.get("freeze_backbone"),
            )
    if "finetune_strategy" in changes:
        changes.pop("unfreeze_last_n_blocks", None)
    return changes


def has_forbidden_field_changes(
    before: TrainingSpec | dict[str, Any],
    after: TrainingSpec | dict[str, Any],
) -> bool:
    """Return True if refinement changed fields outside the allowed scope."""

    before_summary = _summary(before)
    after_summary = _summary(after)
    return any(before_summary.get(field) != after_summary.get(field) for field in FORBIDDEN_REFINEMENT_FIELDS)


def generate_ablation_variants(base_spec: TrainingSpec) -> list[AblationVariant]:
    """Generate deterministic one-component variants for a baseline spec."""

    candidates: list[tuple[str, Any, TrainingSpec]] = []

    optimizer = _optimizer_variant(base_spec)
    if optimizer and optimizer != base_spec.optimizer:
        candidates.append(("optimizer", optimizer, _replace_with_model_config(base_spec, optimizer=optimizer)))

    learning_rate = _learning_rate_variant(base_spec.learning_rate)
    if learning_rate != base_spec.learning_rate:
        candidates.append(("learning_rate", learning_rate, _replace_with_model_config(base_spec, learning_rate=learning_rate)))

    augmentation = _augmentation_variant(base_spec.augmentation)
    if augmentation != base_spec.augmentation:
        candidates.append(("augmentation", augmentation, _replace_with_model_config(base_spec, augmentation=augmentation)))
    augmentation_removal = _augmentation_removal_variant(base_spec.augmentation)
    if augmentation_removal != base_spec.augmentation:
        candidates.append(("augmentation", augmentation_removal, _replace_with_model_config(base_spec, augmentation=augmentation_removal)))
    if base_spec.task_type == "classification" and not base_spec.tta:
        candidates.append(("tta", True, _replace_with_model_config(base_spec, tta=True)))

    finetune_strategy = _finetune_variant(base_spec)
    if finetune_strategy and finetune_strategy != base_spec.finetune_strategy:
        unfreeze_last_n_blocks = 2 if finetune_strategy == "partial" else 0
        candidates.append(
            (
                "finetune_strategy",
                finetune_strategy,
                _replace_with_model_config(
                    base_spec,
                    finetune_strategy=finetune_strategy,
                    freeze_backbone=(finetune_strategy == "head_only"),
                    unfreeze_last_n_blocks=unfreeze_last_n_blocks,
                    train_norm_layers=(finetune_strategy == "partial"),
                ),
            )
        )
    unfreeze_last_n_blocks = _unfreeze_depth_variant(base_spec)
    if unfreeze_last_n_blocks and unfreeze_last_n_blocks != base_spec.unfreeze_last_n_blocks:
        candidates.append(
            (
                "unfreeze_last_n_blocks",
                unfreeze_last_n_blocks,
                _replace_with_model_config(
                    base_spec,
                    finetune_strategy="partial",
                    freeze_backbone=False,
                    unfreeze_last_n_blocks=unfreeze_last_n_blocks,
                    train_norm_layers=True,
                ),
            )
        )

    loss = _loss_variant(base_spec)
    if loss and loss != base_spec.loss:
        candidates.append(("loss", loss, _replace_with_model_config(base_spec, loss=loss)))

    alternative_backbone = _alternative_backbone(base_spec)
    if alternative_backbone and alternative_backbone != base_spec.backbone:
        candidates.append(("backbone", alternative_backbone, _replace_with_model_config(base_spec, backbone=alternative_backbone)))

    alternative_checkpoint = _alternative_checkpoint(base_spec)
    if alternative_checkpoint and alternative_checkpoint != base_spec.checkpoint:
        candidates.append(("checkpoint", alternative_checkpoint, _replace_with_model_config(base_spec, checkpoint=alternative_checkpoint)))

    variants: list[AblationVariant] = []
    base_id = spec_id(base_spec)
    for index, (component, value, variant_spec) in enumerate(candidates, start=1):
        changes = diff_controlled_fields(base_spec, variant_spec)
        if len(changes) != 1 or component not in changes:
            continue
        variants.append(
            AblationVariant(
                variant_id=f"{base_id}_abl{index}_{component}",
                base_spec_id=base_id,
                modified_component=component,
                modified_value=value,
                training_spec=variant_spec,
            )
        )
    return variants


def _summary(value: TrainingSpec | dict[str, Any]) -> dict[str, Any]:
    if isinstance(value, TrainingSpec):
        return training_spec_summary(value)
    return dict(value)


def _optimizer_variant(spec: TrainingSpec) -> str:
    optimizer = spec.optimizer.lower()
    if spec.task_type == "object_detection":
        return "sgd_momentum" if "sgd" not in optimizer else "adamw"
    if spec.task_type == "image_segmentation":
        return "adam" if optimizer != "adam" else "adamw"
    if spec.task_type == "feature_extraction":
        return "adamw" if optimizer != "adamw" else "adam"
    return "sgd_momentum" if "sgd" not in optimizer else "adamw"


def _learning_rate_variant(current: float) -> float:
    if current > 5.0e-4:
        return 3.0e-4
    if current > 1.5e-4:
        return 1.0e-4
    return 3.0e-4


def _augmentation_variant(current: str) -> str:
    current_lower = str(current or "basic").lower()
    return "stronger" if current_lower in {"", "none", "basic", "light"} else "basic"


def _augmentation_removal_variant(current: str) -> str:
    current_lower = str(current or "basic").lower()
    return "none" if current_lower not in {"", "none"} else "basic"


def _finetune_variant(spec: TrainingSpec) -> str | None:
    if spec.task_type == "object_detection":
        return None if spec.finetune_strategy == "full" else "full"
    if spec.finetune_strategy == "full":
        return "partial" if _is_transformer_backbone(spec.backbone) else "head_only"
    if spec.finetune_strategy == "head_only" and _is_transformer_backbone(spec.backbone):
        return "partial"
    return "full"


def _unfreeze_depth_variant(spec: TrainingSpec) -> int | None:
    if spec.finetune_strategy != "partial" or not _is_transformer_backbone(spec.backbone):
        return None
    return 4 if spec.unfreeze_last_n_blocks <= 2 else 2


def _loss_variant(spec: TrainingSpec) -> str | None:
    loss = spec.loss.lower()
    if spec.task_type == "classification":
        if "focal" not in loss:
            return "focal_loss"
        return "cross_entropy_loss"
    if spec.task_type == "image_segmentation":
        if "dice" not in loss:
            return "dice_loss"
        return "cross_entropy_loss"
    if spec.task_type == "object_detection":
        if "focal" not in loss:
            return "focal_loss"
        return "detection_smoke_loss"
    if spec.task_type == "feature_extraction":
        if "contrastive" not in loss:
            return "contrastive_loss"
        return "feature_mse_loss"
    return None


def _alternative_backbone(spec: TrainingSpec) -> str | None:
    for item in spec.alternatives:
        value = _alternative_value(item, keys=("backbone", "model_id", "name"))
        if value and value != spec.backbone:
            return value
    return None


def _alternative_checkpoint(spec: TrainingSpec) -> str | None:
    for item in spec.alternatives:
        value = _alternative_value(item, keys=("checkpoint", "checkpoint_id", "pretrained_hf_id", "hf_id"))
        if value and value != spec.checkpoint:
            return value
    return None


def _alternative_value(item: Any, *, keys: tuple[str, ...]) -> str | None:
    if isinstance(item, str):
        return item if "backbone" in keys or "name" in keys else None
    if not isinstance(item, dict):
        return None
    nested = dict(item.get("model_config") or {})
    for source in (nested, item):
        for key in keys:
            value = source.get(key)
            if value:
                return str(value)
    return None


def _is_transformer_backbone(backbone: str) -> bool:
    name = str(backbone or "").lower()
    return any(token in name for token in ("vit", "swin", "dino", "clip", "deit", "beit", "eva", "siglip"))


def _replace_with_model_config(spec: TrainingSpec, **changes: Any) -> TrainingSpec:
    raw_model_config = dict(spec.raw_model_config)
    for key, value in changes.items():
        if key in {
            "optimizer",
            "learning_rate",
            "augmentation",
            "finetune_strategy",
            "freeze_backbone",
            "unfreeze_last_n_blocks",
            "train_norm_layers",
            "loss",
            "backbone",
            "checkpoint",
            "tta",
        }:
            raw_model_config[key] = value
    return replace(spec, raw_model_config=raw_model_config, **changes)


def _slug(value: str) -> str:
    text = "".join(char if char.isalnum() else "_" for char in str(value).lower())
    return "_".join(part for part in text.split("_") if part) or "spec"

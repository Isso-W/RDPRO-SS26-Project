"""Convert Module 3 candidates into Module 4 TrainingSpec objects."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from .schemas import SUPPORTED_TASK_TYPES, TrainingSpec


TASK_TYPE_ALIASES = {
    "classification": "classification",
    "classifier": "classification",
    "image_classification": "classification",
    "object_detection": "object_detection",
    "detection": "object_detection",
    "detector": "object_detection",
    "segmentation": "image_segmentation",
    "image_segmentation": "image_segmentation",
    "semantic_segmentation": "image_segmentation",
    "feature_extraction": "feature_extraction",
    "features": "feature_extraction",
    "embedding": "feature_extraction",
    "retrieval": "feature_extraction",
}

TASK_DEFAULTS = {
    "classification": {
        "head": "classification_head",
        "loss": "cross_entropy_loss",
        "optimizer": "adamw",
        "num_classes": 3,
    },
    "object_detection": {
        "head": "detection_head_anchor_free",
        "loss": "detection_smoke_loss",
        "optimizer": "adamw",
        "num_classes": 3,
    },
    "image_segmentation": {
        "head": "segmentation_head",
        "loss": "cross_entropy_loss",
        "optimizer": "adamw",
        "num_classes": 3,
    },
    "feature_extraction": {
        "head": "embedding_head",
        "loss": "feature_mse_loss",
        "optimizer": "adamw",
        "num_classes": 3,
    },
}


def build_training_specs(candidates: Sequence[Mapping[str, Any]]) -> list[TrainingSpec]:
    """Build normalized specs from a list of Module 3 candidates.

    The parser accepts both structured and natural-language output styles, and
    some examples use older field names.
    """

    if not isinstance(candidates, Sequence) or isinstance(candidates, (str, bytes)):
        raise TypeError("Module 3 output must be a list of candidate dictionaries.")
    if not candidates:
        raise ValueError("Module 3 output must contain at least one candidate.")

    specs: list[TrainingSpec] = []
    for index, candidate_like in enumerate(candidates, start=1):
        candidate = dict(candidate_like or {}) if isinstance(candidate_like, Mapping) else {}
        model_config = dict(candidate.get("model_config") or {})
        task_overrides = _extract_structured_task_fields(candidate.get("tasks"))
        merged = {**task_overrides, **model_config}
        recipe = merged.get("recipe") if isinstance(merged.get("recipe"), Mapping) else {}

        task_type = _normalize_task_type(
            merged.get("task_type")
            or candidate.get("task_type")
            or candidate.get("task_family")
            or candidate.get("family")
            or _infer_task_type(candidate, merged)
        )
        defaults = TASK_DEFAULTS[task_type]

        raw_strategy = (
            merged.get("finetune_strategy")
            or merged.get("strategy")
            or candidate.get("finetune_strategy")
        )
        has_pretrained = bool(
            merged.get("pretrained_hf_id")
            or merged.get("hf_id")
            or candidate.get("pretrained_hf_id")
        )
        if raw_strategy is None or str(raw_strategy).lower() == "none":
            finetune_strategy = "head_only" if has_pretrained else "full"
        else:
            finetune_strategy = str(raw_strategy).lower()
        if finetune_strategy not in {"head_only", "partial", "full", "either"}:
            finetune_strategy = "head_only" if has_pretrained else "full"

        freeze_backbone = _safe_bool(
            merged.get("freeze_backbone"),
            default=(finetune_strategy == "head_only"),
        )
        if finetune_strategy == "head_only":
            freeze_backbone = True
        elif finetune_strategy == "partial":
            freeze_backbone = False
        elif finetune_strategy == "full" and "freeze_backbone" not in merged:
            freeze_backbone = False

        unfreeze_last_n_blocks = _safe_int(
            merged.get("unfreeze_last_n_blocks")
            or merged.get("unfreeze_last_n_layers")
            or merged.get("unfreeze_last_k")
            or candidate.get("unfreeze_last_n_blocks"),
            default=2 if finetune_strategy == "partial" else 0,
        )
        train_norm_layers = _safe_bool(
            merged.get("train_norm_layers"),
            default=(finetune_strategy == "partial"),
        )

        backbone = (
            merged.get("backbone")
            or candidate.get("backbone")
            or candidate.get("model_id")
            or merged.get("finetune_base")
            or "tiny_cnn"
        )
        constraints = dict(candidate.get("constraints") or {}) if isinstance(candidate.get("constraints"), Mapping) else {}
        class_imbalance_value = merged.get("class_imbalance")
        if class_imbalance_value is None:
            class_imbalance_value = constraints.get("class_imbalance")
        checkpoint = (
            merged.get("checkpoint")
            or merged.get("checkpoint_id")
            or merged.get("pretrained_hf_id")
            or merged.get("hf_id")
            or candidate.get("checkpoint")
            or ""
        )

        spec = TrainingSpec(
            rank=_safe_int(candidate.get("rank"), default=index),
            score=_safe_float(candidate.get("score"), default=0.0),
            task_type=task_type,
            backbone=str(backbone),
            pretrained_hf_id=str(
                merged.get("pretrained_hf_id")
                or merged.get("hf_id")
                or candidate.get("pretrained_hf_id")
                or ""
            ),
            pretrained_name=str(
                merged.get("pretrained_name")
                or merged.get("model_name")
                or candidate.get("pretrained_weights")
                or ""
            ),
            head=str(merged.get("head") or defaults["head"]),
            loss=str(merged.get("loss") or defaults["loss"]),
            optimizer=str(merged.get("optimizer") or defaults["optimizer"]),
            finetune_strategy=finetune_strategy,
            freeze_backbone=freeze_backbone,
            unfreeze_last_n_blocks=unfreeze_last_n_blocks,
            train_norm_layers=train_norm_layers,
            strategy_ablation_group=str(merged.get("strategy_ablation_group") or ""),
            strategy_ablation_variant=str(merged.get("strategy_ablation_variant") or ""),
            scratch_viable=_safe_bool(merged.get("scratch_viable"), default=True),
            params_M=_safe_optional_float(merged.get("params_M")),
            tasks=list(candidate.get("tasks") or []),
            alternatives=list(candidate.get("alternatives") or []),
            learning_rate=_safe_float(
                _first_present(
                    merged.get("learning_rate"),
                    merged.get("lr"),
                    recipe.get("learning_rate"),
                ),
                default=1.0e-3,
            ),
            augmentation=_first_present(
                merged.get("augmentation"),
                constraints.get("augmentation"),
                candidate.get("augmentation"),
                recipe.get("augmentation"),
                "basic",
            ),
            data_size=str(
                merged.get("data_size")
                or constraints.get("data_size")
                or candidate.get("data_size")
                or "medium"
            ).lower(),
            class_imbalance=_safe_bool(class_imbalance_value, default=False),
            checkpoint=str(checkpoint),
            num_classes=_safe_int(
                merged.get("num_classes") or candidate.get("num_classes"),
                default=int(defaults["num_classes"]),
            ),
            embedding_dim=_safe_int(merged.get("embedding_dim"), default=32),
            image_size=_safe_int(
                _first_present(
                    merged.get("image_size"),
                    merged.get("input_size"),
                    recipe.get("image_size"),
                    candidate.get("default_input_size"),
                ),
                default=224,
            ),
            offline_smoke=_safe_bool(merged.get("offline_smoke"), default=True),
            use_pretrained=_safe_bool(
                merged.get("use_pretrained"),
                default=bool(
                    merged.get("pretrained_hf_id")
                    or merged.get("hf_id")
                    or candidate.get("pretrained_hf_id")
                ),
            ),
            raw_model_config=model_config,
        )
        specs.append(spec)
    return _expand_strategy_ablations(specs)


def _expand_strategy_ablations(specs: Sequence[TrainingSpec]) -> list[TrainingSpec]:
    """Add lightweight strategy variants for backbones whose best unfreeze depth is data-sensitive."""

    expanded: list[TrainingSpec] = []
    added_variant = False
    for spec in specs:
        if not spec.strategy_ablation_group and _should_auto_ablate_strategy(spec):
            group = f"rank{spec.rank}_{_slug(spec.backbone)}_strategy"
            base_variant = f"partial_last{max(1, spec.unfreeze_last_n_blocks or 2)}"
            spec = _replace_with_model_config(
                spec,
                strategy_ablation_group=group,
                strategy_ablation_variant=base_variant,
            )
        expanded.append(spec)

        if not _should_auto_ablate_strategy(spec):
            continue
        existing_depths = {
            item.unfreeze_last_n_blocks
            for item in expanded
            if item.strategy_ablation_group == spec.strategy_ablation_group
        }
        next_rank = max((item.rank for item in expanded), default=spec.rank)
        for depth in (2, 4):
            if depth in existing_depths:
                continue
            next_rank += 1
            expanded.append(
                _replace_with_model_config(
                    spec,
                    rank=next_rank,
                    finetune_strategy="partial",
                    freeze_backbone=False,
                    unfreeze_last_n_blocks=depth,
                    train_norm_layers=True,
                    strategy_ablation_variant=f"partial_last{depth}",
                )
            )
            added_variant = True
            existing_depths.add(depth)
    if added_variant:
        return [replace(item, rank=index) for index, item in enumerate(expanded, start=1)]
    return expanded


def _should_auto_ablate_strategy(spec: TrainingSpec) -> bool:
    if str(spec.raw_model_config.get("auto_strategy_ablation", "true")).lower() in {"0", "false", "no", "off"}:
        return False
    return (
        spec.task_type == "classification"
        and spec.backbone.lower() == "dinov3"
        and spec.finetune_strategy == "partial"
    )


def _replace_with_model_config(spec: TrainingSpec, **changes: Any) -> TrainingSpec:
    raw_model_config = dict(spec.raw_model_config)
    for key, value in changes.items():
        if key in {
            "finetune_strategy",
            "freeze_backbone",
            "unfreeze_last_n_blocks",
            "train_norm_layers",
            "strategy_ablation_group",
            "strategy_ablation_variant",
        }:
            raw_model_config[key] = value
    return replace(spec, raw_model_config=raw_model_config, **changes)


def _slug(value: str) -> str:
    text = str(value or "item").lower()
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "item"


def _first_present(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def specs_to_configs(specs: Sequence[TrainingSpec]) -> list[dict[str, Any]]:
    """Serialize specs for embedding into generated code."""

    return [spec.to_config() for spec in specs]


def _extract_structured_task_fields(tasks: Any) -> dict[str, Any]:
    """Extract model fields from Module 3 structured task lists if present."""

    extracted: dict[str, Any] = {}
    if not isinstance(tasks, list):
        return extracted
    for item in tasks:
        if not isinstance(item, Mapping):
            continue
        action = str(item.get("action") or "").lower()
        if action in {"load_pretrained", "train_from_scratch"}:
            if item.get("hf_id"):
                extracted["pretrained_hf_id"] = item.get("hf_id")
            if item.get("model_name"):
                extracted["pretrained_name"] = item.get("model_name")
            if item.get("params_M") is not None:
                extracted["params_M"] = item.get("params_M")
            if item.get("finetune_base"):
                extracted["backbone"] = item.get("finetune_base")
            if item.get("backbone"):
                extracted["backbone"] = item.get("backbone")
        elif action == "set_finetune_strategy":
            extracted["finetune_strategy"] = item.get("strategy")
            extracted["freeze_backbone"] = item.get("freeze_backbone")
            extracted["scratch_viable"] = item.get("scratch_viable")
            for key in ("unfreeze_last_n_blocks", "unfreeze_last_n_layers", "unfreeze_last_k"):
                if item.get(key) is not None:
                    extracted["unfreeze_last_n_blocks"] = item.get(key)
                    break
            if item.get("train_norm_layers") is not None:
                extracted["train_norm_layers"] = item.get("train_norm_layers")
        elif action == "configure_head":
            extracted["head"] = item.get("type") or item.get("name")
        elif action == "configure_loss":
            extracted["loss"] = item.get("type") or item.get("name")
        elif action == "configure_optimizer":
            extracted["optimizer"] = item.get("type") or item.get("name")
    return extracted


def _normalize_task_type(value: Any) -> str:
    key = str(value or "classification").strip().lower()
    normalized = TASK_TYPE_ALIASES.get(key, key)
    if normalized not in SUPPORTED_TASK_TYPES:
        return "classification"
    return normalized


def _infer_task_type(candidate: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    text_parts: list[str] = []
    for key in ("head", "loss", "backbone", "pretrained_name"):
        if config.get(key):
            text_parts.append(str(config.get(key)))
    for task in candidate.get("tasks") or []:
        text_parts.append(str(task))
    text = " ".join(text_parts).lower()
    if any(token in text for token in ("detect", "detection", "yolo", "box")):
        return "object_detection"
    if any(token in text for token in ("segment", "segmentation", "mask")):
        return "image_segmentation"
    if any(token in text for token in ("embedding", "feature", "retrieval", "clip")):
        return "feature_extraction"
    return "classification"


def _safe_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return default


def _safe_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_float(value: Any, default: float) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

"""Recipe layer orchestration."""

from __future__ import annotations

from typing import Any

from .augment import resolve_augmentation
from .tables import (
    CHECKPOINT_IMAGE_DEFAULT,
    FAMILY_IMAGE_DEFAULT,
    IMAGE_DIVISOR,
    LR_BASE,
    derive_recommended_epochs,
    early_stopping_patience,
    family_class,
    snap_image_size,
    training_mode,
)


def _safe_bool(value: Any, default: bool = False) -> bool:
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


def _checkpoint_id(config: dict, backbone_facts: dict | None) -> str | None:
    if config.get("checkpoint"):
        return str(config["checkpoint"])
    facts = backbone_facts or {}
    cp = facts.get("checkpoint")
    if isinstance(cp, dict) and cp.get("id"):
        return str(cp["id"])
    return None


def _checkpoint_default_size(config: dict, backbone_facts: dict | None) -> tuple[int, str]:
    cid = _checkpoint_id(config, backbone_facts)
    if cid and cid in CHECKPOINT_IMAGE_DEFAULT:
        return CHECKPOINT_IMAGE_DEFAULT[cid], f"ckpt_default[{cid}]={CHECKPOINT_IMAGE_DEFAULT[cid]}"

    facts = backbone_facts or {}
    cp = facts.get("checkpoint")
    if isinstance(cp, dict):
        for key in ("expected_image_size", "image_size", "default_input_size"):
            if cp.get(key):
                return int(cp[key]), f"ckpt_node[{key}]={int(cp[key])}"

    family = str(config.get("backbone") or "").lower()
    if family in FAMILY_IMAGE_DEFAULT:
        return FAMILY_IMAGE_DEFAULT[family], f"family_default[{family}]={FAMILY_IMAGE_DEFAULT[family]}"
    return 224, "fallback_default=224"


def _image_divisor(config: dict) -> int | None:
    cid = str(config.get("checkpoint") or "")
    if cid in IMAGE_DIVISOR:
        return IMAGE_DIVISOR[cid]
    family = str(config.get("backbone") or "").lower()
    return IMAGE_DIVISOR.get(family)


def _resolve_image_size(config: dict, input_json: dict, backbone_facts: dict | None,
                        data_stats: dict | None) -> tuple[int, str]:
    data_stats = data_stats or {}
    constraints = input_json.get("constraints", {}) if isinstance(input_json, dict) else {}
    data_size = str(input_json.get("data_size") or config.get("data_size") or "medium").lower()
    priority = str(input_json.get("priority") or "balanced").lower()

    size, note = _checkpoint_default_size(config, backbone_facts)
    notes = [note]
    if data_stats.get("resolution_tier") == "high" and constraints.get("fine_grained"):
        if priority != "speed" and data_size != "large":
            bumped = 384 if size <= 224 else max(size, 384)
            if bumped != size:
                notes.append(f"fine_grained+high_res bump: {size}→{bumped}")
                size = bumped
        else:
            notes.append("high_res bump vetoed by speed_or_large_data")
    elif "resolution_tier" not in data_stats:
        notes.append("signal_missing: resolution_tier")

    divisor = _image_divisor(config)
    snapped = snap_image_size(size, divisor)
    if divisor and snapped != size:
        notes.append(f"snapped /{divisor}: {size}→{snapped}")
    elif divisor:
        notes.append(f"already divisible /{divisor}")
    return snapped, " | ".join(notes)


def build_recipe(
    config: dict,
    input_json: dict,
    backbone_facts: dict,
    data_stats: dict | None = None,
) -> tuple[dict, dict]:
    """Build model-coupled hyperparameters and provenance.

    v0 targets classification. Other task types receive an empty recipe so
    existing detection/segmentation smoke paths remain unchanged.
    """
    input_json = input_json or {}
    task_type = str(input_json.get("task_type") or config.get("task_type") or "classification")
    if task_type != "classification":
        return {}, {"status": f"unsupported_task:{task_type}"}

    constraints = dict(input_json.get("constraints") or {})
    data_stats = data_stats or input_json.get("data_stats") or {}
    data_size = str(input_json.get("data_size") or config.get("data_size") or "medium").lower()
    finetune_strategy = config.get("finetune_strategy")
    use_pretrained = _safe_bool(
        config.get("use_pretrained"),
        default=bool(config.get("pretrained_hf_id") or config.get("checkpoint")),
    )
    mode = training_mode(str(finetune_strategy or ""), use_pretrained)
    family = family_class(config.get("backbone"))

    epochs = derive_recommended_epochs(data_size, str(finetune_strategy or ""), use_pretrained)
    lr = LR_BASE.get((family, mode), 1.0e-3)
    image_size, image_src = _resolve_image_size(config, input_json, backbone_facts, data_stats)
    augmentation, aug_src = resolve_augmentation(
        data_size=data_size,
        finetune_strategy=str(finetune_strategy or ""),
        constraints=constraints,
        data_stats=data_stats,
    )

    recipe = {
        "image_size": image_size,
        "learning_rate": lr,
        "epochs": epochs,
        "early_stopping_patience": early_stopping_patience(data_size),
        "augmentation": augmentation,
    }
    provenance = {
        "image_size": image_src,
        "learning_rate": f"lr_base[{family},{mode}]",
        "epochs": f"epochs_table[{data_size},{mode}]",
        "early_stopping_patience": f"early_stopping[{data_size}]",
        "augmentation": aug_src,
    }
    return recipe, provenance

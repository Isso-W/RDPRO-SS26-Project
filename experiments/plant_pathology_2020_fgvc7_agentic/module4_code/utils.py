"""Utility helpers for generated scripts."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import torch


SUPPORTED_TASK_TYPES = {
    "classification",
    "object_detection",
    "image_segmentation",
    "feature_extraction",
}


def get_value(config: dict[str, Any] | None, key: str, default: Any) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    return default


def as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y"}:
            return True
        if lowered in {"0", "false", "no", "n"}:
            return False
    return default


def task_type(config: dict[str, Any] | None) -> str:
    task = str(get_value(config, "task_type", "classification")).lower()
    task = {
        "detection": "object_detection",
        "segmentation": "image_segmentation",
        "semantic_segmentation": "image_segmentation",
        "features": "feature_extraction",
        "embedding": "feature_extraction",
    }.get(task, task)
    if task not in SUPPORTED_TASK_TYPES:
        return "classification"
    return task


def normalize_config(item: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(item, dict):
        return {}
    config = dict(item)
    model_config = config.get("model_config")
    if isinstance(model_config, dict):
        merged = dict(config)
        for key, value in model_config.items():
            if value is not None or key not in merged:
                merged[key] = value
        config = merged
    return config


def load_config(path: str | None, default_config: dict[str, Any]) -> dict[str, Any]:
    if not path:
        return normalize_config(default_config)
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        if not data:
            raise ValueError("Config list is empty.")
        return normalize_config(data[0])
    if isinstance(data, dict) and isinstance(data.get("candidates"), list):
        if not data["candidates"]:
            raise ValueError("Candidate list is empty.")
        return normalize_config(data["candidates"][0])
    if isinstance(data, dict):
        return normalize_config(data)
    raise ValueError("Config file must contain a dict, a list, or {'candidates': [...]}.")


def load_configs(path: str | None, default_configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not path:
        return [normalize_config(item) for item in default_configs]
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("candidates"), list):
        data = data["candidates"]
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("Experiment input must be a list, dict, or {'candidates': [...]}.")
    return [normalize_config(item) for item in data]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def compact_config_summary(config: dict[str, Any], rank_default: int | None = None) -> dict[str, Any]:
    return {
        "rank": config.get("rank", rank_default),
        "backbone": config.get("backbone", "tiny_cnn"),
        "task_type": config.get("task_type", "classification"),
        "loss": config.get("loss", ""),
        "optimizer": config.get("optimizer", ""),
        "finetune_strategy": config.get("finetune_strategy", ""),
        "unfreeze_last_n_blocks": config.get("unfreeze_last_n_blocks", 0),
        "train_norm_layers": config.get("train_norm_layers", True),
        "strategy_ablation_group": config.get("strategy_ablation_group", ""),
        "strategy_ablation_variant": config.get("strategy_ablation_variant", ""),
        "tta": config.get("tta", False),
    }

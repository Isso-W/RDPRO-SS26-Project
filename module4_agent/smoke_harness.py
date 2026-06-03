"""Synthetic data helpers for Module 4 smoke testing."""

from __future__ import annotations

from typing import Any

import torch


def synthetic_batch(
    task_type: str,
    *,
    batch_size: int = 2,
    channels: int = 3,
    image_size: int = 224,
    num_classes: int = 3,
) -> tuple[Any, Any]:
    """Create a task-specific synthetic batch.

    Shapes follow the Module 4 smoke-test contract:
    classification uses x=(2,3,224,224), y=(2,);
    segmentation uses x=(2,3,H,W), mask=(2,H,W);
    detection uses image tensors and target dictionaries;
    feature extraction uses image tensors and labels for placeholder metrics.
    """

    task = _normalize_task_type(task_type)
    x = torch.randn(batch_size, channels, image_size, image_size)
    num_classes = max(1, int(num_classes))
    if task == "classification":
        labels = torch.arange(batch_size, dtype=torch.long) % num_classes
        return x, labels
    if task == "image_segmentation":
        mask = torch.randint(0, num_classes, (batch_size, image_size, image_size), dtype=torch.long)
        return x, mask
    if task == "object_detection":
        targets = []
        for idx in range(batch_size):
            targets.append(
                {
                    "boxes": torch.tensor([[0.1, 0.1, 0.8, 0.8]], dtype=torch.float32),
                    "class_labels": torch.tensor([idx % num_classes], dtype=torch.long),
                }
            )
        return x, targets
    if task == "feature_extraction":
        return x, torch.zeros(batch_size, dtype=torch.long)
    labels = torch.arange(batch_size, dtype=torch.long) % num_classes
    return x, labels


def smoke_config(task_type: str, **overrides: Any) -> dict[str, Any]:
    """Return a small config dictionary suitable for generated smoke code."""

    config: dict[str, Any] = {
        "rank": 1,
        "task_type": _normalize_task_type(task_type),
        "backbone": "tiny_cnn",
        "loss": "cross_entropy_loss",
        "optimizer": "adamw",
        "finetune_strategy": "head_only",
        "freeze_backbone": True,
        "num_classes": 3,
        "image_size": 224,
        "offline_smoke": True,
        "use_pretrained": False,
    }
    config.update(overrides)
    return config


def _normalize_task_type(task_type: str) -> str:
    task = str(task_type or "classification").lower()
    return {
        "detection": "object_detection",
        "segmentation": "image_segmentation",
        "features": "feature_extraction",
        "embedding": "feature_extraction",
    }.get(task, task)


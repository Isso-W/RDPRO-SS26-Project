"""Synthetic data helpers for local smoke runs."""

from __future__ import annotations

from typing import Any

import torch

from utils import as_int, get_value, task_type


def synthetic_batch(config: dict[str, Any] | None, batch_size: int = 2) -> tuple[Any, Any]:
    """Create a synthetic batch for the configured task."""

    task = task_type(config)
    image_size = as_int(get_value(config, "image_size", 224), 224)
    num_classes = max(1, as_int(get_value(config, "num_classes", 3), 3))
    x = synthetic_image(config, batch_size=batch_size)
    if task == "classification":
        return x, torch.arange(batch_size, dtype=torch.long) % num_classes
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
    return x, torch.arange(batch_size, dtype=torch.long) % num_classes


def synthetic_image(config: dict[str, Any] | None, batch_size: int = 1) -> torch.Tensor:
    image_size = as_int(get_value(config, "image_size", 224), 224)
    return torch.randn(batch_size, 3, image_size, image_size)

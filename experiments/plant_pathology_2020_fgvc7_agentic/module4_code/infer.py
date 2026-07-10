"""Inference entry point for generated configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from model import build_model
from smoke_data import synthetic_image
from train import _classification_logits
from utils import task_type


def predict(weights_path: str | None = None, image: torch.Tensor | None = None, config: dict[str, Any] | None = None, model: torch.nn.Module | None = None) -> dict[str, Any]:
    """Run one forward pass and return a JSON-friendly prediction."""

    config = config or {}
    task = task_type(config)
    if model is None:
        model = build_model(config)
        if weights_path and Path(weights_path).exists():
            checkpoint = torch.load(weights_path, map_location="cpu")
            state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            model.load_state_dict(state_dict, strict=False)
    device = next(model.parameters()).device
    model.eval()
    if image is None:
        image = synthetic_image(config, batch_size=1)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        output = _classification_logits(model, image, config) if task == "classification" else model(image)

    if task == "classification":
        probs = output.softmax(dim=1)
        return {
            "task_type": task,
            "class_id": int(probs.argmax(dim=1)[0].item()),
            "confidence": float(probs.max(dim=1).values[0].item()),
        }
    if task == "image_segmentation":
        mask = output.argmax(dim=1)
        return {
            "task_type": task,
            "mask_shape": list(mask.shape),
            "unique_labels": sorted(int(value) for value in mask.unique().tolist()),
        }
    if task == "object_detection":
        pred_logits = output["pred_logits"][0]
        pred_boxes = output["pred_boxes"][0]
        scores = pred_logits.softmax(dim=-1).max(dim=-1).values
        labels = pred_logits.argmax(dim=-1)
        return {
            "task_type": task,
            "boxes": pred_boxes.cpu().tolist(),
            "labels": labels.cpu().tolist(),
            "scores": scores.cpu().tolist(),
        }
    if task == "feature_extraction":
        return {
            "task_type": task,
            "embedding_shape": list(output.shape),
            "embedding_preview": output[0, : min(5, output.shape[1])].cpu().tolist(),
        }
    return {"task_type": task, "status": "success"}

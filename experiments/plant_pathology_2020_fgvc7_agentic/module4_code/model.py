"""Task-specific model builders.

When offline_smoke is true (default), models use a lightweight TinyBackbone
for fast CPU checks.  When offline_smoke is false, model_utils.load_backbone
loads the real pretrained checkpoint chosen by Module 3.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from utils import as_bool, as_int, get_value, task_type


class TinyBackbone(nn.Module):
    """Small CNN backbone for smoke runs."""

    def __init__(self, in_channels: int = 3, width: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, width // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width // 2, width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out_channels = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ClassificationModel(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module | None = None, out_channels: int = 16) -> None:
        super().__init__()
        self.backbone = backbone if backbone is not None else TinyBackbone()
        _ch = out_channels if backbone is not None else self.backbone.out_channels
        self.head = nn.Linear(_ch, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        return self.head(features)


class SegmentationModel(nn.Module):
    def __init__(self, num_classes: int, backbone: nn.Module | None = None, out_channels: int = 16) -> None:
        super().__init__()
        self.backbone = backbone if backbone is not None else TinyBackbone()
        _ch = out_channels if backbone is not None else self.backbone.out_channels
        self.head = nn.Conv2d(_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() == 2:
            warnings.warn("Backbone returns pooled [B,D] features; segmentation needs spatial output.")
            return torch.zeros(x.shape[0], self.head.out_channels, x.shape[2], x.shape[3],
                               device=x.device, requires_grad=True)
        logits = self.head(features)
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(logits, size=x.shape[-2:], mode="bilinear", align_corners=False)
        return logits


class DetectionModel(nn.Module):
    """Minimal detector that returns DETR-like outputs."""

    def __init__(self, num_classes: int, backbone: nn.Module | None = None, out_channels: int = 16) -> None:
        super().__init__()
        self.backbone = backbone if backbone is not None else TinyBackbone()
        _ch = out_channels if backbone is not None else self.backbone.out_channels
        self.box_head = nn.Linear(_ch, 4)
        self.class_head = nn.Linear(_ch, num_classes)

    def forward(self, x: torch.Tensor, targets: list[dict[str, torch.Tensor]] | None = None) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        pred_boxes = torch.sigmoid(self.box_head(features)).unsqueeze(1)
        pred_logits = self.class_head(features).unsqueeze(1)
        output: dict[str, torch.Tensor] = {
            "pred_boxes": pred_boxes,
            "pred_logits": pred_logits,
        }
        if targets is not None:
            target_boxes = []
            target_classes = []
            for item in targets:
                boxes = item.get("boxes")
                labels = item.get("class_labels", item.get("labels"))
                if boxes is None or boxes.numel() == 0:
                    target_boxes.append(torch.zeros(4, device=x.device))
                else:
                    target_boxes.append(boxes.to(x.device).float()[0])
                if labels is None or labels.numel() == 0:
                    target_classes.append(torch.tensor(0, device=x.device, dtype=torch.long))
                else:
                    target_classes.append(labels.to(x.device).long()[0])
            target_box_tensor = torch.stack(target_boxes, dim=0)
            target_class_tensor = torch.stack(target_classes, dim=0)
            cls_loss = F.cross_entropy(pred_logits[:, 0, :], target_class_tensor)
            box_loss = F.l1_loss(pred_boxes[:, 0, :], target_box_tensor)
            output["loss"] = cls_loss + box_loss
        return output


class FeatureExtractorModel(nn.Module):
    def __init__(self, embedding_dim: int, backbone: nn.Module | None = None, out_channels: int = 16) -> None:
        super().__init__()
        self.backbone = backbone if backbone is not None else TinyBackbone()
        _ch = out_channels if backbone is not None else self.backbone.out_channels
        self.head = nn.Linear(_ch, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        embeddings = self.head(features)
        return F.normalize(embeddings, dim=1)


def _set_backbone_requires_grad(model: nn.Module, requires_grad: bool) -> int:
    changed = 0
    for name, parameter in model.named_parameters():
        if "backbone" in name:
            if parameter.requires_grad != requires_grad:
                changed += 1
            parameter.requires_grad = requires_grad
    return changed


def _unfreeze_last_backbone_blocks(model: nn.Module, last_n: int) -> int:
    if last_n <= 0:
        return 0
    backbone = getattr(model, "backbone", None)
    if not isinstance(backbone, nn.Module):
        return 0
    containers = []
    for attr in ("blocks", "layers", "features", "net"):
        candidate = getattr(backbone, attr, None)
        if isinstance(candidate, (nn.ModuleList, nn.Sequential)) and len(candidate) > 0:
            containers.append(candidate)
    if not containers:
        children = list(backbone.children())
        if children:
            containers.append(nn.Sequential(*children))
    changed = 0
    if containers:
        for block in list(containers[0].children())[-last_n:]:
            for parameter in block.parameters():
                if not parameter.requires_grad:
                    changed += 1
                parameter.requires_grad = True
    return changed


def _unfreeze_module(module: nn.Module) -> int:
    changed = 0
    for parameter in module.parameters():
        if not parameter.requires_grad:
            changed += 1
        parameter.requires_grad = True
    return changed


def _unfreeze_backbone_norm_layers(model: nn.Module) -> int:
    norm_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
        nn.LayerNorm,
        nn.LocalResponseNorm,
    )
    changed = 0
    for module_name, module in model.named_modules():
        if "backbone" in module_name and isinstance(module, norm_types):
            changed += _unfreeze_module(module)
    return changed


def _apply_finetune_strategy(model: nn.Module, config: dict[str, Any] | None) -> nn.Module:
    strategy = str(get_value(config, "finetune_strategy", "head_only")).lower()
    frozen = 0
    unfrozen = 0
    last_n = 0
    if strategy == "head_only":
        frozen = _set_backbone_requires_grad(model, False)
    elif strategy == "partial":
        frozen = _set_backbone_requires_grad(model, False)
        last_n = max(0, as_int(get_value(config, "unfreeze_last_n_blocks", 2), 2))
        unfrozen = _unfreeze_last_backbone_blocks(model, last_n)
        if as_bool(get_value(config, "train_norm_layers", True), True):
            unfrozen += _unfreeze_backbone_norm_layers(model)
        if last_n > 0 and unfrozen == 0:
            unfrozen = _set_backbone_requires_grad(model, True)
    elif strategy in ("full", "either"):
        _set_backbone_requires_grad(model, True)
    else:
        freeze_backbone = as_bool(
            get_value(config, "freeze_backbone", False),
            False,
        )
        if freeze_backbone:
            frozen = _set_backbone_requires_grad(model, False)
    model._frozen_backbone_params = frozen
    model._partial_unfrozen_params = unfrozen
    model._unfreeze_last_n_blocks = last_n
    return model


def build_model(config: dict[str, Any] | None) -> nn.Module:
    """Build a task-compatible model from a config dictionary.

    When offline_smoke is true, uses TinyBackbone for fast CPU checks.
    When false, loads the real backbone via model_utils.load_backbone.
    """
    config = config or {}
    task = task_type(config)
    num_classes = max(1, as_int(get_value(config, "num_classes", 3), 3))
    embedding_dim = max(2, as_int(get_value(config, "embedding_dim", 32), 32))
    offline_smoke = as_bool(get_value(config, "offline_smoke", True), True)

    if offline_smoke:
        backbone = None
        out_channels = 16
    else:
        from model_utils import load_backbone
        backbone, out_channels = load_backbone(config)

    if task == "classification":
        model = ClassificationModel(num_classes, backbone, out_channels)
    elif task == "object_detection":
        model = DetectionModel(num_classes, backbone, out_channels)
    elif task == "image_segmentation":
        model = SegmentationModel(num_classes, backbone, out_channels)
    elif task == "feature_extraction":
        model = FeatureExtractorModel(embedding_dim, backbone, out_channels)
    else:
        model = ClassificationModel(num_classes, backbone, out_channels)

    if offline_smoke:
        _apply_finetune_strategy(model, config)
    else:
        from model_utils import apply_freeze
        apply_freeze(model, config)
    return model

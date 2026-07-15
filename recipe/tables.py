"""Frozen lookup tables for the recipe layer.

The values are v0 defaults. They are intentionally centralized so future
kb_mining recipes.json or A/B results can override a small table instead of
scattering hyperparameter constants through the pipeline.
"""

from __future__ import annotations

import math


RECOMMENDED_EPOCHS = {
    ("small", "head_only"): 25,
    ("small", "finetune"): 40,
    ("small", "scratch"): 50,
    ("medium", "head_only"): 12,
    ("medium", "finetune"): 20,
    ("medium", "scratch"): 30,
    ("large", "head_only"): 8,
    ("large", "finetune"): 15,
    ("large", "scratch"): 20,
}

EARLY_STOPPING_PATIENCE = {
    "small": 3,
    "medium": 5,
    "large": 8,
}

TRANSFORMER_TOKENS = ("vit", "swin", "dino", "clip", "deit", "beit", "eva")

LR_BASE = {
    ("cnn", "head_only"): 1.0e-3,
    ("cnn", "finetune"): 1.0e-4,
    ("cnn", "scratch"): 5.0e-4,
    ("transformer", "head_only"): 1.0e-3,
    ("transformer", "finetune"): 3.0e-5,
    ("transformer", "scratch"): 3.0e-4,
}

FAMILY_IMAGE_DEFAULT = {
    "resnet": 224,
    "efficientnet": 224,
    "mobilenet_v3": 224,
    "vit": 224,
    "swin_transformer": 224,
    "convnext": 224,
    "dinov2": 224,
    "dinov3": 224,
    "clip_vit": 224,
    "yolov8": 640,
    "detr": 800,
    "rt_detr": 640,
    "segformer": 512,
    "mask2former": 512,
    "unet": 512,
}

CHECKPOINT_IMAGE_DEFAULT = {
    "efficientnet_b0_imagenet": 224,
    "efficientnet_lite0": 224,
    "resnet18_imagenet": 224,
    "resnet50_imagenet": 224,
    "mobilenet_v3_imagenet": 224,
    "vit_base_in21k": 224,
    "vit_large_in21k": 224,
    "swin_base_in22k": 224,
    "swin_large_in22k": 224,
    "convnext_base_in22k": 224,
    "convnext_large_in22k": 224,
    "dinov2_base": 224,
    "dinov2_large": 224,
    "dinov3_small_lvd1689m": 224,
    "dinov3_base_lvd1689m": 224,
    "dinov3_large_lvd1689m": 224,
    "clip_vit_base_32": 224,
    "clip_vit_large_14": 224,
    "segformer_b0_ade": 512,
    "segformer_b2_ade": 512,
    "segformer_b5_ade": 640,
}

IMAGE_DIVISOR = {
    "dinov2": 14,
    "dinov2_base": 14,
    "dinov2_large": 14,
    "dinov3": 16,
    "dinov3_small_lvd1689m": 16,
    "dinov3_base_lvd1689m": 16,
    "dinov3_large_lvd1689m": 16,
    "vit": 16,
    "vit_base_in21k": 16,
    "vit_large_in21k": 16,
    "swin_transformer": 32,
    "swin_base_in22k": 32,
    "swin_large_in22k": 32,
}


def training_mode(finetune_strategy: str | None, use_pretrained: bool) -> str:
    if not use_pretrained:
        return "scratch"
    if finetune_strategy == "head_only":
        return "head_only"
    return "finetune"


def derive_recommended_epochs(
    data_size: str,
    finetune_strategy: str | None,
    use_pretrained: bool,
) -> int:
    """Recommend epochs using the legacy table, now owned by recipe."""
    mode = training_mode(finetune_strategy, use_pretrained)
    return RECOMMENDED_EPOCHS.get((str(data_size or "medium").lower(), mode), 15)


def early_stopping_patience(data_size: str | None) -> int:
    """Return the validation patience associated with a dataset-size tier."""
    return EARLY_STOPPING_PATIENCE.get(str(data_size or "medium").lower(), 5)


def family_class(backbone: str | None) -> str:
    lowered = str(backbone or "").lower()
    return "transformer" if any(token in lowered for token in TRANSFORMER_TOKENS) else "cnn"


def snap_image_size(size: int, divisor: int | None) -> int:
    """Snap upward to the next valid divisor multiple without reducing detail."""
    if not divisor or divisor <= 1:
        return int(size)
    return int(math.ceil(int(size) / divisor) * divisor)

"""tables.py — recipe 层的全部冻结查表（唯一数据源）。

值为 v0 手工默认；标注了哪些可被 kb_mining 的 recipes.json / A/B 结果覆盖
（校准 seam，见 recipe_layer_plan.md §5）。纯数据 + 极简派生 helper。
"""

from __future__ import annotations

# ═══ epochs（自 pipeline.py 收编的孤儿，键/值不变，回归测试保证）═══════════════
_RECOMMENDED_EPOCHS: dict[tuple[str, str], int] = {
    ("small",  "head_only"): 25,
    ("small",  "finetune"):  40,
    ("small",  "scratch"):   50,
    ("medium", "head_only"): 12,
    ("medium", "finetune"):  20,
    ("medium", "scratch"):   30,
    ("large",  "head_only"):  8,
    ("large",  "finetune"):  15,
    ("large",  "scratch"):   20,
}


def training_mode(finetune_strategy: str | None, use_pretrained: bool) -> str:
    """(finetune_strategy, use_pretrained) → mode ∈ {head_only, finetune, scratch}。"""
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
    """Recommend training epochs based on data size and training mode.

    （原 pipeline.derive_recommended_epochs；pipeline 现从此 re-export，保持
    run_kaggle_benchmark 的 import 不破。）
    """
    mode = training_mode(finetune_strategy, use_pretrained)
    return _RECOMMENDED_EPOCHS.get((data_size, mode), 15)


# ═══ backbone 家族分类（lr / 默认分辨率用）═══════════════════════════════════
# 与 retrieval/rag_retrieval.py COMPONENTS 的 14 个 backbone id 对齐。
_FAMILY_CLASS: dict[str, str] = {
    "resnet": "cnn", "efficientnet": "cnn", "mobilenet_v3": "cnn",
    "convnext": "cnn", "unet": "cnn", "yolov8": "cnn",
    "vit": "transformer", "swin_transformer": "transformer",
    "dinov2": "transformer", "clip_vit": "transformer",
    "detr": "transformer", "rt_detr": "transformer",
    "segformer": "transformer", "mask2former": "transformer",
}


def family_class(family: str) -> str:
    return _FAMILY_CLASS.get(family, "cnn")   # 未知家族保守当 cnn


# ═══ image_size ═══════════════════════════════════════════════════════════════
# family 默认输入分辨率（无 checkpoint 期望分辨率时的回退）
_FAMILY_IMAGE_DEFAULT: dict[str, int] = {
    "resnet": 224, "efficientnet": 224, "mobilenet_v3": 224, "convnext": 224,
    "vit": 224, "swin_transformer": 224, "dinov2": 224, "clip_vit": 224,
    "unet": 256, "segformer": 512, "mask2former": 512,
    "yolov8": 640, "detr": 800, "rt_detr": 640,
}
DEFAULT_IMAGE_SIZE = 224

# 硬约束：patch/window 整除要求（image_size 必须吸附到最近合法值）。
# 仅列除数明确、不满足会报错的家族；CNN 接受任意尺寸，不入表。
_IMAGE_DIVISOR: dict[str, int] = {
    "dinov2": 14,            # ViT patch 14
    "vit": 16,               # ViT patch 16
    "swin_transformer": 32,  # 4 级下采样 → 32 整除
}

# fine_grained + 高分辨率时的上调梯度
_IMAGE_BUMP = {224: 384, 256: 384, 384: 512}


def family_image_default(family: str) -> int:
    return _FAMILY_IMAGE_DEFAULT.get(family, DEFAULT_IMAGE_SIZE)


def image_divisor(family: str) -> int | None:
    return _IMAGE_DIVISOR.get(family)


def bump_image(size: int) -> int:
    return _IMAGE_BUMP.get(size, size)


def snap_to_divisor(size: int, divisor: int) -> int:
    """吸附到最近的 divisor 倍数（至少一个 divisor）。"""
    return max(divisor, round(size / divisor) * divisor)


# ═══ learning_rate（family_class × mode）═══════════════════════════════════════
# v0 默认，待 A/B / recipes.json 校准。transformer 微调用更低 LR。
_LR_BASE: dict[tuple[str, str], float] = {
    ("cnn", "head_only"): 1e-3,  ("cnn", "finetune"): 1e-4,  ("cnn", "scratch"): 5e-4,
    ("transformer", "head_only"): 1e-3, ("transformer", "finetune"): 3e-5,
    ("transformer", "scratch"): 3e-4,
}


def lr_base(fam_class: str, mode: str) -> float:
    return _LR_BASE.get((fam_class, mode), 1e-4)

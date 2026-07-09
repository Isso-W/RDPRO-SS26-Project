"""Backbone loading and feature extraction utilities.

Provides load_backbone() for reliable model loading with dynamic dimension
inference, and apply_freeze() for finetune strategy. Used by both LLM-generated
and template model.py.
"""

from __future__ import annotations

import sys
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from utils import as_bool, as_int, get_value


class TinyBackbone(nn.Module):
    """Minimal CNN fallback when torchvision model is unavailable."""

    def __init__(self, width: int = 16) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, width // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(width // 2, width, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out_channels = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _annotate_backbone(
    backbone: nn.Module,
    *,
    source: str,
    requested_backbone: str,
    requested_hf_id: str = "",
    actual_model: str = "",
    fallback_reason: str = "",
) -> nn.Module:
    backbone._jiaozi_load_info = {
        "source": source,
        "requested_backbone": requested_backbone,
        "requested_hf_id": requested_hf_id,
        "actual_model": actual_model or backbone.__class__.__name__,
        "fallback_reason": fallback_reason,
        "backbone_class": backbone.__class__.__name__,
        "feature_pooling": str(getattr(backbone, "feature_pooling", "") or ""),
    }
    return backbone


def backbone_load_info(model: nn.Module, config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the real backbone that was constructed for logging/checkpoints."""
    config = config or {}
    backbone = getattr(model, "backbone", model)
    info = dict(getattr(backbone, "_jiaozi_load_info", {}) or {})
    info.setdefault("source", "unknown")
    info.setdefault("requested_backbone", str(get_value(config, "backbone", "")))
    info.setdefault("requested_hf_id", str(get_value(config, "pretrained_hf_id", "") or ""))
    info.setdefault("actual_model", backbone.__class__.__name__ if backbone is not None else "None")
    info.setdefault("fallback_reason", "")
    info.setdefault("feature_pooling", str(getattr(backbone, "feature_pooling", "") or ""))
    info["backbone_class"] = backbone.__class__.__name__ if backbone is not None else "None"
    info["model_class"] = model.__class__.__name__
    info["total_params"] = int(sum(p.numel() for p in model.parameters()))
    info["trainable_params"] = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return info


_TORCHVISION_MODELS: dict[str, str] = {
    "resnet": "resnet50",
    "resnet18": "resnet18",
    "resnet34": "resnet34",
    "resnet50": "resnet50",
    "resnet101": "resnet101",
    "mobilenet_v3": "mobilenet_v3_small",
    "mobilenetv3": "mobilenet_v3_small",
    "efficientnet": "efficientnet_b0",
    "efficientnet_b0": "efficientnet_b0",
    "efficientnet_b1": "efficientnet_b1",
    "efficientnet_b2": "efficientnet_b2",
    "efficientnet_b3": "efficientnet_b3",
    "convnext": "convnext_tiny",
    "convnext_tiny": "convnext_tiny",
    "regnet": "regnet_y_400mf",
    "vit": "vit_b_16",
    "vit_b_16": "vit_b_16",
    "swin": "swin_t",
    "swin_transformer": "swin_t",
    "swin_t": "swin_t",
}


class _SpatialExtractor(nn.Module):
    """Wraps a model's feature layers to output spatial features [B, C, H', W']."""

    def __init__(self, layers: nn.Module) -> None:
        super().__init__()
        self.layers = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _HFBackbone(nn.Module):
    """Wraps a transformers AutoModel to emit plain feature tensors.

    Transformer encoders return [B, seq, D]. Prefer the model's pooled
    image embedding or CLS token; averaging all tokens can mix DINOv3
    register tokens into the classification feature.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.feature_pooling = "pooler_output_or_cls_token"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(pixel_values=x)
        pooled = getattr(out, "pooler_output", None)
        if pooled is not None:
            return pooled
        hidden = getattr(out, "last_hidden_state", None)
        if hidden is None:
            hidden = out[0] if isinstance(out, (tuple, list)) else out
        if hidden.dim() == 3:
            return hidden[:, 0]
        return hidden


def _try_huggingface(
    hf_id: str,
    image_size: int,
    requested_backbone: str,
) -> tuple[nn.Module, int, str | None] | None:
    """Load the exact HuggingFace checkpoint chosen by Module 3.

    Requires the optional ``transformers`` dependency and network access
    on first download. Returns None on any failure so the caller can
    fall back to torchvision.
    """
    try:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(hf_id)
        backbone = _HFBackbone(model)
        _annotate_backbone(
            backbone,
            source="huggingface",
            requested_backbone=requested_backbone,
            requested_hf_id=hf_id,
            actual_model=hf_id,
        )
        channels = _infer_channels(backbone, image_size)
        return backbone, channels, None
    except Exception as exc:
        reason = f"HuggingFace checkpoint {hf_id!r} unavailable: {exc}"
        print(f"[model_utils] {reason}; falling back.", file=sys.stderr)
        return None


def _try_torchvision(name: str, pretrained: bool = False) -> nn.Module | None:
    try:
        import torchvision.models as tv
    except ImportError:
        return None
    model_name = _TORCHVISION_MODELS.get(name.lower())
    if model_name is None:
        return None
    factory = getattr(tv, model_name, None)
    if factory is None:
        return None
    if pretrained:
        try:
            return factory(weights="DEFAULT")
        except Exception:
            pass
    try:
        return factory(weights=None)
    except Exception:
        return None


def _extract_features(model: nn.Module) -> nn.Module:
    """Strip classifier from torchvision model, keep feature extractor."""
    if hasattr(model, "features"):
        return _SpatialExtractor(model.features)
    children = list(model.children())
    if len(children) > 2:
        return _SpatialExtractor(nn.Sequential(*children[:-2]))
    for attr in ("heads", "head", "fc", "classifier"):
        if hasattr(model, attr):
            setattr(model, attr, nn.Identity())
    return model


def _infer_channels(backbone: nn.Module, image_size: int = 224) -> int:
    """Run a dummy forward to determine output channel/feature count."""
    dummy = torch.randn(1, 3, image_size, image_size)
    with torch.no_grad():
        out = backbone(dummy)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if isinstance(out, dict):
        out = next(iter(out.values()))
    if out.dim() == 4:
        return int(out.shape[1])
    if out.dim() == 3:
        return int(out.shape[-1])
    return int(out.shape[-1])


def load_backbone(config: dict[str, Any] | None) -> tuple[nn.Module, int]:
    """Load backbone and return (backbone_module, out_channels).

    The backbone outputs spatial features [B, C, H', W'] for CNN models.
    Transformer models return [B, D]. Falls back to TinyBackbone
    if the requested model is unavailable.

    When ``use_pretrained`` is true in *config*, the exact HuggingFace
    checkpoint in ``pretrained_hf_id`` is loaded first (needs the
    optional ``transformers`` package); failing that, torchvision
    DEFAULT weights for the named backbone. ``offline_smoke`` forces
    random init so smoke runs never download anything.
    """
    config = config or {}
    name = str(get_value(config, "backbone", "tiny_cnn")).lower()
    image_size = as_int(get_value(config, "image_size", 224), 224)
    pretrained = as_bool(get_value(config, "use_pretrained", False), False)
    if as_bool(get_value(config, "offline_smoke", False), False):
        pretrained = False

    if pretrained:
        hf_id = str(get_value(config, "pretrained_hf_id", "") or "").strip()
        if hf_id:
            loaded = _try_huggingface(hf_id, image_size, name)
            if loaded is not None:
                backbone, channels, _reason = loaded
                return backbone, channels

    model = _try_torchvision(name, pretrained=pretrained)
    if model is None:
        bb = TinyBackbone()
        _annotate_backbone(
            bb,
            source="tiny_fallback",
            requested_backbone=name,
            requested_hf_id=str(get_value(config, "pretrained_hf_id", "") or ""),
            actual_model="TinyBackbone",
            fallback_reason="No HuggingFace or torchvision backbone could be loaded.",
        )
        return bb, bb.out_channels

    extractor = _extract_features(model)
    _annotate_backbone(
        extractor,
        source="torchvision_pretrained" if pretrained else "torchvision_random",
        requested_backbone=name,
        requested_hf_id=str(get_value(config, "pretrained_hf_id", "") or ""),
        actual_model=_TORCHVISION_MODELS.get(name, name),
    )
    channels = _infer_channels(extractor, image_size)
    return extractor, channels


def _is_backbone_parameter_name(param_name: str) -> bool:
    return (
        "backbone" in param_name
        or param_name.startswith("features")
        or param_name.startswith("layers")
    )


def _set_backbone_requires_grad(model: nn.Module, requires_grad: bool) -> int:
    changed = 0
    for param_name, param in model.named_parameters():
        if _is_backbone_parameter_name(param_name):
            if param.requires_grad != requires_grad:
                changed += 1
            param.requires_grad = requires_grad
    return changed


def _resolve_module_path(root: nn.Module, path: str):
    current = root
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def _iter_block_containers(model: nn.Module):
    roots = []
    backbone = getattr(model, "backbone", None)
    if isinstance(backbone, nn.Module):
        roots.append(backbone)
    roots.append(model)
    paths = (
        "model.encoder.layer",
        "model.encoder.layers",
        "model.encoder.blocks",
        "model.blocks",
        "model.layers",
        "encoder.layer",
        "encoder.layers",
        "encoder.blocks",
        "blocks",
        "layers",
        "features",
        "net",
    )
    seen: set[int] = set()
    for root in roots:
        for path in paths:
            container = _resolve_module_path(root, path)
            if isinstance(container, (nn.ModuleList, nn.Sequential)) and len(container) > 0:
                key = id(container)
                if key not in seen:
                    seen.add(key)
                    yield container


def _unfreeze_module(module: nn.Module) -> int:
    changed = 0
    for param in module.parameters():
        if not param.requires_grad:
            changed += 1
        param.requires_grad = True
    return changed


def _unfreeze_last_blocks(model: nn.Module, last_n: int) -> int:
    if last_n <= 0:
        return 0
    for container in _iter_block_containers(model):
        blocks = list(container.children())
        changed = 0
        for block in blocks[-last_n:]:
            changed += _unfreeze_module(block)
        if changed > 0:
            return changed
    return 0


def _unfreeze_norm_layers(model: nn.Module) -> int:
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
        if not _is_backbone_parameter_name(module_name):
            continue
        if isinstance(module, norm_types):
            changed += _unfreeze_module(module)
    return changed


def apply_freeze(model: nn.Module, config: dict[str, Any] | None) -> None:
    """Freeze backbone parameters based on finetune_strategy config."""
    config = config or {}
    strategy = str(get_value(config, "finetune_strategy", "head_only")).lower()
    frozen = 0
    unfrozen = 0
    last_n = 0
    # head_only means "freeze backbone, train head" by definition — it must not be
    # overridden by a stray freeze_backbone=false in the config.
    if strategy == "head_only":
        frozen = _set_backbone_requires_grad(model, False)
    elif strategy == "partial":
        frozen = _set_backbone_requires_grad(model, False)
        last_n = max(0, as_int(get_value(config, "unfreeze_last_n_blocks", 2), 2))
        unfrozen += _unfreeze_last_blocks(model, last_n)
        if as_bool(get_value(config, "train_norm_layers", True), True):
            unfrozen += _unfreeze_norm_layers(model)
        if last_n > 0 and unfrozen == 0:
            print(
                "[model_utils] partial finetune could not find block containers; "
                "unfreezing the full backbone instead.",
                file=sys.stderr,
            )
            unfrozen += _set_backbone_requires_grad(model, True)
    elif strategy in ("full", "either"):
        _set_backbone_requires_grad(model, True)
    else:
        freeze = as_bool(get_value(config, "freeze_backbone", False), False)
        if freeze:
            frozen = _set_backbone_requires_grad(model, False)
    model._frozen_backbone_params = frozen
    model._partial_unfrozen_params = unfrozen
    model._unfreeze_last_n_blocks = last_n

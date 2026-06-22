"""Project file generator for Module 4 outputs."""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from textwrap import dedent

from .llm_codegen import generate_model_py, get_last_generation_error, get_provider
from .schemas import GeneratedFiles, TrainingSpec
from .spec_builder import specs_to_configs


REQUIRED_GENERATED_FILES = (
    "configs.json",
    "generation_info.json",
    "utils.py",
    "model_utils.py",
    "smoke_data.py",
    "model.py",
    "train.py",
    "evaluate.py",
    "infer.py",
    "run.py",
    "run_experiments.py",
    "requirements.txt",
    "README_generated.md",
)


def generate_files(
    specs: Sequence[TrainingSpec],
    feedback: str | None = None,
    llm_provider: str | None = None,
) -> GeneratedFiles:
    """Return generated project files keyed by relative path.

    ``llm_provider`` overrides the M4_LLM_PROVIDER environment variable.
    """

    if not specs:
        raise ValueError("At least one TrainingSpec is required.")

    configs_json = json.dumps(specs_to_configs(specs), indent=2, sort_keys=True)
    first_config_json = json.dumps(specs[0].to_config(), indent=2, sort_keys=True)

    provider = (llm_provider or get_provider()).strip().lower()
    # LLM 只生成 model.py（使用 model_utils helper），train/evaluate 始终用模板
    llm_model = generate_model_py(specs[0], feedback=feedback or "", provider=provider)
    model_source = provider if llm_model else "template"
    fallback_reason = get_last_generation_error() if not llm_model and provider != "none" else ""

    files = {
        "configs.json": configs_json + "\n",
        "generation_info.json": _generation_info_json(
            provider,
            model_source,
            llm_model is not None,
            fallback_reason=fallback_reason,
        ),
        "utils.py": _utils_py(),
        "model_utils.py": _model_utils_py(),
        "smoke_data.py": _smoke_data_py(),
        "model.py": llm_model if llm_model else _model_py(),
        "train.py": _train_py(),
        "evaluate.py": _evaluate_py(),
        "infer.py": _infer_py(),
        "run.py": _run_py(first_config_json),
        "run_experiments.py": _run_experiments_py(configs_json),
        "requirements.txt": _requirements_txt(),
        "README_generated.md": _readme_generated_md(specs, feedback=feedback, provider=provider, model_source=model_source),
    }
    return GeneratedFiles(files=files)


def _generation_info_json(
    provider: str,
    model_source: str,
    llm_used: bool,
    *,
    fallback_reason: str = "",
) -> str:
    model_name = ""
    if provider == "openai":
        model_name = os.environ.get("M4_OPENAI_MODEL", "gpt-4o")
    elif provider == "qwen":
        model_name = os.environ.get("M4_QWEN_MODEL", "qwen-plus")
    elif provider == "vertex":
        model_name = os.environ.get("M4_VERTEX_MODEL", "gemini-2.0-flash")
    info = {
        "model_py_source": model_source,
        "llm_provider": provider,
        "llm_model": model_name,
        "llm_attempted": provider != "none",
        "llm_used": llm_used,
        "template_fallback": not llm_used,
        "fallback_reason": fallback_reason,
        "generated_by": "module4_agent",
    }
    return json.dumps(info, indent=2, sort_keys=True) + "\n"


def _utils_py() -> str:
    return dedent(
        '''
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


        def get_recipe_value(config: dict[str, Any] | None, key: str, default: Any) -> Any:
            if isinstance(config, dict):
                recipe = config.get("recipe")
                if isinstance(recipe, dict) and key in recipe and recipe[key] is not None:
                    return recipe[key]
            return default


        def get_value(config: dict[str, Any] | None, key: str, default: Any) -> Any:
            if isinstance(config, dict):
                if key in config and config[key] is not None:
                    return config[key]
                return get_recipe_value(config, key, default)
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
            }
        '''
    ).lstrip()


def _smoke_data_py() -> str:
    return dedent(
        '''
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
        '''
    ).lstrip()


def _model_utils_py() -> str:
    return dedent(
        '''
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


        def apply_freeze(model: nn.Module, config: dict[str, Any] | None) -> None:
            """Freeze backbone parameters based on finetune_strategy config."""
            config = config or {}
            strategy = str(get_value(config, "finetune_strategy", "head_only")).lower()
            # head_only means "freeze backbone, train head" by definition — it must not be
            # overridden by a stray freeze_backbone=false in the config.
            if strategy == "head_only":
                freeze = True
            elif strategy in ("full", "either"):
                freeze = False
            else:
                freeze = as_bool(get_value(config, "freeze_backbone", False), False)

            if freeze:
                for param_name, param in model.named_parameters():
                    if "backbone" in param_name or "features" in param_name or "layers" in param_name:
                        param.requires_grad = False
        '''
    ).lstrip()


def _model_py() -> str:
    return dedent(
        '''
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


        def _apply_finetune_strategy(model: nn.Module, config: dict[str, Any] | None) -> nn.Module:
            strategy = str(get_value(config, "finetune_strategy", "head_only")).lower()
            freeze_backbone = as_bool(
                get_value(config, "freeze_backbone", strategy == "head_only"),
                strategy == "head_only",
            )
            if strategy == "full":
                freeze_backbone = False
            elif strategy == "either":
                freeze_backbone = False
            frozen = 0
            if freeze_backbone:
                for name, parameter in model.named_parameters():
                    if "backbone" in name:
                        parameter.requires_grad = False
                        frozen += 1
            model._frozen_backbone_params = frozen
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
        '''
    ).lstrip()


def _train_py() -> str:
    return dedent(
        '''
        """Training loop for generated configs.

        Supports both smoke mode (synthetic data, 1 step) and real training
        (HuggingFace dataset, multi-epoch, checkpoint saving).
        """

        from __future__ import annotations

        import random
        import sys
        import time
        from pathlib import Path
        from typing import Any

        import torch
        import torch.nn.functional as F

        from model import build_model
        from model_utils import backbone_load_info
        from smoke_data import synthetic_batch
        from utils import as_bool, as_float, as_int, get_value, task_type


        _TRANSFORMER_BACKBONES = ("vit", "swin", "dino", "clip", "deit", "beit", "eva")


        def _build_optimizer(model: torch.nn.Module, config: dict[str, Any] | None) -> torch.optim.Optimizer:
            optimizer_name = str(get_value(config, "optimizer", "adamw")).lower()
            lr = as_float(get_value(config, "learning_rate", 1.0e-3), 1.0e-3)

            # Split trainable params into backbone vs the rest (head etc.).
            backbone_params, other_params = [], []
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                (backbone_params if name.startswith("backbone") else other_params).append(parameter)

            # Finetuning a pretrained transformer backbone needs a LOW backbone LR (the head
            # keeps the full LR), or a high LR catastrophically forgets the pretrained features.
            # CNNs and frozen backbones keep a single group (backbone_lr_scale = 1.0).
            backbone_name = str(get_value(config, "backbone", "")).lower()
            is_transformer = any(tok in backbone_name for tok in _TRANSFORMER_BACKBONES)
            backbone_lr_scale = as_float(get_value(config, "backbone_lr_scale", 0.0), 0.0)
            if backbone_lr_scale <= 0.0:
                backbone_lr_scale = 0.01 if (is_transformer and backbone_params) else 1.0

            if backbone_params and other_params and backbone_lr_scale != 1.0:
                groups = [
                    {"params": backbone_params, "lr": lr * backbone_lr_scale},
                    {"params": other_params, "lr": lr},
                ]
            else:
                params = backbone_params + other_params or list(model.parameters())
                groups = [{"params": params, "lr": lr}]

            if "sgd" in optimizer_name:
                return torch.optim.SGD(groups, lr=lr, momentum=0.9)
            if "rmsprop" in optimizer_name:
                return torch.optim.RMSprop(groups, lr=lr)
            if optimizer_name == "adam":
                return torch.optim.Adam(groups, lr=lr)
            return torch.optim.AdamW(groups, lr=lr)


        def _loss_for_output(
            output: Any,
            target: Any,
            config: dict[str, Any] | None,
            class_weights: torch.Tensor | None = None,
        ) -> torch.Tensor:
            task = task_type(config)
            loss_name = str(get_value(config, "loss", "")).lower()
            label_smoothing = as_float(get_value(config, "label_smoothing", 0.0), 0.0)
            if task == "classification":
                if "focal" in loss_name:
                    ce = F.cross_entropy(
                        output,
                        target,
                        weight=class_weights,
                        label_smoothing=label_smoothing,
                        reduction="none",
                    )
                    pt = torch.exp(-ce)
                    return (((1.0 - pt) ** 2.0) * ce).mean()
                return F.cross_entropy(
                    output,
                    target,
                    weight=class_weights,
                    label_smoothing=label_smoothing,
                )
            if task == "image_segmentation":
                if "focal" in loss_name:
                    ce = F.cross_entropy(output, target, reduction="none")
                    pt = torch.exp(-ce)
                    return (((1.0 - pt) ** 2.0) * ce).mean()
                return F.cross_entropy(output, target)
            if task == "object_detection":
                if isinstance(output, dict) and "loss" in output:
                    return output["loss"]
                return torch.as_tensor(0.0, requires_grad=True)
            if task == "feature_extraction":
                embedding_dim = output.shape[1]
                target_embeddings = F.one_hot(target % embedding_dim, num_classes=embedding_dim).float()
                return F.mse_loss(output, target_embeddings)
            return F.cross_entropy(output, target)


        def _build_image_transform(config: dict[str, Any], split: str):
            from torchvision import transforms

            image_size = as_int(get_value(config, "image_size", 224), 224)
            augmentation_value = get_value(config, "augmentation", "basic")
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            if split == "train" and isinstance(augmentation_value, dict):
                tier = str(augmentation_value.get("tier", "medium") or "medium").lower()
                invariance = augmentation_value.get("invariance") or {}
                scale_min = as_float(invariance.get("crop_scale_min", 0.8), 0.8)
                scale_min = min(max(scale_min, 0.05), 1.0)
                ops = [
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(scale_min, 1.0),
                        ratio=(0.75, 1.3333333333),
                    )
                ]
                if bool(invariance.get("hflip", False)):
                    ops.append(transforms.RandomHorizontalFlip())
                if bool(invariance.get("vflip", False)):
                    ops.append(transforms.RandomVerticalFlip())
                if bool(invariance.get("rot90", False)):
                    ops.append(transforms.RandomChoice([
                        transforms.RandomRotation((0, 0)),
                        transforms.RandomRotation((90, 90)),
                        transforms.RandomRotation((180, 180)),
                        transforms.RandomRotation((270, 270)),
                    ]))
                if bool(invariance.get("randaugment", False)) and hasattr(transforms, "RandAugment"):
                    ops.append(transforms.RandAugment())
                if bool(invariance.get("color", False)):
                    ops.append(transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.05,
                    ))
                ops.extend([transforms.ToTensor(), normalize])
                if tier in {"medium", "heavy"} and bool(invariance.get("random_erasing", tier != "light")):
                    ops.append(transforms.RandomErasing(
                        p=0.2 if tier == "medium" else 0.3,
                        scale=(0.02, 0.15),
                        ratio=(0.3, 3.3),
                    ))
                return transforms.Compose(ops)

            augmentation = str(augmentation_value or "basic").lower()
            if split == "train" and augmentation in {"strong", "competition", "advanced"}:
                return transforms.Compose([
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(0.65, 1.0),
                        ratio=(0.75, 1.3333333333),
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    transforms.RandomRotation(20),
                    transforms.ColorJitter(
                        brightness=0.2,
                        contrast=0.2,
                        saturation=0.2,
                        hue=0.05,
                    ),
                    transforms.ToTensor(),
                    normalize,
                    transforms.RandomErasing(
                        p=0.2,
                        scale=(0.02, 0.15),
                        ratio=(0.3, 3.3),
                    ),
                ])
            if split == "train" and augmentation in {"none", "off", "false"}:
                return transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                    normalize,
                ])
            if split == "train":
                return transforms.Compose([
                    transforms.Resize((image_size, image_size)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            resize_size = max(image_size, round(image_size * 1.1))
            return transforms.Compose([
                transforms.Resize((resize_size, resize_size)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                normalize,
            ])


        def _balanced_class_weights(
            labels: list[int],
            num_classes: int,
            power: float,
        ) -> tuple[torch.Tensor, list[int]]:
            counts = torch.bincount(
                torch.tensor(labels, dtype=torch.long),
                minlength=num_classes,
            ).float()
            safe_counts = counts.clamp_min(1.0)
            weights = (counts.sum() / (max(num_classes, 1) * safe_counts)).pow(power)
            weights = weights / weights.mean().clamp_min(1.0e-12)
            weights[counts == 0] = 0.0
            return weights, [int(value) for value in counts.tolist()]


        def _split_indices(labels: list[int], validation_fraction: float, seed: int):
            grouped: dict[int, list[int]] = {}
            for index, label in enumerate(labels):
                grouped.setdefault(int(label), []).append(index)
            rng = random.Random(seed)
            train_indices: list[int] = []
            validation_indices: list[int] = []
            for class_indices in grouped.values():
                shuffled = list(class_indices)
                rng.shuffle(shuffled)
                if len(shuffled) < 2:
                    train_indices.extend(shuffled)
                    continue
                validation_count = max(1, round(len(shuffled) * validation_fraction))
                validation_count = min(validation_count, len(shuffled) - 1)
                validation_indices.extend(shuffled[:validation_count])
                train_indices.extend(shuffled[validation_count:])
            rng.shuffle(train_indices)
            rng.shuffle(validation_indices)
            return train_indices, validation_indices


        def _fold_split_indices(frame, image_column, fold_file, fold_index):
            """外部注入的 paired 折划分：按样本 id 定 val（其余 train），带完整性校验。

            fold_file JSON: {"folds": [[val_id, ...], ...], ...}（每折 = 该折 val 的 id 列表）。
            两臂引用同一 fold_file + 同一 fold_index → val 集完全一致（paired 保证）。
            """
            import json as _json
            with open(fold_file, "r", encoding="utf-8") as _fh:
                spec = _json.load(_fh)
            folds = spec["folds"]
            expected_id_column = spec.get("id_column")
            if expected_id_column and str(expected_id_column) != str(image_column):
                raise ValueError(
                    f"fold_file id_column={expected_id_column!r} does not match image_column={image_column!r}"
                )
            declared_n_folds = spec.get("n_folds")
            if declared_n_folds is not None and int(declared_n_folds) != len(folds):
                raise ValueError(
                    f"fold_file n_folds={declared_n_folds} but contains {len(folds)} folds"
                )
            if not 0 <= fold_index < len(folds):
                raise ValueError(f"fold_index {fold_index} out of range 0..{len(folds) - 1}")
            all_ids = [str(v) for v in frame[image_column].tolist()]
            id_set = set(all_ids)
            seen: set = set()
            union: set = set()
            for one in folds:
                fs = {str(x) for x in one}
                if seen & fs:
                    raise ValueError("fold_file 有交集：同一 id 出现在多折")
                seen |= fs
                union |= fs
            if union != id_set:
                raise ValueError(
                    f"fold_file 与 CSV id 不一致：缺 {len(id_set - union)} 多 {len(union - id_set)}"
                )
            val_ids = {str(x) for x in folds[fold_index]}
            train_indices = [i for i, x in enumerate(all_ids) if x not in val_ids]
            validation_indices = [i for i, x in enumerate(all_ids) if x in val_ids]
            return train_indices, validation_indices


        def _build_local_dataloader(config: dict[str, Any], split: str, batch_size: int, deterministic: bool = False):
            train_csv = str(get_value(config, "train_csv", "") or "").strip()
            image_dir = str(get_value(config, "image_dir", "") or "").strip()
            if not train_csv and not image_dir:
                return None

            import pandas as pd
            from PIL import Image
            from torchvision import datasets as tv_datasets

            # deterministic=True forces the eval transform (no random augmentation) so
            # features are stable across epochs — required for the frozen-backbone cache.
            transform = _build_image_transform(config, "test" if deterministic else split)
            seed = as_int(get_value(config, "seed", 42), 42)
            validation_fraction = as_float(get_value(config, "validation_fraction", 0.2), 0.2)
            max_samples_key = "max_train_samples" if split == "train" else "max_eval_samples"
            max_samples = as_int(get_value(config, max_samples_key, 0), 0)

            if train_csv:
                csv_path = Path(train_csv).expanduser().resolve()
                if not csv_path.exists():
                    raise FileNotFoundError(f"train_csv does not exist: {csv_path}")
                frame = pd.read_csv(csv_path)
                image_column = str(get_value(config, "image_column", "image") or "image")
                label_column = str(get_value(config, "label_column", "label") or "label")
                if image_column not in frame.columns or label_column not in frame.columns:
                    raise ValueError(
                        f"CSV must contain {image_column!r} and {label_column!r}; "
                        f"available columns: {list(frame.columns)}"
                    )

                label_values = frame[label_column].tolist()
                unique_labels = sorted(set(label_values), key=lambda value: str(value))
                label_to_index = {value: index for index, value in enumerate(unique_labels)}
                encoded_labels = [label_to_index[value] for value in label_values]
                fold_file = str(get_value(config, "fold_file", "") or "").strip()
                fold_index = get_value(config, "fold_index", None)
                if fold_file and fold_index is not None:
                    # 外部 paired 折划分（旁路内部 val_split）
                    train_indices, validation_indices = _fold_split_indices(
                        frame, image_column, fold_file, int(fold_index)
                    )
                else:
                    train_indices, validation_indices = _split_indices(
                        encoded_labels,
                        validation_fraction=validation_fraction,
                        seed=seed,
                    )
                selected_indices = train_indices if split == "train" else validation_indices
                if max_samples > 0:
                    selected_indices = selected_indices[:max_samples]
                selected_frame = frame.iloc[selected_indices].reset_index(drop=True)
                selected_labels = [encoded_labels[index] for index in selected_indices]
                base_dir = Path(image_dir).expanduser().resolve() if image_dir else csv_path.parent
                path_template = str(get_value(config, "image_path_template", "{image}") or "{image}")
                image_extension = str(get_value(config, "image_extension", "") or "")

                class CSVImageDataset(torch.utils.data.Dataset):
                    def __init__(self):
                        self.class_weights = None
                        self.class_counts = []
                        if split == "train" and as_bool(
                            get_value(config, "use_class_weights", False),
                            False,
                        ):
                            power = as_float(get_value(config, "class_weight_power", 0.5), 0.5)
                            self.class_weights, self.class_counts = _balanced_class_weights(
                                selected_labels,
                                len(unique_labels),
                                power,
                            )

                    def __len__(self):
                        return len(selected_frame)

                    def __getitem__(self, index):
                        row = selected_frame.iloc[index]
                        image_value = str(row[image_column])
                        relative = path_template.format(
                            image=image_value,
                            label=str(row[label_column]),
                            stem=Path(image_value).stem,
                        )
                        image_path = base_dir / relative
                        if image_extension and not image_path.suffix:
                            image_path = image_path.with_suffix(image_extension)
                        with Image.open(image_path) as image:
                            tensor = transform(image.convert("RGB"))
                        return tensor, torch.tensor(selected_labels[index], dtype=torch.long)

                dataset = CSVImageDataset()
                workers = as_int(get_value(config, "num_workers", 2), 2)
                return torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=split == "train",
                    num_workers=workers,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=workers > 0,
                )

            image_root = Path(image_dir).expanduser().resolve()
            if not image_root.exists():
                raise FileNotFoundError(f"image_dir does not exist: {image_root}")
            dataset = tv_datasets.ImageFolder(image_root, transform=transform)
            labels = [int(label) for _, label in dataset.samples]
            train_indices, validation_indices = _split_indices(
                labels,
                validation_fraction=validation_fraction,
                seed=seed,
            )
            selected_indices = train_indices if split == "train" else validation_indices
            if max_samples > 0:
                selected_indices = selected_indices[:max_samples]
            subset = torch.utils.data.Subset(dataset, selected_indices)
            if split == "train" and as_bool(get_value(config, "use_class_weights", False), False):
                selected_labels = [labels[index] for index in selected_indices]
                power = as_float(get_value(config, "class_weight_power", 0.5), 0.5)
                subset.class_weights, subset.class_counts = _balanced_class_weights(
                    selected_labels,
                    len(dataset.classes),
                    power,
                )
            workers = as_int(get_value(config, "num_workers", 2), 2)
            return torch.utils.data.DataLoader(
                subset,
                batch_size=batch_size,
                shuffle=split == "train",
                num_workers=workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=workers > 0,
            )


        def _build_dataloader(config: dict[str, Any], split: str = "train", batch_size: int = 32, deterministic: bool = False):
            """Build a DataLoader from local Kaggle files or a HuggingFace dataset.

            Returns None if the dataset cannot be loaded (caller falls back to
            synthetic data).  Currently supports classification and feature_extraction;
            detection / segmentation fall back to synthetic data.

            deterministic=True forces the eval transform on the train split so cached
            frozen-backbone features stay stable across epochs.
            """
            local_loader = _build_local_dataloader(config, split, batch_size, deterministic=deterministic)
            if local_loader is not None:
                return local_loader

            dataset_id = str(get_value(config, "dataset_id", "") or "").strip()
            if not dataset_id:
                return None

            task = task_type(config)
            if task in ("object_detection", "image_segmentation"):
                print(f"[train] Real dataloader for {task} not yet implemented; using synthetic data.")
                return None

            try:
                import importlib.metadata
                _orig_ver = importlib.metadata.version
                def _patched_ver(name):
                    v = _orig_ver(name)
                    if v is None and name == "torch":
                        return torch.__version__.split("+")[0]
                    return v
                if not getattr(importlib.metadata.version, "_patched", False):
                    _patched_ver._patched = True
                    importlib.metadata.version = _patched_ver
            except Exception:
                pass

            try:
                from datasets import load_dataset
            except ImportError:
                print("[train] 'datasets' or 'torchvision' not installed; using synthetic data.")
                return None

            try:
                subset = get_value(config, "dataset_subset", None)
                try:
                    ds = load_dataset(dataset_id, subset, trust_remote_code=True)
                except (TypeError, ValueError):
                    ds = load_dataset(dataset_id, subset)
                requested_split = split
                if requested_split == "train":
                    source_split = "train" if "train" in ds else list(ds.keys())[0]
                else:
                    source_split = next(
                        (name for name in ("validation", "test", "val") if name in ds),
                        "train" if "train" in ds else list(ds.keys())[0],
                    )
                ds_split = ds[source_split]
            except Exception as exc:
                print(f"[train] Failed to load dataset {dataset_id!r}: {exc}")
                return None

            transform = _build_image_transform(config, "test" if deterministic else requested_split)

            cols = ds_split.column_names
            image_col = "image" if "image" in cols else ("img" if "img" in cols else None)
            label_col = "label" if "label" in cols else ("labels" if "labels" in cols else None)
            if image_col is None:
                print("[train] No image column found in dataset; using synthetic data.")
                return None
            seed = as_int(get_value(config, "seed", 42), 42)
            validation_fraction = as_float(get_value(config, "validation_fraction", 0.2), 0.2)
            has_dedicated_eval_split = any(name in ds for name in ("validation", "test", "val"))
            if not has_dedicated_eval_split and len(ds_split) > 1:
                shuffled = ds_split.shuffle(seed=seed)
                validation_count = max(1, round(len(shuffled) * validation_fraction))
                validation_count = min(validation_count, len(shuffled) - 1)
                split_at = len(shuffled) - validation_count
                ds_split = (
                    shuffled.select(range(split_at))
                    if requested_split == "train"
                    else shuffled.select(range(split_at, len(shuffled)))
                )
            max_samples_key = "max_train_samples" if requested_split == "train" else "max_eval_samples"
            max_samples = as_int(get_value(config, max_samples_key, 0), 0)
            if max_samples > 0 and len(ds_split) > max_samples:
                ds_split = ds_split.select(range(max_samples))

            class _HFDataset(torch.utils.data.Dataset):
                def __init__(self, hf_ds, img_col, lbl_col, tfm):
                    self.hf_ds = hf_ds
                    self.img_col = img_col
                    self.lbl_col = lbl_col
                    self.tfm = tfm

                def __len__(self):
                    return len(self.hf_ds)

                def __getitem__(self, idx):
                    row = self.hf_ds[idx]
                    img = row[self.img_col]
                    if not isinstance(img, torch.Tensor):
                        img = img.convert("RGB")
                        img = self.tfm(img)
                    lbl = row[self.lbl_col] if self.lbl_col else 0
                    return img, torch.tensor(lbl, dtype=torch.long)

            wrapped = _HFDataset(ds_split, image_col, label_col, transform)
            workers = as_int(get_value(config, "num_workers", 2), 2)
            return torch.utils.data.DataLoader(
                wrapped,
                batch_size=batch_size,
                shuffle=requested_split == "train",
                num_workers=workers,
                pin_memory=torch.cuda.is_available(),
                drop_last=False,
                persistent_workers=workers > 0,
            )


        def _classification_metrics(
            pred_values: torch.Tensor,
            label_values: torch.Tensor,
            probability_values: torch.Tensor,
            config: dict[str, Any],
        ) -> dict[str, float]:
            """Compute the configured validation metric from predictions/probabilities."""
            accuracy = float((pred_values == label_values).float().mean().item())
            metric_name = str(
                get_value(config, "evaluation_metric", "accuracy") or "accuracy"
            ).lower()
            metric_value = accuracy
            try:
                from sklearn.metrics import cohen_kappa_score, log_loss, roc_auc_score
                labels_np = label_values.numpy()
                probabilities_np = probability_values.numpy()
                if metric_name in {"qwk", "quadratic_weighted_kappa"}:
                    metric_name = "qwk"
                    metric_value = float(
                        cohen_kappa_score(
                            labels_np,
                            pred_values.numpy(),
                            weights="quadratic",
                        )
                    )
                elif metric_name in {"roc_auc", "auc"}:
                    metric_name = "roc_auc"
                    if probabilities_np.shape[1] == 2:
                        metric_value = float(roc_auc_score(labels_np, probabilities_np[:, 1]))
                    else:
                        metric_value = float(
                            roc_auc_score(labels_np, probabilities_np, multi_class="ovr")
                        )
                elif metric_name in {"log_loss", "multiclass_log_loss"}:
                    metric_name = "log_loss"
                    metric_value = float(
                        log_loss(
                            labels_np,
                            probabilities_np,
                            labels=list(range(probabilities_np.shape[1])),
                        )
                    )
            except (ImportError, ValueError) as exc:
                print(f"[train] Validation metric {metric_name} failed: {exc}; using accuracy.")
                metric_name = "accuracy"
                metric_value = accuracy
            return {
                "metric_name": metric_name,
                "metric_value": metric_value,
                "accuracy": accuracy,
            }


        def _classification_validation(
            model: torch.nn.Module,
            dataloader,
            config: dict[str, Any],
            device: torch.device,
        ) -> dict[str, float]:
            model.eval()
            predictions: list[torch.Tensor] = []
            labels: list[torch.Tensor] = []
            probabilities: list[torch.Tensor] = []
            with torch.no_grad():
                for x, target in dataloader:
                    x = x.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)
                    predictions.append(probs.argmax(dim=1).cpu())
                    labels.append(target.cpu())
                    probabilities.append(probs.cpu())
            result = _classification_metrics(
                torch.cat(predictions),
                torch.cat(labels),
                torch.cat(probabilities),
                config,
            )
            model.train()
            return result


        def _build_scheduler(
            optimizer: torch.optim.Optimizer,
            config: dict[str, Any],
            epochs: int,
        ):
            scheduler_name = str(
                get_value(config, "scheduler", "cosine") or "cosine"
            ).lower()
            if scheduler_name in {"none", "off", ""}:
                return None
            min_lr = as_float(get_value(config, "min_learning_rate", 1.0e-6), 1.0e-6)
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, int(epochs)),
                eta_min=min_lr,
            )


        def _log_model_load_info(model_load_info: dict[str, Any]) -> None:
            """Print the actual constructed backbone, not just the requested config name."""
            print(
                "[train] requested_backbone="
                f"{model_load_info.get('requested_backbone') or '<none>'} "
                f"hf_id={model_load_info.get('requested_hf_id') or '<none>'} "
                f"source={model_load_info.get('source') or '<unknown>'} "
                f"actual_model={model_load_info.get('actual_model') or '<unknown>'} "
                f"backbone_class={model_load_info.get('backbone_class') or '<unknown>'} "
                f"feature_pooling={model_load_info.get('feature_pooling') or '<none>'}",
                file=sys.stderr,
            )
            print(
                "[train] params "
                f"total={model_load_info.get('total_params', 0)} "
                f"trainable={model_load_info.get('trainable_params', 0)}",
                file=sys.stderr,
            )
            if model_load_info.get("fallback_reason"):
                print(
                    f"[train] backbone_fallback_reason={model_load_info['fallback_reason']}",
                    file=sys.stderr,
                )


        def _checkpoint_matches_current_model(
            checkpoint: dict[str, Any],
            model_load_info: dict[str, Any],
        ) -> bool:
            """Avoid resuming stale fallback checkpoints into a different real backbone."""
            checkpoint_info = checkpoint.get("model_load_info")
            if not isinstance(checkpoint_info, dict):
                if model_load_info.get("source") == "huggingface":
                    print(
                        "[train] Skipping resume checkpoint without model_load_info "
                        "for current HuggingFace backbone.",
                        file=sys.stderr,
                    )
                    return False
                return True
            keys = ["source", "actual_model", "backbone_class"]
            if model_load_info.get("source") == "huggingface":
                keys.append("feature_pooling")
            same = all(
                str(checkpoint_info.get(key, "")) == str(model_load_info.get(key, ""))
                for key in keys
            )
            if not same:
                print("[train] Skipping resume checkpoint because backbone changed.", file=sys.stderr)
                print(f"[train] checkpoint_model_load_info={checkpoint_info}", file=sys.stderr)
                print(f"[train] current_model_load_info={model_load_info}", file=sys.stderr)
            return same


        def _backbone_is_frozen(model: torch.nn.Module) -> bool:
            """True only when the model exposes a backbone whose params are all frozen."""
            backbone = getattr(model, "backbone", None)
            if backbone is None or not hasattr(model, "head"):
                return False
            params = list(backbone.parameters())
            return len(params) > 0 and all(not p.requires_grad for p in params)


        def _backbone_features(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
            """Replicate the head's input: backbone output, pooled+flattened if spatial."""
            feats = model.backbone(x)
            if feats.dim() == 4:
                feats = F.adaptive_avg_pool2d(feats, 1).flatten(1)
            return feats


        def _extract_features(model: torch.nn.Module, loader, device: torch.device):
            model.eval()
            feats: list[torch.Tensor] = []
            labels: list[torch.Tensor] = []
            with torch.no_grad():
                for x, target in loader:
                    x = x.to(device, non_blocking=True)
                    f = _backbone_features(model, x).detach().float().cpu()
                    feats.append(f)
                    if not isinstance(target, torch.Tensor):
                        target = torch.as_tensor(target)
                    labels.append(target.cpu())
            return torch.cat(feats), torch.cat(labels)


        def _cache_token(value: object) -> str:
            text = str(value or "unknown")
            safe = "".join(ch if ch.isalnum() else "_" for ch in text)
            return safe.strip("_")[:120] or "unknown"


        def _get_or_extract_features(model, loader, device, config, tag, checkpoint_dir):
            """Extract (or load from disk) the frozen-backbone features for one split."""
            cache_dir = Path(
                str(get_value(config, "feature_cache_dir", str(Path(checkpoint_dir) / "feature_cache")))
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            backbone_name = str(get_value(config, "backbone", "backbone"))
            image_size = as_int(get_value(config, "image_size", 224), 224)
            count = len(loader.dataset)
            backbone = getattr(model, "backbone", None)
            info = getattr(backbone, "_jiaozi_load_info", {}) or {}
            cache_sig = _cache_token(
                "|".join(
                    [
                        str(info.get("actual_model") or backbone_name),
                        str(info.get("feature_pooling") or ""),
                    ]
                )
            )
            cache_path = cache_dir / f"feat_{tag}_{backbone_name}_{cache_sig}_{image_size}_{count}.pt"
            if cache_path.exists():
                blob = torch.load(cache_path, map_location="cpu")
                print(f"[train] Loaded cached {tag} features {tuple(blob['X'].shape)} from {cache_path}")
                return blob["X"], blob["y"]
            print(f"[train] Extracting {tag} features (one pass over the data)...")
            X, y = _extract_features(model, loader, device)
            torch.save({"X": X, "y": y}, cache_path)
            print(f"[train] Cached {tag} features {tuple(X.shape)} to {cache_path}")
            return X, y


        def _train_frozen_head(
            model,
            validation_loader,
            config,
            device,
            optimizer,
            scheduler,
            epochs,
            start_epoch,
            checkpoint_dir,
            class_weights,
            metric_name,
            minimize_metric,
            early_stopping_patience,
            gradient_clip_norm,
            save_every_epoch,
            model_load_info,
        ):
            """Extract frozen-backbone features once, then train the head on the cache.

            Returns (best_metric, best_epoch, epoch_losses, validation_history,
            loss_value, total_steps).  Saves full-model checkpoints so evaluate/infer
            stay unchanged.
            """
            batch_size = as_int(get_value(config, "batch_size", 32), 32)
            extract_bs = as_int(get_value(config, "eval_batch_size", batch_size * 2), batch_size * 2)
            # Deterministic train loader (eval transform) so cached features are stable.
            det_train_loader = _build_dataloader(
                config, split="train", batch_size=extract_bs, deterministic=True
            )
            if det_train_loader is None:
                raise RuntimeError("Feature cache path could not build a deterministic train loader.")

            train_X, train_y = _get_or_extract_features(
                model, det_train_loader, device, config, "train", checkpoint_dir
            )
            val_X = val_y = None
            if validation_loader is not None:
                val_X, val_y = _get_or_extract_features(
                    model, validation_loader, device, config, "val", checkpoint_dir
                )

            head_batch = as_int(get_value(config, "head_batch_size", 256), 256)
            feat_loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(train_X, train_y),
                batch_size=head_batch,
                shuffle=True,
            )

            loss_value = 0.0
            total_steps = 0
            epoch_losses: list[float] = []
            validation_history: list[dict[str, Any]] = []
            best_metric: float | None = None
            best_epoch = 0
            epochs_without_improvement = 0

            for epoch in range(start_epoch, max(1, int(epochs))):
                model.train()
                epoch_loss = 0.0
                batch_count = 0
                for feats, target in feat_loader:
                    feats = feats.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    logits = model.head(feats)
                    loss = _loss_for_output(logits, target, config, class_weights=class_weights)
                    loss.backward()
                    if gradient_clip_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.head.parameters(), gradient_clip_norm)
                    optimizer.step()
                    loss_value = float(loss.detach().cpu().item())
                    epoch_loss += loss_value
                    batch_count += 1
                    total_steps += 1
                avg_loss = epoch_loss / max(batch_count, 1)
                epoch_losses.append(avg_loss)

                validation_result = None
                if val_X is not None:
                    model.eval()
                    with torch.no_grad():
                        logits = model.head(val_X.to(device))
                        probs = torch.softmax(logits, dim=1).cpu()
                    validation_result = _classification_metrics(
                        probs.argmax(dim=1), val_y, probs, config
                    )
                    validation_result["epoch"] = epoch + 1
                    validation_history.append(validation_result)
                if scheduler is not None:
                    scheduler.step()

                current_lr = float(optimizer.param_groups[0]["lr"])
                metric_text = ""
                improved = best_metric is None
                if validation_result is not None:
                    current_metric = float(validation_result["metric_value"])
                    improved = (
                        best_metric is None
                        or (current_metric < best_metric if minimize_metric else current_metric > best_metric)
                    )
                    metric_text = (
                        f"  val_{validation_result['metric_name']}={current_metric:.4f}"
                        f"  val_acc={validation_result['accuracy']:.4f}"
                    )
                    if improved:
                        best_metric = current_metric
                        best_epoch = epoch + 1
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1

                print(
                    f"[train] (cached) epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}"
                    f"{metric_text}  lr={current_lr:.2e}  steps={total_steps}"
                )

                checkpoint_payload = {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                    "loss": avg_loss,
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
                    "validation": validation_result,
                    "config": config,
                    "model_load_info": model_load_info,
                    "feature_cached": True,
                }
                torch.save(checkpoint_payload, checkpoint_dir / "last_checkpoint.pt")
                if save_every_epoch:
                    torch.save(checkpoint_payload, checkpoint_dir / f"checkpoint_epoch{epoch + 1}.pt")
                if improved:
                    torch.save(checkpoint_payload, checkpoint_dir / "best_model.pt")
                    print(f"[train] Saved new best checkpoint at epoch {epoch + 1}")

                if (
                    early_stopping_patience > 0
                    and epochs_without_improvement >= early_stopping_patience
                ):
                    print(
                        f"[train] Early stopping after {early_stopping_patience} "
                        "epochs without validation improvement."
                    )
                    break

            return best_metric, best_epoch, epoch_losses, validation_history, loss_value, total_steps


        def train_model(
            config: dict[str, Any] | None,
            data: tuple[Any, Any] | None = None,
            epochs: int = 1,
            max_steps: int = 1,
            save_dir: str | None = None,
        ) -> tuple[torch.nn.Module, dict[str, Any]]:
            """Train a model and return it with a summary.

            Smoke mode (offline_smoke=true or no dataset): runs a quick
            synthetic-data loop.  Real mode: loads the dataset, trains for
            the requested epochs, saves checkpoints, and logs progress.
            """
            start = time.time()
            config = config or {}
            task = task_type(config)
            offline_smoke = as_bool(get_value(config, "offline_smoke", True), True)
            model = build_model(config)
            model_load_info = backbone_load_info(model, config)
            _log_model_load_info(model_load_info)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type == "cuda":
                torch.backends.cudnn.benchmark = True
            model.to(device)
            model.train()
            optimizer = _build_optimizer(model, config)
            scheduler = _build_scheduler(optimizer, config, epochs)
            amp_enabled = (
                device.type == "cuda"
                and as_bool(get_value(config, "mixed_precision", True), True)
            )
            try:
                scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
            except (AttributeError, TypeError):
                scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
            loss_value = 0.0
            total_steps = 0
            epoch_losses: list[float] = []
            validation_history: list[dict[str, Any]] = []
            best_metric: float | None = None
            best_epoch = 0
            start_epoch = 0

            dataloader = None
            validation_loader = None
            if not offline_smoke and data is None:
                batch_size = as_int(get_value(config, "batch_size", 32), 32)
                dataloader = _build_dataloader(config, split="train", batch_size=batch_size)
                if task == "classification":
                    eval_batch_size = as_int(
                        get_value(config, "eval_batch_size", batch_size * 2),
                        batch_size * 2,
                    )
                    validation_loader = _build_dataloader(
                        config,
                        split="test",
                        batch_size=eval_batch_size,
                    )

            if dataloader is not None:
                if save_dir is None:
                    save_dir = str(get_value(config, "checkpoint_dir", "checkpoints"))
                checkpoint_dir = Path(save_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                class_weights = getattr(dataloader.dataset, "class_weights", None)
                class_counts = getattr(dataloader.dataset, "class_counts", [])
                if isinstance(class_weights, torch.Tensor):
                    class_weights = class_weights.to(device)
                    print(
                        "[train] class counts=",
                        class_counts,
                        " weights=",
                        [round(float(value), 4) for value in class_weights.cpu().tolist()],
                    )

                resume_checkpoint = str(
                    get_value(config, "resume_checkpoint", "") or ""
                ).strip()
                if resume_checkpoint.lower() == "auto":
                    resume_checkpoint = str(checkpoint_dir / "last_checkpoint.pt")
                if resume_checkpoint and Path(resume_checkpoint).exists():
                    checkpoint = torch.load(resume_checkpoint, map_location=device)
                    if _checkpoint_matches_current_model(checkpoint, model_load_info):
                    model.load_state_dict(checkpoint["model_state_dict"])
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                    if scheduler is not None and checkpoint.get("scheduler_state_dict"):
                        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                    if checkpoint.get("scaler_state_dict"):
                        scaler.load_state_dict(checkpoint["scaler_state_dict"])
                    start_epoch = int(checkpoint.get("epoch", 0))
                    best_metric = checkpoint.get("best_metric")
                    best_epoch = int(checkpoint.get("best_epoch", 0))
                    print(f"[train] Resuming from {resume_checkpoint} at epoch {start_epoch + 1}.")

                metric_name = str(
                    get_value(config, "evaluation_metric", "accuracy") or "accuracy"
                ).lower()
                minimize_metric = metric_name in {"log_loss", "multiclass_log_loss", "rmse"}
                early_stopping_patience = as_int(
                    get_value(config, "early_stopping_patience", 0),
                    0,
                )
                epochs_without_improvement = 0
                gradient_clip_norm = as_float(
                    get_value(config, "gradient_clip_norm", 1.0),
                    1.0,
                )
                save_every_epoch = as_bool(
                    get_value(config, "save_every_epoch", False),
                    False,
                )

                # Frozen backbone + classification/feature_extraction → extract features once
                # and train only the head on the cached vectors (≈ one data pass instead of N).
                ran_cached = False
                use_feature_cache = (
                    task in ("classification", "feature_extraction")
                    and _backbone_is_frozen(model)
                    and as_bool(get_value(config, "feature_cache", True), True)
                )
                if use_feature_cache:
                    print(
                        "[train] Frozen backbone detected — caching features and training the "
                        "head only (deterministic preprocessing, no random augmentation)."
                    )
                    (
                        best_metric,
                        best_epoch,
                        cached_losses,
                        cached_history,
                        loss_value,
                        cached_steps,
                    ) = _train_frozen_head(
                        model,
                        validation_loader,
                        config,
                        device,
                        optimizer,
                        scheduler,
                        max(1, int(epochs)),
                        start_epoch,
                        checkpoint_dir,
                        class_weights,
                        metric_name,
                        minimize_metric,
                        early_stopping_patience,
                        gradient_clip_norm,
                        save_every_epoch,
                        model_load_info,
                    )
                    epoch_losses.extend(cached_losses)
                    validation_history.extend(cached_history)
                    total_steps += cached_steps
                    ran_cached = True

                for epoch in (
                    range(start_epoch, max(1, int(epochs))) if not ran_cached else range(0)
                ):
                    epoch_loss = 0.0
                    batch_count = 0
                    for x, target in dataloader:
                        x = x.to(device, non_blocking=True)
                        if isinstance(target, torch.Tensor):
                            target = target.to(device, non_blocking=True)
                        optimizer.zero_grad(set_to_none=True)
                        with torch.autocast(
                            device_type=device.type,
                            dtype=torch.float16,
                            enabled=amp_enabled,
                        ):
                            if task == "object_detection":
                                output = model(x, target)
                            else:
                                output = model(x)
                            loss = _loss_for_output(
                                output,
                                target,
                                config,
                                class_weights=class_weights,
                            )
                        scaler.scale(loss).backward()
                        if gradient_clip_norm > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                model.parameters(),
                                gradient_clip_norm,
                            )
                        scaler.step(optimizer)
                        scaler.update()
                        loss_value = float(loss.detach().cpu().item())
                        epoch_loss += loss_value
                        batch_count += 1
                        total_steps += 1
                        if max_steps > 0 and total_steps >= max_steps:
                            break
                    avg_loss = epoch_loss / max(batch_count, 1)
                    epoch_losses.append(avg_loss)
                    validation_result = None
                    if validation_loader is not None:
                        validation_result = _classification_validation(
                            model,
                            validation_loader,
                            config,
                            device,
                        )
                        validation_result["epoch"] = epoch + 1
                        validation_history.append(validation_result)
                    if scheduler is not None:
                        scheduler.step()

                    current_lr = float(optimizer.param_groups[0]["lr"])
                    metric_text = ""
                    improved = best_metric is None
                    if validation_result is not None:
                        current_metric = float(validation_result["metric_value"])
                        improved = (
                            best_metric is None
                            or (current_metric < best_metric if minimize_metric else current_metric > best_metric)
                        )
                        metric_text = (
                            f"  val_{validation_result['metric_name']}={current_metric:.4f}"
                            f"  val_acc={validation_result['accuracy']:.4f}"
                        )
                        if improved:
                            best_metric = current_metric
                            best_epoch = epoch + 1
                            epochs_without_improvement = 0
                        else:
                            epochs_without_improvement += 1

                    print(
                        f"[train] epoch {epoch + 1}/{epochs}  loss={avg_loss:.4f}"
                        f"{metric_text}  lr={current_lr:.2e}"
                        f"  steps={batch_count}  time={time.time() - start:.1f}s"
                    )

                    checkpoint_payload = {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
                        "scaler_state_dict": scaler.state_dict(),
                        "loss": avg_loss,
                        "best_metric": best_metric,
                        "best_epoch": best_epoch,
                        "validation": validation_result,
                        "config": config,
                        "model_load_info": model_load_info,
                    }
                    torch.save(checkpoint_payload, checkpoint_dir / "last_checkpoint.pt")
                    if save_every_epoch:
                        torch.save(
                            checkpoint_payload,
                            checkpoint_dir / f"checkpoint_epoch{epoch + 1}.pt",
                        )
                    if improved:
                        torch.save(checkpoint_payload, checkpoint_dir / "best_model.pt")
                        print(
                            f"[train] Saved new best checkpoint at epoch {epoch + 1}: "
                            f"{checkpoint_dir / 'best_model.pt'}"
                        )

                    if max_steps > 0 and total_steps >= max_steps:
                        break
                    if (
                        early_stopping_patience > 0
                        and epochs_without_improvement >= early_stopping_patience
                    ):
                        print(
                            f"[train] Early stopping after {early_stopping_patience} "
                            "epochs without validation improvement."
                        )
                        break

                best_path = checkpoint_dir / "best_model.pt"
                if best_path.exists():
                    best_checkpoint = torch.load(best_path, map_location=device)
                    model.load_state_dict(best_checkpoint["model_state_dict"])
                print(f"[train] Done. Best model: {best_path}")
            else:
                batch = data if data is not None else synthetic_batch(config)
                steps = max(1, int(max_steps)) if max_steps > 0 else 1
                for _epoch in range(max(1, int(epochs))):
                    for _step in range(steps):
                        x, target = batch
                        x = x.to(device)
                        if isinstance(target, torch.Tensor):
                            target = target.to(device)
                        optimizer.zero_grad(set_to_none=True)
                        if task == "object_detection":
                            output = model(x, target)
                        else:
                            output = model(x)
                        loss = _loss_for_output(output, target, config)
                        loss.backward()
                        optimizer.step()
                        loss_value = float(loss.detach().cpu().item())
                        total_steps += 1

            summary = {
                "status": "success",
                "task_type": task,
                "loss": loss_value,
                "total_steps": total_steps,
                "epoch_losses": epoch_losses,
                "validation_history": validation_history,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "mixed_precision": amp_enabled,
                "runtime_sec": round(time.time() - start, 4),
                "real_data": dataloader is not None,
                "model_load_info": model_load_info,
                "config_summary": {
                    "rank": get_value(config, "rank", None),
                    "backbone": get_value(config, "backbone", "tiny_cnn"),
                    "actual_model": model_load_info.get("actual_model"),
                    "backbone_source": model_load_info.get("source"),
                    "loss": get_value(config, "loss", "cross_entropy_loss"),
                    "optimizer": get_value(config, "optimizer", "adamw"),
                    "finetune_strategy": get_value(config, "finetune_strategy", "head_only"),
                    "frozen_backbone_params": int(getattr(model, "_frozen_backbone_params", 0)),
                },
            }
            return model, summary


        def train_one(config: dict[str, Any] | None, data: tuple[Any, Any] | None = None, epochs: int = 1, max_steps: int = 1) -> dict[str, Any]:
            """Run training and return a summary."""
            _model, summary = train_model(config, data=data, epochs=epochs, max_steps=max_steps)
            return summary
        '''
    ).lstrip()


def _evaluate_py() -> str:
    return dedent(
        '''
        """Evaluation helpers for generated configs."""

        from __future__ import annotations

        from typing import Any

        import torch

        from smoke_data import synthetic_batch
        from utils import as_bool, as_int, get_value, task_type


        def _macro_f1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
            scores = []
            for cls in range(num_classes):
                pred_pos = preds == cls
                label_pos = labels == cls
                tp = torch.logical_and(pred_pos, label_pos).sum().item()
                fp = torch.logical_and(pred_pos, torch.logical_not(label_pos)).sum().item()
                fn = torch.logical_and(torch.logical_not(pred_pos), label_pos).sum().item()
                denom = (2 * tp) + fp + fn
                if denom > 0:
                    scores.append((2 * tp) / denom)
            return float(sum(scores) / len(scores)) if scores else 0.0


        def _mean_iou(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
            values = []
            for cls in range(num_classes):
                pred_mask = preds == cls
                label_mask = labels == cls
                intersection = torch.logical_and(pred_mask, label_mask).sum().item()
                union = torch.logical_or(pred_mask, label_mask).sum().item()
                if union > 0:
                    values.append(intersection / union)
            return float(sum(values) / len(values)) if values else 0.0


        def _dice(preds: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
            values = []
            for cls in range(num_classes):
                pred_mask = preds == cls
                label_mask = labels == cls
                intersection = torch.logical_and(pred_mask, label_mask).sum().item()
                denom = pred_mask.sum().item() + label_mask.sum().item()
                if denom > 0:
                    values.append((2 * intersection) / denom)
            return float(sum(values) / len(values)) if values else 0.0


        def _box_iou(box_a: torch.Tensor, box_b: torch.Tensor) -> torch.Tensor:
            top_left = torch.maximum(box_a[:2], box_b[:2])
            bottom_right = torch.minimum(box_a[2:], box_b[2:])
            wh = (bottom_right - top_left).clamp(min=0)
            inter = wh[0] * wh[1]
            area_a = (box_a[2] - box_a[0]).clamp(min=0) * (box_a[3] - box_a[1]).clamp(min=0)
            area_b = (box_b[2] - box_b[0]).clamp(min=0) * (box_b[3] - box_b[1]).clamp(min=0)
            union = area_a + area_b - inter
            if float(union.item()) <= 0.0:
                return torch.tensor(0.0)
            return inter / union


        def _count_params(model: torch.nn.Module) -> dict[str, int]:
            total = sum(p.numel() for p in model.parameters())
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            return {"total": total, "trainable": trainable}


        def _eval_on_dataloader(model: torch.nn.Module, dataloader, config: dict[str, Any]) -> dict[str, Any]:
            """Evaluate on a full DataLoader (real data path)."""
            task = task_type(config)
            num_classes = max(1, as_int(get_value(config, "num_classes", 3), 3))
            device = next(model.parameters()).device
            model.eval()

            all_preds: list[torch.Tensor] = []
            all_labels: list[torch.Tensor] = []
            all_probabilities: list[torch.Tensor] = []
            with torch.no_grad():
                for x, target in dataloader:
                    x = x.to(device, non_blocking=True)
                    if isinstance(target, torch.Tensor):
                        target = target.to(device, non_blocking=True)
                    if task == "classification":
                        logits = model(x)
                        probabilities = torch.softmax(logits, dim=1)
                        preds = probabilities.argmax(dim=1)
                        all_preds.append(preds)
                        all_labels.append(target)
                        all_probabilities.append(probabilities)
                    elif task == "feature_extraction":
                        output = model(x)
                        all_preds.append(output)
                        all_labels.append(target)
                    else:
                        output = model(x)
                        all_preds.append(output.argmax(dim=1) if output.dim() > 1 else output)
                        all_labels.append(target)

            if task == "classification":
                preds = torch.cat(all_preds).cpu()
                labels = torch.cat(all_labels).cpu()
                probabilities = torch.cat(all_probabilities).cpu()
                accuracy = float((preds == labels).float().mean().item())
                requested_metric = str(get_value(config, "evaluation_metric", "accuracy") or "accuracy").lower()
                metric_name = "accuracy"
                metric_value = accuracy
                try:
                    from sklearn.metrics import cohen_kappa_score, log_loss, roc_auc_score
                    label_values = labels.numpy()
                    probability_values = probabilities.numpy()
                    if requested_metric in {"qwk", "quadratic_weighted_kappa"}:
                        metric_name = "qwk"
                        metric_value = float(
                            cohen_kappa_score(label_values, preds.numpy(), weights="quadratic")
                        )
                    elif requested_metric in {"roc_auc", "auc"}:
                        metric_name = "roc_auc"
                        if probability_values.shape[1] == 2:
                            metric_value = float(roc_auc_score(label_values, probability_values[:, 1]))
                        else:
                            metric_value = float(
                                roc_auc_score(label_values, probability_values, multi_class="ovr")
                            )
                    elif requested_metric in {"log_loss", "multiclass_log_loss"}:
                        metric_name = "log_loss"
                        metric_value = float(
                            log_loss(
                                label_values,
                                probability_values,
                                labels=list(range(num_classes)),
                            )
                        )
                except (ImportError, ValueError) as exc:
                    print(f"[evaluate] Could not compute {requested_metric}: {exc}; using accuracy.")
                export_path = str(get_value(config, "export_preds_path", "") or "").strip()
                if export_path:
                    # 导出 val 预测供离线算指标 bundle（macro_f1 / roc_auc / pr_auc）
                    import json as _json
                    with open(export_path, "w", encoding="utf-8") as _fh:
                        _json.dump(
                            {
                                "y_true": labels.tolist(),
                                "y_prob": probabilities.tolist(),
                                "y_score": probabilities.tolist(),
                            },
                            _fh,
                        )
                return {
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "accuracy": accuracy,
                    "macro_f1": _macro_f1(preds, labels, num_classes),
                    "num_samples": len(labels),
                    "params": _count_params(model),
                    "status": "success",
                }
            if task == "feature_extraction":
                embeddings = torch.cat(all_preds)
                labels = torch.cat(all_labels)
                distances = torch.cdist(embeddings, embeddings)
                distances.fill_diagonal_(float("inf"))
                nearest = distances.argmin(dim=1)
                recall = float((labels[nearest] == labels).float().mean().item())
                return {
                    "metric_name": "recall@1",
                    "metric_value": recall,
                    "num_samples": len(labels),
                    "params": _count_params(model),
                    "status": "success",
                }
            preds = torch.cat(all_preds)
            labels = torch.cat(all_labels)
            accuracy = float((preds == labels).float().mean().item())
            return {
                "metric_name": "accuracy",
                "metric_value": accuracy,
                "num_samples": len(labels),
                "params": _count_params(model),
                "status": "success",
            }


        def evaluate(model: torch.nn.Module, config: dict[str, Any] | None, data: tuple[Any, Any] | None = None) -> dict[str, Any]:
            """Evaluate a model.  Uses real test data when offline_smoke is false."""

            config = config or {}
            task = task_type(config)
            num_classes = max(1, as_int(get_value(config, "num_classes", 3), 3))
            offline_smoke = as_bool(get_value(config, "offline_smoke", True), True)

            if not offline_smoke and data is None:
                from train import _build_dataloader
                dataloader = _build_dataloader(config, split="test", batch_size=64)
                if dataloader is not None:
                    return _eval_on_dataloader(model, dataloader, config)

            x, target = data if data is not None else synthetic_batch(config)
            device = next(model.parameters()).device
            x = x.to(device)
            if isinstance(target, torch.Tensor):
                target = target.to(device)
            elif isinstance(target, list):
                target = [
                    {
                        key: value.to(device) if isinstance(value, torch.Tensor) else value
                        for key, value in item.items()
                    }
                    for item in target
                ]
            model.eval()
            with torch.no_grad():
                output = model(x)

            result: dict[str, Any] = {"params": _count_params(model)}

            if task == "classification":
                preds = output.argmax(dim=1)
                accuracy = float((preds == target).float().mean().item())
                result.update({
                    "metric_name": "accuracy",
                    "metric_value": accuracy,
                    "macro_f1": _macro_f1(preds, target, num_classes),
                    "status": "success",
                })
                return result
            if task == "image_segmentation":
                preds = output.argmax(dim=1)
                result.update({
                    "metric_name": "mIoU",
                    "metric_value": _mean_iou(preds, target, num_classes),
                    "dice": _dice(preds, target, num_classes),
                    "status": "success",
                })
                return result
            if task == "object_detection":
                pred_boxes = output["pred_boxes"][:, 0, :]
                pred_logits = output["pred_logits"][:, 0, :]
                pred_classes = pred_logits.argmax(dim=1)
                hits = []
                for idx, item in enumerate(target):
                    label = item.get("class_labels", item.get("labels"))[0]
                    box = item["boxes"][0]
                    class_hit = int(pred_classes[idx].item()) == int(label.item())
                    box_hit = float(_box_iou(pred_boxes[idx].cpu(), box.cpu()).item()) >= 0.5
                    hits.append(1.0 if class_hit and box_hit else 0.0)
                result.update({
                    "metric_name": "mAP@0.5",
                    "metric_value": float(sum(hits) / len(hits)) if hits else 0.0,
                    "status": "success",
                })
                return result
            if task == "feature_extraction":
                embeddings = output
                distances = torch.cdist(embeddings, embeddings)
                distances.fill_diagonal_(float("inf"))
                nearest = distances.argmin(dim=1)
                recall = float((target[nearest] == target).float().mean().item())
                result.update({
                    "metric_name": "recall@1",
                    "metric_value": recall,
                    "status": "success",
                })
                return result
            result.update({"metric_name": "accuracy", "metric_value": 0.0, "status": "success"})
            return result
        '''
    ).lstrip()


def _infer_py() -> str:
    return dedent(
        '''
        """Inference entry point for generated configs."""

        from __future__ import annotations

        from pathlib import Path
        from typing import Any

        import torch

        from model import build_model
        from smoke_data import synthetic_image
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
                output = model(image)

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
        '''
    ).lstrip()


def _run_py(first_config_json: str) -> str:
    template = dedent(
        '''
        """Single-configuration runner.

        Smoke mode (default):  offline_smoke=true  → synthetic data, 1 epoch, 1 step.
        Real training mode:    offline_smoke=false  → HuggingFace dataset, multi-epoch,
                               checkpoint saving.
        """

        from __future__ import annotations

        import argparse
        import json
        from typing import Any

        from evaluate import evaluate
        from infer import predict
        from train import train_model
        from utils import as_bool, as_int, compact_config_summary, get_recipe_value, get_value, load_config, set_seed


        DEFAULT_CONFIG = json.loads(__DEFAULT_CONFIG_JSON__)


        def main() -> None:
            parser = argparse.ArgumentParser(description="Run one experiment (smoke or real training).")
            parser.add_argument("--config", default="configs.json", help="JSON config path.")
            parser.add_argument("--seed", type=int, default=123)
            parser.add_argument("--epochs", type=int, default=None,
                                help="Training epochs (default: 1 for smoke, 10 for real).")
            parser.add_argument("--dataset", default=None,
                                help="Override dataset_id in config for real training.")
            args = parser.parse_args()

            set_seed(args.seed)
            config = load_config(args.config, DEFAULT_CONFIG)

            if args.dataset:
                config["dataset_id"] = args.dataset

            offline_smoke = as_bool(get_value(config, "offline_smoke", True), True)
            recipe_epochs = get_recipe_value(config, "epochs", 10)
            default_epochs = 1 if offline_smoke else as_int(get_value(config, "recommended_epochs", recipe_epochs), 10)
            epochs = args.epochs if args.epochs is not None else default_epochs
            max_steps = 1 if offline_smoke else 0

            model, train_result = train_model(config, epochs=epochs, max_steps=max_steps)
            eval_result = evaluate(model, config)
            infer_result = predict(config=config, model=model)
            summary = {
                "status": "success",
                "config": compact_config_summary(config),
                "train": train_result,
                "evaluate": eval_result,
                "infer": infer_result,
            }
            print(json.dumps(summary, indent=2, sort_keys=True))


        if __name__ == "__main__":
            main()
        '''
    ).lstrip()
    return template.replace("__DEFAULT_CONFIG_JSON__", repr(first_config_json))


def _run_experiments_py(configs_json: str) -> str:
    template = dedent(
        '''
        """Sweep all Module 3 candidates."""

        from __future__ import annotations

        import argparse
        import json
        from typing import Any

        from evaluate import evaluate
        from train import train_model
        from utils import as_bool, as_int, compact_config_summary, get_recipe_value, get_value, load_configs, set_seed


        DEFAULT_CONFIGS = json.loads(__DEFAULT_CONFIGS_JSON__)


        def run_all(configs: list[dict[str, Any]], seed: int = 123, epochs: int | None = None) -> list[dict[str, Any]]:
            rows = []
            for index, config in enumerate(configs, start=1):
                set_seed(seed)
                offline_smoke = as_bool(get_value(config, "offline_smoke", True), True)
                recipe_epochs = get_recipe_value(config, "epochs", 10)
                default_ep = 1 if offline_smoke else as_int(get_value(config, "recommended_epochs", recipe_epochs), 10)
                ep = epochs if epochs is not None else default_ep
                ms = 1 if offline_smoke else 0
                model, train_result = train_model(config, epochs=ep, max_steps=ms)
                eval_result = evaluate(model, config)
                row = compact_config_summary(config, rank_default=index)
                row.update(
                    {
                        "metric_name": eval_result.get("metric_name"),
                        "metric_value": eval_result.get("metric_value"),
                        "status": "success" if train_result.get("status") == "success" and eval_result.get("status") == "success" else "failed",
                    }
                )
                rows.append(row)
            return rows


        def main() -> None:
            parser = argparse.ArgumentParser(description="Sweep all Module 3 candidate configs.")
            parser.add_argument("--input", default="configs.json", help="JSON file with one or more configs.")
            parser.add_argument("--seed", type=int, default=123)
            parser.add_argument("--epochs", type=int, default=None,
                                help="Training epochs per candidate (default: 1 smoke / 10 real).")
            args = parser.parse_args()
            rows = run_all(load_configs(args.input, DEFAULT_CONFIGS), seed=args.seed, epochs=args.epochs)
            print(json.dumps(rows, indent=2, sort_keys=True))


        if __name__ == "__main__":
            main()
        '''
    ).lstrip()
    return template.replace("__DEFAULT_CONFIGS_JSON__", repr(configs_json))


def _requirements_txt() -> str:
    return "torch\ntorchvision\ntransformers\ndatasets\nPillow\npandas\nscikit-learn\n"


def _readme_generated_md(
    specs: Sequence[TrainingSpec],
    feedback: str | None = None,
    *,
    provider: str = "none",
    model_source: str = "template",
) -> str:
    candidate_lines = "\n".join(
        f"- rank {spec.rank}: {spec.task_type}, backbone={spec.backbone}, "
        f"loss={spec.loss}, optimizer={spec.optimizer}, finetune={spec.finetune_strategy}"
        for spec in specs
    )
    return dedent(
        f"""
        # Generated Module 4 Project

        This folder was generated from Module 3 candidate configurations. The
        structured `model_config` fields drive the generated code; task text is
        kept only for context. The project runs local smoke checks and does not
        perform long training.

        ## Candidates

        {candidate_lines}

        ## Config Contract

        - `configs.json` contains the normalized Module 4 configs consumed by
          the generated scripts.
        - `generation_info.json` records whether `model.py` came from a model
          provider or from the template fallback.
        - `model_config` remains the provenance record from Module 3.
        - If `model_config` and natural-language `tasks` disagree, generated
          code follows the structured config.

        ## Code Generation

        - model.py source: `{model_source}`
        - configured provider: `{provider}`
        - set `M4_LLM_PROVIDER=qwen` to request Qwen generation for `model.py`.
        - set `M4_LLM_PROVIDER=none` for template-only generation.

        ## Files

        - `configs.json`: normalized candidate configs used by this project.
        - `generation_info.json`: records provider and fallback status.
        - `utils.py`: shared config parsing, seed, and task-type helpers.
        - `model_utils.py`: shared backbone loading and freeze helpers.
        - `smoke_data.py`: shared synthetic data helpers for local smoke runs.
        - `model.py`: task-compatible PyTorch models with `build_model(config)`.
          Uses TinyBackbone in smoke mode, real pretrained backbone otherwise.
        - `train.py`: training loop with HuggingFace, Kaggle CSV/image, and
          ImageFolder dataloaders, strong augmentation, class weighting,
          mixed precision, validation, early stopping, resumable training,
          and best/last checkpoint saving when `offline_smoke: false`.
        - `evaluate.py`: metrics by task type.
        - `infer.py`: `predict(weights_path=None, image=None, config=None)`.
        - `run.py`: single-configuration runner (smoke or real).
        - `run_experiments.py`: sweeps every Module 3 candidate.

        ## Usage

        Smoke check (fast, offline, CPU):
        ```bash
        python run.py --config configs.json
        python run_experiments.py --input configs.json
        ```

        Real training (set `offline_smoke: false` in configs.json first):
        ```bash
        python run.py --config configs.json --epochs 20
        python run.py --config configs.json --dataset uoft-cs/cifar10 --epochs 10
        python run_experiments.py --input configs.json --epochs 5
        ```

        ## Smoke vs Real Training

        Smoke runs (`offline_smoke: true`, the default) never download weights:
        backbones are randomly initialized so the checks stay fast and offline.
        The local smoke path verifies tensor shapes, loss computation, backward
        pass, optimizer step, evaluation output, inference output, and
        experiment sweep coverage.

        For real training, set `offline_smoke: false` and keep
        `use_pretrained: true` in the config.  What changes:
        - `model.py` loads the real backbone via `model_utils.load_backbone`
          (HuggingFace checkpoint → torchvision → TinyBackbone fallback)
        - `train.py` loads either the HuggingFace dataset specified by
          `dataset_id`, a local CSV dataset specified by `train_csv`, or an
          ImageFolder dataset specified by `image_dir`
        - Multi-epoch training with per-epoch logging
        - Checkpoints saved to `checkpoints/` after each epoch
        - Requires: `pip install transformers datasets Pillow`

        ## Current Limitations

        - Real dataloader supports classification and feature_extraction;
          detection / segmentation still use synthetic data.
        - Object detection and segmentation metrics are simplified,
          not benchmark scores.
        - Module 3 controls candidate scale; this project only executes the
          supplied configs.

        {_feedback_section(feedback)}
        """
    ).lstrip()


def _feedback_section(feedback: str | None) -> str:
    if not feedback:
        return ""
    sanitized = feedback.strip().replace("```", "'''")
    return f"""## Previous Review Notes

This project was regenerated after these review notes:

```text
{sanitized}
```
"""

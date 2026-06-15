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
    "imagenet_prior.py",
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
        "imagenet_prior.py": _imagenet_prior_py(),
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
                "augmentation": config.get("augmentation", "basic"),
                "scheduler": config.get("scheduler", "cosine"),
                "learning_rate": config.get("learning_rate"),
                "backbone_learning_rate": config.get("backbone_learning_rate"),
                "head_learning_rate": config.get("head_learning_rate"),
                "unfreeze_last_n_blocks": config.get("unfreeze_last_n_blocks", 0),
                "label_smoothing": config.get("label_smoothing", 0.0),
                "mixup_alpha": config.get("mixup_alpha", 0.0),
                "cutmix_alpha": config.get("cutmix_alpha", 0.0),
                "tta_horizontal_flip": config.get("tta_horizontal_flip", False),
                "imagenet_prior_blend": config.get("imagenet_prior_blend", False),
                "imagenet_prior_model": config.get("imagenet_prior_model", ""),
                "fold_count": config.get("fold_count", 1),
                "fold_index": config.get("fold_index", 0),
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
            "inception": "inception_v3",
            "inception_v3": "inception_v3",
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

            Transformer encoders return [B, seq, D]; we mean-pool to [B, D] so
            heads can treat the output like any 2D feature vector.
            """

            def __init__(self, model: nn.Module) -> None:
                super().__init__()
                self.model = model
                self.model_type = str(
                    getattr(getattr(model, "config", None), "model_type", "")
                ).lower()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                out = self.model(pixel_values=x)
                hidden = getattr(out, "last_hidden_state", None)
                if hidden is None:
                    hidden = out[0] if isinstance(out, (tuple, list)) else out
                if hidden.dim() == 3:
                    if self.model_type in {"dinov2", "dinov2_with_registers"}:
                        register_tokens = int(
                            getattr(getattr(self.model, "config", None), "num_register_tokens", 0)
                            or 0
                        )
                        patch_tokens = hidden[:, 1 + register_tokens :]
                        return torch.cat(
                            [hidden[:, 0], patch_tokens.mean(dim=1)],
                            dim=1,
                        )
                    if self.model_type in {"beit", "deit", "vit"}:
                        return hidden[:, 0]
                    pooled = getattr(out, "pooler_output", None)
                    if pooled is not None and pooled.dim() == 2:
                        return pooled
                    return hidden.mean(dim=1)
                return hidden


        def _try_huggingface(hf_id: str, image_size: int) -> tuple[nn.Module, int] | None:
            """Load the exact HuggingFace checkpoint chosen by Module 3.

            Requires the optional ``transformers`` dependency and network access
            on first download. Returns None on any failure so the caller can
            fall back to torchvision.
            """
            try:
                from transformers import AutoModel
                model = AutoModel.from_pretrained(hf_id)
                backbone = _HFBackbone(model)
                channels = _infer_channels(backbone, image_size)
                return backbone, channels
            except Exception as exc:
                print(f"[model_utils] HuggingFace checkpoint {hf_id!r} unavailable ({exc}); falling back.")
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
                    loaded = _try_huggingface(hf_id, image_size)
                    if loaded is not None:
                        return loaded

            model = _try_torchvision(name, pretrained=pretrained)
            if model is None:
                bb = TinyBackbone()
                return bb, bb.out_channels

            extractor = _extract_features(model)
            channels = _infer_channels(extractor, image_size)
            return extractor, channels


        def _resolve_module(root: nn.Module, path: str):
            current = root
            for part in path.split("."):
                current = getattr(current, part, None)
                if current is None:
                    return None
            return current


        def _encoder_blocks(backbone: nn.Module) -> list[nn.Module]:
            for path in (
                "model.encoder.layer",
                "model.encoder.layers",
                "model.blocks",
                "encoder.layer",
                "encoder.layers",
                "blocks",
                "features",
                "layers",
            ):
                value = _resolve_module(backbone, path)
                if isinstance(value, (nn.ModuleList, nn.Sequential, list, tuple)):
                    return list(value)
            return []


        def apply_freeze(model: nn.Module, config: dict[str, Any] | None) -> None:
            """Apply head-only, partial, or full backbone fine-tuning."""
            config = config or {}
            strategy = str(get_value(config, "finetune_strategy", "head_only")).lower()
            backbone = getattr(model, "backbone", None)
            if backbone is None:
                return

            if strategy == "head_only":
                for parameter in backbone.parameters():
                    parameter.requires_grad = False
            elif strategy == "partial":
                for parameter in backbone.parameters():
                    parameter.requires_grad = False
                count = max(1, as_int(get_value(config, "unfreeze_last_n_blocks", 2), 2))
                blocks = _encoder_blocks(backbone)
                if not blocks:
                    raise ValueError(
                        "partial finetuning requires a backbone with identifiable encoder blocks."
                    )
                for block in blocks[-count:]:
                    for parameter in block.parameters():
                        parameter.requires_grad = True
                for path in (
                    "model.layernorm",
                    "model.norm",
                    "model.post_layernorm",
                    "layernorm",
                    "norm",
                ):
                    norm = _resolve_module(backbone, path)
                    if isinstance(norm, nn.Module):
                        for parameter in norm.parameters():
                            parameter.requires_grad = True
            elif strategy in {"full", "either"}:
                for parameter in backbone.parameters():
                    parameter.requires_grad = True
            elif as_bool(get_value(config, "freeze_backbone", False), False):
                for parameter in backbone.parameters():
                    parameter.requires_grad = False

            model._frozen_backbone_params = sum(
                1 for parameter in backbone.parameters() if not parameter.requires_grad
            )
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
        import time
        from pathlib import Path
        from typing import Any

        import torch
        import torch.nn.functional as F

        from model import build_model
        from smoke_data import synthetic_batch
        from utils import as_bool, as_float, as_int, get_value, task_type


        def _build_optimizer(model: torch.nn.Module, config: dict[str, Any] | None) -> torch.optim.Optimizer:
            optimizer_name = str(get_value(config, "optimizer", "adamw")).lower()
            lr = as_float(get_value(config, "learning_rate", 1.0e-3), 1.0e-3)
            backbone_lr = as_float(
                get_value(config, "backbone_learning_rate", lr),
                lr,
            )
            head_lr = as_float(get_value(config, "head_learning_rate", lr), lr)
            backbone_parameters = []
            head_parameters = []
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if name.startswith("backbone."):
                    backbone_parameters.append(parameter)
                else:
                    head_parameters.append(parameter)
            parameter_groups = []
            if backbone_parameters:
                parameter_groups.append({"params": backbone_parameters, "lr": backbone_lr})
            if head_parameters:
                parameter_groups.append({"params": head_parameters, "lr": head_lr})
            if not parameter_groups:
                parameter_groups = [{"params": list(model.parameters()), "lr": lr}]
            if "sgd" in optimizer_name:
                return torch.optim.SGD(parameter_groups, lr=lr, momentum=0.9)
            if "rmsprop" in optimizer_name:
                return torch.optim.RMSprop(parameter_groups, lr=lr)
            if optimizer_name == "adam":
                return torch.optim.Adam(parameter_groups, lr=lr)
            return torch.optim.AdamW(parameter_groups, lr=lr)


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
                if isinstance(target, torch.Tensor) and target.ndim == 2:
                    return -(target * F.log_softmax(output, dim=1)).sum(dim=1).mean()
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
            augmentation = str(get_value(config, "augmentation", "basic") or "basic").lower()
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            if split == "train" and augmentation in {"strong", "competition", "advanced"}:
                return transforms.Compose([
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(0.65, 1.0),
                        ratio=(0.75, 1.3333333333),
                    ),
                    transforms.RandomHorizontalFlip(),
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
            if split == "train" and augmentation in {"randaugment", "rand_aug"}:
                num_ops = as_int(get_value(config, "randaugment_num_ops", 2), 2)
                magnitude = as_int(get_value(config, "randaugment_magnitude", 9), 9)
                return transforms.Compose([
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(0.7, 1.0),
                        ratio=(0.75, 1.3333333333),
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandAugment(num_ops=num_ops, magnitude=magnitude),
                    transforms.ToTensor(),
                    normalize,
                ])
            if split == "train" and augmentation in {"none", "off", "deterministic"}:
                resize_size = max(image_size, round(image_size * 1.1))
                return transforms.Compose([
                    transforms.Resize(resize_size),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    normalize,
                ])
            if split == "train":
                return transforms.Compose([
                    transforms.RandomResizedCrop(
                        image_size,
                        scale=(0.8, 1.0),
                        ratio=(0.75, 1.3333333333),
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ])
            resize_size = max(image_size, round(image_size * 1.1))
            return transforms.Compose([
                transforms.Resize(resize_size),
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


        def _split_indices(
            labels: list[int],
            validation_fraction: float,
            seed: int,
            fold_count: int = 1,
            fold_index: int = 0,
        ):
            grouped: dict[int, list[int]] = {}
            for index, label in enumerate(labels):
                grouped.setdefault(int(label), []).append(index)
            rng = random.Random(seed)
            fold_count = max(1, int(fold_count))
            fold_index = int(fold_index)
            if not 0 <= fold_index < fold_count:
                raise ValueError(
                    f"fold_index must be in [0, {fold_count}); got {fold_index}."
                )
            train_indices: list[int] = []
            validation_indices: list[int] = []
            for class_indices in grouped.values():
                shuffled = list(class_indices)
                rng.shuffle(shuffled)
                if len(shuffled) < 2:
                    train_indices.extend(shuffled)
                    continue
                if fold_count > 1:
                    for position, sample_index in enumerate(shuffled):
                        destination = (
                            validation_indices
                            if position % fold_count == fold_index
                            else train_indices
                        )
                        destination.append(sample_index)
                else:
                    validation_count = max(1, round(len(shuffled) * validation_fraction))
                    validation_count = min(validation_count, len(shuffled) - 1)
                    validation_indices.extend(shuffled[:validation_count])
                    train_indices.extend(shuffled[validation_count:])
            rng.shuffle(train_indices)
            rng.shuffle(validation_indices)
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
            fold_count = max(1, as_int(get_value(config, "fold_count", 1), 1))
            fold_index = as_int(get_value(config, "fold_index", 0), 0)
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
                train_indices, validation_indices = _split_indices(
                    encoded_labels,
                    validation_fraction=validation_fraction,
                    seed=seed,
                    fold_count=fold_count,
                    fold_index=fold_index,
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
                fold_count=fold_count,
                fold_index=fold_index,
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
                from sklearn.metrics import cohen_kappa_score, f1_score, log_loss, roc_auc_score
                labels_np = label_values.numpy()
                probabilities_np = probability_values.numpy()
                macro_f1 = float(f1_score(labels_np, pred_values.numpy(), average="macro"))
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
                macro_f1 = 0.0
            return {
                "metric_name": metric_name,
                "metric_value": metric_value,
                "accuracy": accuracy,
                "macro_f1": macro_f1,
            }


        def _classification_probabilities(
            model: torch.nn.Module,
            x: torch.Tensor,
            config: dict[str, Any],
        ) -> torch.Tensor:
            probabilities = torch.softmax(model(x), dim=1)
            if as_bool(get_value(config, "tta_horizontal_flip", False), False):
                flipped = torch.softmax(model(torch.flip(x, dims=[3])), dim=1)
                probabilities = (probabilities + flipped) / 2.0
            return probabilities


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
                    probs = _classification_probabilities(model, x, config)
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
            warmup_epochs = max(
                0,
                min(
                    as_int(get_value(config, "warmup_epochs", 0), 0),
                    max(0, int(epochs) - 1),
                ),
            )
            if warmup_epochs:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=1.0 / max(2, warmup_epochs),
                    end_factor=1.0,
                    total_iters=warmup_epochs,
                )
                cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(1, int(epochs) - warmup_epochs),
                    eta_min=min_lr,
                )
                return torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[warmup_epochs],
                )
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, int(epochs)),
                eta_min=min_lr,
            )


        def _apply_batch_regularization(
            x: torch.Tensor,
            target: torch.Tensor,
            config: dict[str, Any],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Apply exactly one of MixUp or CutMix and return soft targets."""
            mixup_alpha = as_float(get_value(config, "mixup_alpha", 0.0), 0.0)
            cutmix_alpha = as_float(get_value(config, "cutmix_alpha", 0.0), 0.0)
            if mixup_alpha > 0 and cutmix_alpha > 0:
                raise ValueError("MixUp and CutMix cannot be enabled in the same experiment.")
            if mixup_alpha <= 0 and cutmix_alpha <= 0:
                return x, target
            num_classes = max(1, as_int(get_value(config, "num_classes", 3), 3))
            permutation = torch.randperm(x.size(0), device=x.device)
            alpha = mixup_alpha if mixup_alpha > 0 else cutmix_alpha
            lam = float(torch.distributions.Beta(alpha, alpha).sample().item())
            soft = F.one_hot(target, num_classes=num_classes).float()
            permuted_soft = soft[permutation]
            if mixup_alpha > 0:
                return (
                    (lam * x) + ((1.0 - lam) * x[permutation]),
                    (lam * soft) + ((1.0 - lam) * permuted_soft),
                )
            height, width = x.shape[-2:]
            ratio = (1.0 - lam) ** 0.5
            cut_h, cut_w = int(height * ratio), int(width * ratio)
            center_y = int(torch.randint(0, height, (1,), device=x.device).item())
            center_x = int(torch.randint(0, width, (1,), device=x.device).item())
            y1, y2 = max(0, center_y - cut_h // 2), min(height, center_y + cut_h // 2)
            x1, x2 = max(0, center_x - cut_w // 2), min(width, center_x + cut_w // 2)
            mixed = x.clone()
            mixed[:, :, y1:y2, x1:x2] = x[permutation, :, y1:y2, x1:x2]
            adjusted = 1.0 - ((y2 - y1) * (x2 - x1) / max(height * width, 1))
            return mixed, (adjusted * soft) + ((1.0 - adjusted) * permuted_soft)


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


        def _get_or_extract_features(model, loader, device, config, tag, checkpoint_dir):
            """Extract (or load from disk) the frozen-backbone features for one split."""
            cache_dir = Path(
                str(get_value(config, "feature_cache_dir", str(Path(checkpoint_dir) / "feature_cache")))
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            backbone_name = str(get_value(config, "backbone", "backbone"))
            image_size = as_int(get_value(config, "image_size", 224), 224)
            count = len(loader.dataset)
            cache_path = cache_dir / f"feat_{tag}_{backbone_name}_{image_size}_{count}.pt"
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
                    and str(get_value(config, "augmentation", "basic")).lower()
                    in {"none", "off", "deterministic"}
                    and as_float(get_value(config, "mixup_alpha", 0.0), 0.0) <= 0
                    and as_float(get_value(config, "cutmix_alpha", 0.0), 0.0) <= 0
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
                        if task == "classification":
                            x, target = _apply_batch_regularization(x, target, config)
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
                "config_summary": {
                    "rank": get_value(config, "rank", None),
                    "backbone": get_value(config, "backbone", "tiny_cnn"),
                    "loss": get_value(config, "loss", "cross_entropy_loss"),
                    "optimizer": get_value(config, "optimizer", "adamw"),
                    "finetune_strategy": get_value(config, "finetune_strategy", "head_only"),
                    "unfreeze_last_n_blocks": get_value(config, "unfreeze_last_n_blocks", 0),
                    "augmentation": get_value(config, "augmentation", "basic"),
                    "mixup_alpha": get_value(config, "mixup_alpha", 0.0),
                    "cutmix_alpha": get_value(config, "cutmix_alpha", 0.0),
                    "label_smoothing": get_value(config, "label_smoothing", 0.0),
                    "scheduler": get_value(config, "scheduler", "cosine"),
                    "learning_rate": get_value(config, "learning_rate", 1.0e-3),
                    "backbone_learning_rate": get_value(
                        config, "backbone_learning_rate", None
                    ),
                    "head_learning_rate": get_value(config, "head_learning_rate", None),
                    "tta_horizontal_flip": get_value(config, "tta_horizontal_flip", False),
                    "fold_count": get_value(config, "fold_count", 1),
                    "fold_index": get_value(config, "fold_index", 0),
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


def _imagenet_prior_py() -> str:
    return dedent(
        '''
        """ImageNet dog-category prior projection and validation calibration."""

        from __future__ import annotations

        import re
        from pathlib import Path
        from typing import Any

        import numpy as np
        import torch

        from utils import as_bool, as_int, get_value


        def normalize_category_name(value: Any) -> str:
            """Normalize Kaggle snake-case and ImageNet display names identically."""
            return " ".join(
                re.sub(r"[^a-z0-9]+", " ", str(value).lower()).split()
            )


        def build_label_projection(
            label_names: list[str],
            imagenet_categories: list[str],
        ) -> list[int]:
            """Return ImageNet indices in the exact training-label order."""
            category_to_index = {}
            for index, category in enumerate(imagenet_categories):
                category_to_index.setdefault(
                    normalize_category_name(category),
                    index,
                )
            missing = [
                label
                for label in label_names
                if normalize_category_name(label) not in category_to_index
            ]
            if missing:
                raise ValueError(
                    "ImageNet prior cannot map training labels: "
                    + ", ".join(str(value) for value in missing[:10])
                )
            return [
                category_to_index[normalize_category_name(label)]
                for label in label_names
            ]


        def multiclass_log_loss(labels, probabilities) -> float:
            matrix = np.asarray(probabilities, dtype=np.float64)
            matrix = np.clip(matrix, 1.0e-15, 1.0)
            matrix /= matrix.sum(axis=1, keepdims=True)
            targets = np.asarray(labels, dtype=np.int64)
            return float(-np.log(matrix[np.arange(len(targets)), targets]).mean())


        def temperature_scale_probabilities(
            probabilities,
            temperature: float,
        ) -> np.ndarray:
            matrix = np.clip(
                np.asarray(probabilities, dtype=np.float64),
                1.0e-15,
                1.0,
            )
            scaled = matrix ** (1.0 / max(float(temperature), 1.0e-6))
            scaled /= scaled.sum(axis=1, keepdims=True)
            return scaled.astype("float32")


        def calibrate_temperature(
            probabilities,
            labels,
        ) -> tuple[float, np.ndarray, float]:
            """Choose a compact validation temperature grid for log loss."""
            candidates = np.concatenate(
                [
                    np.linspace(0.5, 1.5, 21),
                    np.asarray([1.75, 2.0], dtype=float),
                ]
            )
            best_temperature = 1.0
            best_matrix = np.asarray(probabilities, dtype=np.float32)
            best_loss = multiclass_log_loss(labels, best_matrix)
            for temperature in candidates:
                scaled = temperature_scale_probabilities(
                    probabilities,
                    float(temperature),
                )
                loss = multiclass_log_loss(labels, scaled)
                if loss < best_loss - 1.0e-12:
                    best_temperature = float(temperature)
                    best_matrix = scaled
                    best_loss = float(loss)
            return best_temperature, best_matrix, best_loss


        def calibrate_probability_ensemble(
            probability_sets: list[np.ndarray],
            labels,
        ) -> tuple[np.ndarray, list[float], list[float], float]:
            """Temperature-calibrate up to two prior models and select their blend."""
            if not probability_sets:
                raise ValueError("At least one prior probability set is required.")
            if len(probability_sets) > 2:
                raise ValueError("At most two ImageNet prior models are supported.")
            calibrated = []
            temperatures = []
            for probabilities in probability_sets:
                temperature, matrix, _loss = calibrate_temperature(
                    probabilities,
                    labels,
                )
                temperatures.append(temperature)
                calibrated.append(matrix)
            if len(calibrated) == 1:
                loss = multiclass_log_loss(labels, calibrated[0])
                return calibrated[0], temperatures, [1.0], loss

            best_matrix = calibrated[0]
            best_weights = [1.0, 0.0]
            best_loss = multiclass_log_loss(labels, best_matrix)
            for unit in range(21):
                first_weight = unit / 20.0
                combined = (
                    first_weight * calibrated[0]
                    + (1.0 - first_weight) * calibrated[1]
                )
                combined = np.clip(combined, 1.0e-15, 1.0)
                combined /= combined.sum(axis=1, keepdims=True)
                loss = multiclass_log_loss(labels, combined)
                if loss < best_loss - 1.0e-12:
                    best_matrix = combined.astype("float32")
                    best_weights = [first_weight, 1.0 - first_weight]
                    best_loss = float(loss)
            return best_matrix, temperatures, best_weights, best_loss


        def calibrate_probability_blend(
            learned_probabilities,
            prior_probabilities,
            labels,
            *,
            step: float = 0.05,
        ) -> tuple[float, np.ndarray, float]:
            """Select a convex prior blend using validation multiclass log loss."""
            learned = np.asarray(learned_probabilities, dtype=np.float64)
            prior = np.asarray(prior_probabilities, dtype=np.float64)
            targets = np.asarray(labels, dtype=np.int64)
            if learned.shape != prior.shape:
                raise ValueError("Learned and ImageNet-prior probabilities must align.")
            if len(learned) != len(targets):
                raise ValueError("Probability rows and labels must align.")
            units = max(1, round(1.0 / float(step)))
            best_alpha = 0.0
            best_matrix = learned
            best_loss = multiclass_log_loss(targets, learned)
            for unit in range(1, units + 1):
                alpha = unit / units
                combined = ((1.0 - alpha) * learned) + (alpha * prior)
                combined = np.clip(combined, 1.0e-15, 1.0)
                combined /= combined.sum(axis=1, keepdims=True)
                loss = multiclass_log_loss(targets, combined)
                if loss < best_loss - 1.0e-12:
                    best_alpha = float(alpha)
                    best_matrix = combined
                    best_loss = float(loss)
            return best_alpha, np.asarray(best_matrix, dtype=np.float32), best_loss


        def calibrate_probability_fusion(
            learned_probabilities,
            prior_probabilities,
            labels,
        ) -> tuple[float, np.ndarray, float, float, float]:
            learned_temperature, learned, _learned_loss = calibrate_temperature(
                learned_probabilities,
                labels,
            )
            prior_temperature, prior, _prior_loss = calibrate_temperature(
                prior_probabilities,
                labels,
            )
            alpha, combined, loss = calibrate_probability_blend(
                learned,
                prior,
                labels,
            )
            return (
                alpha,
                combined,
                loss,
                learned_temperature,
                prior_temperature,
            )


        def _prior_enabled(config: dict[str, Any]) -> bool:
            value = get_value(config, "imagenet_prior_blend", False)
            if isinstance(value, str):
                return value.strip().lower() in {"auto", "true", "yes", "on", "1"}
            return bool(value)


        def prior_model_specs(config: dict[str, Any]) -> list[str]:
            raw = str(
                get_value(
                    config,
                    "imagenet_prior_model",
                    "efficientnet_v2_s",
                )
                or "efficientnet_v2_s"
            )
            specs = [item.strip() for item in raw.split(",") if item.strip()]
            if not specs:
                return ["efficientnet_v2_s"]
            if len(specs) > 2:
                raise ValueError("At most two ImageNet prior models are supported.")
            return specs


        def _load_prior_model(model_spec: str, device: torch.device):
            import torchvision.models as models

            model_name, separator, weight_name = model_spec.partition("@")
            model_name = model_name.strip().lower()
            try:
                weight_enum = models.get_model_weights(model_name)
                weights = (
                    getattr(weight_enum, weight_name.strip())
                    if separator and weight_name.strip()
                    else weight_enum.DEFAULT
                )
                model = models.get_model(model_name, weights=weights)
            except (AttributeError, KeyError, ValueError) as exc:
                raise ValueError(
                    f"Unsupported torchvision ImageNet prior model: {model_spec}"
                ) from exc
            model.to(device).eval()
            for parameter in model.parameters():
                parameter.requires_grad = False
            return model, weights, model_spec


        def _effective_prior_batch_size(model_spec: str, requested: int) -> int:
            model_name = model_spec.partition("@")[0].strip().lower()
            limit = 8 if model_name.startswith(("vit_", "swin")) else 16
            return max(1, min(limit, int(requested)))


        def _csv_prior_loader(
            config: dict[str, Any],
            transform,
            *,
            split: str,
            batch_size: int,
        ):
            import pandas as pd
            from PIL import Image
            from train import _split_indices

            csv_path = Path(str(get_value(config, "train_csv", ""))).expanduser().resolve()
            if not csv_path.is_file():
                raise FileNotFoundError(
                    "ImageNet prior currently requires a local train_csv."
                )
            frame = pd.read_csv(csv_path)
            image_column = str(get_value(config, "image_column", "image") or "image")
            label_column = str(get_value(config, "label_column", "label") or "label")
            label_values = frame[label_column].tolist()
            label_names = sorted(set(label_values), key=lambda value: str(value))
            label_to_index = {value: index for index, value in enumerate(label_names)}
            encoded_labels = [label_to_index[value] for value in label_values]
            train_indices, validation_indices = _split_indices(
                encoded_labels,
                validation_fraction=float(get_value(config, "validation_fraction", 0.2)),
                seed=as_int(get_value(config, "seed", 42), 42),
                fold_count=max(1, as_int(get_value(config, "fold_count", 1), 1)),
                fold_index=as_int(get_value(config, "fold_index", 0), 0),
            )
            selected_indices = train_indices if split == "train" else validation_indices
            max_samples_key = "max_train_samples" if split == "train" else "max_eval_samples"
            max_samples = as_int(get_value(config, max_samples_key, 0), 0)
            if max_samples > 0:
                selected_indices = selected_indices[:max_samples]
            selected_frame = frame.iloc[selected_indices].reset_index(drop=True)
            selected_labels = [encoded_labels[index] for index in selected_indices]
            base_dir_value = str(get_value(config, "image_dir", "") or "")
            base_dir = (
                Path(base_dir_value).expanduser().resolve()
                if base_dir_value
                else csv_path.parent
            )
            path_template = str(
                get_value(config, "image_path_template", "{image}") or "{image}"
            )
            image_extension = str(get_value(config, "image_extension", "") or "")

            class PriorDataset(torch.utils.data.Dataset):
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
                    return tensor, torch.tensor(
                        selected_labels[index],
                        dtype=torch.long,
                    )

            workers = as_int(get_value(config, "num_workers", 2), 2)
            loader = torch.utils.data.DataLoader(
                PriorDataset(),
                batch_size=max(1, int(batch_size)),
                shuffle=False,
                num_workers=workers,
                pin_memory=torch.cuda.is_available(),
                persistent_workers=workers > 0,
            )
            return loader, [str(value) for value in label_names]


        def _project_logits(
            logits: torch.Tensor,
            projection: list[int],
        ) -> torch.Tensor:
            probabilities = torch.softmax(logits, dim=1)[:, projection]
            return probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1.0e-15)


        def validation_prior_probabilities(
            config: dict[str, Any],
            *,
            device: torch.device,
            batch_size: int,
        ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
            """Run and validation-calibrate native ImageNet classifiers."""
            model_specs = prior_model_specs(config)
            probability_sets = []
            labels_reference = None
            label_names_reference = None
            projections = []
            use_tta = as_bool(
                get_value(config, "tta_horizontal_flip", False),
                False,
            )
            for model_spec in model_specs:
                model, weights, _loaded_spec = _load_prior_model(model_spec, device)
                loader, label_names = _csv_prior_loader(
                    config,
                    weights.transforms(),
                    split="test",
                    batch_size=_effective_prior_batch_size(
                        model_spec,
                        batch_size,
                    ),
                )
                projection = build_label_projection(
                    label_names,
                    list(weights.meta["categories"]),
                )
                probability_batches = []
                label_batches = []
                with torch.no_grad():
                    for images, labels in loader:
                        images = images.to(device, non_blocking=True)
                        probabilities = _project_logits(model(images), projection)
                        if use_tta:
                            flipped = _project_logits(
                                model(torch.flip(images, dims=[3])),
                                projection,
                            )
                            probabilities = (probabilities + flipped) / 2.0
                        probability_batches.append(probabilities.cpu())
                        label_batches.append(labels.cpu())
                current_probabilities = (
                    torch.cat(probability_batches).numpy().astype("float32")
                )
                current_labels = torch.cat(label_batches).numpy().astype("int64")
                if labels_reference is None:
                    labels_reference = current_labels
                    label_names_reference = label_names
                elif not np.array_equal(labels_reference, current_labels):
                    raise ValueError("ImageNet prior validation models are not aligned.")
                probability_sets.append(current_probabilities)
                projections.append(projection)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            (
                combined,
                component_temperatures,
                component_weights,
                component_log_loss,
            ) = calibrate_probability_ensemble(
                probability_sets,
                labels_reference,
            )
            metadata = {
                "prior_model": ",".join(model_specs),
                "prior_models": model_specs,
                "prior_component_temperatures": component_temperatures,
                "prior_component_weights": component_weights,
                "prior_component_log_loss": component_log_loss,
                "imagenet_indices": projections[0],
                "imagenet_indices_by_model": projections,
                "label_names": label_names_reference,
            }
            return combined, labels_reference, metadata


        def predict_prior_directory(
            config: dict[str, Any],
            image_dir: str | Path,
            *,
            batch_size: int = 16,
            component_temperatures: list[float] | None = None,
            component_weights: list[float] | None = None,
        ) -> tuple[list[tuple[str, list[float]]], dict[str, Any]]:
            """Run the validation-selected native ImageNet prior ensemble."""
            import pandas as pd
            from PIL import Image

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model_specs = prior_model_specs(config)
            csv_path = Path(str(get_value(config, "train_csv", ""))).expanduser().resolve()
            frame = pd.read_csv(csv_path)
            label_column = str(get_value(config, "label_column", "label") or "label")
            label_names = sorted(
                {str(value) for value in frame[label_column].tolist()},
                key=str,
            )
            extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
            files = sorted(
                path
                for path in Path(image_dir).rglob("*")
                if path.suffix.lower() in extensions
            )
            if not files:
                raise FileNotFoundError(f"No images found under {image_dir}")

            use_tta = as_bool(
                get_value(config, "tta_horizontal_flip", False),
                False,
            )
            prediction_matrices = []
            names_reference = None
            projections = []
            for model_spec in model_specs:
                model, weights, _loaded_spec = _load_prior_model(model_spec, device)
                projection = build_label_projection(
                    label_names,
                    list(weights.meta["categories"]),
                )
                transform = weights.transforms()

                class PriorTestDataset(torch.utils.data.Dataset):
                    def __len__(self):
                        return len(files)

                    def __getitem__(self, index):
                        path = files[index]
                        with Image.open(path) as image:
                            tensor = transform(image.convert("RGB"))
                        return tensor, path.name

                loader = torch.utils.data.DataLoader(
                    PriorTestDataset(),
                    batch_size=_effective_prior_batch_size(
                        model_spec,
                        batch_size,
                    ),
                    shuffle=False,
                    num_workers=as_int(get_value(config, "num_workers", 2), 2),
                    pin_memory=torch.cuda.is_available(),
                )
                probability_batches = []
                current_names = []
                with torch.no_grad():
                    for images, names in loader:
                        images = images.to(device, non_blocking=True)
                        probabilities = _project_logits(model(images), projection)
                        if use_tta:
                            flipped = _project_logits(
                                model(torch.flip(images, dims=[3])),
                                projection,
                            )
                            probabilities = (probabilities + flipped) / 2.0
                        probability_batches.append(probabilities.cpu())
                        current_names.extend(str(name) for name in names)
                matrix = torch.cat(probability_batches).numpy().astype("float32")
                if names_reference is None:
                    names_reference = current_names
                elif names_reference != current_names:
                    raise ValueError("ImageNet prior test models are not aligned.")
                prediction_matrices.append(matrix)
                projections.append(projection)
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            temperatures = list(component_temperatures or [1.0] * len(model_specs))
            weights = list(component_weights or [1.0 / len(model_specs)] * len(model_specs))
            if len(temperatures) != len(model_specs) or len(weights) != len(model_specs):
                raise ValueError("ImageNet prior calibration metadata is not aligned.")
            calibrated = [
                temperature_scale_probabilities(matrix, temperature)
                for matrix, temperature in zip(prediction_matrices, temperatures)
            ]
            normalized_weights = np.asarray(weights, dtype=float)
            normalized_weights /= normalized_weights.sum()
            combined = sum(
                weight * matrix
                for weight, matrix in zip(normalized_weights, calibrated)
            )
            combined = np.clip(combined, 1.0e-15, 1.0)
            combined /= combined.sum(axis=1, keepdims=True)
            results = [
                (name, [float(value) for value in row])
                for name, row in zip(names_reference, combined)
            ]
            return results, {
                "prior_model": ",".join(model_specs),
                "prior_models": model_specs,
                "prior_component_temperatures": temperatures,
                "prior_component_weights": [float(value) for value in normalized_weights],
                "imagenet_indices": projections[0],
                "imagenet_indices_by_model": projections,
                "label_names": label_names,
            }


        def apply_validation_prior(
            learned_probabilities,
            labels,
            config: dict[str, Any],
            *,
            device: torch.device,
            batch_size: int,
        ) -> tuple[np.ndarray, dict[str, Any]]:
            """Return validation-selected learned/native ImageNet probability blend."""
            learned = np.asarray(learned_probabilities, dtype=np.float32)
            if not _prior_enabled(config):
                return learned, {
                    "prior_alpha": 0.0,
                    "prior_model": "",
                    "prior_models": [],
                    "prior_component_temperatures": [],
                    "prior_component_weights": [],
                    "learned_temperature": 1.0,
                    "prior_temperature": 1.0,
                    "imagenet_indices": [],
                    "label_names": [],
                }
            prior, prior_labels, metadata = validation_prior_probabilities(
                config,
                device=device,
                batch_size=batch_size,
            )
            targets = np.asarray(labels, dtype=np.int64)
            if not np.array_equal(targets, prior_labels):
                raise ValueError("ImageNet-prior validation labels are not aligned.")
            (
                alpha,
                combined,
                loss,
                learned_temperature,
                prior_temperature,
            ) = calibrate_probability_fusion(
                learned,
                prior,
                targets,
            )
            metadata.update(
                {
                    "prior_alpha": alpha,
                    "prior_log_loss": multiclass_log_loss(targets, prior),
                    "learned_log_loss": multiclass_log_loss(targets, learned),
                    "learned_temperature": learned_temperature,
                    "prior_temperature": prior_temperature,
                    "combined_log_loss": loss,
                }
            )
            return combined, metadata
        '''
    ).lstrip()


def _evaluate_py() -> str:
    return dedent(
        '''
        """Evaluation helpers for generated configs."""

        from __future__ import annotations

        from pathlib import Path
        from typing import Any

        import numpy as np
        import torch

        from imagenet_prior import apply_validation_prior
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
                        probabilities = torch.softmax(model(x), dim=1)
                        if as_bool(get_value(config, "tta_horizontal_flip", False), False):
                            flipped = torch.softmax(model(torch.flip(x, dims=[3])), dim=1)
                            probabilities = (probabilities + flipped) / 2.0
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
                labels = torch.cat(all_labels).cpu()
                probabilities = torch.cat(all_probabilities).cpu()
                probability_values, prior_metadata = apply_validation_prior(
                    probabilities.numpy(),
                    labels.numpy(),
                    config,
                    device=device,
                    batch_size=min(
                        16,
                        max(1, int(getattr(dataloader, "batch_size", 16) or 16)),
                    ),
                )
                probabilities = torch.from_numpy(probability_values)
                preds = probabilities.argmax(dim=1)
                artifact_dir = Path(
                    str(get_value(config, "checkpoint_dir", "checkpoints"))
                )
                artifact_dir.mkdir(parents=True, exist_ok=True)
                validation_artifact = artifact_dir / "validation_probabilities.npz"
                np.savez_compressed(
                    validation_artifact,
                    probabilities=probabilities.numpy().astype("float32"),
                    labels=labels.numpy().astype("int64"),
                    prior_alpha=np.asarray(
                        prior_metadata.get("prior_alpha", 0.0),
                        dtype="float32",
                    ),
                    prior_model=np.asarray(
                        prior_metadata.get("prior_model", ""),
                    ),
                    prior_models=np.asarray(
                        prior_metadata.get("prior_models", []),
                    ),
                    prior_component_temperatures=np.asarray(
                        prior_metadata.get("prior_component_temperatures", []),
                        dtype="float32",
                    ),
                    prior_component_weights=np.asarray(
                        prior_metadata.get("prior_component_weights", []),
                        dtype="float32",
                    ),
                    learned_temperature=np.asarray(
                        prior_metadata.get("learned_temperature", 1.0),
                        dtype="float32",
                    ),
                    prior_temperature=np.asarray(
                        prior_metadata.get("prior_temperature", 1.0),
                        dtype="float32",
                    ),
                    imagenet_indices=np.asarray(
                        prior_metadata.get("imagenet_indices", []),
                        dtype="int64",
                    ),
                )
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
                return {
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "accuracy": accuracy,
                    "macro_f1": _macro_f1(preds, labels, num_classes),
                    "num_samples": len(labels),
                    "params": _count_params(model),
                    "validation_artifact": str(validation_artifact),
                    "prior_alpha": float(prior_metadata.get("prior_alpha", 0.0)),
                    "prior_model": prior_metadata.get("prior_model", ""),
                    "prior_models": prior_metadata.get("prior_models", []),
                    "prior_component_temperatures": prior_metadata.get(
                        "prior_component_temperatures",
                        [],
                    ),
                    "prior_component_weights": prior_metadata.get(
                        "prior_component_weights",
                        [],
                    ),
                    "learned_temperature": float(
                        prior_metadata.get("learned_temperature", 1.0)
                    ),
                    "prior_temperature": float(
                        prior_metadata.get("prior_temperature", 1.0)
                    ),
                    "prior_log_loss": prior_metadata.get("prior_log_loss"),
                    "learned_log_loss": prior_metadata.get("learned_log_loss"),
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
                if (
                    task == "classification"
                    and as_bool(get_value(config, "tta_horizontal_flip", False), False)
                ):
                    output = (
                        torch.softmax(output, dim=1)
                        + torch.softmax(model(torch.flip(x, dims=[3])), dim=1)
                    ) / 2.0

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
        from utils import as_bool, as_int, compact_config_summary, get_value, load_config, set_seed


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
            default_epochs = 1 if offline_smoke else as_int(get_value(config, "recommended_epochs", 10), 10)
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
        from utils import as_bool, as_int, compact_config_summary, get_value, load_configs, set_seed


        DEFAULT_CONFIGS = json.loads(__DEFAULT_CONFIGS_JSON__)


        def run_all(configs: list[dict[str, Any]], seed: int = 123, epochs: int | None = None) -> list[dict[str, Any]]:
            rows = []
            for index, config in enumerate(configs, start=1):
                set_seed(seed)
                offline_smoke = as_bool(get_value(config, "offline_smoke", True), True)
                default_ep = 1 if offline_smoke else as_int(get_value(config, "recommended_epochs", 10), 10)
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

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


def _augmentation_recipe(config: dict[str, Any]):
    """Return the structured recipe augmentation {tier, invariance, schedule} or None.

    Module 3's recipe layer emits this under model_config["recipe"]; the generated
    config lifts "recipe" to the top level. Falls back to a top-level "augmentation"
    dict if one is present. A legacy string augmentation (or nothing) → None.
    """
    recipe = get_value(config, "recipe", None)
    aug = recipe.get("augmentation") if isinstance(recipe, dict) else None
    if aug is None:
        candidate = get_value(config, "augmentation", None)
        if isinstance(candidate, dict):
            aug = candidate
    return aug if isinstance(aug, dict) else None


def _augmentation_schedule(config: dict[str, Any]) -> str:
    aug = _augmentation_recipe(config)
    return str((aug or {}).get("schedule", "") or "").lower()


def _build_image_transform(config: dict[str, Any], split: str):
    from torchvision import transforms

    image_size = as_int(get_value(config, "image_size", 224), 224)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    if split != "train":
        resize_size = max(image_size, round(image_size * 1.1))
        return transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ])

    # Structured recipe augmentation (Module 3 recipe layer) wins when present: the
    # pipeline is assembled op-by-op from the invariance mask, so grayscale /
    # document / fine-grained / domain vetoes are honored. RandAugment is
    # deliberately NOT used — it bundles color + rotation + shear that would bypass
    # those vetoes; tier controls intensity via crop range + RandomErasing instead.
    recipe_aug = _augmentation_recipe(config)
    if recipe_aug is not None:
        tier = str(recipe_aug.get("tier", "medium")).lower()
        invariance = recipe_aug.get("invariance") or {}
        if tier == "none":
            return transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                normalize,
            ])
        crop_min = as_float(invariance.get("crop_scale_min", 0.65), 0.65)
        crop_min = min(max(crop_min, 0.05), 1.0)
        ops = [transforms.RandomResizedCrop(
            image_size,
            scale=(crop_min, 1.0),
            ratio=(0.75, 1.3333333333),
        )]
        if as_bool(invariance.get("hflip", False), False):
            ops.append(transforms.RandomHorizontalFlip())
        if as_bool(invariance.get("vflip", False), False):
            ops.append(transforms.RandomVerticalFlip())
        if as_bool(invariance.get("rot90", False), False):
            ops.append(transforms.RandomRotation(180))   # orientation-free domain
        if as_bool(invariance.get("color", False), False):
            ops.append(transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05,
            ))
        ops.append(transforms.ToTensor())
        ops.append(normalize)
        if tier == "heavy":
            ops.append(transforms.RandomErasing(
                p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3),
            ))
        return transforms.Compose(ops)

    # Legacy string augmentation (backward compatible).
    augmentation = str(get_value(config, "augmentation", "basic") or "basic").lower()
    if augmentation in {"strong", "competition", "advanced"}:
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
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
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


# Test-time augmentation views. Each maps an input batch to a geometrically
# transformed one; leaves are orientation-free, so flips and 90° rotation are
# label-preserving and average out spatial bias. Dim 2 is height, dim 3 width.
_TTA_TRANSFORMS = {
    "hflip": lambda x: torch.flip(x, dims=[3]),
    "vflip": lambda x: torch.flip(x, dims=[2]),
    "rot90": lambda x: torch.rot90(x, k=1, dims=[2, 3]),
    "rot180": lambda x: torch.rot90(x, k=2, dims=[2, 3]),
    "rot270": lambda x: torch.rot90(x, k=3, dims=[2, 3]),
}


def _resolve_tta_transforms(config: dict[str, Any] | None) -> list[str]:
    """Return the ordered list of extra TTA views requested by ``config``.

    ``tta`` may be a bool (``True`` -> ``["hflip"]``) or a dict
    ``{"enabled": bool, "transforms": [...]}``. Unknown transform names are
    dropped. An empty list means identity-only inference.
    """

    raw = get_value(config, "tta", False)
    if isinstance(raw, dict):
        if not as_bool(raw.get("enabled", True), True):
            return []
        names = raw.get("transforms") or ["hflip"]
    elif as_bool(raw, False):
        names = ["hflip"]
    else:
        return []
    return [str(name).lower() for name in names if str(name).lower() in _TTA_TRANSFORMS]


def _apply_tta(
    model: torch.nn.Module,
    x: torch.Tensor,
    ops: list[str],
) -> torch.Tensor:
    """Average model logits over the identity view plus each TTA view."""

    logits = model(x)
    if not ops:
        return logits
    total = logits
    for name in ops:
        total = total + model(_TTA_TRANSFORMS[name](x))
    return total / (len(ops) + 1)


def _classification_logits(
    model: torch.nn.Module,
    x: torch.Tensor,
    config: dict[str, Any] | None,
) -> torch.Tensor:
    return _apply_tta(model, x, _resolve_tta_transforms(config))


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
            logits = _classification_logits(model, x, config)
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


def _parameter_breakdown(model: torch.nn.Module) -> dict[str, int]:
    breakdown = {
        "total": 0,
        "trainable": 0,
        "backbone_total": 0,
        "backbone_trainable": 0,
        "head_total": 0,
        "head_trainable": 0,
        "other_total": 0,
        "other_trainable": 0,
    }
    for name, parameter in model.named_parameters():
        count = int(parameter.numel())
        trainable = count if parameter.requires_grad else 0
        if name.startswith("backbone"):
            prefix = "backbone"
        elif name.startswith("head"):
            prefix = "head"
        else:
            prefix = "other"
        breakdown["total"] += count
        breakdown["trainable"] += trainable
        breakdown[f"{prefix}_total"] += count
        breakdown[f"{prefix}_trainable"] += trainable
    return breakdown


def _log_model_load_info(
    model: torch.nn.Module,
    model_load_info: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, int]:
    """Print the actual constructed backbone, not just the requested config name."""
    param_breakdown = _parameter_breakdown(model)
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
        f"total={param_breakdown['total']} "
        f"trainable={param_breakdown['trainable']} "
        f"backbone_trainable={param_breakdown['backbone_trainable']} "
        f"head_trainable={param_breakdown['head_trainable']} "
        f"other_trainable={param_breakdown['other_trainable']}",
        file=sys.stderr,
    )
    print(
        "[train] finetune "
        f"strategy={get_value(config, 'finetune_strategy', 'head_only')} "
        f"unfreeze_last_n_blocks={get_value(config, 'unfreeze_last_n_blocks', 0)} "
        f"frozen_backbone_param_tensors={int(getattr(model, '_frozen_backbone_params', 0))} "
        f"partial_unfrozen_param_tensors={int(getattr(model, '_partial_unfrozen_params', 0))}",
        file=sys.stderr,
    )
    if model_load_info.get("fallback_reason"):
        print(
            f"[train] backbone_fallback_reason={model_load_info['fallback_reason']}",
            file=sys.stderr,
        )
    return param_breakdown


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
    param_breakdown = _log_model_load_info(model, model_load_info, config)
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

        # Augmentation taper (recipe schedule): swap the train loader to the
        # eval (no-augmentation) transform for the final 20% of epochs so the
        # model settles on the clean data distribution. Frozen-head path below
        # is already augmentation-free, so this only affects full finetuning.
        aug_schedule = _augmentation_schedule(config)
        _taper_at = int(0.8 * max(1, int(epochs)))
        taper_start_epoch = (
            _taper_at if aug_schedule == "taper_last_20pct" and _taper_at >= 1 else None
        )
        tapered = False

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
            if taper_start_epoch is not None and not tapered and epoch >= taper_start_epoch:
                print(
                    f"[train] Augmentation taper: switching to eval-style "
                    f"preprocessing for the final epochs (from epoch {epoch + 1})."
                )
                dataloader = _build_dataloader(
                    config, split="train", batch_size=batch_size, deterministic=True
                )
                tapered = True
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
        "param_breakdown": param_breakdown,
        "config_summary": {
            "rank": get_value(config, "rank", None),
            "backbone": get_value(config, "backbone", "tiny_cnn"),
            "actual_model": model_load_info.get("actual_model"),
            "backbone_source": model_load_info.get("source"),
            "loss": get_value(config, "loss", "cross_entropy_loss"),
            "optimizer": get_value(config, "optimizer", "adamw"),
            "finetune_strategy": get_value(config, "finetune_strategy", "head_only"),
            "unfreeze_last_n_blocks": int(getattr(model, "_unfreeze_last_n_blocks", 0)),
            "frozen_backbone_params": int(getattr(model, "_frozen_backbone_params", 0)),
            "partial_unfrozen_params": int(getattr(model, "_partial_unfrozen_params", 0)),
            "trainable_params": int(param_breakdown.get("trainable", 0)),
            "backbone_trainable_params": int(param_breakdown.get("backbone_trainable", 0)),
            "head_trainable_params": int(param_breakdown.get("head_trainable", 0)),
        },
    }
    return model, summary


def train_one(config: dict[str, Any] | None, data: tuple[Any, Any] | None = None, epochs: int = 1, max_steps: int = 1) -> dict[str, Any]:
    """Run training and return a summary."""
    _model, summary = train_model(config, data=data, epochs=epochs, max_steps=max_steps)
    return summary

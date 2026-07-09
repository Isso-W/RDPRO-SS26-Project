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


def _classification_logits(
    model: torch.nn.Module,
    x: torch.Tensor,
    config: dict[str, Any] | None,
) -> torch.Tensor:
    logits = model(x)
    if as_bool(get_value(config, "tta", False), False):
        flipped_logits = model(torch.flip(x, dims=[3]))
        logits = (logits + flipped_logits) / 2.0
    return logits


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
                logits = _classification_logits(model, x, config)
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
                    {"y_true": labels.tolist(), "y_prob": probabilities.tolist()},
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
        output = _classification_logits(model, x, config) if task == "classification" else model(x)

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

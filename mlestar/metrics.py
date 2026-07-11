"""Metric-correct, dependency-light scoring for the supported benchmarks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import cohen_kappa_score, log_loss, roc_auc_score

from .contracts import MetricSpec


@dataclass(frozen=True)
class MetricResult:
    """A score together with the direction used to compare candidates."""

    metric: str
    value: float
    greater_is_better: bool


def score_metric(
    metric: str | MetricSpec,
    y_true: Any,
    y_pred: Any,
    **options: Any,
) -> MetricResult:
    """Score predictions using the benchmark's public-evaluation direction.

    ``log_loss`` accepts binary probabilities or an ``(n_samples, n_classes)``
    probability matrix.  ``qwk`` accepts ordinal labels; if probabilities are
    supplied, their argmax is used.  ``detection_map`` accepts one set of boxes
    or a mapping of image id to boxes.
    """

    spec = metric if isinstance(metric, MetricSpec) else MetricSpec(metric)
    if spec.name == "roc_auc":
        value = _roc_auc(y_true, y_pred, **options)
    elif spec.name in {"log_loss", "multiclass_log_loss"}:
        value = _log_loss(y_true, y_pred, **options)
    elif spec.name == "qwk":
        value = _quadratic_weighted_kappa(y_true, y_pred)
    elif spec.name == "rmse":
        value = _rmse(y_true, y_pred)
    elif spec.name == "dice":
        value = _dice(y_true, y_pred, **options)
    elif spec.name == "detection_map":
        value = detection_map(y_true, y_pred, **options)
    else:  # MetricSpec makes this unreachable, retained for static safety.
        raise ValueError(f"Unsupported metric: {spec.name!r}")
    return MetricResult(spec.name, float(value), bool(spec.greater_is_better))


def better_than(
    candidate: float, incumbent: float | None, metric: str | MetricSpec
) -> bool:
    """Return whether ``candidate`` improves on ``incumbent`` for ``metric``."""

    if incumbent is None:
        return True
    spec = metric if isinstance(metric, MetricSpec) else MetricSpec(metric)
    return candidate > incumbent if spec.greater_is_better else candidate < incumbent


def _roc_auc(y_true: Any, y_pred: Any, **options: Any) -> float:
    truth = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    if predictions.ndim == 2 and predictions.shape[1] > 2:
        return float(
            roc_auc_score(
                truth,
                predictions,
                multi_class=options.pop("multi_class", "ovr"),
                average=options.pop("average", "macro"),
                **options,
            )
        )
    if predictions.ndim == 2 and predictions.shape[1] == 2:
        predictions = predictions[:, 1]
    return float(roc_auc_score(truth, predictions, **options))


def _log_loss(y_true: Any, y_pred: Any, **options: Any) -> float:
    predictions = np.asarray(y_pred, dtype=float)
    if predictions.ndim not in {1, 2}:
        raise ValueError("log loss predictions must be a vector or probability matrix")
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        predictions = predictions[:, 0]
    return float(log_loss(y_true, predictions, **options))


def _quadratic_weighted_kappa(y_true: Any, y_pred: Any) -> float:
    truth = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    if predictions.ndim == 2:
        predictions = np.argmax(predictions, axis=1)
    if truth.shape[0] != predictions.shape[0]:
        raise ValueError("qwk y_true and y_pred must have the same number of rows")
    return float(cohen_kappa_score(truth, predictions, weights="quadratic"))


def _rmse(y_true: Any, y_pred: Any) -> float:
    truth = np.asarray(y_true, dtype=float)
    predictions = np.asarray(y_pred, dtype=float)
    if truth.shape != predictions.shape:
        raise ValueError("rmse y_true and y_pred must have matching shapes")
    return float(np.sqrt(np.mean(np.square(truth - predictions))))


def _dice(
    y_true: Any,
    y_pred: Any,
    *,
    threshold: float = 0.5,
    smooth: float = 0.0,
) -> float:
    """Compute foreground Dice over all supplied mask pixels.

    An empty reference and empty prediction has Dice 1.0, which is the standard
    and useful convention for the Ultrasound Nerve benchmark's local folds.
    """

    truth = np.asarray(y_true)
    predictions = np.asarray(y_pred)
    if truth.shape != predictions.shape:
        raise ValueError("dice y_true and y_pred must have matching shapes")
    if smooth < 0:
        raise ValueError("dice smooth must not be negative")
    truth_mask = truth.astype(bool)
    predicted_mask = predictions >= threshold
    intersection = np.logical_and(truth_mask, predicted_mask).sum(dtype=np.float64)
    total = truth_mask.sum(dtype=np.float64) + predicted_mask.sum(dtype=np.float64)
    if total == 0:
        return 1.0
    return float((2.0 * intersection + smooth) / (total + smooth))


def detection_map(
    y_true: Any,
    y_pred: Any,
    *,
    iou_thresholds: Sequence[float] = tuple(np.arange(0.5, 0.76, 0.05)),
) -> float:
    """Global Wheat's per-image mAP at IoU thresholds 0.50 through 0.75.

    Box coordinates are ``[x_min, y_min, x_max, y_max]``.  Predictions can be
    bare boxes, ``{"boxes": ..., "scores": ...}``, or a mapping from image id
    to either representation.  Scores control matching order when present.
    """

    thresholds = tuple(float(value) for value in iou_thresholds)
    if not thresholds or any(value <= 0 or value > 1 for value in thresholds):
        raise ValueError("IoU thresholds must be within (0, 1]")
    truth_by_image = _boxes_by_image(y_true, predictions=False)
    prediction_by_image = _boxes_by_image(y_pred, predictions=True)
    if set(truth_by_image) != set(prediction_by_image):
        raise ValueError("detection y_true and y_pred must contain the same image ids")
    per_image_scores: list[float] = []
    for image_id in truth_by_image:
        truth_boxes, _ = truth_by_image[image_id]
        predicted_boxes, scores = prediction_by_image[image_id]
        order = np.argsort(-scores, kind="stable")
        ordered_predictions = predicted_boxes[order]
        per_threshold: list[float] = []
        for threshold in thresholds:
            matched = np.zeros(len(truth_boxes), dtype=bool)
            true_positives = 0
            false_positives = 0
            for predicted_box in ordered_predictions:
                if len(truth_boxes) == 0:
                    false_positives += 1
                    continue
                overlaps = _iou(predicted_box, truth_boxes)
                best_index = int(np.argmax(overlaps))
                if overlaps[best_index] >= threshold and not matched[best_index]:
                    matched[best_index] = True
                    true_positives += 1
                else:
                    false_positives += 1
            false_negatives = int((~matched).sum())
            denominator = true_positives + false_positives + false_negatives
            per_threshold.append(1.0 if denominator == 0 else true_positives / denominator)
        per_image_scores.append(float(np.mean(per_threshold)))
    return float(np.mean(per_image_scores)) if per_image_scores else 0.0


def _boxes_by_image(value: Any, *, predictions: bool) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    if isinstance(value, Mapping) and "boxes" not in value:
        return {
            str(image_id): _coerce_boxes(boxes, predictions=predictions)
            for image_id, boxes in value.items()
        }
    return {"__single_image__": _coerce_boxes(value, predictions=predictions)}


def _coerce_boxes(value: Any, *, predictions: bool) -> tuple[np.ndarray, np.ndarray]:
    scores: Any = None
    boxes = value
    if isinstance(value, Mapping):
        if "boxes" not in value:
            raise ValueError("detection records must contain a 'boxes' field")
        boxes = value["boxes"]
        scores = value.get("scores")
    array = np.asarray(boxes, dtype=float)
    if array.size == 0:
        array = np.empty((0, 4), dtype=float)
    if array.ndim != 2 or array.shape[1] != 4:
        raise ValueError("detection boxes must have shape (n, 4)")
    if np.any(array[:, 2] < array[:, 0]) or np.any(array[:, 3] < array[:, 1]):
        raise ValueError("detection boxes must be [x_min, y_min, x_max, y_max]")
    if scores is None:
        score_array = np.ones(len(array), dtype=float)
    else:
        score_array = np.asarray(scores, dtype=float)
        if score_array.shape != (len(array),):
            raise ValueError("detection scores must contain one value per box")
    if not predictions and scores is not None:
        raise ValueError("ground-truth boxes must not include confidence scores")
    return array, score_array


def _iou(box: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    left_top = np.maximum(box[:2], candidates[:, :2])
    right_bottom = np.minimum(box[2:], candidates[:, 2:])
    intersection_sizes = np.maximum(0.0, right_bottom - left_top)
    intersection = intersection_sizes[:, 0] * intersection_sizes[:, 1]
    box_area = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
    candidate_sizes = np.maximum(0.0, candidates[:, 2:] - candidates[:, :2])
    candidate_areas = candidate_sizes[:, 0] * candidate_sizes[:, 1]
    union = box_area + candidate_areas - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


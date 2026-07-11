"""OOF-only blending for independently validated candidate projects."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Mapping, Sequence

import numpy as np

from .dataset import score_oof


LOWER_IS_BETTER = {"log_loss", "multiclass_log_loss", "rmse"}


@dataclass(frozen=True)
class BlendResult:
    weights: dict[str, float]
    metric_name: str
    metric_value: float


def fit_simplex_blend(
    *,
    y_true: Sequence[Any],
    oof_by_candidate: Mapping[str, Sequence[Any] | np.ndarray],
    metric_name: str,
    grid_step: float = 0.05,
) -> BlendResult:
    """Select non-negative sum-one weights using OOF predictions only."""

    names = tuple(oof_by_candidate)
    if not names:
        raise ValueError("At least one candidate OOF prediction is required.")
    predictions = [np.asarray(oof_by_candidate[name], dtype=float) for name in names]
    if any(prediction.shape[0] != len(y_true) for prediction in predictions):
        raise ValueError("Every OOF prediction must cover exactly the target rows.")
    if any(prediction.shape != predictions[0].shape for prediction in predictions[1:]):
        raise ValueError("Every OOF prediction must have the same shape and class order.")
    if not 0 < grid_step <= 1:
        raise ValueError("grid_step must be in (0, 1].")
    candidates = _simplex_weights(len(names), grid_step)
    best_weights: tuple[float, ...] | None = None
    best_score: float | None = None
    for weights in candidates:
        blended = sum(weight * prediction for weight, prediction in zip(weights, predictions))
        score = score_oof(metric_name, y_true, blended)
        if best_score is None or _better(metric_name, score, best_score):
            best_score, best_weights = score, weights
    assert best_score is not None and best_weights is not None
    return BlendResult(
        weights={name: float(weight) for name, weight in zip(names, best_weights)},
        metric_name=metric_name,
        metric_value=float(best_score),
    )


def blend_predictions(predictions_by_candidate: Mapping[str, Sequence[Any] | np.ndarray], weights: Mapping[str, float]) -> np.ndarray:
    """Blend aligned test predictions after the OOF weights have been selected."""

    if set(predictions_by_candidate) != set(weights):
        raise ValueError("Prediction candidates and blend weights must match exactly.")
    values = [np.asarray(predictions_by_candidate[name], dtype=float) for name in weights]
    if not values or any(item.shape != values[0].shape for item in values[1:]):
        raise ValueError("Prediction arrays must be non-empty and shape-aligned.")
    total = float(sum(weights.values()))
    if not np.isclose(total, 1.0) or any(weight < 0 for weight in weights.values()):
        raise ValueError("Blend weights must be non-negative and sum to one.")
    return sum(float(weights[name]) * np.asarray(predictions_by_candidate[name], dtype=float) for name in weights)


def _better(metric_name: str, candidate: float, incumbent: float) -> bool:
    return candidate < incumbent if metric_name.lower() in LOWER_IS_BETTER else candidate > incumbent


def _simplex_weights(n_members: int, step: float) -> list[tuple[float, ...]]:
    units = round(1 / step)
    if not np.isclose(units * step, 1.0):
        raise ValueError("grid_step must divide one exactly.")
    if n_members == 1:
        return [(1.0,)]
    if n_members > 4:
        # Exhaustive simplex grids become unhelpfully large; evenly weighted
        # blends remain deterministic until a later optimizer is configured.
        return [tuple([1.0 / n_members] * n_members)]
    weights: list[tuple[float, ...]] = []
    for parts in product(range(units + 1), repeat=n_members - 1):
        last = units - sum(parts)
        if last >= 0:
            weights.append(tuple([part / units for part in (*parts, last)]))
    return weights

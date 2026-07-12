"""OOF-only ensemble selection for independently evaluated MLE-STAR runs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Callable, Mapping, Sequence

import numpy as np

from .contracts import MetricSpec
from .metrics import MetricResult, score_metric


@dataclass(frozen=True)
class EnsembleResult:
    weights: dict[str, float]
    score: MetricResult


def align_oof_rows(
    oof_by_candidate: Mapping[str, tuple[Sequence[object], Sequence[float] | np.ndarray]]
) -> dict[str, np.ndarray]:
    """Align candidate OOF predictions and reject partial or reordered rows."""

    if not oof_by_candidate:
        raise ValueError("At least one OOF candidate is required.")
    expected_ids: tuple[object, ...] | None = None
    aligned: dict[str, np.ndarray] = {}
    for name, (row_ids, predictions) in oof_by_candidate.items():
        ids = tuple(row_ids)
        if len(ids) != len(set(ids)):
            raise ValueError("OOF row ids must be unique.")
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError("All OOF candidates must have the same row ids in the same order.")
        aligned[name] = np.asarray(predictions, dtype=float)
    shapes = {values.shape for values in aligned.values()}
    if len(shapes) != 1:
        raise ValueError("All OOF candidates must have identical prediction shapes.")
    return aligned


def _simplex(count: int, step: float) -> tuple[tuple[float, ...], ...]:
    if not 0 < step <= 1 or not np.isclose(round(1 / step) * step, 1.0):
        raise ValueError("grid_step must divide one exactly.")
    units = round(1 / step)
    if count == 1:
        return ((1.0,),)
    if count > 4:
        return (tuple([1.0 / count] * count),)
    values: list[tuple[float, ...]] = []
    for head in product(range(units + 1), repeat=count - 1):
        tail = units - sum(head)
        if tail >= 0:
            values.append(tuple(value / units for value in (*head, tail)))
    return tuple(values)


def select_ensemble(
    oof_by_candidate: Mapping[str, tuple[Sequence[object], Sequence[float] | np.ndarray]],
    y_true: Sequence[object],
    metric: str | MetricSpec,
    *,
    grid_step: float = 0.05,
    score_transform: "Callable[[np.ndarray], np.ndarray] | None" = None,
) -> EnsembleResult:
    """Choose non-negative, sum-one blending weights by OOF score only."""

    aligned = align_oof_rows(oof_by_candidate)
    names = tuple(aligned)
    arrays = tuple(aligned[name] for name in names)
    if arrays[0].shape[0] != len(y_true):
        raise ValueError("OOF predictions and y_true must have the same number of rows.")
    best: EnsembleResult | None = None
    for weights in _simplex(len(names), grid_step):
        prediction = sum(weight * array for weight, array in zip(weights, arrays))
        if score_transform is not None:
            prediction = score_transform(prediction)
        score = score_metric(metric, y_true, prediction)
        candidate = EnsembleResult(dict(zip(names, weights)), score)
        if best is None or (
            candidate.score.value > best.score.value
            if candidate.score.greater_is_better
            else candidate.score.value < best.score.value
        ):
            best = candidate
    assert best is not None
    return best


def blend_test_predictions(
    predictions_by_candidate: Mapping[str, Sequence[float] | np.ndarray], weights: Mapping[str, float]
) -> np.ndarray:
    """Apply OOF-selected weights to aligned test predictions."""

    if set(predictions_by_candidate) != set(weights):
        raise ValueError("Test prediction candidates must exactly match ensemble weights.")
    if not np.isclose(sum(weights.values()), 1.0) or any(weight < 0 for weight in weights.values()):
        raise ValueError("Ensemble weights must be non-negative and sum to one.")
    arrays = {name: np.asarray(value, dtype=float) for name, value in predictions_by_candidate.items()}
    if len({value.shape for value in arrays.values()}) != 1:
        raise ValueError("Test predictions must have identical shapes.")
    return sum(float(weights[name]) * arrays[name] for name in weights)

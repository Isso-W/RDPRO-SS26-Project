"""Validation-selected probability ensembles for classification experiments."""

from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

import numpy as np


def save_probability_artifact(
    path: str | Path,
    *,
    probabilities,
    labels=None,
    ids=None,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "probabilities": np.asarray(probabilities, dtype=np.float32),
    }
    if labels is not None:
        payload["labels"] = np.asarray(labels, dtype=np.int64)
    if ids is not None:
        payload["ids"] = np.asarray([str(value) for value in ids])
    np.savez_compressed(destination, **payload)
    return destination


def combine_prediction_sets(
    prediction_sets: list[list[tuple[str, list[float]]]],
    weights: list[float],
) -> list[tuple[str, list[float]]]:
    if not prediction_sets or len(prediction_sets) != len(weights):
        raise ValueError("Prediction sets and weights must be non-empty and aligned.")
    normalized_weights = np.asarray(weights, dtype=float)
    if not np.isfinite(normalized_weights).all() or (normalized_weights < 0).any():
        raise ValueError("Ensemble weights must be finite and non-negative.")
    if float(normalized_weights.sum()) <= 0:
        raise ValueError("At least one ensemble weight must be positive.")
    normalized_weights /= normalized_weights.sum()

    reference_ids = [name for name, _ in prediction_sets[0]]
    reference_set = set(reference_ids)
    matrices = []
    for predictions in prediction_sets:
        by_id = {name: probabilities for name, probabilities in predictions}
        if set(by_id) != reference_set:
            raise ValueError("All prediction sets must contain the same IDs.")
        matrices.append(np.asarray([by_id[name] for name in reference_ids], dtype=float))
    shape = matrices[0].shape
    if any(matrix.shape != shape for matrix in matrices):
        raise ValueError("All prediction matrices must have the same shape.")

    combined = sum(weight * matrix for weight, matrix in zip(normalized_weights, matrices))
    combined = np.clip(combined, 1.0e-12, 1.0)
    combined /= combined.sum(axis=1, keepdims=True)
    return [
        (name, [float(value) for value in row])
        for name, row in zip(reference_ids, combined)
    ]


def _log_loss(labels: np.ndarray, probabilities: np.ndarray) -> float:
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1.0e-15, 1.0)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    labels = np.asarray(labels, dtype=np.int64)
    return float(-np.log(probabilities[np.arange(len(labels)), labels]).mean())


def _weight_grid(count: int, step: float):
    units = max(1, round(1.0 / step))
    for values in product(range(units + 1), repeat=count):
        if sum(values) == units:
            yield np.asarray(values, dtype=float) / units


def optimize_validation_ensemble(
    members: list[dict[str, Any]],
    *,
    step: float = 0.05,
    max_members: int = 3,
) -> dict[str, Any]:
    usable = [
        dict(member)
        for member in members
        if member.get("validation_artifact")
        and Path(member["validation_artifact"]).is_file()
    ][: max(1, int(max_members))]
    if not usable:
        return {
            "improved": False,
            "members": [],
            "best_single_log_loss": None,
            "ensemble_log_loss": None,
        }

    matrices = []
    labels = None
    single_losses = []
    for member in usable:
        artifact = np.load(member["validation_artifact"], allow_pickle=False)
        probabilities = np.asarray(artifact["probabilities"], dtype=float)
        member_labels = np.asarray(artifact["labels"], dtype=np.int64)
        if labels is None:
            labels = member_labels
        elif not np.array_equal(labels, member_labels):
            raise ValueError("Validation artifacts must use the same labels and order.")
        matrices.append(probabilities)
        single_losses.append(_log_loss(member_labels, probabilities))

    best_single_index = int(np.argmin(single_losses))
    best_single_loss = float(single_losses[best_single_index])
    best_loss = best_single_loss
    best_weights = np.eye(len(usable), dtype=float)[best_single_index]
    for weights in _weight_grid(len(usable), step):
        combined = sum(weight * matrix for weight, matrix in zip(weights, matrices))
        loss = _log_loss(labels, combined)
        if loss < best_loss:
            best_loss = loss
            best_weights = weights

    selected_members = []
    for member, weight in zip(usable, best_weights):
        if weight <= 0:
            continue
        selected = dict(member)
        selected["weight"] = float(weight)
        selected_members.append(selected)
    return {
        "improved": best_loss < best_single_loss - 1.0e-8,
        "members": selected_members,
        "best_single": usable[best_single_index]["name"],
        "best_single_log_loss": best_single_loss,
        "ensemble_log_loss": float(best_loss),
        "grid_step": float(step),
    }

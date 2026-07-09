"""Shared artifact helpers for the producer/consumer ensemble pipeline."""

from __future__ import annotations

import json
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


ARTIFACT_SCHEMA_VERSION = 1
PROB_PREFIX = "prob_"
TRUE_PREFIX = "true_"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return out


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def probability_columns(label_columns: Iterable[str]) -> list[str]:
    return [f"{PROB_PREFIX}{column}" for column in label_columns]


def truth_columns(label_columns: Iterable[str]) -> list[str]:
    return [f"{TRUE_PREFIX}{column}" for column in label_columns]


def candidate_id(index: int) -> str:
    return f"candidate_{index}"


def one_hot(indices: Iterable[int], num_classes: int) -> np.ndarray:
    index_list = [int(index) for index in indices]
    values = np.zeros((len(index_list), num_classes), dtype=float)
    for row, index in enumerate(index_list):
        values[row, int(index)] = 1.0
    return values


def mean_column_auc(
    y_true: np.ndarray | pd.DataFrame,
    y_prob: np.ndarray | pd.DataFrame,
    label_columns: list[str],
) -> dict[str, Any]:
    """Compute Plant Pathology-style mean per-column ROC AUC."""

    from sklearn.metrics import roc_auc_score

    true_values = np.asarray(y_true, dtype=float)
    prob_values = np.asarray(y_prob, dtype=float)
    per_column: dict[str, float] = {}
    skipped: list[str] = []
    for index, label in enumerate(label_columns):
        column_true = true_values[:, index]
        if len(np.unique(column_true)) < 2:
            skipped.append(label)
            continue
        per_column[label] = float(roc_auc_score(column_true, prob_values[:, index]))
    mean_auc = float(np.mean(list(per_column.values()))) if per_column else None
    return {
        "metric_name": "mean_column_auc",
        "metric_value": mean_auc,
        "per_column_auc": per_column,
        "skipped_columns": skipped,
    }


def clipped_probabilities(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def weight_grid(num_models: int, step: float) -> list[list[float]]:
    """Return non-negative weight vectors summing to 1.0."""

    if num_models <= 0:
        return []
    if num_models == 1:
        return [[1.0]]
    if step <= 0:
        return [[1.0 / num_models] * num_models]
    units = max(1, int(round(1.0 / step)))
    grids: list[list[float]] = []
    for combo in product(range(units + 1), repeat=num_models):
        if sum(combo) == units:
            grids.append([value / units for value in combo])
    return grids

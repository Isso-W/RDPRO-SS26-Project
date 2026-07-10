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

    true_values = np.asarray(y_true, dtype=float)
    prob_values = np.asarray(y_prob, dtype=float)
    per_column: dict[str, float] = {}
    skipped: list[str] = []
    for index, label in enumerate(label_columns):
        column_true = true_values[:, index]
        if len(np.unique(column_true)) < 2:
            skipped.append(label)
            continue
        per_column[label] = _binary_auc(column_true, prob_values[:, index])
    mean_auc = float(np.mean(list(per_column.values()))) if per_column else None
    return {
        "metric_name": "mean_column_auc",
        "metric_value": mean_auc,
        "per_column_auc": per_column,
        "skipped_columns": skipped,
    }


def _binary_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Binary ROC AUC with average ranks, used to avoid a hard sklearn runtime dependency."""

    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    positives = y_true == 1.0
    pos_count = int(positives.sum())
    neg_count = int(len(y_true) - pos_count)
    if pos_count == 0 or neg_count == 0:
        raise ValueError("AUC requires both positive and negative examples")

    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(len(y_score), dtype=float)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end

    positive_rank_sum = float(ranks[positives].sum())
    auc = (positive_rank_sum - (pos_count * (pos_count + 1) / 2.0)) / (pos_count * neg_count)
    return float(auc)


def clipped_probabilities(values: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), 0.0, 1.0)


def rank_normalize(values: np.ndarray) -> np.ndarray:
    """Per-column average-rank transform scaled to [0, 1].

    The competition metric is mean column-wise ROC AUC, which depends only on
    the within-column ordering of scores. Rank-normalising each candidate's
    columns before blending makes models comparable on a common scale, so a
    model with a narrow probability band is not drowned out by a wide-range
    one. It is monotonic per column, so a single model's AUC is unchanged.
    """

    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    out = np.empty_like(arr, dtype=float)
    n = arr.shape[0]
    denom = max(n - 1, 1)
    for col in range(arr.shape[1]):
        column = arr[:, col]
        order = np.argsort(column, kind="mergesort")
        ordered = column[order]
        ranks = np.empty(n, dtype=float)
        start = 0
        while start < n:
            end = start + 1
            while end < n and ordered[end] == ordered[start]:
                end += 1
            # zero-based average rank so the range is exactly [0, n-1]
            average_rank = (start + end - 1) / 2.0
            ranks[order[start:end]] = average_rank
            start = end
        out[:, col] = ranks / denom
    return out


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

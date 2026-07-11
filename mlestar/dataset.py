"""Dataset inventory, immutable fold assignments, and local metric scoring."""

from __future__ import annotations

from hashlib import sha256
import json
from math import sqrt
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    log_loss as sklearn_log_loss,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from .contracts import DatasetInventory


MAX_FILE_HASH_BYTES = 64 * 1024 * 1024


def inspect_dataset(data_root: str | Path, *, output_path: str | Path | None = None) -> DatasetInventory:
    """Create a deterministic inventory without changing the supplied data."""

    root = Path(data_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        item: dict[str, Any] = {"path": relative, "size_bytes": path.stat().st_size}
        if item["size_bytes"] <= MAX_FILE_HASH_BYTES:
            item["sha256"] = _file_digest(path)
        if path.suffix.lower() == ".csv":
            try:
                item["columns"] = list(pd.read_csv(path, nrows=0).columns)
            except (OSError, UnicodeDecodeError, pd.errors.ParserError) as exc:
                item["csv_error"] = f"{type(exc).__name__}: {exc}"
        files.append(item)
    fingerprint = sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    inventory = DatasetInventory(data_root=str(root), fingerprint=fingerprint, files=tuple(files))
    if output_path is not None:
        _write_json(output_path, inventory.to_dict())
    return inventory


def make_folds(
    frame: pd.DataFrame,
    *,
    target: str,
    strategy: str,
    n_splits: int,
    seed: int,
    output_path: str | Path,
    id_column: str | None = None,
    group_column: str | None = None,
    time_column: str | None = None,
) -> pd.DataFrame:
    """Return and persist one immutable fold assignment per input row."""

    if target not in frame.columns:
        raise ValueError(f"Target column {target!r} is absent.")
    if n_splits < 2:
        raise ValueError("n_splits must be at least two.")
    if len(frame) < n_splits:
        raise ValueError("n_splits cannot exceed the number of rows.")
    strategy = strategy.lower().replace("-", "_")
    indices = np.arange(len(frame))
    assignment = np.full(len(frame), -1, dtype=int)

    if strategy == "stratified":
        _require_class_counts(frame[target], n_splits)
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(indices, frame[target])
    elif strategy == "kfold":
        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(indices)
    elif strategy == "group":
        groups = _column(frame, group_column, "group")
        splitter = GroupKFold(n_splits=n_splits)
        split_iter = splitter.split(indices, frame[target], groups)
    elif strategy == "stratified_group":
        groups = _column(frame, group_column, "group")
        _require_class_counts(frame[target], n_splits)
        try:
            from sklearn.model_selection import StratifiedGroupKFold
        except ImportError as exc:  # pragma: no cover - guarded by supported sklearn versions
            raise RuntimeError("StratifiedGroupKFold requires a newer scikit-learn version.") from exc
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(indices, frame[target], groups)
    elif strategy in {"time", "chronological"}:
        times = _column(frame, time_column, "time")
        ordered = np.argsort(pd.to_datetime(times, errors="raise").to_numpy(), kind="stable")
        for fold, validation_indices in enumerate(np.array_split(ordered, n_splits)):
            assignment[validation_indices] = fold
        split_iter = ()
    else:
        raise ValueError(f"Unsupported fold strategy {strategy!r}.")

    for fold, (_, validation_indices) in enumerate(split_iter):
        assignment[validation_indices] = fold
    if np.any(assignment < 0):
        raise RuntimeError("Fold construction did not assign every row exactly once.")

    output = pd.DataFrame({"row_index": frame.index.to_list(), "fold": assignment})
    if id_column is not None:
        if id_column not in frame.columns:
            raise ValueError(f"ID column {id_column!r} is absent.")
        output.insert(0, id_column, frame[id_column].to_list())
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output.to_parquet(path, index=False)
    return output


def score_oof(metric_name: str, y_true: Sequence[Any], y_pred: Sequence[Any] | np.ndarray) -> float:
    """Score only matched OOF labels/predictions using the declared metric."""

    if len(y_true) != len(y_pred):
        raise ValueError("OOF labels and predictions must have identical row counts.")
    name = metric_name.lower().replace("-", "_")
    if name == "roc_auc":
        return float(roc_auc_score(y_true, y_pred))
    if name in {"multiclass_log_loss", "log_loss"}:
        return float(sklearn_log_loss(y_true, y_pred))
    if name == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if name == "qwk":
        return float(cohen_kappa_score(y_true, y_pred, weights="quadratic"))
    if name == "rmse":
        return float(sqrt(mean_squared_error(y_true, y_pred)))
    if name == "dice":
        return _dice(y_true, y_pred)
    if name == "map_iou":
        return _mean_iou(y_true, y_pred)
    raise ValueError(f"Unsupported OOF metric {metric_name!r}.")


def _file_digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _column(frame: pd.DataFrame, column: str | None, label: str) -> pd.Series:
    if not column or column not in frame.columns:
        raise ValueError(f"{label.capitalize()} column is required for this fold strategy.")
    return frame[column]


def _require_class_counts(labels: Iterable[Any], n_splits: int) -> None:
    smallest = int(pd.Series(labels).value_counts(dropna=False).min())
    if smallest < n_splits:
        raise ValueError(f"n_splits={n_splits} exceeds the smallest class count ({smallest}).")


def _dice(y_true: Sequence[Any], y_pred: Sequence[Any] | np.ndarray) -> float:
    truth = np.asarray(y_true).astype(bool).ravel()
    prediction = (np.asarray(y_pred) >= 0.5).ravel()
    denominator = int(truth.sum() + prediction.sum())
    return 1.0 if denominator == 0 else float(2 * np.logical_and(truth, prediction).sum() / denominator)


def _mean_iou(y_true: Sequence[Any], y_pred: Sequence[Any] | np.ndarray) -> float:
    """Return a simple aligned-box IoU mean for adapter-level local checks.

    Detection adapters provide the competition's complete thresholded mAP later;
    this scorer remains intentionally strict and only accepts aligned boxes.
    """

    truths = np.asarray(y_true, dtype=float)
    predictions = np.asarray(y_pred, dtype=float)
    if truths.shape != predictions.shape or truths.ndim != 2 or truths.shape[1] != 4:
        raise ValueError("map_iou expects aligned N x 4 [x1, y1, x2, y2] boxes.")
    left_top = np.maximum(truths[:, :2], predictions[:, :2])
    right_bottom = np.minimum(truths[:, 2:], predictions[:, 2:])
    intersection = np.prod(np.maximum(right_bottom - left_top, 0.0), axis=1)
    truth_area = np.prod(np.maximum(truths[:, 2:] - truths[:, :2], 0.0), axis=1)
    prediction_area = np.prod(np.maximum(predictions[:, 2:] - predictions[:, :2], 0.0), axis=1)
    union = truth_area + prediction_area - intersection
    return float(np.mean(np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)))

"""Immutable data, validation, metric, and submission contracts for benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mlestar.contracts import MetricSpec, MODALITIES


@dataclass(frozen=True)
class SubmissionContract:
    kind: str
    id_column: str
    prediction_columns: tuple[str, ...] = ()
    filename: str = "submission.csv"
    rle_order: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"csv", "rle_csv", "detection_csv", "image_directory"}:
            raise ValueError(f"Unsupported submission kind {self.kind!r}.")
        if not self.id_column:
            raise ValueError("Submission ID column is required.")
        if self.kind != "image_directory" and not self.prediction_columns:
            raise ValueError("CSV submissions require at least one prediction column.")
        if self.kind == "rle_csv" and not self.rle_order:
            raise ValueError("RLE submissions require a flatten order.")

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "id_column": self.id_column, "prediction_columns": list(self.prediction_columns), "filename": self.filename, "rle_order": self.rle_order}


@dataclass(frozen=True)
class FoldContract:
    strategy: str
    n_splits: int = 5
    group_column: str | None = None
    time_column: str | None = None

    def __post_init__(self) -> None:
        if self.strategy not in {"stratified", "kfold", "group", "stratified_group", "time"}:
            raise ValueError(f"Unsupported fold strategy {self.strategy!r}.")
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least two.")
        if self.strategy in {"group", "stratified_group"} and not self.group_column:
            raise ValueError("Group fold strategies require group_column.")
        if self.strategy == "time" and not self.time_column:
            raise ValueError("Time folds require time_column.")

    def to_dict(self) -> dict[str, Any]:
        return {"strategy": self.strategy, "n_splits": self.n_splits, "group_column": self.group_column, "time_column": self.time_column}


@dataclass(frozen=True)
class BenchmarkContract:
    key: str
    competition: str
    modality: str
    metric: MetricSpec
    labels: tuple[str, ...]
    submission: SubmissionContract
    folds: FoldContract
    query: str
    data_hints: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.key or not self.competition or not self.query:
            raise ValueError("Benchmark key, competition, and query are required.")
        if self.modality not in MODALITIES:
            raise ValueError(f"Unsupported modality {self.modality!r}.")
        object.__setattr__(self, "labels", tuple(self.labels))
        object.__setattr__(self, "data_hints", tuple((str(key), str(value)) for key, value in self.data_hints))

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key, "competition": self.competition, "modality": self.modality,
            "metric": self.metric.to_dict(), "labels": list(self.labels),
            "submission": self.submission.to_dict(), "folds": self.folds.to_dict(),
            "query": self.query, "data_hints": dict(self.data_hints),
        }

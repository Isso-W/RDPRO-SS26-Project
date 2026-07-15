"""Immutable contracts shared by MLE-STAR benchmark runs.

The contracts deliberately contain metadata only.  Model objects, tensors and
data frames belong in run artifacts, which keeps every execution phase
serialisable and reproducible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, ClassVar, Mapping


_METRIC_DIRECTIONS: dict[str, bool] = {
    "roc_auc": True,
    "log_loss": False,
    "multiclass_log_loss": False,
    "qwk": True,
    "rmse": False,
    "dice": True,
    "detection_map": True,
}

_METRIC_ALIASES: dict[str, str] = {
    "logloss": "log_loss",
    "multiclass_logloss": "multiclass_log_loss",
    "map": "detection_map",
    "competition_detection_map": "detection_map",
}


@dataclass(frozen=True)
class MetricSpec:
    """The validation metric and its explicit optimisation direction."""

    name: str
    greater_is_better: bool | None = None

    def __post_init__(self) -> None:
        canonical_name = self.name.lower().replace("-", "_")
        canonical_name = _METRIC_ALIASES.get(canonical_name, canonical_name)
        if canonical_name not in _METRIC_DIRECTIONS:
            raise ValueError(f"Unsupported metric: {self.name!r}")
        expected_direction = _METRIC_DIRECTIONS[canonical_name]
        if self.greater_is_better is None:
            object.__setattr__(self, "greater_is_better", expected_direction)
        elif self.greater_is_better != expected_direction:
            raise ValueError(
                f"Metric {canonical_name!r} has greater_is_better="
                f"{expected_direction}, not {self.greater_is_better}"
            )
        object.__setattr__(self, "name", canonical_name)

    @classmethod
    def known_directions(cls) -> Mapping[str, bool]:
        """Return a copy so callers cannot change the contract registry."""

        return dict(_METRIC_DIRECTIONS)


@dataclass(frozen=True)
class FoldSpec:
    """A deterministic cross-validation split policy."""

    n_splits: int
    strategy: str = "stratified_kfold"
    seed: int = 13
    shuffle: bool = True
    group_column: str | None = None

    def __post_init__(self) -> None:
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        if not self.strategy:
            raise ValueError("fold strategy must not be empty")
        if self.group_column == "":
            raise ValueError("group_column must be a name or None")


@dataclass(frozen=True)
class SubmissionSpec:
    """The schema a local submission must satisfy before it may be uploaded."""

    id_columns: tuple[str, ...]
    prediction_columns: tuple[str, ...]
    filename: str = "submission.csv"
    prediction_from_sample: bool = False

    def __post_init__(self) -> None:
        if not self.id_columns:
            raise ValueError("submission needs at least one id column")
        if not self.prediction_columns and not self.prediction_from_sample:
            raise ValueError(
                "submission needs prediction columns unless they are read from "
                "sample_submission.csv"
            )
        if not self.filename or "/" in self.filename or "\\" in self.filename:
            raise ValueError("submission filename must be a file name")

    @property
    def columns(self) -> tuple[str, ...]:
        return self.id_columns + self.prediction_columns


@dataclass(frozen=True)
class TaskSpec:
    """Everything required to run one benchmark without ambient assumptions."""

    key: str
    competition: str
    modality: str
    metric: MetricSpec
    fold: FoldSpec
    submission: SubmissionSpec
    target_columns: tuple[str, ...]
    description: str = ""

    def __post_init__(self) -> None:
        if not self.key or not self.key.replace("_", "").isalnum():
            raise ValueError("task key must contain only letters, digits and underscores")
        if not self.competition:
            raise ValueError("competition must not be empty")
        if not self.modality:
            raise ValueError("modality must not be empty")
        if not self.target_columns:
            raise ValueError("task needs at least one target column")
        if len(set(self.target_columns)) != len(self.target_columns):
            raise ValueError("target column names must be unique")

    def to_dict(self) -> dict[str, Any]:
        """Produce JSON-compatible primitive values only."""

        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TaskSpec":
        try:
            metric = MetricSpec(**data["metric"])
            fold = FoldSpec(**data["fold"])
            submission_data = dict(data["submission"])
            submission_data["id_columns"] = tuple(submission_data["id_columns"])
            submission_data["prediction_columns"] = tuple(
                submission_data["prediction_columns"]
            )
            return cls(
                key=str(data["key"]),
                competition=str(data["competition"]),
                modality=str(data["modality"]),
                metric=metric,
                fold=fold,
                submission=SubmissionSpec(**submission_data),
                target_columns=tuple(data["target_columns"]),
                description=str(data.get("description", "")),
            )
        except (KeyError, TypeError) as error:
            raise ValueError("invalid task specification") from error

    @classmethod
    def from_json(cls, source: str) -> "TaskSpec":
        try:
            value = json.loads(source)
        except json.JSONDecodeError as error:
            raise ValueError("task specification is not valid JSON") from error
        if not isinstance(value, Mapping):
            raise ValueError("task specification JSON must contain an object")
        return cls.from_dict(value)


@dataclass(frozen=True)
class ExperimentReceipt:
    """An append-only record for one attempted candidate evaluation."""

    experiment_id: str
    parent_experiment_id: str | None
    phase: str
    candidate_id: str
    metric_value: float | None
    fold_scores: tuple[float, ...]
    seed: int
    oof_path: str | None
    test_path: str | None
    error: str | None

    _PATH_FIELDS: ClassVar[tuple[str, ...]] = ("oof_path", "test_path")

    def __post_init__(self) -> None:
        if not self.experiment_id:
            raise ValueError("experiment_id must not be empty")
        if not self.phase:
            raise ValueError("phase must not be empty")
        if not self.candidate_id:
            raise ValueError("candidate_id must not be empty")
        if self.metric_value is not None and not self.fold_scores:
            raise ValueError("metric receipt has no fold scores")
        for field_name in self._PATH_FIELDS:
            path = getattr(self, field_name)
            path_parts = Path(path).parts if path is not None else ()
            if path is not None and (
                not path or Path(path).is_absolute() or ".." in path_parts
            ):
                raise ValueError(f"{field_name} must be a non-empty relative path")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ExperimentReceipt":
        try:
            receipt_data = dict(data)
            receipt_data["fold_scores"] = tuple(receipt_data["fold_scores"])
            return cls(**receipt_data)
        except (KeyError, TypeError) as error:
            raise ValueError("invalid experiment receipt") from error

    @classmethod
    def from_json(cls, source: str) -> "ExperimentReceipt":
        try:
            value = json.loads(source)
        except json.JSONDecodeError as error:
            raise ValueError("experiment receipt is not valid JSON") from error
        if not isinstance(value, Mapping):
            raise ValueError("experiment receipt JSON must contain an object")
        return cls.from_dict(value)

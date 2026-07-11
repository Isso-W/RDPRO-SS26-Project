"""JSON-friendly, validated artifacts shared by the MLE-STAR workflow.

The orchestration layer deliberately persists these small contracts rather than
passing unstructured dictionaries between search, generation, execution, and
the competition runner.  This makes every selected result traceable to a task,
dataset fingerprint, generated project, and out-of-fold artifact.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import re
from typing import Any, Iterable


COMPONENT_NAMES = (
    "data_loading",
    "data_preparation",
    "model",
    "training",
    "prediction",
)

MODALITIES = {
    "tabular",
    "image_classification",
    "object_detection",
    "image_segmentation",
    "image_to_image",
}

_SHA256_HEX = re.compile(r"[0-9a-fA-F]{64}")


def _now() -> str:
    """Return an ISO-8601 timestamp using the Python 3.10 UTC API."""

    return datetime.now(timezone.utc).isoformat()


def _string_tuple(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(str(value) for value in values)


def _require_text(value: object, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required.")


def _require_sha256(value: object, field_name: str = "code_sha256") -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required.")
    if _SHA256_HEX.fullmatch(value) is None:
        raise ValueError(f"{field_name} must be a 64-character SHA-256 hexadecimal digest.")


def _require_bool(value: object, field_name: str = "greater_is_better") -> None:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a bool.")


@dataclass(frozen=True)
class MetricSpec:
    """The task metric and the direction used for experiment comparison."""

    name: str
    greater_is_better: bool

    def __post_init__(self) -> None:
        _require_text(self.name, "Metric name")
        _require_bool(self.greater_is_better)

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "greater_is_better": self.greater_is_better}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetricSpec":
        return cls(name=str(data.get("name") or ""), greater_is_better=data.get("greater_is_better"))


@dataclass(frozen=True)
class Component:
    """One stable code-marker component in a generated ML project."""

    name: str
    description: str = ""

    def __post_init__(self) -> None:
        if self.name not in COMPONENT_NAMES:
            raise ValueError(f"Unsupported component {self.name!r}.")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Component":
        return cls(name=str(data.get("name") or ""), description=str(data.get("description") or ""))


@dataclass(frozen=True)
class TaskContract:
    """Immutable description of a task that the agent is permitted to solve."""

    task_id: str
    modality: str
    target_columns: tuple[str, ...]
    id_column: str
    metric: MetricSpec
    components: tuple[Component, ...]
    description: str = ""
    constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "target_columns", _string_tuple(self.target_columns))
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "constraints", _string_tuple(self.constraints))
        if not self.task_id.strip():
            raise ValueError("Task ID must be non-empty.")
        if self.modality not in MODALITIES:
            raise ValueError(f"Unsupported modality {self.modality!r}.")
        if not self.id_column.strip():
            raise ValueError("ID column must be non-empty.")
        if not self.target_columns or any(not name.strip() for name in self.target_columns):
            raise ValueError("At least one non-empty target column is required.")
        names = tuple(component.name for component in self.components)
        if len(names) != len(COMPONENT_NAMES) or set(names) != set(COMPONENT_NAMES):
            raise ValueError("Task contracts must contain each MLE-STAR component exactly once.")

    @property
    def component_names(self) -> tuple[str, ...]:
        return tuple(component.name for component in self.components)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "modality": self.modality,
            "target_columns": list(self.target_columns),
            "id_column": self.id_column,
            "metric": self.metric.to_dict(),
            "components": [component.to_dict() for component in self.components],
            "description": self.description,
            "constraints": list(self.constraints),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskContract":
        metric = data.get("metric") or {}
        components = data.get("components") or ()
        return cls(
            task_id=str(data.get("task_id") or ""),
            modality=str(data.get("modality") or ""),
            target_columns=tuple(data.get("target_columns") or ()),
            id_column=str(data.get("id_column") or ""),
            metric=metric if isinstance(metric, MetricSpec) else MetricSpec.from_dict(dict(metric)),
            components=tuple(
                item if isinstance(item, Component) else Component.from_dict(dict(item)) for item in components
            ),
            description=str(data.get("description") or ""),
            constraints=tuple(data.get("constraints") or ()),
        )


@dataclass(frozen=True)
class DatasetInventory:
    data_root: str
    fingerprint: str
    files: tuple[dict[str, Any], ...] = ()
    created_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", tuple(dict(item) for item in self.files))
        if not self.data_root or not self.fingerprint:
            raise ValueError("Dataset inventory requires data_root and fingerprint.")

    def to_dict(self) -> dict[str, Any]:
        return {"data_root": self.data_root, "fingerprint": self.fingerprint, "files": [dict(item) for item in self.files], "created_at": self.created_at}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetInventory":
        return cls(str(data.get("data_root") or ""), str(data.get("fingerprint") or ""), tuple(data.get("files") or ()), str(data.get("created_at") or _now()))


@dataclass(frozen=True)
class SearchEvidence:
    title: str
    url: str
    summary: str
    model_hint: str = ""
    example_code: str = ""
    license_note: str = ""
    retrieved_at: str = field(default_factory=_now)

    def __post_init__(self) -> None:
        if not self.title or not self.url or not self.summary:
            raise ValueError("Search evidence requires title, URL, and summary.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title, "url": self.url, "summary": self.summary,
            "model_hint": self.model_hint, "example_code": self.example_code,
            "license_note": self.license_note, "retrieved_at": self.retrieved_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SearchEvidence":
        return cls(
            title=str(data.get("title") or ""), url=str(data.get("url") or ""),
            summary=str(data.get("summary") or ""), model_hint=str(data.get("model_hint") or ""),
            example_code=str(data.get("example_code") or ""), license_note=str(data.get("license_note") or ""),
            retrieved_at=str(data.get("retrieved_at") or _now()),
        )


@dataclass(frozen=True)
class CandidateProject:
    candidate_id: str
    project_dir: str
    code_sha256: str
    components: tuple[Component, ...]
    created_at: str = field(default_factory=_now)
    evidence_urls: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "components", tuple(self.components))
        object.__setattr__(self, "evidence_urls", _string_tuple(self.evidence_urls))
        _require_text(self.candidate_id, "Candidate project ID")
        _require_text(self.project_dir, "Candidate project directory")
        _require_sha256(self.code_sha256)

    def to_dict(self) -> dict[str, Any]:
        return {"candidate_id": self.candidate_id, "project_dir": self.project_dir, "code_sha256": self.code_sha256, "components": [item.to_dict() for item in self.components], "created_at": self.created_at, "evidence_urls": list(self.evidence_urls)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CandidateProject":
        return cls(
            candidate_id=str(data.get("candidate_id") or ""), project_dir=str(data.get("project_dir") or ""),
            code_sha256=str(data.get("code_sha256") or ""),
            components=tuple(item if isinstance(item, Component) else Component.from_dict(dict(item)) for item in (data.get("components") or ())),
            created_at=str(data.get("created_at") or _now()), evidence_urls=tuple(data.get("evidence_urls") or ()),
        )


@dataclass(frozen=True)
class ExperimentReceipt:
    experiment_id: str
    candidate_id: str
    component: str
    stage: str
    metric_name: str
    greater_is_better: bool
    metric_value: float | None
    elapsed_seconds: float
    status: str
    code_sha256: str
    data_fingerprint: str
    oof_path: str | None
    created_at: str = field(default_factory=_now)
    prediction_path: str | None = None
    submission_path: str | None = None
    parent_experiment_id: str | None = None
    fold: int | None = None
    seed: int | None = None
    error_text: str | None = None

    def __post_init__(self) -> None:
        for value, field_name in (
            (self.experiment_id, "experiment_id"),
            (self.candidate_id, "candidate_id"),
            (self.component, "component"),
            (self.stage, "stage"),
            (self.metric_name, "metric_name"),
            (self.status, "status"),
            (self.data_fingerprint, "data_fingerprint"),
        ):
            _require_text(value, field_name)
        if self.component not in COMPONENT_NAMES:
            raise ValueError(f"Unsupported component {self.component!r}.")
        _require_bool(self.greater_is_better)
        _require_sha256(self.code_sha256)

    @property
    def success(self) -> bool:
        return self.status == "success" and self.metric_value is not None and bool(self.oof_path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id, "candidate_id": self.candidate_id,
            "component": self.component, "stage": self.stage, "metric_name": self.metric_name,
            "greater_is_better": self.greater_is_better, "metric_value": self.metric_value,
            "elapsed_seconds": self.elapsed_seconds, "status": self.status,
            "code_sha256": self.code_sha256, "data_fingerprint": self.data_fingerprint,
            "oof_path": self.oof_path, "created_at": self.created_at,
            "prediction_path": self.prediction_path, "submission_path": self.submission_path,
            "parent_experiment_id": self.parent_experiment_id, "fold": self.fold,
            "seed": self.seed, "error_text": self.error_text,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExperimentReceipt":
        return cls(
            experiment_id=str(data.get("experiment_id") or ""), candidate_id=str(data.get("candidate_id") or ""),
            component=str(data.get("component") or ""), stage=str(data.get("stage") or ""),
            metric_name=str(data.get("metric_name") or ""), greater_is_better=data.get("greater_is_better"),
            metric_value=None if data.get("metric_value") is None else float(data["metric_value"]),
            elapsed_seconds=float(data.get("elapsed_seconds") or 0.0), status=str(data.get("status") or ""),
            code_sha256=str(data.get("code_sha256") or ""), data_fingerprint=str(data.get("data_fingerprint") or ""),
            oof_path=data.get("oof_path"), created_at=str(data.get("created_at") or _now()),
            prediction_path=data.get("prediction_path"), submission_path=data.get("submission_path"),
            parent_experiment_id=data.get("parent_experiment_id"), fold=data.get("fold"), seed=data.get("seed"),
            error_text=data.get("error_text"),
        )


@dataclass(frozen=True)
class AuditFinding:
    audit_name: str
    status: str
    message: str
    created_at: str = field(default_factory=_now)
    code: str = ""
    severity: str = "warning"

    def to_dict(self) -> dict[str, Any]:
        return {"audit_name": self.audit_name, "status": self.status, "message": self.message, "created_at": self.created_at, "code": self.code, "severity": self.severity}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AuditFinding":
        return cls(str(data.get("audit_name") or ""), str(data.get("status") or ""), str(data.get("message") or ""), str(data.get("created_at") or _now()), str(data.get("code") or ""), str(data.get("severity") or "warning"))


@dataclass(frozen=True)
class EnsembleReceipt:
    ensemble_id: str
    candidate_weights: dict[str, float]
    metric_name: str
    metric_value: float | None
    oof_path: str | None
    created_at: str = field(default_factory=_now)
    prediction_path: str | None = None
    submission_path: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidate_weights", {str(key): float(value) for key, value in self.candidate_weights.items()})

    def to_dict(self) -> dict[str, Any]:
        return {"ensemble_id": self.ensemble_id, "candidate_weights": dict(self.candidate_weights), "metric_name": self.metric_name, "metric_value": self.metric_value, "oof_path": self.oof_path, "created_at": self.created_at, "prediction_path": self.prediction_path, "submission_path": self.submission_path}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EnsembleReceipt":
        return cls(
            ensemble_id=str(data.get("ensemble_id") or ""), candidate_weights=dict(data.get("candidate_weights") or {}),
            metric_name=str(data.get("metric_name") or ""),
            metric_value=None if data.get("metric_value") is None else float(data["metric_value"]),
            oof_path=data.get("oof_path"), created_at=str(data.get("created_at") or _now()),
            prediction_path=data.get("prediction_path"), submission_path=data.get("submission_path"),
        )

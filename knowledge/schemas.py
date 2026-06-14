"""JSON-friendly contracts for the MCP knowledge and experiment loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


VALID_TASK_TYPES = {
    "classification",
    "object_detection",
    "image_segmentation",
    "feature_extraction",
}
VALID_COMPONENTS = {
    "augmentation",
    "loss",
    "optimizer",
    "scheduler",
    "inference",
    "ensemble",
    "backbone",
    "finetune",
}
VALID_RISK_LEVELS = {"low", "medium", "high"}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class Evidence:
    source_id: str
    note: str
    url: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Evidence":
        return cls(**value)


@dataclass
class SourceRecord:
    id: str
    source_name: str
    source_type: str
    content_path: str
    url: str
    content_sha256: str
    created_at: str = field(default_factory=utc_now)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourceRecord":
        return cls(**value)


@dataclass
class SourceSummary:
    id: str
    source_id: str
    summary: str = ""
    models: list[str] = field(default_factory=list)
    augmentations: list[str] = field(default_factory=list)
    losses: list[str] = field(default_factory=list)
    optimizers: list[str] = field(default_factory=list)
    schedulers: list[str] = field(default_factory=list)
    inference: list[str] = field(default_factory=list)
    ensemble: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SourceSummary":
        return cls(**value)


@dataclass
class StrategyCard:
    id: str
    task_type: str
    domain: str
    strategy_name: str
    component: str
    summary: str
    use_when: list[str] = field(default_factory=list)
    avoid_when: list[str] = field(default_factory=list)
    compatible_with: list[str] = field(default_factory=list)
    target_metrics: list[str] = field(default_factory=list)
    experiment_template: dict[str, Any] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)
    risk: str = ""
    risk_level: str = "medium"
    priority: float = 0.0
    observed_results: list[dict[str, Any]] = field(default_factory=list)

    def validate(self) -> None:
        if not self.id or not self.strategy_name or not self.summary:
            raise ValueError("Strategy card id, strategy_name, and summary are required.")
        if self.task_type not in VALID_TASK_TYPES:
            raise ValueError(f"Unsupported task_type: {self.task_type}")
        if self.component not in VALID_COMPONENTS:
            raise ValueError(f"Unsupported component: {self.component}")
        if self.risk_level not in VALID_RISK_LEVELS:
            raise ValueError(f"Unsupported risk_level: {self.risk_level}")
        if not 0.0 <= float(self.priority) <= 1.0:
            raise ValueError("priority must be between 0 and 1.")
        if not isinstance(self.experiment_template, dict):
            raise ValueError("experiment_template must be a dict.")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StrategyCard":
        data = dict(value)
        data["evidence"] = [
            item if isinstance(item, Evidence) else Evidence.from_dict(item)
            for item in data.get("evidence", [])
        ]
        card = cls(**data)
        card.validate()
        return card


@dataclass
class ExperimentProposal:
    experiment_name: str
    strategy_card_ids: list[str]
    config: dict[str, Any]
    changed_fields: list[str]
    config_hash: str
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentComparison:
    best_experiment: str | None
    improved: bool
    target_metric: str
    baseline_value: float | None
    best_value: float | None
    metric_delta: float | None
    keep_strategy_card_ids: list[str] = field(default_factory=list)
    discard_strategy_card_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

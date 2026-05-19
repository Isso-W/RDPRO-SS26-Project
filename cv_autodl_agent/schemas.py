from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal

from .exceptions import InputValidationError

TaskFamily = Literal["classification", "segmentation", "detection"]
ReviewStatus = Literal["pass", "revise", "fallback_candidate"]
ExecutionStatus = Literal["success", "failed"]

_VALID_TASKS = {"classification", "segmentation", "detection"}


@dataclass(slots=True)
class DatasetManifest:
    dataset_name: str
    task_family: TaskFamily
    train_path: str
    val_path: str
    test_path: str
    annotation_format: str
    recommended_metric: str
    num_classes: int | None = None
    class_names: list[str] = field(default_factory=list)
    label_source: str | None = None
    image_size_hint: int | None = None
    hf_dataset_id: str | None = None
    hf_config_name: str | None = None
    image_column: str | None = None
    label_column: str | None = None
    train_split: str | None = None
    val_split: str | None = None
    test_split: str | None = None
    max_train_samples: int | None = None
    max_val_samples: int | None = None
    max_epochs: int | None = None
    mask_format: str | None = None
    ignore_index: int | None = None
    categories: list[str] = field(default_factory=list)
    bbox_format: str | None = None
    coco_json_path: str | None = None

    def validate(self) -> None:
        required = {
            "dataset_name": self.dataset_name,
            "task_family": self.task_family,
            "train_path": self.train_path,
            "val_path": self.val_path,
            "test_path": self.test_path,
            "annotation_format": self.annotation_format,
            "recommended_metric": self.recommended_metric,
        }
        for name, value in required.items():
            if not value:
                raise InputValidationError(f"DatasetManifest missing required field: {name}")
        if self.task_family not in _VALID_TASKS:
            raise InputValidationError(f"Unsupported task_family: {self.task_family}")
        if self.task_family in {"classification", "segmentation"} and not self.num_classes:
            raise InputValidationError("num_classes is required for classification and segmentation")
        if self.task_family == "classification" and not self.label_source:
            raise InputValidationError("label_source is required for classification")
        if self.task_family == "segmentation" and not self.mask_format:
            raise InputValidationError("mask_format is required for segmentation")
        if self.task_family == "detection":
            if not self.bbox_format:
                raise InputValidationError("bbox_format is required for detection")
            if not (self.categories or self.coco_json_path):
                raise InputValidationError("detection requires categories or coco_json_path")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetManifest":
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class RetrievedModelCandidate:
    rank: int
    model_id: str
    source: str
    task_family: TaskFamily
    library: str
    processor_or_transform: str
    default_input_size: int | None
    pretrained_weights: str | None
    license: str | None
    training_notes: str | None
    install_deps: list[str] = field(default_factory=list)
    reference_code_ref: str | None = None
    selection_reason: str | None = None

    def validate(self) -> None:
        required = {
            "rank": self.rank,
            "model_id": self.model_id,
            "source": self.source,
            "task_family": self.task_family,
            "library": self.library,
            "processor_or_transform": self.processor_or_transform,
        }
        for name, value in required.items():
            if value in (None, "", []):
                raise InputValidationError(f"RetrievedModelCandidate missing required field: {name}")
        if self.task_family not in _VALID_TASKS:
            raise InputValidationError(f"Unsupported candidate task_family: {self.task_family}")
        if self.rank < 1:
            raise InputValidationError("Candidate rank must be >= 1")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "RetrievedModelCandidate":
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrainingSpec:
    selected_model_id: str
    template_id: str
    task_family: TaskFamily
    dataset_loader_strategy: str
    transforms: list[str]
    loss_fn: str
    metric: str
    optimizer: dict[str, Any]
    scheduler: dict[str, Any]
    epochs: int
    batch_size: int
    freeze_strategy: str
    checkpoint_policy: str
    early_stopping: dict[str, Any]
    edit_regions: dict[str, list[str]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TrainingSpec":
        return cls(**payload)


@dataclass(slots=True)
class ExecutionResult:
    status: ExecutionStatus
    stdout: str
    stderr: str
    runtime_sec: float
    primary_metric_name: str
    primary_metric_value: float | None
    checkpoint_path: str | None
    artifacts_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AblationVariant:
    component: str
    label: str
    overrides: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AblationTrial:
    run_id: str
    component: str
    label: str
    overrides: dict[str, Any]
    result: ExecutionResult

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["result"] = self.result.to_dict()
        return payload


@dataclass(slots=True)
class AblationPlan:
    baseline_run_id: str
    target_component: str
    variants: list[AblationVariant]
    edit_budget: int
    expected_signal: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_run_id": self.baseline_run_id,
            "target_component": self.target_component,
            "variants": [variant.to_dict() for variant in self.variants],
            "edit_budget": self.edit_budget,
            "expected_signal": self.expected_signal,
        }


@dataclass(slots=True)
class AblationSummary:
    best_component_to_change: str
    evidence: str
    tested_variants: list[dict[str, Any]]
    winner_variant: dict[str, Any] | None
    recommended_edit_region: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReviewFinding:
    severity: Literal["error", "warning"]
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ReviewReport:
    status: ReviewStatus
    findings: list[ReviewFinding]
    required_fixes: list[str]
    can_run_in_colab: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "findings": [finding.to_dict() for finding in self.findings],
            "required_fixes": list(self.required_fixes),
            "can_run_in_colab": self.can_run_in_colab,
        }


@dataclass(slots=True)
class GeneratedProject:
    project_dir: str
    manifest_path: str
    spec_path: str
    train_script_path: str
    dataset_script_path: str
    inference_script_path: str
    requirements_path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class WorkflowResult:
    project_dir: str
    selected_candidate: RetrievedModelCandidate
    training_spec: TrainingSpec
    baseline_result: ExecutionResult
    final_result: ExecutionResult
    ablation_summary: AblationSummary
    review_report: ReviewReport
    notebook_path: str
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_dir": self.project_dir,
            "selected_candidate": self.selected_candidate.to_dict(),
            "training_spec": self.training_spec.to_dict(),
            "baseline_result": self.baseline_result.to_dict(),
            "final_result": self.final_result.to_dict(),
            "ablation_summary": self.ablation_summary.to_dict(),
            "review_report": self.review_report.to_dict(),
            "notebook_path": self.notebook_path,
            "artifacts": self.artifacts,
        }

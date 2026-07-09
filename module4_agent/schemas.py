"""Lightweight data contracts for Module 4.

The dataclasses stay small and JSON-friendly. Module 4 consumes Module 3 output,
normalizes it into TrainingSpec objects, generates a local project, runs smoke
tests, and reviews the generated files.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SUPPORTED_TASK_TYPES = {
    "classification",
    "object_detection",
    "image_segmentation",
    "feature_extraction",
}

ALLOWED_REFINEMENT_FIELDS = {
    "optimizer",
    "learning_rate",
    "augmentation",
    "finetune_strategy",
    "loss",
    "checkpoint",
    "backbone",
    "tta",
}


@dataclass
class Module3Candidate:
    """Raw Module 3 candidate wrapper."""

    raw: dict[str, Any]


@dataclass
class TrainingSpec:
    """Internal normalized training specification used by Module 4."""

    rank: int = 1
    score: float = 0.0
    task_type: str = "classification"
    backbone: str = "tiny_cnn"
    pretrained_hf_id: str = ""
    pretrained_name: str = ""
    head: str = "classification_head"
    loss: str = "cross_entropy_loss"
    optimizer: str = "adamw"
    finetune_strategy: str = "head_only"
    freeze_backbone: bool = True
    unfreeze_last_n_blocks: int = 0
    train_norm_layers: bool = True
    strategy_ablation_group: str = ""
    strategy_ablation_variant: str = ""
    scratch_viable: bool = True
    params_M: float | None = None
    tasks: list[Any] = field(default_factory=list)
    alternatives: list[Any] = field(default_factory=list)
    learning_rate: float = 1.0e-3
    augmentation: str = "basic"
    data_size: str = "medium"
    class_imbalance: bool = False
    checkpoint: str = ""
    num_classes: int = 3
    embedding_dim: int = 32
    image_size: int = 224
    tta: bool = False
    offline_smoke: bool = True
    use_pretrained: bool = False
    raw_model_config: dict[str, Any] = field(default_factory=dict)

    def to_config(self) -> dict[str, Any]:
        """Return a portable config dictionary for generated code."""

        config = asdict(self)
        config.pop("raw_model_config", None)
        config["model_config"] = dict(self.raw_model_config)
        return config


@dataclass
class GeneratedFiles:
    """Generated project files keyed by relative path."""

    files: dict[str, str]


@dataclass
class CommandResult:
    """Result for one subprocess smoke command."""

    command: list[str]
    return_code: int
    stdout: str
    stderr: str
    runtime_sec: float
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.return_code == 0 and not self.timed_out

    def to_summary(self) -> dict[str, Any]:
        """Return a compact JSON-serializable command summary."""

        return {
            "command": self.command,
            "return_code": self.return_code,
            "runtime_sec": self.runtime_sec,
            "timed_out": self.timed_out,
            "success": self.success,
            "stdout_tail": self.stdout[-2000:],
            "stderr_tail": self.stderr[-2000:],
        }


@dataclass
class SmokeResult:
    """Aggregate smoke-test result."""

    success: bool
    command_results: list[CommandResult] = field(default_factory=list)
    runtime_sec: float = 0.0
    skipped: bool = False

    @property
    def stdout(self) -> str:
        return "\n".join(result.stdout for result in self.command_results if result.stdout)

    @property
    def stderr(self) -> str:
        return "\n".join(result.stderr for result in self.command_results if result.stderr)

    def to_summary(self) -> dict[str, Any]:
        """Return a JSON-serializable smoke-test summary."""

        return {
            "success": self.success,
            "skipped": self.skipped,
            "runtime_sec": self.runtime_sec,
            "commands": [result.to_summary() for result in self.command_results],
        }


@dataclass
class ReviewResult:
    """Deterministic reviewer output."""

    is_approved: bool
    feedback: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_summary(self) -> dict[str, Any]:
        """Return a JSON-serializable review summary."""

        return {
            "is_approved": self.is_approved,
            "feedback": self.feedback,
            "errors": self.errors,
            "warnings": self.warnings,
        }


@dataclass
class IterationRecord:
    """One generation, execution, and review iteration."""

    iteration: int
    smoke_success: bool
    smoke_skipped: bool
    is_approved: bool
    feedback: str
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_summary(self) -> dict[str, Any]:
        """Return a JSON-serializable iteration summary."""

        return asdict(self)


@dataclass
class WorkflowResult:
    """End-to-end Module 4 workflow output."""

    output_dir: str
    specs: list[TrainingSpec]
    generated_files: list[str]
    smoke_result: SmokeResult
    review_result: ReviewResult
    iterations: int
    iteration_history: list[IterationRecord] = field(default_factory=list)
    refinement_summary: RefinementSummary | None = None

    @property
    def is_approved(self) -> bool:
        return self.review_result.is_approved

    def to_summary(self) -> dict[str, Any]:
        """Return a compact JSON-serializable summary."""

        return {
            "status": "approved" if self.is_approved else "rejected",
            "output_dir": self.output_dir,
            "iterations": self.iterations,
            "num_candidates": len(self.specs),
            "generated_files": self.generated_files,
            "smoke_success": self.smoke_result.success,
            "review_feedback": self.review_result.feedback,
            "errors": self.review_result.errors,
            "warnings": self.review_result.warnings,
            "smoke": self.smoke_result.to_summary(),
            "review": self.review_result.to_summary(),
            "iteration_history": [record.to_summary() for record in self.iteration_history],
            "refinement_summary": self.refinement_summary.to_summary() if self.refinement_summary else None,
            "candidates": [
                {
                    "rank": spec.rank,
                    "score": spec.score,
                    "task_type": spec.task_type,
                    "backbone": spec.backbone,
                    "loss": spec.loss,
                    "optimizer": spec.optimizer,
                    "finetune_strategy": spec.finetune_strategy,
                }
                for spec in self.specs
            ],
        }


@dataclass
class ExperimentResult:
    """One proxy-evaluated experiment in the refinement loop."""

    experiment_id: str
    spec_id: str
    parent_id: str | None
    stage: str
    modified_component: str | None
    metric_name: str
    metric_value: float
    status: str
    config_summary: dict[str, Any]
    notes: str | None = None

    def to_summary(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AblationVariant:
    """A controlled one-component variant of a base TrainingSpec."""

    variant_id: str
    base_spec_id: str
    modified_component: str
    modified_value: Any
    training_spec: TrainingSpec

    def to_summary(self) -> dict[str, Any]:
        data = asdict(self)
        data["training_spec"] = self.training_spec.to_config()
        return data


@dataclass
class RefinementSummary:
    """Summary of the baseline -> ablation -> refinement loop."""

    baseline_result: ExperimentResult
    ablation_results: list[ExperimentResult]
    selected_component: str | None
    refined_result: ExperimentResult | None
    best_result: ExperimentResult
    improvement: float
    stopped_reason: str
    iterations: int = 0

    def to_summary(self) -> dict[str, Any]:
        return {
            "baseline_result": self.baseline_result.to_summary(),
            "ablation_results": [result.to_summary() for result in self.ablation_results],
            "selected_component": self.selected_component,
            "refined_result": self.refined_result.to_summary() if self.refined_result else None,
            "best_result": self.best_result.to_summary(),
            "improvement": self.improvement,
            "stopped_reason": self.stopped_reason,
            "iterations": self.iterations,
        }

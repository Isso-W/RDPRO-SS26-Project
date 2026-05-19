from __future__ import annotations

import py_compile
from pathlib import Path

from .planner import HeuristicTrainingSpecPlanner, clone_spec_with_updates
from .schemas import DatasetManifest, ExecutionResult, ReviewFinding, ReviewReport, TrainingSpec

_ALLOWED_METRICS = {
    "classification": {"accuracy", "f1", "roc_auc", "top1"},
    "segmentation": {"miou", "iou", "dice"},
    "detection": {"map", "map50", "map_50"},
}

_ALLOWED_LOSSES = {
    "classification": {"cross_entropy", "focal"},
    "segmentation": {"dice_ce", "dice", "cross_entropy"},
    "detection": {"detection_default", "focal"},
}


class ProjectReviewer:
    def review(
        self,
        manifest: DatasetManifest,
        spec: TrainingSpec,
        project_dir: str | Path,
        execution_result: ExecutionResult,
        fallback_available: bool,
    ) -> ReviewReport:
        findings: list[ReviewFinding] = []
        required_fixes: list[str] = []
        root = Path(project_dir)

        for file_name in ("train.py", "dataset.py", "inference.py"):
            path = root / file_name
            if not path.exists():
                findings.append(ReviewFinding("error", f"Missing required file: {file_name}"))
                required_fixes.append(f"Regenerate {file_name}")
                continue
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as exc:
                findings.append(ReviewFinding("error", f"py_compile failed for {file_name}: {exc.msg}"))
                required_fixes.append(f"Fix syntax error in {file_name}")

        if manifest.task_family != spec.task_family:
            findings.append(ReviewFinding("error", "Task family mismatch between manifest and training spec"))
            required_fixes.append("Align training_spec.task_family with manifest.task_family")

        if spec.metric not in _ALLOWED_METRICS[manifest.task_family]:
            findings.append(ReviewFinding("error", f"Metric '{spec.metric}' is invalid for {manifest.task_family}"))
            required_fixes.append("Replace metric with a task-compatible metric")

        if spec.loss_fn not in _ALLOWED_LOSSES[manifest.task_family]:
            findings.append(ReviewFinding("error", f"Loss '{spec.loss_fn}' is invalid for {manifest.task_family}"))
            required_fixes.append("Replace loss_fn with a task-compatible loss")

        if spec.batch_size <= 0:
            findings.append(ReviewFinding("error", "batch_size must be positive"))
            required_fixes.append("Set batch_size to a positive integer")

        if execution_result.status != "success":
            findings.append(ReviewFinding("error", f"Training execution failed: {execution_result.stderr}"))
            required_fixes.append("Fix execution failure or fall back to the next candidate")
        elif execution_result.primary_metric_value is None:
            findings.append(ReviewFinding("error", "Training did not produce a primary metric"))
            required_fixes.append("Ensure train.py emits a result payload with metric")

        if manifest.task_family == "segmentation" and not manifest.mask_format:
            findings.append(ReviewFinding("error", "Segmentation manifest requires mask_format"))
            required_fixes.append("Provide mask_format in DatasetManifest")

        if manifest.task_family == "detection" and not manifest.bbox_format:
            findings.append(ReviewFinding("error", "Detection manifest requires bbox_format"))
            required_fixes.append("Provide bbox_format in DatasetManifest")

        can_run_in_colab = (root / "requirements.txt").exists() and not any(
            finding.severity == "error" for finding in findings if "syntax" in finding.message.lower()
        )

        if any(finding.severity == "error" for finding in findings):
            status = "fallback_candidate" if fallback_available else "revise"
        else:
            status = "pass"

        return ReviewReport(
            status=status,
            findings=findings,
            required_fixes=required_fixes,
            can_run_in_colab=can_run_in_colab,
        )

    def apply_fixes(self, manifest: DatasetManifest, spec: TrainingSpec, report: ReviewReport) -> TrainingSpec:
        updates: dict[str, object] = {}
        if spec.metric not in _ALLOWED_METRICS[manifest.task_family]:
            updates["metric"] = HeuristicTrainingSpecPlanner._default_metric(manifest.task_family)
        if spec.loss_fn not in _ALLOWED_LOSSES[manifest.task_family]:
            updates["loss_fn"] = HeuristicTrainingSpecPlanner._default_loss(manifest.task_family)
        if spec.batch_size <= 0:
            updates["batch_size"] = HeuristicTrainingSpecPlanner._default_batch_size(manifest.task_family, manifest.image_size_hint or 224)
        if manifest.task_family == "detection" and "score_threshold" not in spec.optimizer:
            optimizer = dict(spec.optimizer)
            optimizer["score_threshold"] = 0.25
            updates["optimizer"] = optimizer
        return clone_spec_with_updates(spec, updates) if updates else spec

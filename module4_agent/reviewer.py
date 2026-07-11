"""Deterministic reviewer for generated Module 4 projects."""

from __future__ import annotations

import json
from pathlib import Path

from .ablation import CONTROLLED_FIELDS, diff_controlled_fields, has_forbidden_field_changes
from .code_generator import REQUIRED_GENERATED_FILES
from .schemas import GeneratedFiles, ReviewResult, SmokeResult, TrainingSpec


METRICS_BY_TASK = {
    "classification": {"accuracy", "macro-F1", "macro_f1"},
    "object_detection": {"mAP@0.5"},
    "image_segmentation": {"mIoU", "Dice", "dice"},
    "feature_extraction": {"recall@1", "recall@k", "kNN accuracy", "knn_accuracy"},
}


IMPORTANT_FIELDS = (
    "task_type",
    "backbone",
    "pretrained_hf_id",
    "head",
    "loss",
    "optimizer",
    "finetune_strategy",
    "freeze_backbone",
    "params_M",
    "rank",
    "score",
)


def review_generated(
    generated: GeneratedFiles,
    specs: list[TrainingSpec],
    smoke_result: SmokeResult | None = None,
    output_dir: str | Path | None = None,
    refinement_enabled: bool = False,
) -> ReviewResult:
    """Run deterministic checks and return a structured review."""

    errors: list[str] = []
    warnings: list[str] = []
    files = generated.files

    _check_required_files(files, errors)
    _check_compiles(files, errors)
    _check_experiment_sweep(files, specs, errors)
    _check_external_config_support(files, errors)
    _check_configs_json(files, specs, errors)
    _check_generated_readme(files, specs, warnings)
    _check_important_fields(files, specs, errors, warnings)
    _check_finetune_strategy(files, specs, errors, Path(output_dir) if output_dir is not None else None)

    if smoke_result is None:
        errors.append("Smoke result is missing.")
    elif smoke_result.skipped:
        warnings.append("Smoke tests were skipped by request; approval is static-only.")
    elif not smoke_result.success:
        errors.append("Smoke test failed.")
        if smoke_result.stderr:
            warnings.append(smoke_result.stderr[-2000:])
    else:
        _check_smoke_metrics(smoke_result, specs, errors)

    if output_dir is not None:
        _check_written_files(Path(output_dir), errors)
        if refinement_enabled:
            _check_refinement_artifacts(Path(output_dir), errors)
    elif refinement_enabled:
        errors.append("Refinement review requires an output directory.")

    approved = not errors
    feedback = "Approved: generated code is complete, smoke-tested, and consistent." if approved else "Rejected: " + "; ".join(errors)
    return ReviewResult(is_approved=approved, feedback=feedback, errors=errors, warnings=warnings)


def _check_required_files(files: dict[str, str], errors: list[str]) -> None:
    missing = [name for name in REQUIRED_GENERATED_FILES if name not in files]
    if missing:
        errors.append(f"Missing required files: {missing}")


def _check_compiles(files: dict[str, str], errors: list[str]) -> None:
    for filename, content in files.items():
        if filename.endswith(".py"):
            try:
                compile(content, filename, "exec")
            except SyntaxError as exc:
                errors.append(f"{filename} does not compile: {exc}")


def _check_experiment_sweep(files: dict[str, str], specs: list[TrainingSpec], errors: list[str]) -> None:
    content = files.get("run_experiments.py", "")
    if "DEFAULT_CONFIGS" not in content:
        errors.append("run_experiments.py does not embed candidate configs.")
    if "for index, config in enumerate(configs" not in content and "for config in configs" not in content:
        errors.append("run_experiments.py does not visibly loop over all configs.")


def _check_external_config_support(files: dict[str, str], errors: list[str]) -> None:
    run_py = files.get("run.py", "")
    run_experiments_py = files.get("run_experiments.py", "")
    if "--config" not in run_py or "load_config" not in run_py:
        errors.append("run.py does not expose an external --config input path.")
    if "--input" not in run_experiments_py or "load_configs" not in run_experiments_py:
        errors.append("run_experiments.py does not expose an external --input path.")


def _check_configs_json(files: dict[str, str], specs: list[TrainingSpec], errors: list[str]) -> None:
    content = files.get("configs.json")
    if content is None:
        errors.append("configs.json is missing.")
        return
    try:
        configs = json.loads(content)
    except json.JSONDecodeError as exc:
        errors.append(f"configs.json is not valid JSON: {exc}")
        return
    if not isinstance(configs, list):
        errors.append("configs.json must contain a list of candidate configs.")
        return
    if len(configs) != len(specs):
        errors.append(f"configs.json contains {len(configs)} configs, expected {len(specs)}.")
        return
    for config, spec in zip(configs, specs, strict=False):
        if not isinstance(config, dict):
            errors.append("configs.json contains a non-object config.")
            continue
        expected = spec.to_config()
        for field in ("rank", "score", "task_type", "backbone", "loss", "optimizer", "finetune_strategy"):
            if config.get(field) != expected.get(field):
                errors.append(f"configs.json field {field!r} does not match rank {spec.rank}.")


def _check_generated_readme(files: dict[str, str], specs: list[TrainingSpec], warnings: list[str]) -> None:
    readme = files.get("README_generated.md", "")
    for required in (
        "configs.json",
        "utils.py",
        "smoke_data.py",
        "run.py",
        "run_experiments.py",
        "Smoke vs Real Training",
        "Current Limitations",
    ):
        if required not in readme:
            warnings.append(f"README_generated.md does not mention {required!r}.")
    for spec in specs:
        if str(spec.rank) not in readme or spec.backbone not in readme:
            warnings.append(f"README_generated.md may not describe rank {spec.rank} / {spec.backbone}.")


def _check_important_fields(
    files: dict[str, str],
    specs: list[TrainingSpec],
    errors: list[str],
    warnings: list[str],
) -> None:
    combined = "\n".join(files.values())
    for field in IMPORTANT_FIELDS:
        if field not in combined:
            errors.append(f"Important model_config field is not represented: {field}")
    for spec in specs:
        for value in (spec.task_type, spec.backbone, spec.loss, spec.optimizer, spec.finetune_strategy):
            if value and str(value) not in combined:
                warnings.append(f"Spec value is not visible in generated files: {value}")


def _check_finetune_strategy(
    files: dict[str, str],
    specs: list[TrainingSpec],
    errors: list[str],
    output_dir: Path | None,
) -> None:
    needs_check = any(spec.finetune_strategy in {"head_only", "full"} for spec in specs)
    if not needs_check:
        return

    model_py = files.get("model.py", "")
    if any(spec.finetune_strategy == "head_only" for spec in specs) and (
        "parameter.requires_grad = False" not in model_py
        or "\"backbone\" in name" not in model_py
        or "head_only" not in model_py
        or "_frozen_backbone_params" not in model_py
    ):
        errors.append("head_only finetune strategy did not freeze backbone-like parameters.")
    if any(spec.finetune_strategy == "full" for spec in specs) and "strategy == \"full\"" not in model_py:
        errors.append("full finetune strategy is not explicitly handled.")


def _check_smoke_metrics(smoke_result: SmokeResult, specs: list[TrainingSpec], errors: list[str]) -> None:
    experiment_output = ""
    for command_result in smoke_result.command_results:
        if command_result.command and command_result.command[-1] == "run_experiments.py":
            experiment_output = command_result.stdout
            break
    if not experiment_output:
        errors.append("run_experiments.py did not produce stdout.")
        return
    try:
        rows = json.loads(experiment_output)
    except json.JSONDecodeError as exc:
        errors.append(f"run_experiments.py stdout is not JSON: {exc}")
        return
    if not isinstance(rows, list):
        errors.append("run_experiments.py output must be a JSON list.")
        return
    if len(rows) != len(specs):
        errors.append(f"run_experiments.py swept {len(rows)} candidates, expected {len(specs)}.")
        return
    for row, spec in zip(rows, specs, strict=False):
        required_row_fields = {
            "rank",
            "backbone",
            "task_type",
            "loss",
            "optimizer",
            "finetune_strategy",
            "metric_name",
            "metric_value",
            "status",
        }
        missing = required_row_fields - row.keys()
        if missing:
            errors.append(f"Experiment row for rank {spec.rank} is missing fields: {sorted(missing)}.")
        metric_name = str(row.get("metric_name"))
        allowed = METRICS_BY_TASK.get(spec.task_type, set())
        if metric_name not in allowed:
            errors.append(f"Metric {metric_name!r} does not match task type {spec.task_type!r}.")
        if row.get("status") != "success":
            errors.append(f"Experiment row for rank {spec.rank} did not succeed.")


def _check_written_files(output_dir: Path, errors: list[str]) -> None:
    for filename in REQUIRED_GENERATED_FILES:
        if not (output_dir / filename).exists():
            errors.append(f"Required file was not written to output dir: {filename}")


def _check_refinement_artifacts(output_dir: Path, errors: list[str]) -> None:
    jsonl_path = output_dir / "experiments.jsonl"
    summary_path = output_dir / "refinement_summary.json"
    leaderboard_path = output_dir / "leaderboard.json"
    best_config_path = output_dir / "best_config.json"

    missing = [
        path.name
        for path in (jsonl_path, summary_path, leaderboard_path, best_config_path)
        if not path.exists()
    ]
    if missing:
        errors.append(f"Missing refinement artifacts: {missing}")
        return

    rows = _load_experiment_rows(jsonl_path, errors)
    summary = _load_json_object(summary_path, "refinement_summary.json", errors)
    leaderboard = _load_json_object(leaderboard_path, "leaderboard.json", errors)
    best_config = _load_json_object(best_config_path, "best_config.json", errors)
    if not rows or summary is None or leaderboard is None or best_config is None:
        return

    _check_experiment_rows(rows, errors)
    _check_parent_child_refinement(rows, errors)
    _check_refinement_summary(summary, errors)
    _check_leaderboard(leaderboard, errors)
    _check_best_config(best_config, summary, errors)


def _load_experiment_rows(path: Path, errors: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"experiments.jsonl line {line_number} is invalid JSON: {exc}")
            continue
        if not isinstance(row, dict):
            errors.append(f"experiments.jsonl line {line_number} is not an object.")
            continue
        rows.append(row)
    if not rows:
        errors.append("experiments.jsonl contains no experiment rows.")
    return rows


def _load_json_object(path: Path, label: str, errors: list[str]) -> dict[str, object] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        errors.append(f"{label} is invalid JSON: {exc}")
        return None
    if not isinstance(data, dict):
        errors.append(f"{label} must contain a JSON object.")
        return None
    return data


def _check_experiment_rows(rows: list[dict[str, object]], errors: list[str]) -> None:
    required = {
        "experiment_id",
        "stage",
        "parent_id",
        "task_type",
        "backbone",
        "loss",
        "optimizer",
        "finetune_strategy",
        "modified_component",
        "metric_name",
        "metric_value",
        "status",
        "notes",
        "config_summary",
    }
    seen_baseline = False
    for row in rows:
        missing = required - row.keys()
        if missing:
            errors.append(f"Experiment row {row.get('experiment_id')!r} is missing fields: {sorted(missing)}.")
        stage = row.get("stage")
        if stage not in {"baseline", "ablation", "refinement"}:
            errors.append(f"Experiment row {row.get('experiment_id')!r} has invalid stage {stage!r}.")
        seen_baseline = seen_baseline or stage == "baseline"
        modified_component = row.get("modified_component")
        if stage != "baseline" and modified_component not in CONTROLLED_FIELDS:
            errors.append(
                f"Experiment row {row.get('experiment_id')!r} reports unsupported modified_component "
                f"{modified_component!r}."
            )
        task_type = str(row.get("task_type"))
        metric_name = str(row.get("metric_name"))
        if metric_name not in METRICS_BY_TASK.get(task_type, set()):
            errors.append(f"Refinement metric {metric_name!r} does not match task type {task_type!r}.")
        if row.get("status") != "success":
            errors.append(f"Experiment row {row.get('experiment_id')!r} did not succeed.")
        if not isinstance(row.get("config_summary"), dict):
            errors.append(f"Experiment row {row.get('experiment_id')!r} has no config_summary object.")
    if not seen_baseline:
        errors.append("Refinement artifacts do not include a baseline experiment.")


def _check_parent_child_refinement(rows: list[dict[str, object]], errors: list[str]) -> None:
    by_id = {str(row.get("experiment_id")): row for row in rows}
    for row in rows:
        stage = row.get("stage")
        if stage == "baseline":
            if row.get("parent_id") is not None:
                errors.append("Baseline experiment must not have a parent_id.")
            continue
        parent_id = row.get("parent_id")
        if not parent_id or str(parent_id) not in by_id:
            errors.append(f"Experiment row {row.get('experiment_id')!r} has an unknown parent_id.")
            continue
        parent = by_id[str(parent_id)]
        parent_summary = parent.get("config_summary")
        row_summary = row.get("config_summary")
        if not isinstance(parent_summary, dict) or not isinstance(row_summary, dict):
            continue
        if has_forbidden_field_changes(parent_summary, row_summary):
            errors.append(f"Experiment row {row.get('experiment_id')!r} changed a forbidden field.")
        changes = diff_controlled_fields(parent_summary, row_summary)
        modified_component = row.get("modified_component")
        if len(changes) > 1:
            errors.append(
                f"Experiment row {row.get('experiment_id')!r} changed multiple components: {sorted(changes)}."
            )
        if stage == "ablation" and len(changes) != 1:
            errors.append(f"Ablation row {row.get('experiment_id')!r} must modify exactly one component.")
        if changes and modified_component not in changes:
            errors.append(
                f"Experiment row {row.get('experiment_id')!r} reports {modified_component!r} "
                f"but changed {sorted(changes)}."
            )


def _check_refinement_summary(summary: dict[str, object], errors: list[str]) -> None:
    required = {"baseline_result", "best_result", "improvement", "stopped_reason"}
    missing = required - summary.keys()
    if missing:
        errors.append(f"refinement_summary.json is missing fields: {sorted(missing)}.")
    if not isinstance(summary.get("best_result"), dict):
        errors.append("refinement_summary.json does not report a best_result object.")
    if not isinstance(summary.get("baseline_result"), dict):
        errors.append("refinement_summary.json does not report a baseline_result object.")
    try:
        float(summary.get("improvement"))
    except (TypeError, ValueError):
        errors.append("refinement_summary.json improvement is not numeric.")
    if not summary.get("stopped_reason"):
        errors.append("refinement_summary.json stopped_reason is missing.")


def _check_leaderboard(leaderboard: dict[str, object], errors: list[str]) -> None:
    rows = leaderboard.get("rows")
    if not isinstance(rows, list) or not rows:
        errors.append("leaderboard.json must contain a non-empty rows list.")
        return
    metric_values: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            errors.append("leaderboard.json rows must be objects.")
            return
        try:
            metric_values.append(float(row.get("metric_value")))
        except (TypeError, ValueError):
            errors.append("leaderboard.json row metric_value is not numeric.")
            return
    if metric_values != sorted(metric_values, reverse=True):
        errors.append("leaderboard.json rows are not sorted by descending metric_value.")
    if "best_result" not in leaderboard or "improvement" not in leaderboard:
        errors.append("leaderboard.json does not report best_result and improvement.")


def _check_best_config(
    best_config: dict[str, object],
    summary: dict[str, object],
    errors: list[str],
) -> None:
    required = {"task_type", "backbone", "loss", "optimizer", "finetune_strategy", "_module4_refinement"}
    missing = required - best_config.keys()
    if missing:
        errors.append(f"best_config.json is missing fields: {sorted(missing)}.")
    metadata = best_config.get("_module4_refinement")
    if not isinstance(metadata, dict):
        errors.append("best_config.json does not contain _module4_refinement metadata.")
        return
    best_result = summary.get("best_result")
    if isinstance(best_result, dict):
        if metadata.get("experiment_id") != best_result.get("experiment_id"):
            errors.append("best_config.json metadata does not match refinement_summary best_result.")
        try:
            if float(metadata.get("metric_value")) != float(best_result.get("metric_value")):
                errors.append("best_config.json metric_value does not match refinement_summary best_result.")
        except (TypeError, ValueError):
            errors.append("best_config.json metric_value is not numeric.")

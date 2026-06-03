"""End-to-end Module 4 Coder -> Executor -> Reviewer workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .code_generator import generate_files
from .executor import run_smoke, write_generated_files
from .experiment_loop import run_refinement_loop
from .reviewer import review_generated
from .schemas import IterationRecord, SmokeResult, WorkflowResult
from .spec_builder import build_training_specs


MAX_ITER = 2


def load_m3_configs(input_path: str | Path) -> list[dict[str, Any]]:
    """Load Module 3 candidate configs from JSON."""

    data = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("candidates"), list):
        data = data["candidates"]
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        raise ValueError("Input JSON must be a candidate list, one candidate object, or {'candidates': [...]}.")
    return [dict(item or {}) if isinstance(item, dict) else {} for item in data]


def run_workflow(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    max_iter: int = MAX_ITER,
    timeout: int = 60,
    skip_smoke: bool = False,
    run_refinement: bool = False,
    max_refinement_iters: int = 2,
    improvement_threshold: float = 0.01,
) -> WorkflowResult:
    """Run deterministic generation, execution, and review."""

    m3_configs = load_m3_configs(input_path)
    specs = build_training_specs(m3_configs)
    feedback = ""
    generated = None
    smoke_result = None
    review_result = None
    refinement_summary = None
    iteration_history: list[IterationRecord] = []

    iterations = max(1, int(max_iter))
    for iteration in range(1, iterations + 1):
        generated = generate_files(specs, feedback=feedback)
        write_generated_files(generated, output_dir)
        smoke_result = SmokeResult(success=True, skipped=True) if skip_smoke else run_smoke(output_dir, timeout=timeout)
        review_result = review_generated(generated, specs, smoke_result=smoke_result, output_dir=output_dir)
        feedback = review_result.feedback
        iteration_history.append(
            IterationRecord(
                iteration=iteration,
                smoke_success=smoke_result.success,
                smoke_skipped=smoke_result.skipped,
                is_approved=review_result.is_approved,
                feedback=review_result.feedback,
                errors=list(review_result.errors),
                warnings=list(review_result.warnings),
            )
        )
        if review_result.is_approved:
            iterations = iteration
            break
    assert generated is not None
    assert smoke_result is not None
    assert review_result is not None
    if run_refinement and review_result.is_approved:
        refinement_summary = run_refinement_loop(
            specs,
            output_dir,
            max_iters=max_refinement_iters,
            improvement_threshold=improvement_threshold,
        )
        review_result = review_generated(
            generated,
            specs,
            smoke_result=smoke_result,
            output_dir=output_dir,
            refinement_enabled=True,
        )
    generated_files = sorted(generated.files)
    if "module4_summary.json" not in generated_files:
        generated_files.append("module4_summary.json")
    if refinement_summary is not None:
        for artifact in ("experiments.jsonl", "leaderboard.json", "refinement_summary.json", "best_config.json"):
            if artifact not in generated_files:
                generated_files.append(artifact)
    result = WorkflowResult(
        output_dir=str(Path(output_dir)),
        specs=specs,
        generated_files=generated_files,
        smoke_result=smoke_result,
        review_result=review_result,
        iterations=iterations,
        iteration_history=iteration_history,
        refinement_summary=refinement_summary,
    )
    _write_workflow_summary(result, output_dir)
    return result


def _write_workflow_summary(result: WorkflowResult, output_dir: str | Path) -> None:
    """Write the final workflow summary into the generated project."""

    path = Path(output_dir) / "module4_summary.json"
    path.write_text(json.dumps(result.to_summary(), indent=2, sort_keys=True), encoding="utf-8")

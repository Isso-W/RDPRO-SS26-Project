"""Baseline -> ablation -> targeted refinement loop for Module 4."""

from __future__ import annotations

from pathlib import Path

from .ablation import generate_ablation_variants, spec_id
from .experiment_tracker import write_experiment_artifacts
from .refinement import apply_targeted_refinement, proxy_evaluate, select_best_result
from .schemas import ExperimentResult, RefinementSummary, TrainingSpec


def run_refinement_loop(
    specs: list[TrainingSpec],
    output_dir: str | Path,
    *,
    max_iters: int = 2,
    improvement_threshold: float = 0.01,
    seed: int = 123,
) -> RefinementSummary:
    """Run the experiment refinement loop and write artifacts."""

    if not specs:
        raise ValueError("At least one TrainingSpec is required for refinement.")

    baseline_spec = sorted(specs, key=lambda item: item.rank)[0]
    baseline_id = f"{spec_id(baseline_spec)}_baseline"
    baseline_result = proxy_evaluate(
        baseline_spec,
        experiment_id=baseline_id,
        stage="baseline",
        seed=seed,
        smoke_project_dir=output_dir,
    )

    all_results: list[ExperimentResult] = [baseline_result]
    ablation_results: list[ExperimentResult] = []
    best_result = baseline_result
    refined_result: ExperimentResult | None = None
    selected_component: str | None = None
    current_spec = baseline_spec
    current_parent_id = baseline_result.experiment_id
    stopped_reason = "max_iterations_reached"
    completed_iterations = 0

    if max_iters <= 0:
        summary = _build_summary(
            baseline_result=baseline_result,
            ablation_results=ablation_results,
            selected_component=selected_component,
            refined_result=refined_result,
            best_result=best_result,
            stopped_reason=stopped_reason,
            iterations=completed_iterations,
        )
        write_experiment_artifacts(output_dir, all_results, summary)
        return summary

    for iteration in range(1, max_iters + 1):
        completed_iterations = iteration
        variants = generate_ablation_variants(current_spec)
        if not variants:
            stopped_reason = "no_ablation_variants_available"
            break

        iteration_results: list[ExperimentResult] = []
        variant_by_result_id = {}
        for variant_index, variant in enumerate(variants, start=1):
            experiment_id = f"iter{iteration}_abl{variant_index}_{variant.modified_component}"
            result = proxy_evaluate(
                variant.training_spec,
                experiment_id=experiment_id,
                stage="ablation",
                parent_id=current_parent_id,
                modified_component=variant.modified_component,
                seed=seed,
                notes="Ablation changes exactly one controlled component; proxy metric only.",
                smoke_project_dir=output_dir,
            )
            iteration_results.append(result)
            variant_by_result_id[result.experiment_id] = variant

        all_results.extend(iteration_results)
        ablation_results.extend(iteration_results)
        best_ablation = select_best_result(iteration_results)
        if best_ablation.metric_value <= best_result.metric_value:
            stopped_reason = "no_variant_improves_over_baseline"
            break

        selected_variant = variant_by_result_id[best_ablation.experiment_id]
        selected_component = selected_variant.modified_component
        refined_spec = apply_targeted_refinement(selected_variant.training_spec, selected_component)
        refined_result = proxy_evaluate(
            refined_spec,
            experiment_id=f"iter{iteration}_refine_{selected_component}",
            stage="refinement",
            parent_id=best_ablation.experiment_id,
            modified_component=selected_component,
            seed=seed,
            notes="Targeted refinement of the selected ablation component; proxy metric only.",
            smoke_project_dir=output_dir,
        )
        all_results.append(refined_result)

        candidate_best = select_best_result([best_ablation, refined_result])
        if candidate_best.metric_value > best_result.metric_value:
            best_result = candidate_best
            current_parent_id = candidate_best.experiment_id
            current_spec = refined_spec if candidate_best.experiment_id == refined_result.experiment_id else selected_variant.training_spec

        improvement = round(best_result.metric_value - baseline_result.metric_value, 6)
        if improvement >= improvement_threshold:
            stopped_reason = "improvement_threshold_reached"
            break
    else:
        stopped_reason = "max_iterations_reached"

    summary = _build_summary(
        baseline_result=baseline_result,
        ablation_results=ablation_results,
        selected_component=selected_component,
        refined_result=refined_result,
        best_result=best_result,
        stopped_reason=stopped_reason,
        iterations=completed_iterations,
    )
    write_experiment_artifacts(output_dir, all_results, summary)
    return summary


def _build_summary(
    *,
    baseline_result: ExperimentResult,
    ablation_results: list[ExperimentResult],
    selected_component: str | None,
    refined_result: ExperimentResult | None,
    best_result: ExperimentResult,
    stopped_reason: str,
    iterations: int,
) -> RefinementSummary:
    return RefinementSummary(
        baseline_result=baseline_result,
        ablation_results=ablation_results,
        selected_component=selected_component,
        refined_result=refined_result,
        best_result=best_result,
        improvement=round(best_result.metric_value - baseline_result.metric_value, 6),
        stopped_reason=stopped_reason,
        iterations=iterations,
    )

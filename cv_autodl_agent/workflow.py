from __future__ import annotations

from pathlib import Path
from typing import Any

from .ablation import AblationEngine
from .codegen import CodeGenerator
from .exceptions import CandidateSelectionError, InputValidationError, WorkflowExecutionError
from .executor import GeneratedProjectExecutor
from .io_utils import slugify, write_json
from .planner import HeuristicTrainingSpecPlanner
from .review import ProjectReviewer
from .notebook import NotebookExporter
from .schemas import (
    DatasetManifest,
    RetrievedModelCandidate,
    ReviewReport,
    TrainingSpec,
    WorkflowResult,
)
from .selectors import rank_candidates


class CVAutoDLWorkflow:
    def __init__(
        self,
        planner: HeuristicTrainingSpecPlanner | None = None,
        code_generator: CodeGenerator | None = None,
        executor: GeneratedProjectExecutor | None = None,
        ablation_engine: AblationEngine | None = None,
        reviewer: ProjectReviewer | None = None,
        notebook_exporter: NotebookExporter | None = None,
    ) -> None:
        self.planner = planner or HeuristicTrainingSpecPlanner()
        self.code_generator = code_generator or CodeGenerator()
        self.executor = executor or GeneratedProjectExecutor()
        self.ablation_engine = ablation_engine or AblationEngine()
        self.reviewer = reviewer or ProjectReviewer()
        self.notebook_exporter = notebook_exporter or NotebookExporter()

    def run(
        self,
        manifest: DatasetManifest,
        candidates: list[RetrievedModelCandidate],
        output_root: str | Path,
        execution_mode: str = "simulate",
        notebook_execution_mode: str | None = None,
    ) -> WorkflowResult:
        self._validate_inputs(manifest, candidates)
        ranked = rank_candidates(manifest, candidates)
        if not ranked or ranked[0].score == float("-inf"):
            raise CandidateSelectionError("No compatible candidates were provided")

        root = Path(output_root)
        root.mkdir(parents=True, exist_ok=True)
        failures: list[str] = []

        for index, scored in enumerate(ranked, start=1):
            if scored.score == float("-inf"):
                continue
            candidate = scored.candidate
            project_dir = root / f"candidate_{index:02d}_{slugify(candidate.model_id)}"
            project_dir.mkdir(parents=True, exist_ok=True)
            write_json(project_dir / "selected_candidate.json", candidate)

            spec = self.planner.plan(manifest, candidate)
            generated = self.code_generator.generate(project_dir, manifest, spec)
            baseline_result = self.executor.run(
                project=generated,
                spec_path=generated.spec_path,
                run_dir=project_dir / "runs" / "baseline",
                execution_mode=execution_mode,
            )
            write_json(project_dir / "baseline_result.json", baseline_result)

            if baseline_result.status != "success":
                failures.append(f"Baseline failed for {candidate.model_id}: {baseline_result.stderr}")
                continue

            ablation_plan = self.ablation_engine.build_plan(manifest, spec, baseline_run_id="baseline")
            write_json(project_dir / "ablation_plan.json", ablation_plan)
            ablation_trials = self.ablation_engine.run(
                project=generated,
                spec=spec,
                plan=ablation_plan,
                executor=self.executor,
                root_dir=project_dir / "runs" / "ablation",
                execution_mode=execution_mode,
            )
            write_json(project_dir / "ablation_trials.json", [trial.to_dict() for trial in ablation_trials])

            ablation_summary = self.ablation_engine.summarize(baseline_result, ablation_trials, spec)
            write_json(project_dir / "ablation_summary.json", ablation_summary)
            refined_spec = self.ablation_engine.apply_summary(spec, ablation_summary)
            active_spec_path = Path(generated.spec_path)
            final_result = baseline_result

            if refined_spec.to_dict() != spec.to_dict():
                active_spec_path = write_json(project_dir / "refined_training_spec.json", refined_spec)
                final_result = self.executor.run(
                    project=generated,
                    spec_path=active_spec_path,
                    run_dir=project_dir / "runs" / "refined",
                    execution_mode=execution_mode,
                )
                write_json(project_dir / "refined_result.json", final_result)
            else:
                refined_spec = spec

            review_report = self.reviewer.review(
                manifest=manifest,
                spec=refined_spec,
                project_dir=project_dir,
                execution_result=final_result,
                fallback_available=index < len(ranked),
            )
            write_json(project_dir / "review_report.json", review_report)

            if review_report.status == "revise":
                fixed_spec = self.reviewer.apply_fixes(manifest, refined_spec, review_report)
                if fixed_spec.to_dict() != refined_spec.to_dict():
                    active_spec_path = write_json(project_dir / "fixed_training_spec.json", fixed_spec)
                    final_result = self.executor.run(
                        project=generated,
                        spec_path=active_spec_path,
                        run_dir=project_dir / "runs" / "fixed",
                        execution_mode=execution_mode,
                    )
                    write_json(project_dir / "fixed_result.json", final_result)
                    review_report = self.reviewer.review(
                        manifest=manifest,
                        spec=fixed_spec,
                        project_dir=project_dir,
                        execution_result=final_result,
                        fallback_available=index < len(ranked),
                    )
                    write_json(project_dir / "review_report_after_fix.json", review_report)
                    refined_spec = fixed_spec

            if review_report.status == "pass":
                best_run_dir = (
                    "runs/fixed"
                    if (project_dir / "runs" / "fixed").exists()
                    else "runs/refined"
                    if (project_dir / "runs" / "refined").exists()
                    else "runs/baseline"
                )
                notebook_path = self.notebook_exporter.export(
                    project_dir=project_dir,
                    manifest=manifest,
                    spec=refined_spec,
                    active_spec_file=Path(active_spec_path).name,
                    best_run_dir=best_run_dir,
                    execution_mode=notebook_execution_mode or execution_mode,
                )
                return WorkflowResult(
                    project_dir=str(project_dir),
                    selected_candidate=candidate,
                    training_spec=refined_spec,
                    baseline_result=baseline_result,
                    final_result=final_result,
                    ablation_summary=ablation_summary,
                    review_report=review_report,
                    notebook_path=str(notebook_path),
                    artifacts={
                        "selected_candidate_json": str(project_dir / "selected_candidate.json"),
                        "training_spec_json": str(active_spec_path),
                        "baseline_result_json": str(project_dir / "baseline_result.json"),
                        "ablation_plan_json": str(project_dir / "ablation_plan.json"),
                        "ablation_summary_json": str(project_dir / "ablation_summary.json"),
                        "review_report_json": str(project_dir / "review_report.json"),
                    },
                )

            failures.append(f"Candidate {candidate.model_id} finished with review status {review_report.status}")

        raise WorkflowExecutionError("Workflow failed to produce a passing candidate. " + " | ".join(failures))

    def _validate_inputs(
        self,
        manifest: DatasetManifest,
        candidates: list[RetrievedModelCandidate],
    ) -> None:
        manifest.validate()
        if not candidates:
            raise InputValidationError("RetrievedModelCandidate[] cannot be empty")
        for candidate in candidates:
            candidate.validate()

    def build_langgraph_workflow(self) -> Any | None:
        try:
            from langgraph.graph import END, START, StateGraph
        except ImportError:
            return None

        class WorkflowState(dict):
            pass

        graph = StateGraph(WorkflowState)

        def passthrough(state: WorkflowState) -> WorkflowState:
            return state

        for node_name in (
            "select_candidate",
            "build_training_spec",
            "generate_baseline_code",
            "run_baseline",
            "run_ablation",
            "summarize_ablation",
            "targeted_refine",
            "review_and_validate",
            "export_notebook",
        ):
            graph.add_node(node_name, passthrough)
        graph.add_edge(START, "select_candidate")
        graph.add_edge("select_candidate", "build_training_spec")
        graph.add_edge("build_training_spec", "generate_baseline_code")
        graph.add_edge("generate_baseline_code", "run_baseline")
        graph.add_edge("run_baseline", "run_ablation")
        graph.add_edge("run_ablation", "summarize_ablation")
        graph.add_edge("summarize_ablation", "targeted_refine")
        graph.add_edge("targeted_refine", "review_and_validate")
        graph.add_edge("review_and_validate", "export_notebook")
        graph.add_edge("export_notebook", END)
        return graph.compile()

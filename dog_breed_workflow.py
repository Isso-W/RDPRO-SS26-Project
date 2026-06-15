"""End-to-end Dog Breed AutoPipeline + MCP experiment workflow."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import cost_meter
from autopipeline import flatten_candidate_config, select_candidate
from agents.knowledge_learner_agent import learn_fixed_sources
from agents.mle_experiment_agent import run_experiment_loop
from ensemble import optimize_validation_ensemble
from kaggle_submit import predict_and_submit
from module4_agent.result_parser import extract_last_json
from pipeline import run_kaggle_pipeline
from recommender import OutcomeMemory
from recommender.fingerprint import dataset_fingerprint


def flatten_config(config: dict[str, Any]) -> dict[str, Any]:
    return flatten_candidate_config(config)


def train_baseline(
    project_dir: str | Path,
    baseline_config: dict[str, Any] | None = None,
) -> tuple[dict, dict, Path]:
    project = Path(project_dir).resolve()
    if baseline_config is None:
        configs = json.loads((project / "configs.json").read_text(encoding="utf-8"))
        baseline_config = configs[0] if isinstance(configs, list) else configs
    baseline = flatten_config(baseline_config)
    baseline["checkpoint_dir"] = str(project / ".jiaozi_experiments" / "baseline" / "checkpoints")
    baseline["resume_checkpoint"] = ""
    config_path = project / ".jiaozi_experiments" / "baseline.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(baseline, indent=2, sort_keys=True), encoding="utf-8")
    completed = subprocess.run(
        [sys.executable, "-u", "run.py", "--config", str(config_path)],
        cwd=project,
        text=True,
        capture_output=True,
    )
    log_path = config_path.with_suffix(".log")
    log_path.write_text(
        completed.stdout + ("\n[stderr]\n" + completed.stderr if completed.stderr else ""),
        encoding="utf-8",
    )
    if completed.returncode:
        raise RuntimeError(f"Baseline training failed; see {log_path}\n{completed.stderr[-2000:]}")
    summary = extract_last_json(completed.stdout)
    if not summary:
        raise RuntimeError(f"Baseline did not emit a JSON summary; see {log_path}")
    evaluate = summary.get("evaluate") or {}
    metrics = {
        "experiment_name": "baseline",
        "status": summary.get("status", "success"),
        "metric_name": evaluate.get("metric_name"),
        "metric_value": evaluate.get("metric_value"),
        "accuracy": evaluate.get("accuracy"),
        "macro_f1": evaluate.get("macro_f1"),
        "validation_artifact": evaluate.get("validation_artifact"),
        "best_epoch": (summary.get("train") or {}).get("best_epoch"),
        "runtime_sec": (summary.get("train") or {}).get("runtime_sec"),
    }
    return baseline, metrics, config_path


def _selected_experiment(loop: dict, baseline_path: Path) -> tuple[str, Path]:
    selected = (loop.get("comparison") or {}).get("best_experiment") or "baseline"
    if selected == "baseline":
        return selected, baseline_path
    for run in loop.get("runs", []):
        if run.get("experiment_name") == selected:
            return selected, Path(run["config_path"])
    return "baseline", baseline_path


async def execute_dog_breed_workflow(
    *,
    data_root: str | Path,
    workspace_root: str | Path,
    report_path: str | Path,
    run_knowledge_learner: bool = True,
    submit_to_kaggle: bool = True,
) -> dict:
    cost_meter.reset()
    workspace = Path(workspace_root).resolve()
    os.environ["JIAOZI_WORKSPACE_ROOT"] = str(workspace)
    kb_root = Path(
        os.getenv("JIAOZI_KB_ROOT", str(workspace.parent / "knowledge_base"))
    ).resolve()
    os.environ["JIAOZI_KB_ROOT"] = str(kb_root)
    os.environ["JIAOZI_OUTCOME_MEMORY"] = str(kb_root / "experiments" / "outcomes.jsonl")
    project = workspace / "dog_breed" / "module4_code"
    project.parent.mkdir(parents=True, exist_ok=True)

    knowledge = await learn_fixed_sources(min_successful_sources=3) if run_knowledge_learner else None
    pipeline_result = run_kaggle_pipeline(
        "dog_breed",
        data_root,
        module4_output=project,
        recommender_memory=str(
            kb_root / "experiments" / "outcomes.jsonl"
        ),
    )
    fingerprint = dataset_fingerprint(
        pipeline_result["m2_report"],
        pipeline_result["module3_input"],
    )
    generated_configs = json.loads(
        (project / "configs.json").read_text(encoding="utf-8")
    )
    if not isinstance(generated_configs, list):
        generated_configs = [generated_configs]
    candidate_calibration = select_candidate(
        project,
        generated_configs,
        target_metric="log_loss",
        probe_epochs=2,
        max_candidates=3,
    )
    baseline, baseline_metrics, baseline_path = train_baseline(
        project,
        candidate_calibration["selected_config"],
    )
    for trial in candidate_calibration["trials"]:
        if trial["status"] == "success":
            cost_meter.record_training(
                epochs=int(trial.get("probe_epochs", 0) or 0),
                runs=1,
            )
    cost_meter.record_training(
        epochs=int(baseline.get("recommended_epochs", 0) or 0),
        runs=1,
    )
    memory = OutcomeMemory(kb_root / "experiments" / "outcomes.jsonl")
    for trial in candidate_calibration["trials"]:
        memory.log(
            fingerprint,
            trial["config"],
            {
                "metric_name": trial.get("metric_name"),
                "metric_value": trial.get("metric_value"),
                "accuracy": trial.get("accuracy"),
                "macro_f1": trial.get("macro_f1"),
                "status": trial.get("status"),
            },
            dataset_id="dog_breed",
            cost={"wall_clock_sec": trial.get("runtime_sec")},
            metadata={
                "experiment_name": f"autopipeline_probe_{trial['candidate_index']}",
                "stage": "candidate_calibration",
                "status": trial.get("status"),
            },
        )
    memory.log(
        fingerprint,
        baseline,
        {
            "metric_name": baseline_metrics["metric_name"],
            "metric_value": baseline_metrics["metric_value"],
            "accuracy": baseline_metrics["accuracy"],
            "macro_f1": baseline_metrics["macro_f1"],
            "best_epoch": baseline_metrics["best_epoch"],
            "status": baseline_metrics["status"],
        },
        dataset_id="dog_breed",
        cost={"wall_clock_sec": baseline_metrics["runtime_sec"]},
        metadata={"experiment_name": "baseline", "status": baseline_metrics["status"]},
    )
    loop = await run_experiment_loop(
        project_dir=str(project),
        baseline_config=baseline,
        baseline_metrics=baseline_metrics,
        fingerprint=fingerprint,
        dataset_id="dog_breed",
        execute=True,
    )
    for run in loop.get("runs", []):
        if run.get("status") != "success":
            continue
        summary = run.get("summary") or {}
        train = summary.get("train") or {}
        actual_epochs = len(train.get("validation_history") or [])
        if not actual_epochs:
            actual_epochs = int(
                next(
                    (
                        proposal["config"].get("recommended_epochs", 0)
                        for proposal in loop.get("proposals", [])
                        if proposal["experiment_name"] == run.get("experiment_name")
                    ),
                    0,
                )
                or 0
            )
        cost_meter.record_training(epochs=actual_epochs, runs=1)
    selected_name, selected_config = _selected_experiment(loop, baseline_path)
    ensemble_candidates = [
        {
            "name": "baseline",
            "config_path": str(baseline_path),
            "validation_artifact": baseline_metrics.get("validation_artifact"),
            "metric_value": baseline_metrics.get("metric_value"),
        }
    ]
    metrics_by_name = {
        item.get("experiment_name"): item for item in loop.get("metrics", [])
    }
    for run in loop.get("runs", []):
        measured = metrics_by_name.get(run.get("experiment_name")) or {}
        if run.get("status") != "success" or measured.get("metric_value") is None:
            continue
        ensemble_candidates.append(
            {
                "name": run["experiment_name"],
                "config_path": run["config_path"],
                "validation_artifact": measured.get("validation_artifact"),
                "metric_value": measured.get("metric_value"),
            }
        )
    ensemble_candidates.sort(
        key=lambda item: (
            item.get("metric_value") is None,
            float(item.get("metric_value") or float("inf")),
        )
    )
    ensemble_plan = optimize_validation_ensemble(
        ensemble_candidates,
        step=0.05,
        max_members=3,
    )
    submission_members = (
        ensemble_plan["members"]
        if ensemble_plan.get("improved") and len(ensemble_plan.get("members", [])) > 1
        else None
    )
    submission_selection = "validation_ensemble" if submission_members else selected_name
    submission = predict_and_submit(
        "dog_breed",
        project,
        data_root,
        message=f"Jiaozi MCP {submission_selection}",
        do_submit=submit_to_kaggle,
        config_path=selected_config,
        selected_experiment=submission_selection,
        ensemble_members=submission_members,
        score_timeout_sec=1800,
        metadata_path=Path(report_path).with_name("submission_result.json"),
    )
    cost = cost_meter.report()
    if knowledge:
        cost["llm_calls"] += int(knowledge.get("llm_calls", 0) or 0)
        cost["llm_tokens"] += int(knowledge.get("llm_tokens", 0) or 0)
    report = {
        "benchmark": "Dog Breed Identification",
        "standard_reference_only": pipeline_result.get("benchmark_reference"),
        "autopipeline_selected": pipeline_result["recommendations"][
            candidate_calibration["selected_index"]
        ],
        "candidate_calibration": candidate_calibration,
        "module3_input": pipeline_result["module3_input"],
        "m2_report": pipeline_result["m2_report"],
        "knowledge": knowledge,
        "baseline_config": baseline,
        "baseline_metrics": baseline_metrics,
        "mcp_calls": [
            "search_strategy_cards",
            "get_past_experiments",
            "generate_experiment_configs",
            "run_experiment",
            "read_metrics",
            "compare_results",
            "write_experiment_result",
        ],
        "experiment_loop": loop,
        "selected_experiment": selected_name,
        "selected_config_path": str(selected_config),
        "ensemble": ensemble_plan,
        "submission_selection": submission_selection,
        "submission": submission,
        "cost": cost,
    }
    destination = Path(report_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return report


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run Dog Breed MCP AutoPipeline.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--skip-knowledge", action="store_true")
    parser.add_argument("--no-submit", action="store_true")
    args = parser.parse_args()
    result = asyncio.run(
        execute_dog_breed_workflow(
            data_root=args.data_root,
            workspace_root=args.workspace_root,
            report_path=args.report,
            run_knowledge_learner=not args.skip_knowledge,
            submit_to_kaggle=not args.no_submit,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()

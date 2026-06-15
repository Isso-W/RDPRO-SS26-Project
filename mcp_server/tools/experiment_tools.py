"""Controlled experiment planning, execution, comparison, and write-back."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from module4_agent.result_parser import extract_last_json
from recommender import generate_proposals


def get_past_experiments_service(context, dataset_id: str, top_k: int = 10) -> list[dict]:
    return context.memory.recent(dataset_id=dataset_id, k=min(int(top_k), 10))


def generate_experiment_configs_service(
    context,
    baseline_config: dict,
    strategy_cards: list[dict],
    past_experiments: list[dict] | None = None,
    max_experiments: int = 3,
    max_changed_variables: int = 2,
) -> list[dict]:
    return [
        proposal.to_dict()
        for proposal in generate_proposals(
            baseline_config,
            strategy_cards,
            past_experiments,
            max_experiments=min(int(max_experiments), 3),
            max_changed_variables=min(int(max_changed_variables), 2),
        )
    ]


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def run_experiment_service(
    context,
    project_dir: str,
    experiment_name: str,
    config: dict[str, Any],
) -> dict:
    project = Path(project_dir).expanduser().resolve()
    if not _inside(project, context.workspace_root):
        raise ValueError("project_dir must be inside JIAOZI_WORKSPACE_ROOT.")
    run_py = project / "run.py"
    if not run_py.is_file():
        raise FileNotFoundError(f"Generated run.py not found: {run_py}")
    safe_name = "".join(ch for ch in experiment_name if ch.isalnum() or ch in {"_", "-"})[:80]
    if not safe_name:
        raise ValueError("experiment_name must contain safe filename characters.")
    config_dir = project / ".jiaozi_experiments"
    config_dir.mkdir(parents=True, exist_ok=True)
    controlled_config = dict(config)
    controlled_config["checkpoint_dir"] = str(config_dir / safe_name / "checkpoints")
    controlled_config["resume_checkpoint"] = ""
    config_path = config_dir / f"{safe_name}.json"
    config_path.write_text(
        json.dumps(controlled_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    started = time.monotonic()
    completed = subprocess.run(
        [sys.executable, "-u", "run.py", "--config", str(config_path)],
        cwd=project,
        text=True,
        capture_output=True,
        timeout=int(controlled_config.get("execution_timeout_sec", 21600)),
    )
    runtime = round(time.monotonic() - started, 4)
    log_path = config_dir / f"{safe_name}.log"
    log_path.write_text(
        completed.stdout + ("\n[stderr]\n" + completed.stderr if completed.stderr else ""),
        encoding="utf-8",
    )
    summary = extract_last_json(completed.stdout)
    return {
        "experiment_name": experiment_name,
        "status": "success" if completed.returncode == 0 and summary else "failed",
        "return_code": completed.returncode,
        "runtime_sec": runtime,
        "config_path": str(config_path),
        "log_path": str(log_path),
        "summary": summary,
        "stderr_tail": completed.stderr[-2000:],
    }


def read_metrics_service(run_result: dict) -> dict:
    summary = run_result.get("summary") or {}
    evaluate = summary.get("evaluate") or {}
    train = summary.get("train") or {}
    return {
        "experiment_name": run_result.get("experiment_name"),
        "status": run_result.get("status"),
        "metric_name": evaluate.get("metric_name"),
        "metric_value": evaluate.get("metric_value"),
        "accuracy": evaluate.get("accuracy"),
        "macro_f1": evaluate.get("macro_f1"),
        "validation_artifact": evaluate.get("validation_artifact"),
        "best_epoch": train.get("best_epoch"),
        "runtime_sec": run_result.get("runtime_sec"),
    }


def compare_results_service(
    baseline_metrics: dict,
    experiment_metrics: list[dict],
    target_metric: str = "log_loss",
) -> dict:
    minimize = target_metric.lower() in {"log_loss", "multiclass_log_loss", "rmse"}
    baseline_value = baseline_metrics.get("metric_value")
    valid = [item for item in experiment_metrics if item.get("metric_value") is not None]
    if not valid:
        return {
            "best_experiment": "baseline",
            "improved": False,
            "target_metric": target_metric,
            "baseline_value": baseline_value,
            "best_value": baseline_value,
            "metric_delta": 0.0 if baseline_value is not None else None,
        }
    best = min(valid, key=lambda item: item["metric_value"]) if minimize else max(
        valid, key=lambda item: item["metric_value"]
    )
    best_value = float(best["metric_value"])
    improved = baseline_value is None or (
        best_value < float(baseline_value) if minimize else best_value > float(baseline_value)
    )
    if not improved:
        return {
            "best_experiment": "baseline",
            "improved": False,
            "target_metric": target_metric,
            "baseline_value": baseline_value,
            "best_value": baseline_value,
            "metric_delta": 0.0,
        }
    delta = (
        float(baseline_value) - best_value
        if minimize and baseline_value is not None
        else best_value - float(baseline_value)
        if baseline_value is not None
        else None
    )
    return {
        "best_experiment": best.get("experiment_name"),
        "improved": True,
        "target_metric": target_metric,
        "baseline_value": baseline_value,
        "best_value": best_value,
        "metric_delta": delta,
    }


def write_experiment_result_service(
    context,
    dataset_id: str,
    fingerprint: dict,
    proposal: dict,
    metrics: dict,
    baseline_metric: float | None,
) -> dict:
    metric = metrics.get("metric_value")
    improved = metric is not None and (
        baseline_metric is None or float(metric) < float(baseline_metric)
    )
    metadata = {
        "experiment_name": proposal.get("experiment_name"),
        "strategy_card_ids": proposal.get("strategy_card_ids", []),
        "changed_fields": proposal.get("changed_fields", []),
        "parent": "baseline",
        "status": metrics.get("status", "unknown"),
        "metric_delta": (
            float(baseline_metric) - float(metric)
            if metric is not None and baseline_metric is not None
            else None
        ),
    }
    context.memory.log(
        fingerprint,
        proposal.get("config", {}),
        {
            "metric_name": metrics.get("metric_name", "log_loss"),
            "metric_value": metric,
            "accuracy": metrics.get("accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "best_epoch": metrics.get("best_epoch"),
            "status": metrics.get("status"),
        },
        dataset_id=dataset_id,
        cost={"wall_clock_sec": metrics.get("runtime_sec")},
        metadata=metadata,
    )
    observation = {
        "dataset_id": dataset_id,
        "experiment_name": proposal.get("experiment_name"),
        "metric_value": metric,
        "metric_delta": metadata["metric_delta"],
        "timestamp": time.time(),
    }
    for card_id in proposal.get("strategy_card_ids", []):
        context.store.record_observation(card_id, observation, improved=improved)
    return {"written": True, "improved": improved, **metadata}

"""Experiment artifact writer for Module 4 refinement runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import ExperimentResult, RefinementSummary


def write_experiment_artifacts(
    output_dir: str | Path,
    results: list[ExperimentResult],
    summary: RefinementSummary,
) -> None:
    """Write JSONL, leaderboard, and summary files into the generated project."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_path / "experiments.jsonl", [_experiment_row(result) for result in results])
    _write_json(output_path / "leaderboard.json", _leaderboard(results, summary))
    _write_json(output_path / "refinement_summary.json", summary.to_summary())
    _write_json(output_path / "best_config.json", _best_config(summary))


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def _leaderboard(results: list[ExperimentResult], summary: RefinementSummary) -> dict[str, Any]:
    rows = sorted(
        [_experiment_row(result) for result in results if result.status == "success"],
        key=lambda row: row["metric_value"],
        reverse=True,
    )
    return {
        "higher_is_better": True,
        "rows": rows,
        "best_result": summary.best_result.to_summary(),
        "baseline_result": summary.baseline_result.to_summary(),
        "improvement": summary.improvement,
        "stopped_reason": summary.stopped_reason,
    }


def _best_config(summary: RefinementSummary) -> dict[str, Any]:
    config = dict(summary.best_result.config_summary)
    config["_module4_refinement"] = {
        "experiment_id": summary.best_result.experiment_id,
        "stage": summary.best_result.stage,
        "metric_name": summary.best_result.metric_name,
        "metric_value": summary.best_result.metric_value,
        "improvement": summary.improvement,
        "stopped_reason": summary.stopped_reason,
        "proxy_metric": True,
    }
    return config


def _experiment_row(result: ExperimentResult) -> dict[str, Any]:
    config = result.config_summary
    return {
        "experiment_id": result.experiment_id,
        "spec_id": result.spec_id,
        "stage": result.stage,
        "parent_id": result.parent_id,
        "rank": config.get("rank"),
        "task_type": config.get("task_type"),
        "backbone": config.get("backbone"),
        "checkpoint": config.get("checkpoint"),
        "loss": config.get("loss"),
        "optimizer": config.get("optimizer"),
        "finetune_strategy": config.get("finetune_strategy"),
        "learning_rate": config.get("learning_rate"),
        "augmentation": config.get("augmentation"),
        "modified_component": result.modified_component,
        "metric_name": result.metric_name,
        "metric_value": result.metric_value,
        "status": result.status,
        "notes": result.notes,
        "config_summary": config,
    }

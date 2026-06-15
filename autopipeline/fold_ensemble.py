"""Controlled selected-configuration fold training for final prediction averaging."""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

from module4_agent.result_parser import extract_last_json


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _normalized_inverse_loss_weights(losses: list[float]) -> list[float]:
    if not losses:
        return []
    inverse = [1.0 / max(float(loss), 1.0e-6) for loss in losses]
    ordered = sorted(inverse)
    median = ordered[len(ordered) // 2]
    clipped = [
        min(max(value, 0.5 * median), 2.0 * median)
        for value in inverse
    ]
    total = sum(clipped)
    return [value / total for value in clipped]


def train_selected_folds(
    project_dir: str | Path,
    selected_config_path: str | Path,
    *,
    fold_count: int = 3,
) -> dict[str, Any]:
    """Train one Module 4-selected configuration on deterministic stratified folds."""

    project = Path(project_dir).expanduser().resolve()
    config_source = Path(selected_config_path).expanduser().resolve()
    if not (project / "run.py").is_file():
        raise FileNotFoundError(f"Generated run.py not found: {project / 'run.py'}")
    if not _inside(config_source, project):
        raise ValueError("Selected config must stay inside the generated project.")
    if not config_source.is_file():
        raise FileNotFoundError(f"Selected config does not exist: {config_source}")
    count = max(2, int(fold_count))
    source = json.loads(config_source.read_text(encoding="utf-8"))
    if isinstance(source, list):
        if not source:
            raise ValueError("Selected config list is empty.")
        source = source[0]
    if not isinstance(source, dict):
        raise ValueError("Selected config must contain a JSON object.")
    nested = source.get("model_config")
    if isinstance(nested, dict):
        merged = dict(source)
        for key, value in nested.items():
            if value is not None or key not in merged:
                merged[key] = value
        source = merged

    fold_root = project / ".jiaozi_folds"
    fold_root.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    successful: list[dict[str, Any]] = []
    for fold_index in range(count):
        controlled = dict(source)
        checkpoint_dir = fold_root / f"fold_{fold_index}" / "checkpoints"
        controlled.update(
            {
                "fold_count": count,
                "fold_index": fold_index,
                "checkpoint_dir": str(checkpoint_dir),
                "resume_checkpoint": "",
            }
        )
        config_path = fold_root / f"fold_{fold_index}.json"
        config_path.write_text(
            json.dumps(controlled, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [sys.executable, "-u", "run.py", "--config", str(config_path)],
            cwd=project,
            text=True,
            capture_output=True,
        )
        log_path = config_path.with_suffix(".log")
        log_path.write_text(
            completed.stdout
            + ("\n[stderr]\n" + completed.stderr if completed.stderr else ""),
            encoding="utf-8",
        )
        summary = extract_last_json(completed.stdout)
        evaluate = (summary or {}).get("evaluate") or {}
        train = (summary or {}).get("train") or {}
        metric_value = evaluate.get("metric_value")
        valid_metric = (
            metric_value is not None
            and math.isfinite(float(metric_value))
        )
        status = (
            "success"
            if completed.returncode == 0 and summary and valid_metric
            else "failed"
        )
        run = {
            "name": f"selected_fold_{fold_index}",
            "fold_index": fold_index,
            "fold_count": count,
            "status": status,
            "config_path": str(config_path),
            "log_path": str(log_path),
            "checkpoint_dir": str(checkpoint_dir),
            "validation_artifact": evaluate.get("validation_artifact"),
            "metric_name": evaluate.get("metric_name"),
            "metric_value": metric_value,
            "accuracy": evaluate.get("accuracy"),
            "macro_f1": evaluate.get("macro_f1"),
            "prior_alpha": evaluate.get("prior_alpha"),
            "prior_model": evaluate.get("prior_model"),
            "best_epoch": train.get("best_epoch"),
            "runtime_sec": train.get("runtime_sec"),
            "actual_epochs": len(train.get("validation_history") or []),
            "stderr_tail": completed.stderr[-2000:],
        }
        runs.append(run)
        if status == "success":
            successful.append(run)

    weights = _normalized_inverse_loss_weights(
        [float(run["metric_value"]) for run in successful]
    )
    members = []
    for run, weight in zip(successful, weights):
        members.append(
            {
                "name": run["name"],
                "config_path": run["config_path"],
                "validation_artifact": run["validation_artifact"],
                "metric_value": run["metric_value"],
                "weight": weight,
                "fold_index": run["fold_index"],
            }
        )
    status = (
        "success"
        if len(successful) == count
        else "partial"
        if successful
        else "failed"
    )
    return {
        "status": status,
        "fold_count": count,
        "selected_config_path": str(config_source),
        "runs": runs,
        "members": members,
    }

"""Short, controlled validation probes for Module 3 candidate selection."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from module4_agent.result_parser import extract_last_json


def flatten_candidate_config(config: dict[str, Any]) -> dict[str, Any]:
    """Merge the generated portable config with its original Module 3 payload."""

    flattened = dict(config)
    nested = config.get("model_config")
    if isinstance(nested, dict):
        for key, value in nested.items():
            if value is not None or key not in flattened:
                flattened[key] = value
    return flattened


def select_candidate(
    project_dir: str | Path,
    configs: list[dict[str, Any]],
    *,
    target_metric: str = "log_loss",
    probe_epochs: int = 2,
    max_candidates: int = 3,
) -> dict[str, Any]:
    """Probe retrieved candidates on the same split and select by validation metric."""

    project = Path(project_dir).expanduser().resolve()
    if not (project / "run.py").is_file():
        raise FileNotFoundError(f"Generated run.py not found: {project / 'run.py'}")
    if not configs:
        raise ValueError("At least one candidate config is required.")

    probe_dir = project / ".jiaozi_candidates"
    probe_dir.mkdir(parents=True, exist_ok=True)
    minimize = target_metric.lower() in {"log_loss", "multiclass_log_loss", "rmse"}
    trials: list[dict[str, Any]] = []

    for index, raw_config in enumerate(configs[: max(1, int(max_candidates))]):
        original = flatten_candidate_config(raw_config)
        controlled = dict(original)
        controlled["recommended_epochs"] = max(1, int(probe_epochs))
        controlled["checkpoint_dir"] = str(probe_dir / f"candidate_{index}" / "checkpoints")
        controlled["resume_checkpoint"] = ""
        config_path = probe_dir / f"candidate_{index}.json"
        config_path.write_text(
            json.dumps(controlled, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        started = time.monotonic()
        completed = subprocess.run(
            [sys.executable, "-u", "run.py", "--config", str(config_path)],
            cwd=project,
            text=True,
            capture_output=True,
        )
        runtime = round(time.monotonic() - started, 4)
        log_path = config_path.with_suffix(".log")
        log_path.write_text(
            completed.stdout + ("\n[stderr]\n" + completed.stderr if completed.stderr else ""),
            encoding="utf-8",
        )
        summary = extract_last_json(completed.stdout)
        evaluate = (summary or {}).get("evaluate") or {}
        metric_value = evaluate.get("metric_value")
        status = (
            "success"
            if completed.returncode == 0 and summary and metric_value is not None
            else "failed"
        )
        trials.append(
            {
                "candidate_index": index,
                "status": status,
                "metric_name": evaluate.get("metric_name"),
                "metric_value": metric_value,
                "accuracy": evaluate.get("accuracy"),
                "macro_f1": evaluate.get("macro_f1"),
                "runtime_sec": runtime,
                "probe_epochs": max(1, int(probe_epochs)),
                "config_path": str(config_path),
                "log_path": str(log_path),
                "config": original,
                "stderr_tail": completed.stderr[-2000:],
            }
        )

    valid = [trial for trial in trials if trial["status"] == "success"]
    if not valid:
        raise RuntimeError("All AutoPipeline candidate probes failed.")
    selected = (
        min(valid, key=lambda item: float(item["metric_value"]))
        if minimize
        else max(valid, key=lambda item: float(item["metric_value"]))
    )
    selected_index = int(selected["candidate_index"])
    return {
        "target_metric": target_metric,
        "minimize": minimize,
        "probe_epochs": max(1, int(probe_epochs)),
        "selected_index": selected_index,
        "selected_config": flatten_candidate_config(configs[selected_index]),
        "trials": trials,
    }

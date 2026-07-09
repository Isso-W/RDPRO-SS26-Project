"""Thin Kaggle run manifests, submission receipts, and memory logging.

This module is intentionally small: it records enough evidence to connect a
Jiaozi-generated Kaggle project to a later submission score without taking on
full Kaggle kernel/offload orchestration.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    commit = result.stdout.strip()
    return commit or None


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def write_run_manifest(
    output_dir: str | Path,
    *,
    benchmark_key: str,
    info: dict,
    module3_input: dict,
    recommendations: list[dict],
    module4: dict,
) -> Path:
    """Write the Jiaozi Kaggle run manifest and return its path."""

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "kaggle_run_manifest.json"
    manifest = {
        "created_at": _now(),
        "git_commit": _git_commit(),
        "benchmark_key": benchmark_key,
        "competition": info.get("competition"),
        "metric": info.get("metric"),
        "data": {
            "train_csv": info.get("train_csv"),
            "image_dir": info.get("image_dir"),
            "image_column": info.get("image_column"),
            "label_column": info.get("label_column"),
            "label_columns": info.get("label_columns", []),
            "sample_submission": info.get("sample_submission"),
            "test_dir": info.get("test_dir"),
        },
        "module3_input": module3_input,
        "recommendations": recommendations,
        "module4": module4,
    }
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=_jsonable), encoding="utf-8")
    return path


def write_submission_receipt(
    receipt_path: str | Path,
    *,
    benchmark_key: str,
    competition: str,
    submission_csv: str | Path,
    submitted: bool,
    message: str | None = None,
    status: str | None = None,
    public_score: float | str | None = None,
    details: dict | None = None,
) -> Path:
    """Write a submission receipt, even when the CSV was not submitted."""

    path = Path(receipt_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = {
        "created_at": _now(),
        "git_commit": _git_commit(),
        "benchmark_key": benchmark_key,
        "competition": competition,
        "submission_csv": str(submission_csv),
        "submitted": bool(submitted),
        "message": message,
        "status": status or ("submitted" if submitted else "not_submitted"),
        "public_score": public_score,
        "details": details or {},
    }
    path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, default=_jsonable), encoding="utf-8")
    return path


def _read_json(path: str | Path | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def _score_as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _project_config(project_dir: str | Path | None) -> dict:
    if not project_dir:
        return {}
    path = Path(project_dir) / "configs.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = data[0] if isinstance(data, list) and data else {}
    if not isinstance(cfg, dict):
        return {}
    flat = dict(cfg)
    model_config = cfg.get("model_config")
    if isinstance(model_config, dict):
        for key, value in model_config.items():
            if value is not None or key not in flat:
                flat[key] = value
    return flat


def _fingerprint_from_manifest(manifest: dict) -> dict:
    m3_input = manifest.get("module3_input") or {}
    constraints = m3_input.get("constraints") or {}
    return {
        "task_type": m3_input.get("task_type", "classification"),
        "num_classes": int(m3_input.get("num_classes", 0) or 0),
        "data_size": m3_input.get("data_size", "medium"),
        "total_images": int((manifest.get("data") or {}).get("total_images", 0) or 0),
        "class_imbalance": bool(constraints.get("class_imbalance", False)),
        "resolution_tier": (m3_input.get("data_stats") or {}).get("resolution_tier", "medium"),
        "color_mode": (m3_input.get("data_stats") or {}).get("color_mode", "rgb"),
    }


def log_kaggle_outcome_if_scored(
    receipt_path: str | Path,
    *,
    run_manifest_path: str | Path | None = None,
    project_dir: str | Path | None = None,
    memory_path: str | Path | None = None,
) -> dict:
    """Append a scored Kaggle submission to OutcomeMemory when a score exists.

    Returns a small status dict. Missing scores are not errors because Kaggle
    often processes submissions asynchronously.
    """

    receipt = _read_json(receipt_path)
    score = _score_as_float(receipt.get("public_score"))
    if score is None:
        return {"logged": False, "reason": "missing_public_score"}

    manifest = _read_json(run_manifest_path)
    config = _project_config(project_dir)
    if not config and manifest.get("recommendations"):
        config = dict(manifest["recommendations"][0])

    result = {
        "metric_name": manifest.get("metric") or "public_score",
        "metric_value": score,
        "status": receipt.get("status"),
        "submission_csv": receipt.get("submission_csv"),
        "receipt_path": str(receipt_path),
    }

    from recommender import OutcomeMemory

    memory = OutcomeMemory(memory_path) if memory_path else OutcomeMemory()
    memory.log(
        _fingerprint_from_manifest(manifest),
        config,
        result,
        dataset_id=receipt.get("competition") or manifest.get("competition"),
        cost={"kaggle_submission": 1},
    )
    return {"logged": True, "memory_path": str(memory.path), "metric_value": score}

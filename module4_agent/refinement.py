"""Proxy evaluation and targeted refinement helpers."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

from .ablation import training_spec_summary
from .executor import subprocess_env
from .schemas import ExperimentResult, TrainingSpec


# proxy_ 前缀强调这是本地启发式代理分数，不是真实 benchmark 指标
METRIC_BY_TASK = {
    "classification": "proxy_accuracy",
    "object_detection": "proxy_mAP@0.5",
    "image_segmentation": "proxy_mIoU",
    "feature_extraction": "proxy_recall@1",
}


def proxy_evaluate(
    spec: TrainingSpec,
    *,
    experiment_id: str,
    stage: str,
    parent_id: str | None = None,
    modified_component: str | None = None,
    seed: int = 123,
    notes: str | None = None,
    smoke_project_dir: str | Path | None = None,
) -> ExperimentResult:
    """Evaluate a TrainingSpec with a local proxy metric."""

    metric_name = METRIC_BY_TASK.get(spec.task_type, "proxy_accuracy")
    config_summary = training_spec_summary(spec)
    smoke_loss, smoke_note = _smoke_loss_signal(spec, smoke_project_dir=smoke_project_dir, seed=seed)
    score = _proxy_score(spec, seed=seed, smoke_loss=smoke_loss)
    result_notes = notes or "Local proxy metric; not a benchmark score."
    if smoke_note:
        result_notes = f"{result_notes} {smoke_note}"
    return ExperimentResult(
        experiment_id=experiment_id,
        spec_id=f"rank{spec.rank}_{spec.task_type}_{spec.backbone}",
        parent_id=parent_id,
        stage=stage,
        modified_component=modified_component,
        metric_name=metric_name,
        metric_value=score,
        status="success",
        config_summary=config_summary,
        notes=result_notes,
    )


def apply_targeted_refinement(spec: TrainingSpec, selected_component: str | None) -> TrainingSpec:
    """Apply a small refinement to the selected component."""

    if selected_component == "learning_rate":
        tuned_lr = _tune_learning_rate(spec.learning_rate)
        return replace(spec, learning_rate=tuned_lr)
    if selected_component == "augmentation" and spec.augmentation == "stronger":
        return replace(spec, augmentation="stronger_v2")
    if selected_component == "finetune_strategy" and spec.finetune_strategy == "full":
        return replace(spec, finetune_strategy="full", freeze_backbone=False)
    return spec


def select_best_result(results: list[ExperimentResult]) -> ExperimentResult:
    """Select the highest successful proxy result."""

    successful = [result for result in results if result.status == "success"]
    if not successful:
        raise ValueError("No successful experiment results to select from.")
    return max(successful, key=lambda result: result.metric_value)


def _proxy_score(spec: TrainingSpec, *, seed: int, smoke_loss: float | None = None) -> float:
    score = {
        "classification": 0.54,
        "object_detection": 0.40,
        "image_segmentation": 0.43,
        "feature_extraction": 0.48,
    }.get(spec.task_type, 0.50)

    optimizer = spec.optimizer.lower()
    loss = spec.loss.lower()
    strategy = spec.finetune_strategy.lower()
    augmentation = str(spec.augmentation or "").lower()
    data_size = spec.data_size.lower()

    if spec.task_type == "classification":
        score += 0.045 if optimizer == "adamw" else 0.015 if "sgd" in optimizer else 0.025
        if "focal" in loss:
            score += 0.055 if spec.class_imbalance else 0.015
        elif "cross" in loss:
            score += 0.030
        if augmentation.startswith("stronger"):
            score += 0.020
    elif spec.task_type == "object_detection":
        score += 0.060 if "sgd" in optimizer else 0.025 if optimizer == "adamw" else 0.015
        score += 0.050 if "focal" in loss else 0.020
    elif spec.task_type == "image_segmentation":
        score += 0.045 if optimizer in {"adam", "adamw"} else 0.020
        score += 0.065 if "dice" in loss else 0.025 if "cross" in loss else 0.010
        if augmentation.startswith("stronger"):
            score += 0.018
    elif spec.task_type == "feature_extraction":
        score += 0.040 if optimizer == "adamw" else 0.020
        score += 0.045 if any(token in loss for token in ("contrastive", "mse")) else 0.015

    if data_size in {"small", "tiny", "low"}:
        score += 0.050 if strategy == "head_only" else -0.020
    elif data_size in {"large", "big", "high"}:
        score += 0.050 if strategy == "full" else 0.005
    else:
        score += 0.030 if strategy in {"head_only", "full"} else 0.010

    if spec.pretrained_hf_id or spec.checkpoint or not spec.scratch_viable:
        score += 0.030
    if not spec.backbone or spec.backbone == "tiny_cnn":
        score -= 0.025
    if not spec.loss:
        score -= 0.020
    if not spec.optimizer:
        score -= 0.020

    if smoke_loss is not None:
        # Lower synthetic smoke loss gets a small bonus.
        score += max(-0.020, min(0.030, (2.0 - smoke_loss) * 0.010))

    score += _stable_jitter(training_spec_summary(spec), seed)
    return round(max(0.0, min(0.99, score)), 6)


def _smoke_loss_signal(
    spec: TrainingSpec,
    *,
    smoke_project_dir: str | Path | None,
    seed: int,
) -> tuple[float | None, str | None]:
    if smoke_project_dir is None:
        return None, None
    project_dir = Path(smoke_project_dir)
    if not (project_dir / "train.py").exists():
        return None, None

    code = """
import json
import random
import sys

import torch

from train import train_one

payload = json.load(sys.stdin)
seed = int(payload.pop("_seed", 123))
random.seed(seed)
torch.manual_seed(seed)
result = train_one(payload, epochs=1, max_steps=1)
print(json.dumps({"status": result.get("status"), "loss": result.get("loss")}))
"""
    payload = spec.to_config()
    payload["_seed"] = seed
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(project_dir),
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=30,
            env=subprocess_env(),
            check=False,
        )
    except subprocess.TimeoutExpired:
        return None, "Smoke-loss signal timed out."
    if completed.returncode != 0:
        return None, "Smoke-loss signal unavailable for this variant."
    try:
        result = json.loads(completed.stdout)
        if result.get("status") != "success":
            return None, "Smoke-loss signal returned a non-success status."
        loss = float(result.get("loss"))
    except (json.JSONDecodeError, TypeError, ValueError):
        return None, "Smoke-loss signal could not be parsed."
    return loss, f"Smoke-loss signal={loss:.6f}."


def _tune_learning_rate(current: float) -> float:
    if current > 5.0e-4:
        return 3.0e-4
    if current < 2.0e-4:
        return 3.0e-4
    return 2.0e-4


def _stable_jitter(config_summary: dict[str, Any], seed: int) -> float:
    payload = json.dumps({"config": config_summary, "seed": seed}, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 1000) / 100000.0

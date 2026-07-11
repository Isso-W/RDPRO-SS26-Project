"""Tests for the persisted MLE-STAR run contracts."""

from __future__ import annotations

from dataclasses import replace
import subprocess
import sys

import pytest

from mlestar.contracts import (
    COMPONENT_NAMES,
    AuditFinding,
    CandidateProject,
    Component,
    DatasetInventory,
    EnsembleReceipt,
    ExperimentReceipt,
    MetricSpec,
    SearchEvidence,
    TaskContract,
)


def _task() -> TaskContract:
    return TaskContract(
        task_id="tiny_binary",
        modality="image_classification",
        target_columns=["target"],
        id_column="image_id",
        metric=MetricSpec(name="roc_auc", greater_is_better=True),
        components=[Component(name=name) for name in COMPONENT_NAMES],
    )


def test_task_contract_round_trip_and_component_set() -> None:
    task = _task()

    assert TaskContract.from_dict(task.to_dict()) == task
    assert task.component_names == COMPONENT_NAMES


def test_task_contract_rejects_missing_or_duplicate_component_markers() -> None:
    with pytest.raises(ValueError, match="exactly"):
        TaskContract(
            task_id="x",
            modality="tabular",
            target_columns=["y"],
            id_column="id",
            metric=MetricSpec("accuracy", True),
            components=[Component("model")],
        )

    with pytest.raises(ValueError, match="exactly"):
        TaskContract(
            task_id="x",
            modality="tabular",
            target_columns=["y"],
            id_column="id",
            metric=MetricSpec("accuracy", True),
            components=[Component(name) for name in COMPONENT_NAMES[:-1]]
            + [Component("model")],
        )


@pytest.mark.parametrize(
    ("contract", "factory"),
    [
        (
            DatasetInventory(
                data_root="data",
                fingerprint="abc123",
                files=[{"path": "train.csv", "size_bytes": 12}],
                created_at="2025-01-02T03:04:05+00:00",
            ),
            DatasetInventory.from_dict,
        ),
        (
            SearchEvidence(
                title="Useful model paper",
                url="https://example.test/paper",
                summary="A concise, task-specific result.",
                license_note="CC BY 4.0",
                retrieved_at="2025-01-02T03:04:05+00:00",
            ),
            SearchEvidence.from_dict,
        ),
        (
            CandidateProject(
                candidate_id="baseline",
                project_dir="projects/baseline",
                code_sha256="a" * 64,
                components=[Component(name) for name in COMPONENT_NAMES],
                created_at="2025-01-02T03:04:05+00:00",
            ),
            CandidateProject.from_dict,
        ),
        (
            ExperimentReceipt(
                experiment_id="experiment-1",
                candidate_id="baseline",
                component="model",
                stage="initial",
                metric_name="roc_auc",
                greater_is_better=True,
                metric_value=0.9,
                elapsed_seconds=1.2,
                status="success",
                code_sha256="a" * 64,
                data_fingerprint="abc123",
                oof_path="oof/baseline.parquet",
                created_at="2025-01-02T03:04:05+00:00",
            ),
            ExperimentReceipt.from_dict,
        ),
        (
            AuditFinding(
                audit_name="leakage",
                status="passed",
                message="No leakage found.",
                created_at="2025-01-02T03:04:05+00:00",
            ),
            AuditFinding.from_dict,
        ),
        (
            EnsembleReceipt(
                ensemble_id="blend-1",
                candidate_weights={"baseline": 1.0},
                metric_name="roc_auc",
                metric_value=0.91,
                oof_path="ensemble/oof.parquet",
                created_at="2025-01-02T03:04:05+00:00",
            ),
            EnsembleReceipt.from_dict,
        ),
    ],
)
def test_contracts_round_trip(contract: object, factory: object) -> None:
    to_dict = getattr(contract, "to_dict")
    assert factory(to_dict()) == contract


def test_receipt_success_requires_status_metric_and_oof_path() -> None:
    receipt = ExperimentReceipt(
        experiment_id="experiment-1",
        candidate_id="baseline",
        component="model",
        stage="initial",
        metric_name="roc_auc",
        greater_is_better=True,
        metric_value=0.9,
        elapsed_seconds=1.2,
        status="success",
        code_sha256="a" * 64,
        data_fingerprint="abc123",
        oof_path="oof/baseline.parquet",
    )
    assert receipt.success is True
    assert ExperimentReceipt.from_dict({**receipt.to_dict(), "oof_path": None}).success is False
    assert ExperimentReceipt.from_dict({**receipt.to_dict(), "metric_value": None}).success is False
    assert ExperimentReceipt.from_dict({**receipt.to_dict(), "status": "failed"}).success is False


@pytest.mark.parametrize("value", [1, 0, "true", None])
def test_metric_spec_rejects_non_boolean_direction(value: object) -> None:
    with pytest.raises(ValueError, match="greater_is_better"):
        MetricSpec(name="roc_auc", greater_is_better=value)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="greater_is_better"):
        MetricSpec.from_dict({"name": "roc_auc", "greater_is_better": value})


@pytest.mark.parametrize("value", [1, 0, "false", None])
def test_experiment_receipt_rejects_non_boolean_direction(value: object) -> None:
    receipt = ExperimentReceipt(
        experiment_id="experiment-1",
        candidate_id="baseline",
        component="model",
        stage="initial",
        metric_name="roc_auc",
        greater_is_better=True,
        metric_value=0.9,
        elapsed_seconds=1.2,
        status="success",
        code_sha256="a" * 64,
        data_fingerprint="abc123",
        oof_path="oof/baseline.parquet",
    )

    with pytest.raises(ValueError, match="greater_is_better"):
        replace(receipt, greater_is_better=value)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="greater_is_better"):
        ExperimentReceipt.from_dict({**receipt.to_dict(), "greater_is_better": value})


@pytest.mark.parametrize("digest", ["a" * 63, "g" * 64, "a" * 65])
def test_candidate_and_experiment_require_sha256_hex_digests(digest: str) -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        CandidateProject(
            candidate_id="baseline",
            project_dir="projects/baseline",
            code_sha256=digest,
            components=[Component(name) for name in COMPONENT_NAMES],
        )

    with pytest.raises(ValueError, match="SHA-256"):
        ExperimentReceipt(
            experiment_id="experiment-1",
            candidate_id="baseline",
            component="model",
            stage="initial",
            metric_name="roc_auc",
            greater_is_better=True,
            metric_value=0.9,
            elapsed_seconds=1.2,
            status="success",
            code_sha256=digest,
            data_fingerprint="abc123",
            oof_path="oof/baseline.parquet",
        )


@pytest.mark.parametrize(
    "field",
    [
        "experiment_id",
        "candidate_id",
        "component",
        "stage",
        "metric_name",
        "status",
        "code_sha256",
        "data_fingerprint",
    ],
)
def test_experiment_receipt_requires_core_fields(field: str) -> None:
    receipt = ExperimentReceipt(
        experiment_id="experiment-1",
        candidate_id="baseline",
        component="model",
        stage="initial",
        metric_name="roc_auc",
        greater_is_better=True,
        metric_value=0.9,
        elapsed_seconds=1.2,
        status="success",
        code_sha256="a" * 64,
        data_fingerprint="abc123",
        oof_path="oof/baseline.parquet",
    )

    with pytest.raises(ValueError, match="required"):
        ExperimentReceipt.from_dict({**receipt.to_dict(), field: ""})


def test_experiment_receipt_rejects_unknown_component() -> None:
    with pytest.raises(ValueError, match="Unsupported component"):
        ExperimentReceipt(
            experiment_id="experiment-1",
            candidate_id="baseline",
            component="all_the_code",
            stage="initial",
            metric_name="roc_auc",
            greater_is_better=True,
            metric_value=0.9,
            elapsed_seconds=1.2,
            status="success",
            code_sha256="a" * 64,
            data_fingerprint="abc123",
            oof_path="oof/baseline.parquet",
        )


def test_module_cli_exposes_a_runnable_help_command() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "mlestar", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "mlestar" in completed.stdout

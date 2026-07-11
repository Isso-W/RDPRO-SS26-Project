"""Tests for leakage and data-usage gates."""

from __future__ import annotations

from mlestar.contracts import COMPONENT_NAMES, Component, MetricSpec, TaskContract
from mlestar.audits import audit_project, has_audit_errors


def _task() -> TaskContract:
    return TaskContract(
        task_id="tiny", modality="tabular", target_columns=["target"], id_column="id",
        metric=MetricSpec("roc_auc", True), components=[Component(name) for name in COMPONENT_NAMES],
    )


def _project(tmp_path, source: str):
    path = tmp_path / "project"
    path.mkdir()
    (path / "pipeline.py").write_text(source, encoding="utf-8")
    return path


def test_audit_flags_test_statistics_used_for_imputation(tmp_path) -> None:
    project = _project(tmp_path, "import pandas as pd\nall_rows = pd.concat([train, test])\nmean = all_rows.x.mean()\n")

    findings = audit_project(project, _task(), {"files": [{"path": "train.csv"}, {"path": "test.csv"}]})

    assert any(item.code == "test_statistics" and item.severity == "error" for item in findings)
    assert has_audit_errors(findings)


def test_audit_flags_unreferenced_required_mask_directory_and_writes_jsonl(tmp_path) -> None:
    project = _project(tmp_path, "import pandas as pd\ntrain = pd.read_csv('train.csv')\nseed = 7\n")
    output = tmp_path / "audit.jsonl"

    findings = audit_project(project, _task(), {"files": [{"path": "train.csv"}, {"path": "masks/001.png"}]}, output_path=output)

    assert any(item.code == "unused_data_source" for item in findings)
    assert output.read_text(encoding="utf-8").count("\n") == len(findings)

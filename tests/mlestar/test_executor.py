"""Tests for isolated generated-project execution and receipt persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mlestar.contracts import COMPONENT_NAMES, Component, MetricSpec, TaskContract
from mlestar.executor import execute_project


@pytest.fixture()
def tiny_contract() -> TaskContract:
    return TaskContract(
        task_id="tiny",
        modality="tabular",
        target_columns=["target"],
        id_column="row_id",
        metric=MetricSpec("accuracy", True),
        components=[Component(name) for name in COMPONENT_NAMES],
    )


def _write_project(root: Path, terminal: dict[str, object]) -> Path:
    project = root / "project"
    project.mkdir()
    project_file = """\
import json
import os
from pathlib import Path

class _Skb:
    def eval(self):
        print('generated progress')
        Path('observed_environment.json').write_text(json.dumps({
            'openai': os.environ.get('OPENAI_API_KEY'),
            'kaggle': os.environ.get('KAGGLE_KEY'),
            'kaggle_api_token': os.environ.get('KAGGLE_API_TOKEN'),
            'python_no_user_site': os.environ.get('PYTHONNOUSERSITE'),
        }))
        return __TERMINAL__

class _Plan:
    skb = _Skb()

def build_dataops_plan(config_path, mode):
    assert Path(config_path).parent.resolve() == Path.cwd().resolve()
    assert json.loads(Path(config_path).read_text())['task_id'] == 'tiny'
    assert mode == 'validate'
    return _Plan()
""".replace("__TERMINAL__", repr(terminal))
    (project / "project.py").write_text(project_file, encoding="utf-8")
    return project


def _terminal(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "metric_name": "accuracy",
        "metric_value": 0.875,
        "oof_path": "oof.parquet",
        "prediction_path": None,
        "submission_path": None,
        "component_trace": list(COMPONENT_NAMES),
    }
    value.update(overrides)
    return value


def test_executor_parses_real_metric_sanitizes_environment_and_persists_artifacts(tmp_path, tiny_contract, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-project")
    monkeypatch.setenv("KAGGLE_KEY", "must-not-reach-project")
    monkeypatch.setenv("KAGGLE_API_TOKEN", "must-not-reach-project")
    project = _write_project(tmp_path, _terminal())

    receipt = execute_project(project, tiny_contract, workspace_root=tmp_path, timeout_seconds=10)

    assert receipt.status == "success"
    assert receipt.metric_value == 0.875
    assert receipt.oof_path and receipt.oof_path.endswith("oof.parquet")
    assert receipt.success
    observed = json.loads((project / "observed_environment.json").read_text())
    assert observed == {
        "openai": None,
        "kaggle": None,
        "kaggle_api_token": None,
        "python_no_user_site": "1",
    }
    assert json.loads((project / "executor_stdout.txt").read_text())["metric_value"] == 0.875
    assert "generated progress" in (project / "executor_stderr.txt").read_text()
    assert json.loads((project / "execution_receipt.json").read_text())["status"] == "success"
    assert not list(project.glob(".mlestar-task-contract-*.json"))


def test_executor_rejects_wrong_terminal_schema_as_a_failed_receipt(tmp_path, tiny_contract) -> None:
    project = _write_project(tmp_path, _terminal(submission_path="missing-key"))
    source = (project / "project.py").read_text(encoding="utf-8")
    (project / "project.py").write_text(source.replace("'submission_path': 'missing-key', ", ""), encoding="utf-8")

    receipt = execute_project(project, tiny_contract, workspace_root=tmp_path, timeout_seconds=10)

    assert receipt.status == "failed"
    assert receipt.metric_value is None
    assert receipt.error_text and "terminal" in receipt.error_text.lower()
    assert json.loads((project / "execution_receipt.json").read_text())["status"] == "failed"


def test_executor_rejects_artifact_paths_outside_workspace(tmp_path, tiny_contract) -> None:
    project = _write_project(tmp_path, _terminal(oof_path="../../outside.parquet"))

    receipt = execute_project(project, tiny_contract, workspace_root=tmp_path, timeout_seconds=10)

    assert receipt.status == "failed"
    assert receipt.error_text and "workspace" in receipt.error_text


def test_executor_rejects_a_project_outside_its_workspace(tmp_path, tiny_contract) -> None:
    project = _write_project(tmp_path, _terminal())
    separate_workspace = tmp_path / "workspace"
    separate_workspace.mkdir()

    with pytest.raises(ValueError, match="inside workspace"):
        execute_project(project, tiny_contract, workspace_root=separate_workspace, timeout_seconds=10)


def test_executor_runs_the_generated_pipeline_dataops_protocol(tmp_path, tiny_contract) -> None:
    project = tmp_path / "generated"
    project.mkdir()
    (project / "pipeline.py").write_text(
        """from pathlib import Path

def load_data(state):
    return dict(state)

def prepare_data(state):
    return dict(state)

def build_model(state):
    return dict(state)

def train_model(state):
    return dict(state)

def predict_or_submit(state):
    root = Path(state['run_context']['workspace_root'])
    (root / 'oof.parquet').write_text('safe placeholder')
    return {**state, 'metric_name': 'accuracy', 'metric_value': 0.75,
            'oof_path': 'oof.parquet', 'prediction_path': None, 'submission_path': None}
""",
        encoding="utf-8",
    )

    receipt = execute_project(project, tiny_contract, workspace_root=tmp_path, timeout_seconds=10)

    assert receipt.status == "success"
    assert receipt.metric_value == 0.75

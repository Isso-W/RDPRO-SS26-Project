"""Tests for the generated-project skrub DataOps execution boundary."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("skrub")

from mlestar.contracts import COMPONENT_NAMES, Component, MetricSpec, TaskContract
from mlestar.dataops import ProjectProtocolError, build_dataops_plan, run_dataops_project


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


def _write_project(path: Path, *, mutate: str = "") -> Path:
    project = path / "project"
    project.mkdir()
    functions = []
    function_names = {
        "data_loading": "load_data",
        "data_preparation": "prepare_data",
        "model": "build_model",
        "training": "train_model",
        "prediction": "predict_or_submit",
    }
    for name in COMPONENT_NAMES:
        body = mutate if name == "training" and mutate else f"return {{**state, 'last_step': '{name}'}}"
        functions.append(f"def {function_names[name]}(state):\n    {body}\n")
    (project / "pipeline.py").write_text("\n".join(functions), encoding="utf-8")
    return project


def test_plan_is_lazy_has_named_components_and_executes_once(tmp_path, tiny_contract) -> None:
    project = _write_project(tmp_path)
    sentinel = project / "imported.txt"
    pipeline = project / "pipeline.py"
    pipeline.write_text(f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('yes')\n" + pipeline.read_text(), encoding="utf-8")

    plan = build_dataops_plan(tiny_contract, project, workspace_root=tmp_path)

    assert sentinel.exists() is False
    description = plan.skb.describe_steps()
    assert all(component in description for component in COMPONENT_NAMES)
    result = run_dataops_project(tiny_contract, project, workspace_root=tmp_path)
    assert sentinel.exists()
    assert result["mode"] == "validate"
    assert result["component_trace"] == list(COMPONENT_NAMES)
    assert result["last_step"] == "prediction"
    report = json.loads((project / "dataops_report.json").read_text())
    assert report["full_report"]["status"] == "skipped_to_prevent_second_evaluation"


def test_project_cannot_spoof_framework_owned_trace_or_context(tmp_path, tiny_contract) -> None:
    project = _write_project(tmp_path, mutate="state['component_trace'].append('training')\n    return state")

    with pytest.raises(ProjectProtocolError, match="component_trace"):
        run_dataops_project(tiny_contract, project, workspace_root=tmp_path)


def test_plan_evaluates_when_the_caller_supplies_the_bound_run_context(tmp_path, tiny_contract) -> None:
    project = _write_project(tmp_path)
    pipeline = project / "pipeline.py"
    pipeline.write_text(
        pipeline.read_text().replace(
            "def prepare_data(state):\n    return {**state, 'last_step': 'data_preparation'}",
            "def prepare_data(state):\n    assert state['previous']['last_step'] == 'data_loading'\n    return {**state, 'last_step': 'data_preparation'}",
        ),
        encoding="utf-8",
    )
    plan = build_dataops_plan(tiny_contract, project, workspace_root=tmp_path)
    state = {
        "run_context": {
            "contract": tiny_contract.to_dict(),
            "project_dir": str(project.resolve()),
            "workspace_root": str(tmp_path.resolve()),
        },
        "mode": "validate",
        "component_trace": [],
        "artifacts": {},
        "previous": None,
    }

    result = plan.skb.eval({"run_context": state})

    assert result["component_trace"] == list(COMPONENT_NAMES)


def test_project_must_be_inside_workspace_and_have_component_protocol(tmp_path, tiny_contract) -> None:
    project = _write_project(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError, match="inside workspace"):
        build_dataops_plan(tiny_contract, project, workspace_root=outside)

    (project / "pipeline.py").write_text("def load_data(state):\n    return state\n", encoding="utf-8")
    with pytest.raises(ProjectProtocolError, match="prepare_data"):
        run_dataops_project(tiny_contract, project, workspace_root=tmp_path)

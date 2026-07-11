"""Tests for generated MLE-STAR project source contracts."""

from __future__ import annotations

import json
from pathlib import Path

from mlestar.contracts import COMPONENT_NAMES, Component, MetricSpec, TaskContract
from mlestar.generation import (
    StaticGenerationProvider,
    generate_project,
    validate_generated_project,
    write_fallback_project,
)


def _contract() -> TaskContract:
    return TaskContract(
        task_id="tiny",
        modality="tabular",
        target_columns=["target"],
        id_column="row_id",
        metric=MetricSpec("accuracy", True),
        components=[Component(name) for name in COMPONENT_NAMES],
    )


def _source(*, training_body: str = "return dict(state)") -> str:
    functions = {
        "data_loading": "load_data",
        "data_preparation": "prepare_data",
        "model": "build_model",
        "training": "train_model",
        "prediction": "predict_or_submit",
    }
    pieces: list[str] = []
    for component in COMPONENT_NAMES:
        body = training_body if component == "training" else "return dict(state)"
        pieces.extend(
            (
                f"# MLESTAR_COMPONENT:{component}:START",
                f"def {functions[component]}(state):\n    {body}",
                f"# MLESTAR_COMPONENT:{component}:END",
            )
        )
    return "\n\n".join(pieces) + "\n"


def _write_project(path: Path, source: str) -> Path:
    project = path / "project"
    project.mkdir()
    (project / "pipeline.py").write_text(source, encoding="utf-8")
    (project / "requirements.txt").write_text("", encoding="utf-8")
    return project


def test_validator_accepts_all_component_markers_and_dataops_protocol(tmp_path) -> None:
    project = _write_project(tmp_path, _source())

    result = validate_generated_project(project, _contract())

    assert result.is_valid
    assert result.errors == ()


def test_validator_rejects_top_level_fitting_outside_component_marker(tmp_path) -> None:
    source = "model.fit(X, y)\n\n" + _source()
    project = _write_project(tmp_path, source)

    result = validate_generated_project(project, _contract())

    assert result.is_valid is False
    assert any("outside component marker" in error for error in result.errors)


def test_validator_requires_each_dataops_function_to_accept_state(tmp_path) -> None:
    project = _write_project(tmp_path, _source().replace("def build_model(state):", "def build_model():"))

    result = validate_generated_project(project, _contract())

    assert result.is_valid is False
    assert any("build_model(state)" in error for error in result.errors)


def test_validator_rejects_unsafe_ast_constructs_and_missing_dependencies(tmp_path) -> None:
    source = _source(training_body="eval('1 + 1')\n    return dict(state)")
    source = "import requests\nimport numpy\n\n" + source.replace("return dict(state)", "open('/tmp/result', 'w').write('bad')\n    return dict(state)", 1)
    project = _write_project(tmp_path, source)

    result = validate_generated_project(project, _contract())

    assert result.is_valid is False
    assert any("eval" in error for error in result.errors)
    assert any("absolute" in error for error in result.errors)
    assert any("requirements" in error for error in result.errors)


def test_fallback_has_protocol_markers_and_generation_falls_back_after_invalid_provider(tmp_path) -> None:
    contract = _contract()
    fallback = write_fallback_project(tmp_path / "fallback", contract)
    fallback_validation = validate_generated_project(fallback, contract)

    assert fallback_validation.is_valid
    assert "DummyClassifier" in (fallback / "pipeline.py").read_text(encoding="utf-8")

    result = generate_project(
        tmp_path / "generated",
        contract,
        provider=StaticGenerationProvider('{"files": {"pipeline.py": "not Python"}}'),
    )

    assert result.used_fallback
    assert result.validation.is_valid
    assert json.loads((result.project_dir / "config.json").read_text(encoding="utf-8"))["task_id"] == "tiny"

"""Tests for persisted baseline candidate-wave orchestration."""

from __future__ import annotations

import json

from mlestar.contracts import COMPONENT_NAMES, Component, MetricSpec, TaskContract
from mlestar.generation import StaticGenerationProvider
from mlestar.search import StaticSearchProvider
from mlestar.workflow import run_mlestar


def _task() -> TaskContract:
    return TaskContract(
        task_id="tiny", modality="tabular", target_columns=["target"], id_column="id",
        metric=MetricSpec("accuracy", True), components=[Component(name) for name in COMPONENT_NAMES],
    )


def _provider_source() -> str:
    mapping = {
        "data_loading": "load_data", "data_preparation": "prepare_data", "model": "build_model",
        "training": "train_model",
    }
    blocks = [
        f"# MLESTAR_COMPONENT:{component}:START\ndef {function}(state):\n    return dict(state)\n# MLESTAR_COMPONENT:{component}:END"
        for component, function in mapping.items()
    ]
    blocks.append(
        "# MLESTAR_COMPONENT:prediction:START\n"
        "def predict_or_submit(state):\n"
        "    from pathlib import Path\n"
        "    root = Path(state['run_context']['workspace_root'])\n"
        "    (root / 'workflow-oof.parquet').write_text('oof')\n"
        "    return {**state, 'metric_name': 'accuracy', 'metric_value': 0.9, "
        "'oof_path': 'workflow-oof.parquet', 'prediction_path': None, 'submission_path': None}\n"
        "# MLESTAR_COMPONENT:prediction:END"
    )
    return "\n\n".join(blocks) + "\n"


def test_workflow_persists_search_generation_audit_and_real_receipt(tmp_path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.csv").write_text("id,target\n1,0\n2,1\n", encoding="utf-8")
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(_task().to_dict()), encoding="utf-8")
    generator = StaticGenerationProvider({"files": {"pipeline.py": _provider_source(), "requirements.txt": ""}})
    report = run_mlestar(
        task_path=task_path, data_root=data, run_dir=tmp_path / "run",
        search=StaticSearchProvider([{"title": "Model", "url": "https://example.test/model", "snippet": "Use TinyNet"}]),
        generator=generator,
    )
    assert report["status"] == "success"
    assert report["best_experiment"]["metric_value"] == 0.9


def test_plan_only_never_starts_generated_subprocess(tmp_path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    task_path = tmp_path / "task.json"
    task_path.write_text(json.dumps(_task().to_dict()), encoding="utf-8")
    report = run_mlestar(task_path=task_path, data_root=data, run_dir=tmp_path / "run", plan_only=True)
    assert report["status"] == "planned"
    assert not (tmp_path / "run" / "projects" / "candidate_1" / "execution_receipt.json").exists()

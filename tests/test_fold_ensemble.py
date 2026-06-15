import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from autopipeline.fold_ensemble import train_selected_folds


def test_selected_config_folds_use_fixed_runner_and_preserve_hyperparameters(
    tmp_path, monkeypatch
):
    project = tmp_path / "project"
    project.mkdir()
    (project / "run.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    selected = project / "selected.json"
    selected.write_text(
        json.dumps(
            {
                "backbone": "dinov2_vits14",
                "learning_rate": 0.0003,
                "image_size": 336,
                "recommended_epochs": 20,
                "checkpoint_dir": "old",
            }
        ),
        encoding="utf-8",
    )
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        config = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        fold_index = config["fold_index"]
        artifact = Path(config["checkpoint_dir"]) / "validation_probabilities.npz"
        summary = {
            "status": "success",
            "train": {
                "best_epoch": 7 + fold_index,
                "runtime_sec": 10.0 + fold_index,
                "validation_history": [{"epoch": 1}],
            },
            "evaluate": {
                "metric_name": "log_loss",
                "metric_value": 0.10 + (0.01 * fold_index),
                "accuracy": 0.9,
                "macro_f1": 0.88,
                "validation_artifact": str(artifact),
            },
        }
        return SimpleNamespace(returncode=0, stdout=json.dumps(summary), stderr="")

    monkeypatch.setattr("autopipeline.fold_ensemble.subprocess.run", fake_run)
    result = train_selected_folds(project, selected, fold_count=3)

    assert result["status"] == "success"
    assert len(result["members"]) == 3
    assert sum(member["weight"] for member in result["members"]) == pytest.approx(1.0)
    for fold_index, (command, kwargs) in enumerate(commands):
        assert command[:3] == [command[0], "-u", "run.py"]
        assert command[3] == "--config"
        assert kwargs["cwd"] == project.resolve()
        config = json.loads(Path(command[-1]).read_text(encoding="utf-8"))
        assert config["fold_count"] == 3
        assert config["fold_index"] == fold_index
        assert config["backbone"] == "dinov2_vits14"
        assert config["learning_rate"] == pytest.approx(0.0003)
        assert config["image_size"] == 336
        assert config["recommended_epochs"] == 20


def test_selected_config_must_stay_inside_generated_project(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    (project / "run.py").write_text("", encoding="utf-8")
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="inside the generated project"):
        train_selected_folds(project, outside)

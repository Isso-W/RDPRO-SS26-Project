import json
from types import SimpleNamespace

from dog_breed_workflow import (
    _selected_experiment,
    _submission_plan,
    flatten_config,
    train_baseline,
)


def test_flatten_config_preserves_runtime_fields():
    config = {
        "backbone": "resnet50",
        "model_config": {"recommended_epochs": 12, "evaluation_metric": "log_loss"},
    }
    result = flatten_config(config)
    assert result["backbone"] == "resnet50"
    assert result["recommended_epochs"] == 12


def test_selected_experiment_falls_back_to_baseline(tmp_path):
    baseline = tmp_path / "baseline.json"
    selected, path = _selected_experiment(
        {"comparison": {"best_experiment": "missing"}, "runs": []},
        baseline,
    )
    assert selected == "baseline"
    assert path == baseline


def test_train_baseline_uses_calibrated_candidate_config(tmp_path, monkeypatch):
    (tmp_path / "run.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    summary = {
        "status": "success",
        "train": {"best_epoch": 3, "runtime_sec": 4.5},
        "evaluate": {
            "metric_name": "log_loss",
            "metric_value": 0.42,
            "accuracy": 0.9,
            "macro_f1": 0.88,
        },
    }

    def fake_run(command, **kwargs):
        config = json.loads(open(command[-1], encoding="utf-8").read())
        assert config["backbone"] == "selected"
        assert config["recommended_epochs"] == 30
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(summary),
            stderr="",
        )

    monkeypatch.setattr("dog_breed_workflow.subprocess.run", fake_run)
    config, metrics, path = train_baseline(
        tmp_path,
        {"backbone": "selected", "recommended_epochs": 30},
    )

    assert config["backbone"] == "selected"
    assert metrics["metric_value"] == 0.42
    assert path.name == "baseline.json"


def test_submission_plan_prefers_selected_config_fold_ensemble():
    selected, members = _submission_plan(
        "exp_3_prior",
        {
            "improved": True,
            "members": [
                {"name": "baseline", "weight": 0.5},
                {"name": "exp_3_prior", "weight": 0.5},
            ],
        },
        {
            "members": [
                {"name": "fold_0", "weight": 0.34},
                {"name": "fold_1", "weight": 0.33},
                {"name": "fold_2", "weight": 0.33},
            ]
        },
    )

    assert selected == "exp_3_prior_3fold"
    assert [member["name"] for member in members] == [
        "fold_0",
        "fold_1",
        "fold_2",
    ]


def test_submission_plan_falls_back_to_validation_ensemble():
    selected, members = _submission_plan(
        "baseline",
        {
            "improved": True,
            "members": [
                {"name": "baseline", "weight": 0.6},
                {"name": "exp_1", "weight": 0.4},
            ],
        },
        {"members": [{"name": "fold_0", "weight": 1.0}]},
    )

    assert selected == "validation_ensemble"
    assert len(members) == 2

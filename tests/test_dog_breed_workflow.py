from dog_breed_workflow import _selected_experiment, flatten_config


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

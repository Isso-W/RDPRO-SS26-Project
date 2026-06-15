import json
from types import SimpleNamespace

from autopipeline.candidate_selector import select_candidate


def _summary(metric):
    return json.dumps(
        {
            "status": "success",
            "train": {"best_epoch": 1},
            "evaluate": {
                "metric_name": "log_loss",
                "metric_value": metric,
                "accuracy": 0.8,
                "macro_f1": 0.79,
            },
        }
    )


def test_selector_probes_all_candidates_and_minimizes_log_loss(tmp_path, monkeypatch):
    (tmp_path / "run.py").write_text("raise SystemExit(0)\n", encoding="utf-8")
    metrics = {0: 0.8, 1: 0.4, 2: 0.6}
    commands = []

    def fake_run(command, **kwargs):
        commands.append((command, kwargs))
        config = json.loads(open(command[-1], encoding="utf-8").read())
        index = int(command[-1].split("candidate_")[1].split(".")[0])
        assert config["recommended_epochs"] == 2
        assert config["seed"] == 42
        return SimpleNamespace(returncode=0, stdout=_summary(metrics[index]), stderr="")

    monkeypatch.setattr("autopipeline.candidate_selector.subprocess.run", fake_run)
    configs = [
        {"backbone": "a", "seed": 42, "recommended_epochs": 20},
        {"backbone": "b", "seed": 42, "recommended_epochs": 30},
        {"backbone": "c", "seed": 42, "recommended_epochs": 40},
    ]

    result = select_candidate(tmp_path, configs, target_metric="log_loss")

    assert result["selected_index"] == 1
    assert result["selected_config"]["recommended_epochs"] == 30
    assert len(result["trials"]) == 3
    assert all(command[0][1:3] == ["-u", "run.py"] for command in commands)
    assert all(command[1]["cwd"] == tmp_path.resolve() for command in commands)


def test_selector_ignores_failed_probe_when_another_candidate_succeeds(
    tmp_path, monkeypatch
):
    (tmp_path / "run.py").write_text("raise SystemExit(0)\n", encoding="utf-8")

    def fake_run(command, **kwargs):
        index = int(command[-1].split("candidate_")[1].split(".")[0])
        if index == 0:
            return SimpleNamespace(returncode=1, stdout="", stderr="out of memory")
        return SimpleNamespace(returncode=0, stdout=_summary(0.5), stderr="")

    monkeypatch.setattr("autopipeline.candidate_selector.subprocess.run", fake_run)
    result = select_candidate(
        tmp_path,
        [{"backbone": "large"}, {"backbone": "base"}],
        probe_epochs=1,
    )

    assert result["selected_index"] == 1
    assert result["trials"][0]["status"] == "failed"
    assert result["trials"][1]["status"] == "success"


def test_selector_rejects_metric_fallback_for_lower_is_better_target(
    tmp_path, monkeypatch
):
    (tmp_path / "run.py").write_text("raise SystemExit(0)\n", encoding="utf-8")

    def fake_run(command, **kwargs):
        index = int(command[-1].split("candidate_")[1].split(".")[0])
        summary = json.loads(_summary(0.5))
        if index == 0:
            summary["evaluate"]["metric_name"] = "accuracy"
            summary["evaluate"]["metric_value"] = 0.2
        return SimpleNamespace(
            returncode=0,
            stdout=json.dumps(summary),
            stderr="",
        )

    monkeypatch.setattr("autopipeline.candidate_selector.subprocess.run", fake_run)
    result = select_candidate(
        tmp_path,
        [{"backbone": "wrong-metric"}, {"backbone": "valid-log-loss"}],
        target_metric="log_loss",
    )

    assert result["selected_index"] == 1
    assert result["trials"][0]["status"] == "failed"
    assert result["trials"][0]["metric_matches"] is False
    assert result["trials"][1]["metric_matches"] is True

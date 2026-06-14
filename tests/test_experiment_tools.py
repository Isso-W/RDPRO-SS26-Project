import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mcp_server.tools.experiment_tools import (
    compare_results_service,
    run_experiment_service,
)


def test_compare_results_minimizes_log_loss():
    result = compare_results_service(
        {"metric_value": 1.2},
        [
            {"experiment_name": "a", "metric_value": 1.1},
            {"experiment_name": "b", "metric_value": 1.3},
        ],
    )
    assert result["best_experiment"] == "a"
    assert result["metric_delta"] == pytest.approx(0.1)


def test_runner_rejects_workspace_escape(tmp_path):
    context = SimpleNamespace(workspace_root=(tmp_path / "workspace").resolve())
    context.workspace_root.mkdir()
    with pytest.raises(ValueError, match="inside"):
        run_experiment_service(context, str(tmp_path / "outside"), "x", {})


def test_runner_uses_fixed_run_py_command(tmp_path):
    root = tmp_path / "workspace"
    project = root / "project"
    project.mkdir(parents=True)
    (project / "run.py").write_text(
        "import json\nprint(json.dumps({'evaluate': {'metric_value': 0.5}}))\n",
        encoding="utf-8",
    )
    context = SimpleNamespace(workspace_root=root.resolve())
    result = run_experiment_service(context, str(project), "safe", {"recommended_epochs": 1})
    assert result["status"] == "success"
    assert json.loads(Path(result["config_path"]).read_text())["recommended_epochs"] == 1

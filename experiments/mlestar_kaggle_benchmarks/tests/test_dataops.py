from __future__ import annotations

import json

import pytest

from mlestar.dataops import build_run_graph, run_phase


def test_graph_evaluates_each_component_once_and_keeps_only_paths(tmp_path) -> None:
    graph = build_run_graph(tmp_path, ("load", "folds", "train", "evaluate"))

    result = graph.skb.eval({"run_context": {"run_dir": str(tmp_path)}})

    assert result["component_trace"] == ["load", "folds", "train", "evaluate"]
    assert all(isinstance(value, str) for value in result["artifacts"].values())
    assert all(path.startswith(str(tmp_path)) for path in result["artifacts"].values())
    json.dumps(result, allow_nan=False)


def test_run_phase_copies_metadata_and_rejects_outside_artifacts(tmp_path) -> None:
    source = {
        "run_context": {"run_dir": str(tmp_path), "task_key": "leaf"},
        "component_trace": [],
        "artifacts": {"folds": "folds.json"},
    }

    result = run_phase(source, "train", tmp_path)

    assert source["component_trace"] == []
    assert source["artifacts"] == {"folds": "folds.json"}
    assert result["component_trace"] == ["train"]
    assert result["artifacts"]["folds"] == str(tmp_path / "folds.json")

    with pytest.raises(ValueError, match="inside the run directory"):
        run_phase(
            {"run_context": {"run_dir": str(tmp_path)}, "artifacts": {"bad": "../bad"}},
            "evaluate",
            tmp_path,
        )


def test_graph_rejects_non_json_metadata_and_duplicate_phases(tmp_path) -> None:
    graph = build_run_graph(tmp_path, ("load",))
    with pytest.raises(TypeError, match="JSON metadata"):
        graph.skb.eval({"run_context": {"run_dir": str(tmp_path), "model": object()}})

    with pytest.raises(ValueError, match="unique"):
        build_run_graph(tmp_path, ("load", "load"))

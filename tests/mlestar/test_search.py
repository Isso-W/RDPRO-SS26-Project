"""Tests for normalized, citation-ready initial model evidence."""

from __future__ import annotations

import json

from mlestar.contracts import COMPONENT_NAMES, Component, MetricSpec, TaskContract
from mlestar.search import StaticSearchProvider, retrieve_model_evidence


def _task() -> TaskContract:
    return TaskContract(
        task_id="tiny",
        modality="image_classification",
        target_columns=["target"],
        id_column="id",
        metric=MetricSpec("roc_auc", True),
        components=[Component(name) for name in COMPONENT_NAMES],
        description="Detect a rare visual class.",
    )


def test_retrieval_deduplicates_urls_and_keeps_model_plus_example_code(tmp_path) -> None:
    provider = StaticSearchProvider(
        [
            {"title": "Efficient model", "url": "https://Example.test/a/?utm=x", "snippet": "Use TinyNet.", "code": "TinyNet()"},
            {"title": "Duplicate", "url": "https://example.test/a", "snippet": "Duplicate", "code": ""},
            {"title": "Missing source", "url": "", "snippet": "Ignore"},
        ]
    )

    evidence = retrieve_model_evidence(_task(), provider, limit=4, output_path=tmp_path / "search_evidence.json")

    assert len(evidence) == 1
    assert evidence[0].url == "https://example.test/a"
    assert evidence[0].model_hint == "TinyNet"
    assert evidence[0].example_code == "TinyNet()"
    assert json.loads((tmp_path / "search_evidence.json").read_text())[0]["title"] == "Efficient model"

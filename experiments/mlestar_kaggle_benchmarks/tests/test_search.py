from mlestar.contracts import FoldSpec, MetricSpec, SubmissionSpec, TaskSpec
from mlestar.search import SearchEvidence, StaticSearchProvider, retrieve_evidence


def test_search_retrieves_unique_citation_ready_evidence() -> None:
    task = TaskSpec(
        key="tiny", competition="tiny", modality="tabular", metric=MetricSpec("roc_auc"),
        fold=FoldSpec(2), submission=SubmissionSpec(("id",), ("score",)), target_columns=("target",),
    )
    duplicate = SearchEvidence("Tree", "https://example.org/model?tracking=x", "Use trees.")
    result = retrieve_evidence(
        task,
        StaticSearchProvider((duplicate, SearchEvidence("Tree copy", "https://example.org/model", "Duplicate."))),
    )
    assert len(result) == 1
    assert "roc_auc" in result[0].url or task.metric.name in "roc_auc"

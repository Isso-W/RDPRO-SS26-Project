import json

from knowledge import Evidence, KnowledgeStore, StrategyCard


def _card(card_id="strategy_aug_test_001"):
    return StrategyCard(
        id=card_id,
        task_type="classification",
        domain="fine_grained_classification",
        strategy_name="RandAugment Test",
        component="augmentation",
        summary="A compact test card.",
        use_when=["small dataset"],
        target_metrics=["log_loss"],
        experiment_template={"augmentation": "randaugment"},
        evidence=[Evidence(source_id="source_1", note="evidence")],
        priority=0.5,
    )


def test_strategy_card_round_trip():
    card = _card()
    assert StrategyCard.from_dict(card.to_dict()) == card


def test_store_merges_evidence_and_searches_without_raw_text(tmp_path):
    store = KnowledgeStore(tmp_path)
    store.upsert_card(_card())
    second = _card("different_id")
    second.evidence = [Evidence(source_id="source_2", note="other evidence")]
    merged = store.upsert_card(second)
    assert merged.id == "strategy_aug_test_001"
    assert len(merged.evidence) == 2
    result = store.search_cards(
        "RandAugment small dataset",
        domain="fine_grained_classification",
        target_metric="log_loss",
    )
    assert result[0]["id"] == merged.id
    assert "evidence" not in result[0]


def test_index_is_rebuildable_from_json(tmp_path):
    store = KnowledgeStore(tmp_path)
    store.upsert_card(_card())
    store.index_path.unlink()
    store.rebuild_index()
    assert store.index_path.exists()
    assert json.loads((store.strategy_cards / "strategy_aug_test_001.json").read_text())["priority"] == 0.5

from mcp_server.tools.knowledge_tools import _component, _priority_value, _risk_level


def test_llm_priority_labels_are_normalized():
    assert _priority_value("high") == 0.8
    assert _priority_value("moderate") == 0.5
    assert _priority_value("0.7") == 0.7
    assert _priority_value("unexpected") == 0.5


def test_llm_card_enums_are_normalized():
    assert _component("Data Augmentation") == "augmentation"
    assert _component("Fine Tuning") == "finetune"
    assert _risk_level("moderate") == "medium"
    assert _risk_level("unexpected") == "medium"

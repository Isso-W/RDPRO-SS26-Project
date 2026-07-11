import pytest

from mlestar.ensemble import blend_test_predictions, select_ensemble


def test_ensemble_rejects_missing_or_reordered_oof_rows() -> None:
    with pytest.raises(ValueError, match="same row ids"):
        select_ensemble({"a": ([1, 2], [0.1, 0.9]), "b": ([2, 1], [0.8, 0.2])}, [0, 1], "roc_auc")


def test_ensemble_selects_oof_weights_and_reuses_them_for_test() -> None:
    result = select_ensemble({"a": ([1, 2, 3, 4], [0.1, 0.8, 0.2, 0.9])}, [0, 1, 0, 1], "roc_auc")
    assert result.weights == {"a": 1.0}
    assert result.score.value == 1.0
    assert blend_test_predictions({"a": [0.4, 0.6]}, result.weights).tolist() == [0.4, 0.6]

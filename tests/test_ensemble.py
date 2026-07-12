import numpy as np
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


def test_ensemble_score_transform_is_applied_before_scoring() -> None:
    # Ordinal (qwk) predictions must be rounded to discrete integer grades
    # before scoring -- cohen_kappa_score rejects continuous floats outright.
    # Candidate "b" is exactly 1.0 above the integer labels everywhere, so
    # rounding its raw predictions recovers a perfect qwk match once its
    # weight dominates the blend.
    y_true = [0, 1, 2, 3]
    oof_by_candidate = {
        "a": ([1, 2, 3, 4], [-0.4, 0.6, 1.6, 2.6]),
        "b": ([1, 2, 3, 4], [0.4, 1.4, 2.4, 3.4]),
    }

    # Without a transform, qwk cannot score these continuous, non-integer
    # predictions at all -- this is exactly the "meaningless kappa" failure
    # mode the bug produced (sklearn actually raises here rather than
    # silently misbehaving, which underscores why the ensemble path needs
    # the same rounding every other arm already applies).
    with pytest.raises(ValueError, match="continuous"):
        select_ensemble(oof_by_candidate, y_true, "qwk", grid_step=1.0)

    # With the ordinal rounding transform applied, scoring succeeds and
    # correctly identifies "b" (which rounds to the true labels exactly) as
    # the winning weight with a perfect score.
    rounded = select_ensemble(
        oof_by_candidate, y_true, "qwk", grid_step=1.0, score_transform=lambda prediction: np.round(prediction)
    )
    assert rounded.weights == {"a": 0.0, "b": 1.0}
    assert rounded.score.value == pytest.approx(1.0)

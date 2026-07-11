"""Tests for OOF-safe non-negative blending."""

from __future__ import annotations

import pytest

from mlestar.ensemble import blend_predictions, fit_simplex_blend


def test_blend_uses_oof_rows_and_returns_non_negative_sum_one_weights() -> None:
    result = fit_simplex_blend(
        y_true=[0, 1, 0, 1],
        oof_by_candidate={"a": [0.1, 0.8, 0.2, 0.7], "b": [0.2, 0.9, 0.1, 0.8]},
        metric_name="roc_auc",
    )

    assert set(result.weights) == {"a", "b"}
    assert all(weight >= 0 for weight in result.weights.values())
    assert sum(result.weights.values()) == pytest.approx(1.0)


def test_blend_rejects_misaligned_predictions_and_illegal_weights() -> None:
    with pytest.raises(ValueError, match="cover"):
        fit_simplex_blend(y_true=[0, 1], oof_by_candidate={"a": [0.1], "b": [0.2, 0.8]}, metric_name="roc_auc")

    with pytest.raises(ValueError, match="sum to one"):
        blend_predictions({"a": [0.1], "b": [0.2]}, {"a": 0.8, "b": 0.8})

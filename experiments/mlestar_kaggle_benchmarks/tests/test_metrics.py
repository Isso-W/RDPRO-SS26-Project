import math

import pytest

from mlestar.metrics import better_than, score_metric


def test_metric_directions_are_explicit() -> None:
    assert score_metric("roc_auc", [0, 1], [0.1, 0.9]).greater_is_better is True
    assert score_metric("rmse", [0.0, 1.0], [0.0, 0.5]).greater_is_better is False


def test_binary_and_multiclass_log_loss_are_minimised() -> None:
    binary = score_metric("log_loss", [0, 1], [0.1, 0.9])
    multi = score_metric(
        "multiclass_log_loss",
        [0, 1, 2],
        [[0.9, 0.05, 0.05], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]],
    )
    assert binary.value < 0.2
    assert multi.value < 0.2
    assert not binary.greater_is_better and not multi.greater_is_better


def test_qwk_rmse_and_dice_have_expected_values() -> None:
    assert score_metric("qwk", [0, 1, 2], [0, 1, 2]).value == pytest.approx(1.0)
    assert score_metric("rmse", [0.0, 1.0], [0.0, 0.0]).value == pytest.approx(math.sqrt(0.5))
    assert score_metric("dice", [[1, 0]], [[0.9, 0.1]]).value == pytest.approx(1.0)
    assert score_metric("dice", [[0, 0]], [[0.0, 0.0]]).value == pytest.approx(1.0)


def test_global_wheat_metric_matches_perfect_and_missed_predictions() -> None:
    truth = {"image": [[0, 0, 10, 10]]}
    assert score_metric("detection_map", truth, {"image": [[0, 0, 10, 10]]}).value == pytest.approx(1.0)
    assert score_metric("detection_map", truth, {"image": []}).value == pytest.approx(0.0)


def test_metric_comparison_uses_the_right_direction() -> None:
    assert better_than(0.8, 0.7, "roc_auc")
    assert better_than(0.2, 0.3, "log_loss")

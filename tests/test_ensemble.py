import numpy as np
import pytest

from ensemble import (
    combine_prediction_sets,
    optimize_validation_ensemble,
    save_probability_artifact,
)


def test_combine_prediction_sets_aligns_ids_and_normalizes():
    first = [("a.jpg", [0.8, 0.2]), ("b.jpg", [0.4, 0.6])]
    second = [("b.jpg", [0.2, 0.8]), ("a.jpg", [0.6, 0.4])]

    combined = combine_prediction_sets([first, second], [0.75, 0.25])

    assert [name for name, _ in combined] == ["a.jpg", "b.jpg"]
    assert combined[0][1] == pytest.approx([0.75, 0.25])
    assert combined[1][1] == pytest.approx([0.35, 0.65])
    assert all(sum(probabilities) == pytest.approx(1.0) for _, probabilities in combined)


def test_combine_prediction_sets_rejects_mismatched_ids():
    with pytest.raises(ValueError, match="same IDs"):
        combine_prediction_sets(
            [[("a.jpg", [0.5, 0.5])], [("b.jpg", [0.5, 0.5])]],
            [0.5, 0.5],
        )


def test_validation_ensemble_uses_only_locally_improving_weights(tmp_path):
    labels = np.asarray([0, 1])
    first = np.asarray([[0.9, 0.1], [0.6, 0.4]], dtype=np.float32)
    second = np.asarray([[0.4, 0.6], [0.1, 0.9]], dtype=np.float32)
    first_path = save_probability_artifact(
        tmp_path / "first.npz", probabilities=first, labels=labels
    )
    second_path = save_probability_artifact(
        tmp_path / "second.npz", probabilities=second, labels=labels
    )

    result = optimize_validation_ensemble(
        [
            {"name": "first", "validation_artifact": str(first_path)},
            {"name": "second", "validation_artifact": str(second_path)},
        ],
        step=0.1,
    )

    assert result["improved"] is True
    assert result["ensemble_log_loss"] < result["best_single_log_loss"]
    assert len(result["members"]) == 2
    assert sum(member["weight"] for member in result["members"]) == pytest.approx(1.0)

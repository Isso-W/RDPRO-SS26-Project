import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from kaggle_submit import (
    _submission_status,
    _submission_value,
    apply_calibrated_imagenet_prior,
    write_submission,
)


def test_dog_breed_probability_submission_has_exact_order_and_sums(tmp_path):
    sample = tmp_path / "sample_submission.csv"
    pd.DataFrame(
        {
            "id": ["a", "b"],
            "akita": [0.0, 0.0],
            "beagle": [0.0, 0.0],
            "corgi": [0.0, 0.0],
        }
    ).to_csv(sample, index=False)
    output = tmp_path / "submission.csv"
    write_submission(
        [("a.jpg", [0.6, 0.3, 0.1]), ("b.jpg", [0.1, 0.2, 0.7])],
        {0: "akita", 1: "beagle", 2: "corgi"},
        sample,
        output,
    )
    result = pd.read_csv(output)
    assert list(result.columns) == ["id", "akita", "beagle", "corgi"]
    assert result.iloc[:, 1:].sum(axis=1).tolist() == pytest.approx([1.0, 1.0])
    assert result.loc[1, "corgi"] == pytest.approx(0.7)


def test_submission_rejects_missing_ids(tmp_path):
    sample = tmp_path / "sample.csv"
    pd.DataFrame({"id": ["missing"], "akita": [0.0]}).to_csv(sample, index=False)
    with pytest.raises(ValueError, match="no prediction"):
        write_submission([], {0: "akita"}, sample, tmp_path / "out.csv")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("complete", "complete"),
        ("SubmissionStatus.COMPLETE", "complete"),
        ("SubmissionStatus.ERROR", "error"),
    ],
)
def test_submission_status_normalizes_kaggle_enum_strings(value, expected):
    assert _submission_status(value) == expected


def test_submission_status_prefers_enum_name_over_numeric_value():
    status = SimpleNamespace(name="COMPLETE", value=1)

    assert _submission_status(status) == "complete"


def test_submission_value_supports_new_kaggle_private_fields():
    submission = SimpleNamespace(_public_score="0.82967", _private_score="0.82967")

    assert _submission_value(
        submission, "publicScore", "public_score", "_public_score"
    ) == "0.82967"


def test_apply_calibrated_imagenet_prior_uses_saved_validation_alpha(
    tmp_path, monkeypatch
):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    np.savez_compressed(
        checkpoint_dir / "validation_probabilities.npz",
        probabilities=np.asarray([[0.5, 0.5]], dtype=np.float32),
        labels=np.asarray([0], dtype=np.int64),
        prior_alpha=np.asarray(0.25, dtype=np.float32),
    )
    module = ModuleType("imagenet_prior")
    module.predict_prior_directory = lambda *_args, **_kwargs: (
        [("a.jpg", [0.8, 0.2]), ("b.jpg", [0.2, 0.8])],
        {"prior_model": "efficientnet_v2_s"},
    )
    module.temperature_scale_probabilities = (
        lambda probabilities, _temperature: np.asarray(
            probabilities,
            dtype=np.float32,
        )
    )
    monkeypatch.setitem(sys.modules, "imagenet_prior", module)

    combined, metadata = apply_calibrated_imagenet_prior(
        tmp_path,
        {
            "imagenet_prior_blend": "auto",
            "imagenet_prior_model": "efficientnet_v2_s",
            "checkpoint_dir": str(checkpoint_dir),
        },
        tmp_path / "test",
        [("a.jpg", [0.6, 0.4]), ("b.jpg", [0.4, 0.6])],
        batch_size=8,
    )

    assert combined[0][1] == pytest.approx([0.65, 0.35])
    assert combined[1][1] == pytest.approx([0.35, 0.65])
    assert metadata["alpha"] == pytest.approx(0.25)
    assert metadata["prior_model"] == "efficientnet_v2_s"

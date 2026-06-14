from types import SimpleNamespace

import pandas as pd
import pytest

from kaggle_submit import _submission_status, _submission_value, write_submission


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

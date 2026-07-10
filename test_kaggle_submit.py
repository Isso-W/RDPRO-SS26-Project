"""Tests for kaggle_submit submission formatting."""

from __future__ import annotations

import pandas as pd

from kaggle_submit import write_submission


def _sample(tmp_path, columns, rows):
    path = tmp_path / "sample_submission.csv"
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
    return path


def test_binary_positive_class_writes_probability(tmp_path):
    # Competition scored on P(positive) in a single target column (e.g. ROC AUC).
    sample = _sample(tmp_path, ["id", "label"], [["a", 0], ["b", 0]])
    predictions = [
        ("a", 1, [0.2, 0.8]),  # P(class '1') = 0.8
        ("b", 0, [0.7, 0.3]),  # P(class '1') = 0.3
    ]
    out = tmp_path / "submission.csv"
    write_submission(
        predictions,
        {0: "0", 1: "1"},
        sample,
        out,
        label_columns=None,
        positive_class="1",
    )
    result = pd.read_csv(out).set_index("id")["label"].to_dict()
    assert result["a"] == 0.8
    assert result["b"] == 0.3


def test_multicolumn_submission_auto_maps_probabilities_by_name(tmp_path):
    # Dog Breed / Leaf style: id + one probability column per class, no
    # label_columns configured. Columns are matched to classes by name.
    sample = _sample(tmp_path, ["id", "cat", "dog", "fox"], [["a", 0, 0, 0], ["b", 0, 0, 0]])
    predictions = [
        ("a", 1, [0.1, 0.7, 0.2]),
        ("b", 2, [0.3, 0.2, 0.5]),
    ]
    out = tmp_path / "submission.csv"
    write_submission(predictions, {0: "cat", 1: "dog", 2: "fox"}, sample, out, label_columns=None)
    result = pd.read_csv(out).set_index("id")
    assert result.loc["a", "dog"] == 0.7
    assert result.loc["a", "cat"] == 0.1
    assert result.loc["b", "fox"] == 0.5


def test_single_target_without_positive_class_writes_label(tmp_path):
    # Default behaviour (e.g. APTOS diagnosis): write the predicted class label.
    sample = _sample(tmp_path, ["id_code", "diagnosis"], [["a", 0], ["b", 0]])
    predictions = [("a", 3, [0.0, 0.0, 0.0, 1.0, 0.0]), ("b", 1, [0.0, 1.0, 0.0, 0.0, 0.0])]
    out = tmp_path / "submission.csv"
    write_submission(
        predictions,
        {0: "0", 1: "1", 2: "2", 3: "3", 4: "4"},
        sample,
        out,
        label_columns=None,
    )
    result = pd.read_csv(out).set_index("id_code")["diagnosis"].to_dict()
    assert result["a"] == 3
    assert result["b"] == 1

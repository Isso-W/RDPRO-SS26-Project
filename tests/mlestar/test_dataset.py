"""Tests for persistent folds, inventory, and metric-correct OOF evaluation."""

from __future__ import annotations

import pandas as pd
import pytest

from mlestar.dataset import inspect_dataset, make_folds, score_oof


def test_stratified_folds_are_persistent_and_cover_each_row_once(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "row_id": range(12),
            "target": [0, 1] * 6,
        }
    )

    folds = make_folds(
        frame,
        target="target",
        strategy="stratified",
        n_splits=3,
        seed=7,
        output_path=tmp_path / "folds.parquet",
        id_column="row_id",
    )

    assert sorted(folds["fold"].tolist()) == [0] * 4 + [1] * 4 + [2] * 4
    assert folds["row_id"].tolist() == list(range(12))
    assert (tmp_path / "folds.parquet").exists()


def test_stratified_folds_reject_insufficient_minority_class(tmp_path) -> None:
    frame = pd.DataFrame({"target": [0, 0, 0, 1]})

    with pytest.raises(ValueError, match="smallest class"):
        make_folds(frame, target="target", strategy="stratified", n_splits=2, seed=7, output_path=tmp_path / "folds.parquet")


@pytest.mark.parametrize(
    ("metric", "y_true", "y_pred", "expected"),
    [
        ("roc_auc", [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], 1.0),
        ("accuracy", [0, 1, 0], [0, 1, 1], 2 / 3),
        ("qwk", [0, 1, 2], [0, 1, 2], 1.0),
        ("rmse", [0.0, 2.0], [0.0, 0.0], 2**0.5),
        ("dice", [0, 1, 1, 0], [0.1, 0.9, 0.8, 0.2], 1.0),
    ],
)
def test_score_oof_uses_metric_correctly(metric, y_true, y_pred, expected) -> None:
    assert score_oof(metric, y_true, y_pred) == pytest.approx(expected)


def test_inspect_dataset_records_relative_files_and_fingerprint(tmp_path) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "one.jpg").write_bytes(b"image")
    pd.DataFrame({"id": ["one"], "target": [1]}).to_csv(tmp_path / "train.csv", index=False)

    inventory = inspect_dataset(tmp_path, output_path=tmp_path / "inventory.json")

    assert inventory.fingerprint
    assert [item["path"] for item in inventory.files] == ["images/one.jpg", "train.csv"]
    assert inventory.files[1]["columns"] == ["id", "target"]
    assert (tmp_path / "inventory.json").exists()

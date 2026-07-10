"""Tests for kaggle_loader helpers that don't require network access."""

from __future__ import annotations

import pandas as pd

from ingestion.kaggle_loader import build_label_csv_from_filenames


def test_build_label_csv_from_filenames(tmp_path):
    image_dir = tmp_path / "train"
    image_dir.mkdir()
    for name in ("cat.1.jpg", "cat.2.jpg", "dog.1.jpg", "bird.1.jpg"):
        (image_dir / name).touch()

    csv_path = build_label_csv_from_filenames(
        image_dir,
        pattern=r"^(cat|dog)\.",
        image_column="id",
        label_column="label",
        out_path=tmp_path / "train_from_names.csv",
    )
    frame = pd.read_csv(csv_path)

    # Only files matching the pattern are labelled; unmatched files are skipped.
    labels = dict(zip(frame["id"].astype(str), frame["label"].astype(str)))
    assert labels["cat.1"] == "cat"
    assert labels["dog.1"] == "dog"
    assert "bird.1" not in labels
    assert len(frame) == 3

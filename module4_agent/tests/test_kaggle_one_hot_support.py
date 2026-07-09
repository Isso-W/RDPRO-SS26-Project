import pandas as pd

from ingestion.kaggle_loader import _materialize_label_csv
from kaggle_submit import write_submission


def test_materialize_label_csv_from_one_hot_columns(tmp_path):
    source = tmp_path / "train.csv"
    pd.DataFrame(
        {
            "image_id": ["Train_0", "Train_1", "Train_2"],
            "healthy": [0, 1, 0],
            "multiple_diseases": [0, 0, 0],
            "rust": [0, 0, 1],
            "scab": [1, 0, 0],
        }
    ).to_csv(source, index=False)

    derived = _materialize_label_csv(
        source,
        "__jiaozi_label",
        ["healthy", "multiple_diseases", "rust", "scab"],
    )

    frame = pd.read_csv(derived)
    assert frame["__jiaozi_label"].tolist() == ["scab", "healthy", "rust"]


def test_write_submission_supports_one_hot_target_columns(tmp_path):
    sample = tmp_path / "sample_submission.csv"
    out = tmp_path / "submission.csv"
    pd.DataFrame(
        {
            "image_id": ["Test_0", "Test_1"],
            "healthy": [0.0, 0.0],
            "multiple_diseases": [0.0, 0.0],
            "rust": [0.0, 0.0],
            "scab": [0.0, 0.0],
        }
    ).to_csv(sample, index=False)

    write_submission(
        [
            ("Test_0.jpg", 2, [0.1, 0.2, 0.6, 0.1]),
            ("Test_1.jpg", 0, [0.7, 0.1, 0.1, 0.1]),
        ],
        {0: "healthy", 1: "multiple_diseases", 2: "rust", 3: "scab"},
        sample,
        out,
        label_columns=["healthy", "multiple_diseases", "rust", "scab"],
    )

    frame = pd.read_csv(out)
    assert frame.loc[0, ["healthy", "multiple_diseases", "rust", "scab"]].tolist() == [0.1, 0.2, 0.6, 0.1]
    assert frame.loc[1, ["healthy", "multiple_diseases", "rust", "scab"]].tolist() == [0.7, 0.1, 0.1, 0.1]

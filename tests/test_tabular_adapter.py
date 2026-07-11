import pandas as pd

from benchmarks.catalog import get_task
from mlestar.adapters.tabular import LeafClassificationAdapter
from mlestar.initialization import CandidateSpec


def _write_leaf_data(root) -> None:
    rows = []
    for label, center in (("A", 0.0), ("B", 4.0), ("C", 8.0)):
        for index in range(5):
            rows.append({"id": f"{label}{index}", "feature_1": center + index / 100, "feature_2": label, "species": label})
    pd.DataFrame(rows).to_csv(root / "train.csv", index=False)
    pd.DataFrame(
        [{"id": "t0", "feature_1": 0.1, "feature_2": "A"}, {"id": "t1", "feature_1": 7.9, "feature_2": "C"}]
    ).to_csv(root / "test.csv", index=False)
    pd.DataFrame({"id": ["t0", "t1"], "A": [0.0, 0.0], "B": [0.0, 0.0], "C": [0.0, 0.0]}).to_csv(
        root / "sample_submission.csv", index=False
    )


def test_leaf_adapter_writes_fixed_fold_oof_test_and_submission(tmp_path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    _write_leaf_data(data_root)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    adapter = LeafClassificationAdapter(data_root, run_dir, get_task("leaf_classification"))
    candidate = CandidateSpec("extra_trees", (("model", "extra_trees"),))

    result = adapter.run(candidate, phase="baseline", seed=13)

    assert result.receipt.error is None
    assert len(result.receipt.fold_scores) == 5
    assert (run_dir / result.receipt.oof_path).is_file()
    assert (run_dir / result.receipt.test_path).is_file()
    assert list(pd.read_csv(run_dir / result.receipt.test_path).columns) == ["id", "A", "B", "C"]

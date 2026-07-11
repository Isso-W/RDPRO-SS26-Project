from tests.test_tabular_adapter import _write_leaf_data

from mlestar.experiment import compare


def test_compare_writes_paired_baseline_and_mlestar_oof_report(tmp_path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _write_leaf_data(data)
    report = compare(benchmark="leaf_classification", data_root=data, run_root=tmp_path / "runs", seeds=(13,))

    assert report["arms"] == ["baseline", "mlestar_initial", "mlestar_refined", "mlestar_ensemble"]
    assert report["paired_folds"] is True
    assert (tmp_path / "runs" / "comparison.csv").is_file()
    assert (tmp_path / "runs" / "comparison.json").is_file()

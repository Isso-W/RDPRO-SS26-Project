from pathlib import Path

import pytest

from tests.test_tabular_adapter import _write_leaf_data

from mlestar.adapters.vision import PlantPathologyAdapter
from mlestar.contracts import MetricSpec
from mlestar.experiment import ADAPTER_CLASSES, _summary, compare


def test_compare_writes_paired_baseline_and_mlestar_oof_report(tmp_path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _write_leaf_data(data)
    report = compare(benchmark="leaf_classification", data_root=data, run_root=tmp_path / "runs", seeds=(13,))

    assert report["arms"] == ["baseline", "mlestar_initial", "mlestar_refined", "mlestar_ensemble"]
    assert report["paired_folds"] is True
    assert (tmp_path / "runs" / "comparison.csv").is_file()
    assert (tmp_path / "runs" / "comparison.json").is_file()


def test_adapter_classes_registry_covers_all_seven_implemented_benchmarks():
    assert set(ADAPTER_CLASSES) == {
        "leaf_classification",
        "plant_pathology_2020",
        "aptos_2019",
        "dog_breed",
        "aerial_cactus",
        "dogs_vs_cats",
        "histopathologic_cancer",
    }
    assert ADAPTER_CLASSES["plant_pathology_2020"] is PlantPathologyAdapter


def test_compare_still_rejects_unimplemented_benchmarks(tmp_path):
    with pytest.raises(NotImplementedError, match="global_wheat"):
        compare(benchmark="global_wheat", data_root=tmp_path, run_root=tmp_path)


def test_summary_wins_are_direction_aware_for_greater_is_better_metrics() -> None:
    metric = MetricSpec("roc_auc")
    rows = [
        {"seed": 13, "arm": "baseline", "metric_value": 0.70},
        {"seed": 13, "arm": "mlestar_refined", "metric_value": 0.90},  # higher: a real win
        {"seed": 29, "arm": "baseline", "metric_value": 0.70},
        {"seed": 29, "arm": "mlestar_refined", "metric_value": 0.50},  # lower: not a win
    ]
    summary = _summary(rows, metric)
    assert summary["mlestar_refined"]["wins"] == 1


def test_compare_runs_plant_pathology_end_to_end(tmp_path):
    report = compare(
        benchmark="plant_pathology_2020",
        data_root=Path("examples/synthetic_plant_pathology"),
        run_root=tmp_path,
        seeds=(13,),
        adapter_kwargs={"pretrained": False},
    )
    assert report["benchmark"] == "plant_pathology_2020"
    assert report["summary"]["baseline"]["failures"] == 0
    assert report["summary"]["mlestar_refined"]["failures"] == 0


def test_compare_runs_aptos_end_to_end(tmp_path):
    report = compare(
        benchmark="aptos_2019",
        data_root=Path("examples/synthetic_aptos"),
        run_root=tmp_path,
        seeds=(13,),
        adapter_kwargs={"pretrained": False},
    )
    assert report["benchmark"] == "aptos_2019"
    assert report["summary"]["baseline"]["failures"] == 0
    assert report["summary"]["mlestar_refined"]["failures"] == 0

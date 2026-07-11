from pathlib import Path

import pytest

from tests.test_tabular_adapter import _write_leaf_data

from mlestar.adapters.vision import PlantPathologyAdapter
from mlestar.experiment import ADAPTER_CLASSES, compare


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

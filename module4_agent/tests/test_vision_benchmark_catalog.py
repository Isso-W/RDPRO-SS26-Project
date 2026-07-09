from vision_benchmark_catalog import BENCHMARKS, get_benchmark


def test_catalog_contains_requested_competitions_and_ten_extra_datasets():
    requested = {
        "cassava",
        "state_farm",
        "siim_isic",
        "diabetic_retinopathy",
        "plant_pathology_2020",
    }

    assert len(BENCHMARKS) == 15
    assert requested.issubset(BENCHMARKS)
    assert sum(item["source"] == "kaggle" for item in BENCHMARKS.values()) == 5
    assert sum(item["source"] == "huggingface" for item in BENCHMARKS.values()) == 10


def test_catalog_entries_have_runnable_source_metadata():
    for key, item in BENCHMARKS.items():
        assert item["name"]
        assert item["query"]
        assert item["metric"]
        assert item["num_classes"] >= 2
        assert item["backbone"]
        assert item["loss"]
        if item["source"] == "kaggle":
            assert item["competition"]
            assert item["csv_globs"]
            assert item["image_dir_globs"]
            assert item["image_column"]
            assert item["label_column"]
        else:
            assert item["dataset_id"]
        assert get_benchmark(key) == item


def test_catalog_accepts_plant_pathology_slug_alias():
    benchmark = get_benchmark("plant-pathology-2020-fgv")
    assert benchmark["competition"] == "plant-pathology-2020-fgvc7"
    assert benchmark["label_columns"] == ["healthy", "multiple_diseases", "rust", "scab"]

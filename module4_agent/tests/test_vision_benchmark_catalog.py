from vision_benchmark_catalog import BENCHMARKS, get_benchmark


def test_catalog_contains_requested_competitions_and_ten_extra_datasets():
    requested = {
        "cassava",
        "state_farm",
        "siim_isic",
        "diabetic_retinopathy",
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

from vision_benchmark_catalog import BENCHMARKS, get_benchmark


def test_catalog_contains_requested_competitions_and_ten_extra_datasets():
    requested = {
        "cassava",
        "state_farm",
        "siim_isic",
        "diabetic_retinopathy",
        "plant_pathology_2020",
    }

    assert requested.issubset(BENCHMARKS)
    assert sum(item["source"] == "huggingface" for item in BENCHMARKS.values()) == 10


def test_catalog_contains_the_ten_target_competitions():
    """The competitions from the MLE-STAR / MLE-bench task sheet."""
    target = {
        "plant_pathology_2020",
        "aptos2019",
        "dog_breed",
        "leaf_classification",
        "aerial_cactus",
        "dogs_vs_cats_redux",
        "histopathologic_cancer",
        "global_wheat",
        "ultrasound_nerve",
        "denoising_dirty_documents",
    }
    assert target.issubset(BENCHMARKS)


def test_catalog_entries_have_runnable_source_metadata():
    for key, item in BENCHMARKS.items():
        assert item["name"]
        assert item["query"]
        assert item["metric"]
        task_type = item.get("task_type", "classification")
        if item["source"] == "kaggle":
            assert item["competition"]
            assert item["image_dir_globs"]
            if task_type == "classification":
                # A train CSV is required unless labels come from filenames.
                assert item.get("csv_globs") or item.get("label_from_filename")
                assert item["image_column"]
                assert item["label_column"]
                assert item["num_classes"] >= 2
                assert item["backbone"]
                assert item["loss"]
            else:
                # detection / segmentation / denoising: flagged, not classification-runnable
                assert item.get("notes")
        else:
            assert item["dataset_id"]
            assert item["num_classes"] >= 2
            assert item["backbone"]
            assert item["loss"]
        assert get_benchmark(key) == item


def test_catalog_accepts_plant_pathology_slug_alias():
    benchmark = get_benchmark("plant-pathology-2020-fgv")
    assert benchmark["competition"] == "plant-pathology-2020-fgvc7"
    assert benchmark["label_columns"] == ["healthy", "multiple_diseases", "rust", "scab"]

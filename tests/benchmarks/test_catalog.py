"""The screenshot competitions must remain distinct benchmark contracts."""

from benchmarks.catalog import get_benchmark, list_benchmarks


def test_screenshot_competitions_have_distinct_modality_metric_and_submission_contracts() -> None:
    catalog = {item.key: item for item in list_benchmarks()}
    assert set(catalog) == {
        "plant_pathology_2020", "aptos_2019", "dog_breed", "global_wheat", "ultrasound_nerve",
        "leaf_classification", "aerial_cactus", "dogs_vs_cats", "histopathologic_cancer", "denoising_dirty_documents",
    }
    assert catalog["aptos_2019"].metric.name == "qwk"
    assert catalog["global_wheat"].modality == "object_detection"
    assert catalog["denoising_dirty_documents"].submission.kind == "image_directory"


def test_alias_resolves_to_canonical_contract() -> None:
    assert get_benchmark("plant-pathology-2020-fgvc7").key == "plant_pathology_2020"

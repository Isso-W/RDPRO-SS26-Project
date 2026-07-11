from benchmarks.catalog import BENCHMARKS, get_task


def test_catalog_contains_ten_unique_metric_correct_tasks() -> None:
    assert len(BENCHMARKS) == 10
    assert len({task.competition for task in BENCHMARKS.values()}) == 10
    assert get_task("global_wheat").metric.name == "detection_map"
    assert get_task("denoising_dirty_documents").metric.greater_is_better is False

from pathlib import Path

import numpy as np

from benchmarks.catalog import get_task
from mlestar.adapters.vision import PlantPathologyAdapter
from mlestar.initialization import CandidateSpec

FIXTURE = Path("examples/synthetic_plant_pathology")


def test_plant_pathology_load_dataset_reads_multilabel_csv():
    task = get_task("plant_pathology_2020")
    adapter = PlantPathologyAdapter(FIXTURE, "/tmp/mlestar-vision-test-load", task, pretrained=False)
    paths, labels, ids = adapter._load_dataset(adapter.data_root)
    assert len(paths) == 6
    assert labels.shape == (6, 4)
    assert ids == ["Train_0", "Train_1", "Train_2", "Train_3", "Train_4", "Train_5"]
    assert paths[0].name == "Train_0.jpg"
    assert paths[0].exists()


def test_plant_pathology_run_produces_a_metric(tmp_path):
    task = get_task("plant_pathology_2020")
    adapter = PlantPathologyAdapter(
        FIXTURE, tmp_path, task, pretrained=False, epochs=1, max_train_samples=None
    )
    candidate = CandidateSpec("resnet18", (("model", "resnet18"),))
    result = adapter.run(candidate, phase="test", seed=13)
    assert result.receipt.error is None, result.receipt.error
    assert result.receipt.metric_value is not None
    assert 0.0 <= result.receipt.metric_value <= 1.0
    assert result.receipt.oof_path is not None
    assert result.receipt.test_path is None
    assert result.oof.shape == (6, 4)
    oof_csv = tmp_path / result.receipt.oof_path
    assert oof_csv.exists()


def test_evaluate_wraps_a_failure_instead_of_raising(tmp_path):
    task = get_task("plant_pathology_2020")
    adapter = PlantPathologyAdapter(tmp_path / "does-not-exist", tmp_path, task, pretrained=False)
    candidate = CandidateSpec("resnet18", (("model", "resnet18"),))
    receipt = adapter.evaluate(candidate, phase="test", seed=13)
    assert receipt.metric_value is None
    assert receipt.error is not None
    assert "FileNotFoundError" in receipt.error or "train.csv" in receipt.error


def test_merge_names_an_ensemble_candidate():
    task = get_task("plant_pathology_2020")
    adapter = PlantPathologyAdapter(FIXTURE, "/tmp/mlestar-vision-test-merge", task, pretrained=False)
    incumbent = CandidateSpec("resnet18", (("model", "resnet18"),))
    addition = CandidateSpec("efficientnet_b0", (("model", "efficientnet_b0"),))
    merged = adapter.merge(incumbent, addition)
    assert merged.candidate_id == "resnet18+efficientnet_b0"
    assert merged.block("model") == "ensemble:resnet18+efficientnet_b0"


def test_score_inputs_ordinal_clips_to_label_range_not_num_classes():
    from mlestar.adapters.vision import ImageClassificationAdapter
    from mlestar.contracts import FoldSpec, MetricSpec, SubmissionSpec, TaskSpec

    task = TaskSpec(
        key="fake_ordinal",
        competition="fake-ordinal",
        modality="image_ordinal",
        metric=MetricSpec("qwk"),
        fold=FoldSpec(n_splits=2),
        submission=SubmissionSpec(id_columns=("id",), prediction_columns=("diagnosis",)),
        target_columns=("diagnosis",),
    )
    adapter = ImageClassificationAdapter("/tmp/does-not-matter", "/tmp/does-not-matter-run", task, pretrained=False)
    labels = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    oof = np.array([0.2, 1.6, 2.4, 3.9, 3.4])  # raw regression outputs, unrounded
    _, rounded = adapter._score_inputs(labels, oof)
    assert rounded.tolist() == [0.0, 2.0, 2.0, 4.0, 3.0]

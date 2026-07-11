import pytest

from mlestar.artifacts import RunArtifacts
from mlestar.contracts import (
    ExperimentReceipt,
    FoldSpec,
    MetricSpec,
    SubmissionSpec,
    TaskSpec,
)


def test_task_spec_has_a_stable_json_round_trip() -> None:
    task = TaskSpec(
        key="tiny_task",
        competition="tiny-competition",
        modality="tabular_binary",
        metric=MetricSpec("roc_auc"),
        fold=FoldSpec(n_splits=3, seed=7),
        submission=SubmissionSpec(("id",), ("target",)),
        target_columns=("target",),
    )
    assert TaskSpec.from_json(task.to_json()) == task


def test_receipt_has_a_stable_json_round_trip() -> None:
    receipt = ExperimentReceipt(
        experiment_id="run-001",
        parent_experiment_id=None,
        phase="initial",
        candidate_id="tree",
        metric_value=0.9,
        fold_scores=(0.8, 1.0),
        seed=7,
        oof_path="oof/tree.parquet",
        test_path="test/tree.parquet",
        error=None,
    )
    assert ExperimentReceipt.from_json(receipt.to_json()) == receipt


def test_metric_direction_cannot_be_overridden() -> None:
    with pytest.raises(ValueError, match="greater_is_better"):
        MetricSpec("rmse", greater_is_better=True)


def test_artifact_cannot_escape_run_directory(tmp_path) -> None:
    with pytest.raises(ValueError, match="run directory"):
        RunArtifacts(tmp_path).resolve("../submission.csv")


def test_artifact_path_can_use_a_contained_absolute_path(tmp_path) -> None:
    artifacts = RunArtifacts(tmp_path)
    assert artifacts.relative(tmp_path / "oof" / "predictions.parquet") == "oof/predictions.parquet"


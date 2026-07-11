from dataclasses import replace

import pytest

from benchmarks.catalog import get_task
from mlestar.contracts import ExperimentReceipt, FoldSpec, MetricSpec, SubmissionSpec, TaskSpec
from mlestar.initialization import CandidateSpec, choose_best, initialize_solution


class _Evaluator:
    values = {"linear": 0.60, "tree": 0.80, "bad": 0.40, "tree+linear": 0.81}

    def evaluate(self, candidate, *, phase, seed, parent_experiment_id=None):
        from mlestar.contracts import ExperimentReceipt

        return ExperimentReceipt(
            experiment_id=f"{phase}-{candidate.candidate_id}", parent_experiment_id=parent_experiment_id,
            phase=phase, candidate_id=candidate.candidate_id,
            metric_value=self.values.get(candidate.candidate_id, 0.79), fold_scores=(0.7,), seed=seed,
            oof_path="oof.csv", test_path="test.csv", error=None,
        )

    def merge(self, incumbent, addition):
        return CandidateSpec(
            candidate_id=f"{incumbent.candidate_id}+{addition.candidate_id}", blocks=incumbent.blocks
        )


def _task() -> TaskSpec:
    return TaskSpec(
        key="synthetic", competition="synthetic", modality="tabular", metric=MetricSpec("roc_auc"),
        fold=FoldSpec(2), submission=SubmissionSpec(("id",), ("score",)), target_columns=("target",),
    )


def test_initialization_evaluates_candidates_and_accepts_improving_merge() -> None:
    blocks = (("model", "model"),)
    result = initialize_solution(
        _task(), _Evaluator(), [CandidateSpec(name, blocks) for name in ("linear", "tree", "bad")], seed=7
    )
    assert result.best.candidate_id == "tree+linear"
    assert [item.candidate_id for item in result.receipts] == ["linear", "tree", "bad"]
    assert result.best_receipt.metric_value == 0.81


def test_choose_best_error_message_includes_candidate_errors():
    task = get_task("leaf_classification")
    candidate_a = CandidateSpec("a", (("model", "a"),))
    candidate_b = CandidateSpec("b", (("model", "b"),))
    receipt_a = ExperimentReceipt(
        experiment_id="a-1", parent_experiment_id=None, phase="initial", candidate_id="a",
        metric_value=None, fold_scores=(), seed=13, oof_path=None, test_path=None,
        error="FileNotFoundError: train.csv missing",
    )
    receipt_b = ExperimentReceipt(
        experiment_id="b-1", parent_experiment_id=None, phase="initial", candidate_id="b",
        metric_value=None, fold_scores=(), seed=13, oof_path=None, test_path=None,
        error="ValueError: bad label",
    )
    with pytest.raises(RuntimeError) as excinfo:
        choose_best([(candidate_a, receipt_a), (candidate_b, receipt_b)], task)
    message = str(excinfo.value)
    assert "FileNotFoundError: train.csv missing" in message
    assert "ValueError: bad label" in message

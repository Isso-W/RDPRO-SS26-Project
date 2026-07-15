"""Search-informed initial solution evaluation and conservative merging."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol, Sequence

from .contracts import ExperimentReceipt, TaskSpec


@dataclass(frozen=True)
class CandidateSpec:
    """A serialisable candidate produced from one retrieved model description."""

    candidate_id: str
    blocks: tuple[tuple[str, str], ...]
    evidence_urls: tuple[str, ...] = ()

    def block(self, name: str) -> str:
        return dict(self.blocks)[name]


class CandidateEvaluator(Protocol):
    """Task adapter boundary used by all MLE-STAR search phases."""

    def evaluate(
        self,
        candidate: CandidateSpec,
        *,
        phase: str,
        seed: int,
        parent_experiment_id: str | None = None,
    ) -> ExperimentReceipt: ...

    def merge(self, incumbent: CandidateSpec, addition: CandidateSpec) -> CandidateSpec: ...


@dataclass(frozen=True)
class InitializationResult:
    best: CandidateSpec
    best_receipt: ExperimentReceipt
    receipts: tuple[ExperimentReceipt, ...]
    merge_receipts: tuple[ExperimentReceipt, ...]


def choose_best(
    candidates: Iterable[tuple[CandidateSpec, ExperimentReceipt]], task: TaskSpec
) -> tuple[CandidateSpec, ExperimentReceipt]:
    """Choose the best successful receipt without treating a failure as zero."""

    pairs = list(candidates)
    valid = [(candidate, receipt) for candidate, receipt in pairs if receipt.metric_value is not None]
    if not valid:
        details = "; ".join(
            f"{candidate.candidate_id}: {receipt.error}"
            for candidate, receipt in pairs
            if receipt.error is not None
        )
        suffix = f" Candidate errors -- {details}" if details else ""
        raise RuntimeError(f"No candidate produced a validation metric.{suffix}")
    direction = bool(task.metric.greater_is_better)
    return max(valid, key=lambda item: float(item[1].metric_value)) if direction else min(
        valid, key=lambda item: float(item[1].metric_value)
    )


def improves(candidate: ExperimentReceipt, incumbent: ExperimentReceipt, task: TaskSpec) -> bool:
    """Return true only for a strict metric improvement."""

    if candidate.metric_value is None:
        return False
    if incumbent.metric_value is None:
        return True
    return (
        candidate.metric_value > incumbent.metric_value
        if task.metric.greater_is_better
        else candidate.metric_value < incumbent.metric_value
    )


def initialize_solution(
    task: TaskSpec,
    evaluator: CandidateEvaluator,
    candidates: Sequence[CandidateSpec],
    *,
    seed: int,
) -> InitializationResult:
    """Evaluate candidates then accept metric-improving incremental merges.

    This mirrors MLE-STAR's initialization policy: candidate models are
    evaluated individually; the current best is then merged with the next
    ranked candidate only if its validation metric improves.
    """

    if not candidates:
        raise ValueError("At least one initial candidate is required.")
    evaluated = tuple(
        (candidate, evaluator.evaluate(candidate, phase="initial", seed=seed)) for candidate in candidates
    )
    best, best_receipt = choose_best(evaluated, task)
    ranked = sorted(
        ((candidate, receipt) for candidate, receipt in evaluated if receipt.metric_value is not None),
        key=lambda item: float(item[1].metric_value),
        reverse=bool(task.metric.greater_is_better),
    )
    merge_receipts: list[ExperimentReceipt] = []
    for candidate, _ in ranked:
        if candidate.candidate_id == best.candidate_id:
            continue
        merged = evaluator.merge(best, candidate)
        receipt = evaluator.evaluate(
            merged,
            phase="initial_merge",
            seed=seed,
            parent_experiment_id=best_receipt.experiment_id,
        )
        merge_receipts.append(receipt)
        if improves(receipt, best_receipt, task):
            best, best_receipt = merged, receipt
    return InitializationResult(
        best=best,
        best_receipt=best_receipt,
        receipts=tuple(receipt for _, receipt in evaluated),
        merge_receipts=tuple(merge_receipts),
    )

"""Ablation-guided targeted refinement for MLE-STAR candidates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .contracts import ExperimentReceipt, TaskSpec
from .initialization import CandidateEvaluator, CandidateSpec, improves


COMPONENTS = ("data_loading", "data_preparation", "model", "training", "prediction")


class RefinementPlanner(Protocol):
    """Propose replacement source for one and only one named component."""

    def propose(self, *, component: str, candidate: CandidateSpec, history: Sequence[ExperimentReceipt]) -> Sequence[str]: ...


def replace_block(candidate: CandidateSpec, component: str, source: str) -> CandidateSpec:
    """Return a new candidate with exactly one block changed."""

    if component not in COMPONENTS:
        raise ValueError(f"Unknown component {component!r}.")
    blocks = dict(candidate.blocks)
    if component not in blocks:
        raise ValueError(f"Candidate has no {component!r} block.")
    blocks[component] = source
    return CandidateSpec(
        candidate_id=f"{candidate.candidate_id}-{component}",
        blocks=tuple((name, blocks[name]) for name in COMPONENTS if name in blocks),
        evidence_urls=candidate.evidence_urls,
    )


def select_target_block(
    baseline: ExperimentReceipt, ablations: dict[str, ExperimentReceipt], task: TaskSpec
) -> str:
    """Select the block whose no-op ablation damages the metric most."""

    if baseline.metric_value is None:
        raise ValueError("Ablation needs a baseline metric.")
    valid = {name: receipt for name, receipt in ablations.items() if receipt.metric_value is not None}
    if not valid:
        raise RuntimeError("No ablation produced a metric.")
    if task.metric.greater_is_better:
        return min(valid, key=lambda name: float(valid[name].metric_value))
    return max(valid, key=lambda name: float(valid[name].metric_value))


@dataclass(frozen=True)
class RefinementResult:
    candidate: CandidateSpec
    receipt: ExperimentReceipt
    target_blocks: tuple[str, ...]
    ablations: tuple[ExperimentReceipt, ...]
    rejected_receipts: tuple[ExperimentReceipt, ...]


def refine_solution(
    task: TaskSpec,
    evaluator: CandidateEvaluator,
    planner: RefinementPlanner,
    candidate: CandidateSpec,
    receipt: ExperimentReceipt,
    *,
    outer_rounds: int = 4,
    inner_rounds: int = 4,
    seed: int,
) -> RefinementResult:
    """Run paper-style ablation outer loops and component-scoped inner loops."""

    if outer_rounds < 1 or inner_rounds < 1:
        raise ValueError("outer_rounds and inner_rounds must be positive.")
    current, current_receipt = candidate, receipt
    targets: list[str] = []
    ablation_history: list[ExperimentReceipt] = []
    rejected: list[ExperimentReceipt] = []
    for _ in range(outer_rounds):
        ablations: dict[str, ExperimentReceipt] = {}
        for component in COMPONENTS:
            if component not in dict(current.blocks):
                continue
            ablated = replace_block(current, component, "pass")
            item = evaluator.evaluate(
                ablated, phase="ablation", seed=seed, parent_experiment_id=current_receipt.experiment_id
            )
            ablations[component] = item
            ablation_history.append(item)
        target = select_target_block(current_receipt, ablations, task)
        targets.append(target)
        for source in planner.propose(component=target, candidate=current, history=tuple(ablation_history))[:inner_rounds]:
            proposed = replace_block(current, target, source)
            attempted = evaluator.evaluate(
                proposed, phase="refinement", seed=seed, parent_experiment_id=current_receipt.experiment_id
            )
            if improves(attempted, current_receipt, task):
                current, current_receipt = proposed, attempted
            else:
                rejected.append(attempted)
    return RefinementResult(
        candidate=current,
        receipt=current_receipt,
        target_blocks=tuple(targets),
        ablations=tuple(ablation_history),
        rejected_receipts=tuple(rejected),
    )

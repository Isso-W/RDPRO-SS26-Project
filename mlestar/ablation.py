"""Real-metric component impact ranking and marker-based ablation helpers."""

from __future__ import annotations

from typing import Mapping

from .contracts import COMPONENT_NAMES
from .refinement import apply_component_patch


def rank_component_impact(
    baseline_score: float,
    ablation_scores: Mapping[str, float],
    *,
    greater_is_better: bool,
) -> str:
    """Select the ablation that hurts the parent score most."""

    unknown = set(ablation_scores).difference(COMPONENT_NAMES)
    if unknown:
        raise ValueError(f"Unknown ablation components: {sorted(unknown)}")
    if not ablation_scores:
        raise ValueError("At least one ablation score is required.")
    def impact(component: str) -> float:
        score = ablation_scores[component]
        return baseline_score - score if greater_is_better else score - baseline_score
    return max((component for component in COMPONENT_NAMES if component in ablation_scores), key=impact)


def make_ablation_source(source: str, component: str, no_op_body: str) -> str:
    """Create a candidate that changes exactly one marked component."""

    return apply_component_patch(source, component, no_op_body)

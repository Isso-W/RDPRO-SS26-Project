from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from .io_utils import slugify, write_json
from .planner import clone_spec_with_updates
from .schemas import (
    AblationPlan,
    AblationSummary,
    AblationTrial,
    AblationVariant,
    DatasetManifest,
    ExecutionResult,
    GeneratedProject,
    TrainingSpec,
)

_COMPONENT_POOLS = {
    "classification": ["transforms", "freeze_strategy", "optimizer/scheduler", "batch_size", "learning_rate"],
    "segmentation": ["transforms", "loss_fn", "freeze_strategy", "learning_rate", "image_size"],
    "detection": ["transforms", "learning_rate", "image_size", "freeze_strategy", "score_threshold"],
}


class AblationEngine:
    def build_plan(self, manifest: DatasetManifest, spec: TrainingSpec, baseline_run_id: str) -> AblationPlan:
        variants = self._variants_for_spec(manifest, spec)
        target_component = variants[0].component if variants else "none"
        return AblationPlan(
            baseline_run_id=baseline_run_id,
            target_component=target_component,
            variants=variants,
            edit_budget=len(variants),
            expected_signal="Find the single most promising component-level refinement.",
        )

    def run(
        self,
        project: GeneratedProject,
        spec: TrainingSpec,
        plan: AblationPlan,
        executor: Any,
        root_dir: str | Path,
        execution_mode: str,
    ) -> list[AblationTrial]:
        root = Path(root_dir)
        root.mkdir(parents=True, exist_ok=True)
        trials: list[AblationTrial] = []
        for index, variant in enumerate(plan.variants, start=1):
            variant_id = f"ablation_{index:02d}_{slugify(variant.component)}_{slugify(variant.label)}"
            variant_spec = clone_spec_with_updates(spec, variant.overrides)
            variant_spec_path = write_json(root / f"{variant_id}_spec.json", variant_spec)
            result = executor.run(
                project=project,
                spec_path=variant_spec_path,
                run_dir=root / variant_id,
                execution_mode=execution_mode,
            )
            trials.append(
                AblationTrial(
                    run_id=variant_id,
                    component=variant.component,
                    label=variant.label,
                    overrides=variant.overrides,
                    result=result,
                )
            )
        return trials

    def summarize(
        self,
        baseline_result: ExecutionResult,
        trials: list[AblationTrial],
        spec: TrainingSpec,
    ) -> AblationSummary:
        baseline_value = baseline_result.primary_metric_value or 0.0
        tested_variants: list[dict[str, Any]] = []
        winner: AblationTrial | None = None
        best_delta = 0.0

        for trial in trials:
            value = trial.result.primary_metric_value or 0.0
            delta = round(value - baseline_value, 4)
            tested_variants.append(
                {
                    "run_id": trial.run_id,
                    "component": trial.component,
                    "label": trial.label,
                    "metric": trial.result.primary_metric_name,
                    "metric_value": value,
                    "delta_vs_baseline": delta,
                    "overrides": trial.overrides,
                }
            )
            if delta > best_delta:
                best_delta = delta
                winner = trial

        if winner is None:
            return AblationSummary(
                best_component_to_change="none",
                evidence="No ablation variant outperformed the baseline.",
                tested_variants=tested_variants,
                winner_variant=None,
                recommended_edit_region="none",
            )

        region = self._component_to_region(winner.component)
        evidence = (
            f"Best ablation improved {winner.result.primary_metric_name} by {best_delta:.4f} "
            f"through component '{winner.component}' using variant '{winner.label}'."
        )
        return AblationSummary(
            best_component_to_change=winner.component,
            evidence=evidence,
            tested_variants=tested_variants,
            winner_variant={
                "run_id": winner.run_id,
                "component": winner.component,
                "label": winner.label,
                "overrides": winner.overrides,
                "metric_value": winner.result.primary_metric_value,
            },
            recommended_edit_region=region,
        )

    def apply_summary(self, spec: TrainingSpec, summary: AblationSummary) -> TrainingSpec:
        if not summary.winner_variant:
            return spec
        overrides = summary.winner_variant.get("overrides", {})
        return clone_spec_with_updates(spec, overrides)

    def _variants_for_spec(self, manifest: DatasetManifest, spec: TrainingSpec) -> list[AblationVariant]:
        image_size = manifest.image_size_hint or 224
        if manifest.task_family == "classification":
            return [
                AblationVariant("transforms", "stronger_aug", {"transforms": [f"resize:{image_size}", "center_crop", "horizontal_flip", "normalize", "randaugment"]}),
                AblationVariant("freeze_strategy", "freeze_backbone", {"freeze_strategy": "freeze_backbone"}),
                AblationVariant("optimizer/scheduler", "adamw_cosine", {"optimizer": {"name": "adamw", "learning_rate": 3e-4, "weight_decay": 1e-4}, "scheduler": {"name": "cosine", "warmup_epochs": 1}}),
                AblationVariant("batch_size", "batch_32", {"batch_size": 32}),
                AblationVariant("learning_rate", "lr_3e4", {"optimizer.learning_rate": 3e-4}),
            ]
        if manifest.task_family == "segmentation":
            return [
                AblationVariant("transforms", "random_crop", {"transforms": [f"resize:{image_size}", "random_crop", "horizontal_flip", "normalize"]}),
                AblationVariant("loss_fn", "dice_ce", {"loss_fn": "dice_ce"}),
                AblationVariant("freeze_strategy", "freeze_backbone", {"freeze_strategy": "freeze_backbone"}),
                AblationVariant("learning_rate", "lr_2e4", {"optimizer.learning_rate": 2e-4}),
                AblationVariant("image_size", "resize_512", {"transforms": ["resize:512", "random_crop", "horizontal_flip", "normalize"]}),
            ]
        return [
            AblationVariant("transforms", "flip_only", {"transforms": [f"resize:{image_size}", "horizontal_flip", "normalize"]}),
            AblationVariant("learning_rate", "lr_1e4", {"optimizer.learning_rate": 1e-4}),
            AblationVariant("image_size", "resize_640", {"transforms": ["resize:640", "horizontal_flip", "normalize"]}),
            AblationVariant("freeze_strategy", "freeze_backbone", {"freeze_strategy": "freeze_backbone"}),
            AblationVariant("score_threshold", "threshold_025", {"optimizer.score_threshold": 0.25}),
        ]

    @staticmethod
    def _component_to_region(component: str) -> str:
        mapping = {
            "transforms": "training_spec.transforms",
            "freeze_strategy": "training_spec.freeze_strategy",
            "optimizer/scheduler": "training_spec.optimizer+training_spec.scheduler",
            "batch_size": "training_spec.batch_size",
            "learning_rate": "training_spec.optimizer.learning_rate",
            "loss_fn": "training_spec.loss_fn",
            "image_size": "training_spec.transforms",
            "score_threshold": "training_spec.optimizer.score_threshold",
        }
        return mapping.get(component, "training_spec")

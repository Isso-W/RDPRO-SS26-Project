"""Paired baseline-versus-MLE-STAR comparisons for runnable adapters."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Sequence

import pandas as pd

from benchmarks.catalog import get_task

from .adapters.tabular import LeafClassificationAdapter
from .ensemble import select_ensemble
from .initialization import CandidateSpec, initialize_solution
from .refinement import RefinementPlanner, refine_solution


def _candidate(candidate_id: str, model: str) -> CandidateSpec:
    return CandidateSpec(candidate_id, (("model", model),))


class _LeafPlanner:
    def propose(self, *, component, candidate, history):
        del component, history
        return ("random_forest",) if candidate.block("model") == "extra_trees" else ("extra_trees",)


def _summary(rows: list[dict[str, object]]) -> dict[str, dict[str, float | int]]:
    output: dict[str, dict[str, float | int]] = {}
    seed_count = len({int(row["seed"]) for row in rows})
    for arm in ("baseline", "mlestar_initial", "mlestar_refined", "mlestar_ensemble"):
        values = [float(row["metric_value"]) for row in rows if row["arm"] == arm and row["metric_value"] is not None]
        output[arm] = {
            "mean": mean(values) if values else float("nan"),
            "sem": stdev(values) / len(values) ** 0.5 if len(values) > 1 else 0.0,
            "wins": 0,
            "failures": seed_count - len(values),
        }
    baseline = {int(row["seed"]): float(row["metric_value"]) for row in rows if row["arm"] == "baseline" and row["metric_value"] is not None}
    for arm in ("mlestar_initial", "mlestar_refined", "mlestar_ensemble"):
        output[arm]["wins"] = sum(
            float(row["metric_value"]) < baseline[int(row["seed"])]
            for row in rows
            if row["arm"] == arm and row["metric_value"] is not None and int(row["seed"]) in baseline
        )
    return output


def compare(
    *, benchmark: str, data_root: str | Path, run_root: str | Path, seeds: Sequence[int] = (13, 29, 47),
    outer_rounds: int = 1, inner_rounds: int = 1,
) -> dict[str, object]:
    """Run paired real Leaf baseline/initial/refinement/OOF-ensemble arms.

    Other catalog entries deliberately fail loudly until their modality adapter
    is installed; no synthetic metric is reported for an unavailable task.
    """

    task = get_task(benchmark)
    if benchmark != "leaf_classification":
        raise NotImplementedError(
            f"{benchmark} requires its {task.modality} adapter; use the catalog for a schema-only preflight."
        )
    root = Path(run_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    receipts: list[dict[str, object]] = []
    for seed in seeds:
        seed_root = root / f"seed_{seed}"
        seed_root.mkdir(parents=True, exist_ok=True)
        adapter = LeafClassificationAdapter(data_root, seed_root, task)
        baseline_candidate = _candidate("extra_trees", "extra_trees")
        baseline_run = adapter.run(baseline_candidate, phase="baseline", seed=seed)
        initial = initialize_solution(
            task, adapter, (baseline_candidate, _candidate("random_forest", "random_forest")), seed=seed
        )
        initial_run = adapter.run(initial.best, phase="initial_selected", seed=seed)
        refined = refine_solution(
            task, adapter, _LeafPlanner(), initial.best, initial.best_receipt,
            outer_rounds=outer_rounds, inner_rounds=inner_rounds, seed=seed,
        )
        refined_run = adapter.run(refined.candidate, phase="refined_selected", seed=seed)
        ensemble = select_ensemble(
            {
                "baseline": (range(len(baseline_run.y_true)), baseline_run.oof),
                "refined": (range(len(refined_run.y_true)), refined_run.oof),
            },
            baseline_run.y_true,
            task.metric,
        )
        arm_values = {
            "baseline": baseline_run.receipt.metric_value,
            "mlestar_initial": initial_run.receipt.metric_value,
            "mlestar_refined": refined_run.receipt.metric_value,
            "mlestar_ensemble": ensemble.score.value,
        }
        rows.extend({"seed": seed, "arm": arm, "metric_value": value} for arm, value in arm_values.items())
        receipts.extend(
            asdict(receipt)
            for receipt in (
                baseline_run.receipt, *initial.receipts, *initial.merge_receipts,
                initial_run.receipt, *refined.ablations, *refined.rejected_receipts,
                refined.receipt, refined_run.receipt,
            )
        )
    report: dict[str, object] = {
        "benchmark": benchmark,
        "metric": task.metric.name,
        "paired_folds": True,
        "seeds": list(seeds),
        "arms": ["baseline", "mlestar_initial", "mlestar_refined", "mlestar_ensemble"],
        "summary": _summary(rows),
        "status": "offline_oof_complete",
    }
    pd.DataFrame(rows).to_csv(root / "comparison.csv", index=False)
    (root / "comparison.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    (root / "receipts.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in receipts), encoding="utf-8")
    return report

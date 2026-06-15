"""Stage 2: retrieve local knowledge and run at most three controlled experiments."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from .mcp_client import call_tool, mcp_session


async def run_experiment_loop(
    *,
    project_dir: str,
    baseline_config: dict,
    baseline_metrics: dict,
    fingerprint: dict,
    dataset_id: str = "dog_breed",
    execute: bool = True,
) -> dict:
    async with mcp_session() as session:
        cards = await call_tool(
            session,
            "search_strategy_cards",
            {
                "query": (
                    "fine grained dog breed partial finetune higher resolution "
                    "label smoothing TTA log loss"
                ),
                "task_type": "classification",
                "domain": "fine_grained_classification",
                "target_metric": "log_loss",
                "top_k": 5,
            },
        )
        history = await call_tool(
            session,
            "get_past_experiments",
            {"dataset_id": dataset_id, "top_k": 10},
        )
        proposals = await call_tool(
            session,
            "generate_experiment_configs",
            {
                "baseline_config": baseline_config,
                "strategy_cards": cards,
                "past_experiments": history,
                "max_experiments": 3,
                "max_changed_variables": 2,
            },
        )
        metrics = []
        runs = []
        if execute:
            for proposal in proposals:
                run = await call_tool(
                    session,
                    "run_experiment",
                    {
                        "project_dir": project_dir,
                        "experiment_name": proposal["experiment_name"],
                        "config": proposal["config"],
                    },
                )
                measured = await call_tool(session, "read_metrics", {"run_result": run})
                await call_tool(
                    session,
                    "write_experiment_result",
                    {
                        "dataset_id": dataset_id,
                        "fingerprint": fingerprint,
                        "proposal": proposal,
                        "metrics": measured,
                        "baseline_metric": baseline_metrics.get("metric_value"),
                    },
                )
                runs.append(run)
                metrics.append(measured)
        comparison = await call_tool(
            session,
            "compare_results",
            {
                "baseline_metrics": baseline_metrics,
                "experiment_metrics": metrics,
                "target_metric": "log_loss",
            },
        )
        return {
            "strategy_cards": cards,
            "past_experiments": history,
            "proposals": proposals,
            "runs": runs,
            "metrics": metrics,
            "comparison": comparison,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the low-token MLE experiment agent.")
    parser.add_argument("--project-dir", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--baseline-metrics", required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--dataset-id", default="dog_breed")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    load = lambda path: json.loads(Path(path).read_text(encoding="utf-8"))
    result = asyncio.run(
        run_experiment_loop(
            project_dir=args.project_dir,
            baseline_config=load(args.baseline),
            baseline_metrics=load(args.baseline_metrics),
            fingerprint=load(args.fingerprint),
            dataset_id=args.dataset_id,
            execute=not args.plan_only,
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

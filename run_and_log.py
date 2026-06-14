"""Generate -> train -> log: a real run that feeds the recommender's outcome memory.

Each call runs the full pipeline (generate a real-training project), trains it, parses
the run summary, and appends the (dataset fingerprint, config, achieved metric) to the
outcome memory — so the recommender gets better the more this is used.

  python run_and_log.py --dataset dpdl-benchmark/cassava \
    --query "Classify cassava leaf images, balance accuracy and speed" --epochs 12
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from module4_agent.result_parser import extract_last_json


def _project_config(project: Path) -> dict:
    import json
    cfg = json.loads((project / "configs.json").read_text(encoding="utf-8"))[0]
    flat = dict(cfg)
    mc = cfg.get("model_config")
    if isinstance(mc, dict):
        for k, v in mc.items():
            if v is not None or k not in flat:
                flat[k] = v
    keep = ("backbone", "pretrained_hf_id", "pretrained_name", "finetune_strategy",
            "image_size", "params_M", "num_classes", "loss", "optimizer")
    return {k: flat.get(k) for k in keep if flat.get(k) is not None}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the pipeline, train, and log the outcome.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--query", default="Classify images, balancing accuracy and training speed.")
    parser.add_argument("--subset", default=None)
    parser.add_argument("--output", default="./run_and_log_out")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--memory", default=None, help="Outcome-memory JSONL (default: recommender/outcomes.jsonl).")
    parser.add_argument("--use-recommender", action="store_true", help="Also re-rank with the recommender.")
    parser.add_argument("--llm-provider", default=None, choices=["none", "qwen", "openai", "vertex"])
    args = parser.parse_args()

    import cost_meter
    from pipeline import parse_dataset_id, run_pipeline
    from module4_agent.result_parser import extract_last_json
    from recommender import OutcomeMemory, log_from_summary

    dataset_id, parsed_subset = parse_dataset_id(args.dataset)
    subset = args.subset or parsed_subset
    out = Path(args.output).resolve()

    cost_meter.reset()   # start timing + counting LLM calls/tokens for this run
    print("[run+log] Generating real-training project via the pipeline ...")
    result = run_pipeline(
        args.query, dataset_id, fmt="nl", subset=subset,
        module4_output=out, module4_real_training=True, module4_skip_smoke=True,
        module4_llm_provider=args.llm_provider,
        use_recommender=args.use_recommender, recommender_memory=args.memory,
    )
    if result.get("module3_input") is None:
        print("[run+log] Module 1 failed; aborting.", file=sys.stderr)
        return 1

    project = out
    print(f"[run+log] Training (epochs={args.epochs}) ...")
    completed = subprocess.run(
        [sys.executable, "-u", "run.py", "--epochs", str(args.epochs)],
        cwd=str(project), text=True, capture_output=True,
    )
    print(completed.stdout)
    if completed.stderr:
        print(completed.stderr, file=sys.stderr)
    completed.check_returncode()
    cost_meter.record_training(epochs=args.epochs, runs=1)
    cost = cost_meter.report()

    summary = extract_last_json(completed.stdout)
    if not summary or "evaluate" not in summary:
        print("[run+log] Could not parse a run summary; nothing logged.", file=sys.stderr)
        return 2

    memory = OutcomeMemory(args.memory) if args.memory else OutcomeMemory()
    fingerprint = log_from_summary(
        summary, result["m2_report"], result["module3_input"],
        config=_project_config(project), dataset_id=dataset_id, memory=memory, cost=cost,
    )
    if fingerprint is None:
        print("[run+log] No usable metric in the summary; nothing logged.", file=sys.stderr)
        return 2

    metric = summary["evaluate"]
    print(f"\n[run+log] Logged outcome: {result['module3_input'].get('task_type')} "
          f"{fingerprint.get('num_classes')} classes | "
          f"{metric.get('metric_name')}={metric.get('metric_value')} -> {memory.path}")
    print(f"[run+log] Cost: {cost['llm_calls']} LLM calls, {cost['llm_tokens']} tokens, "
          f"{cost['epochs']} epochs, {cost['wall_clock_sec']}s wall-clock.")
    print(f"[run+log] Memory now holds {len(memory.all())} outcome(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

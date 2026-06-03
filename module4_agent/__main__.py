"""CLI entry point for Module 4."""

from __future__ import annotations

import argparse
import json
import sys

from .workflow import run_workflow


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and smoke-test Module 4 training code.")
    parser.add_argument("--input", required=True, help="Path to Module 3 candidate JSON output.")
    parser.add_argument("--output", required=True, help="Directory for generated files.")
    parser.add_argument("--max-iter", type=int, default=2, help="Maximum generation/review iterations.")
    parser.add_argument("--timeout", type=int, default=60, help="Per-command smoke-test timeout in seconds.")
    parser.add_argument("--no-smoke", action="store_true", help="Generate and review files without running smoke tests.")
    parser.add_argument(
        "--run-refinement",
        action="store_true",
        help="Run baseline -> ablation -> targeted refinement after code review passes.",
    )
    parser.add_argument(
        "--max-refinement-iters",
        type=int,
        default=2,
        help="Maximum experiment refinement iterations when --run-refinement is enabled.",
    )
    parser.add_argument(
        "--improvement-threshold",
        type=float,
        default=0.01,
        help="Stop refinement once proxy improvement over baseline reaches this value.",
    )
    args = parser.parse_args()

    result = run_workflow(
        args.input,
        args.output,
        max_iter=args.max_iter,
        timeout=args.timeout,
        skip_smoke=args.no_smoke,
        run_refinement=args.run_refinement,
        max_refinement_iters=args.max_refinement_iters,
        improvement_threshold=args.improvement_threshold,
    )
    print(json.dumps(result.to_summary(), indent=2, sort_keys=True))
    return 0 if result.is_approved else 1


if __name__ == "__main__":
    sys.exit(main())

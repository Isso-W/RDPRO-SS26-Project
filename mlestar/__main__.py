"""Command-line entry point for the MLE-STAR package.

The workflow commands are added with the orchestrator.  Keeping this entry
point functional now lets users verify the installed package independently of
optional ML runtime dependencies.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jiaozi-mlestar",
        description="MLE-STAR DataOps workflow utilities.",
    )
    parser.add_argument("--version", action="store_true", help="print the package entry-point version")
    subparsers = parser.add_subparsers(dest="command")
    run = subparsers.add_parser("run", help="generate and execute an MLE-STAR candidate wave")
    run.add_argument("--task", required=True, help="TaskContract JSON path")
    run.add_argument("--data-root", required=True, help="read-only task data directory")
    run.add_argument("--run-dir", required=True, help="directory for generated projects and artifacts")
    run.add_argument("--initial-candidates", type=int, default=1)
    run.add_argument("--plan-only", action="store_true")
    run.add_argument("--timeout-seconds", type=float, default=30 * 60)
    run.add_argument("--llm-provider", choices=("none", "qwen", "openai"), default="none")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the minimal package CLI and return a conventional exit status."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.version:
        print("mlestar 0.1.0")
    elif args.command == "run":
        from .workflow import run_mlestar
        from .generation import ConfiguredGenerationProvider

        report = run_mlestar(
            task_path=Path(args.task),
            data_root=Path(args.data_root),
            run_dir=Path(args.run_dir),
            initial_candidates=args.initial_candidates,
            plan_only=args.plan_only,
            timeout_seconds=args.timeout_seconds,
            generator=None if args.llm_provider == "none" else ConfiguredGenerationProvider(args.llm_provider),
        )
        print(report["run_dir"])
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by Python's module runner.
    raise SystemExit(main())

"""Command line entry point for the MLE-STAR experiment package."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json

from benchmarks.catalog import BENCHMARKS

from . import __version__
from .experiment import compare


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mlestar", description="MLE-STAR Kaggle benchmark experiments.")
    parser.add_argument("--version", action="store_true")
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("benchmarks", help="list benchmark contracts")
    compare_parser = commands.add_parser("compare", help="run a paired offline OOF comparison")
    compare_parser.add_argument("--benchmark", required=True, choices=tuple(BENCHMARKS))
    compare_parser.add_argument("--data-root", required=True)
    compare_parser.add_argument("--run-root", required=True)
    compare_parser.add_argument("--seeds", type=int, nargs="+", default=[13, 29, 47])
    compare_parser.add_argument("--outer-rounds", type=int, default=1)
    compare_parser.add_argument("--inner-rounds", type=int, default=1)
    compare_parser.add_argument("--no-submit", action="store_true", help="accepted for explicit safe invocation; submissions are never made")
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
    elif args.command == "benchmarks":
        print("\n".join(BENCHMARKS))
    elif args.command == "compare":
        report = compare(
            benchmark=args.benchmark, data_root=args.data_root, run_root=args.run_root,
            seeds=tuple(args.seeds), outer_rounds=args.outer_rounds, inner_rounds=args.inner_rounds,
        )
        print(json.dumps(report, sort_keys=True))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

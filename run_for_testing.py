"""
Jiaozi test runner — provide a dataset and a query to run the full pipeline.

Usage:
    python run_for_testing.py --dataset uoft-cs/cifar10 --query "classify images on mobile device"

    # Local image folder (imagefolder layout, one subfolder per class):
    python run_for_testing.py --dataset ./my_images --query "detect vehicles in images"

    # Also generate Module 4 training code:
    python run_for_testing.py --dataset uoft-cs/cifar10 --query "..." --module4

Results are saved under test_runs/<timestamp>/:
    run_info.json         Run parameters
    module3_input.json    Merged Module 1+2 input for Module 3
    recommendations.json  Top 3 model recommendations (with score details)
    task_lists.json       Module 4 task lists
    module4_code/         (with --module4) Generated train/eval/inference code

Prerequisites:
    Copy .env.example to .env and fill in your API key (Module 1 needs an LLM).
    This script auto-loads .env; no manual export needed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def load_env_file(path: Path) -> bool:
    if not path.is_file():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))
    return True


def check_llm_config() -> list[str]:
    problems = []
    provider = os.getenv("JIAOZI_LLM_PROVIDER", os.getenv("M1_LLM_PROVIDER", "qwen")).strip().lower()
    if provider == "qwen":
        key = os.getenv("JIAOZI_DASHSCOPE_API_KEY", "")
        if not key or key.startswith("replace_with"):
            problems.append(
                "JIAOZI_DASHSCOPE_API_KEY is not set or still a placeholder. "
                "Copy .env.example to .env and fill in your DashScope key."
            )
    elif provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "")
        if not key or key.startswith("replace_with"):
            problems.append("OPENAI_API_KEY is not set or still a placeholder; fill it in .env.")
    else:
        problems.append(f"Unknown JIAOZI_LLM_PROVIDER={provider!r} (supported: qwen / openai).")
    return problems


def resolve_dataset(arg: str) -> str:
    local = Path(arg)
    if local.exists():
        if not local.is_dir():
            sys.exit(f"[Error] {arg} is a file, not a directory. "
                     "Local datasets must be an imagefolder directory (one subfolder per class).")
        print(f"[Tester] Detected local dataset folder: {local.resolve()}")
        print("[Tester] Note: folder must use imagefolder layout (one subfolder per class).")
        return local.resolve().as_posix()
    if "/" not in arg:
        print(f"[Tester] Hint: {arg!r} is not a local path; treating as HuggingFace dataset ID "
              "(typically org/name, e.g. uoft-cs/cifar10).")
    return arg


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def print_summary(recommendations: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("Final Recommendations (Top 3)")
    print("=" * 70)
    if not recommendations:
        print("No recommendations — constraints may be too strict (e.g. zero_shot) and filtered all candidates.")
        return
    for i, rec in enumerate(recommendations, 1):
        detail = rec.get("score_detail", {})
        print(f"\n#{i}  {rec.get('backbone')}   score={rec.get('score')} "
              f"(structured={detail.get('structured')} / vector={detail.get('vector')})")
        print(f"    checkpoint: {rec.get('pretrained') or 'None (train from scratch)'}")
        print(f"    head: {rec.get('head')}   loss: {rec.get('loss')}   optimizer: {rec.get('optimizer')}")
        print(f"    finetune: {rec.get('finetune_strategy')}   freeze_viable: {rec.get('freeze_viable')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jiaozi test entry: dataset + query -> Top 3 model recommendations (optional code generation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True,
                        help="HuggingFace dataset ID (e.g. uoft-cs/cifar10); supports org/name:subset format")
    parser.add_argument("--subset", default=None,
                        help="Dataset config/subset name (required for multi-config datasets; or use --dataset org/name:subset)")
    parser.add_argument("--query", required=True, help="Natural language task description")
    parser.add_argument("--output", default=None,
                        help="Output directory (default: test_runs/<timestamp>)")
    parser.add_argument("--module4", action="store_true",
                        help="Also run Module 4 to generate training code (slower)")
    parser.add_argument("--no-smoke", action="store_true",
                        help="Module 4: generate only, skip smoke tests (faster)")
    parser.add_argument("--fmt", default="nl", choices=["structured", "nl"],
                        help="Module 4 task list format")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))

    if load_env_file(REPO_ROOT / ".env"):
        print("[Tester] Loaded .env")
    else:
        print("[Tester] No .env found (relying on existing environment variables)")

    problems = check_llm_config()
    if problems:
        print("\n[Error] LLM config check failed:")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)

    from pipeline import parse_dataset_id

    raw_dataset = resolve_dataset(args.dataset)
    dataset_id, parsed_subset = parse_dataset_id(raw_dataset)
    subset = args.subset or parsed_subset

    output_dir = Path(args.output) if args.output else \
        REPO_ROOT / "test_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Tester] Results will be saved to: {output_dir}")

    save_json(output_dir / "run_info.json", {
        "query": args.query,
        "dataset": args.dataset,
        "subset": subset,
        "resolved_dataset": dataset_id,
        "module4": args.module4,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    })

    from pipeline import run_pipeline

    try:
        result = run_pipeline(
            args.query,
            dataset_id,
            fmt=args.fmt,
            subset=subset,
            module4_output=(output_dir / "module4_code") if args.module4 else None,
            module4_skip_smoke=args.no_smoke,
        )
    except FileNotFoundError as e:
        traceback.print_exc()
        sys.exit(f"\n[Error] Failed to load dataset: {e}\n"
                 f"  - For HuggingFace IDs, check spelling (format: org/name) and network access\n"
                 f"  - For local folders, verify the path exists and uses imagefolder layout")
    except Exception as e:
        traceback.print_exc()
        hint = ""
        msg = str(e).lower()
        if "401" in msg or "authentication" in msg or "api key" in msg:
            hint = "\n  Likely an API key issue — check the key in your .env file."
        elif "connect" in msg or "timeout" in msg or "resolve" in msg:
            hint = "\n  Likely a network issue (dataset download or LLM call failed) — check your connection."
        sys.exit(f"\n[Error] Pipeline failed: {e}{hint}\n  See traceback above for details.")

    if not result.get("module3_input"):
        sys.exit("\n[Error] Module 1 parsing failed (empty result) — check your API key or try rephrasing the query.")

    save_json(output_dir / "module3_input.json", result["module3_input"])
    save_json(output_dir / "recommendations.json", result["recommendations"])
    save_json(output_dir / "task_lists.json", result["task_lists"])

    print_summary(result["recommendations"])

    if result.get("module4"):
        print(f"\nModule 4 code generated to: {result['module4']['output_dir']}")
        save_json(output_dir / "module4_summary.json", result["module4"]["summary"])

    print(f"\nAll results saved to: {output_dir}")
    print("To report issues, zip and share the entire output directory.")


if __name__ == "__main__":
    main()

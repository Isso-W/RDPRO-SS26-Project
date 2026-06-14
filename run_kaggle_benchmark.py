"""Run a Kaggle competition through Jiaozi's complete Module 1-4 pipeline.

Flow:
    ingest competition data (Kaggle API)
      -> read CSV labels -> Module 3 input (data_size / class_imbalance / num_classes)
      -> Module 3 retrieval (our backbone/head/loss/checkpoint selection)
      -> Module 4 code generation, configured for real training on the local Kaggle CSV

It stops after generating the project. Train it with:
    cd <output>/module4_code && python -u run.py --epochs N
Then predict + submit with kaggle_submit.py.

Prereqs: Kaggle credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY) and
having accepted the competition rules. See docs/usage_guide.md.

Usage:
    python run_kaggle_benchmark.py cassava --data-root ./kaggle_data --output ./kaggle_run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def prepare_project(
    benchmark_key: str,
    data_root: str | Path,
    output_dir: str | Path,
    priority: str = "balanced",
    llm_provider: str | None = None,
    force_download: bool = False,
) -> dict:
    del priority
    from pipeline import run_kaggle_pipeline

    return run_kaggle_pipeline(
        benchmark_key,
        data_root,
        module4_output=Path(output_dir) / "module4_code",
        force_download=force_download,
        module4_llm_provider=llm_provider,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a Kaggle benchmark through Module 3 + Module 4.")
    parser.add_argument("benchmark", help="Catalog key, e.g. cassava / state_farm / siim_isic")
    parser.add_argument("--data-root", default="./kaggle_data", help="Where to download/extract.")
    parser.add_argument("--output", default="./kaggle_run", help="Where to write the generated project.")
    parser.add_argument("--priority", default="balanced", choices=["speed", "accuracy", "balanced"],
                        help="Module 3 priority (balanced favours a finetuneable CNN).")
    parser.add_argument("--llm-provider", default=None,
                        choices=["none", "qwen", "openai", "vertex"],
                        help="Module 4 model.py provider; defaults to env var or template.")
    parser.add_argument("--force-download", action="store_true", help="Re-download even if present.")
    args = parser.parse_args()

    result = prepare_project(
        args.benchmark,
        args.data_root,
        args.output,
        priority=args.priority,
        llm_provider=args.llm_provider,
        force_download=args.force_download,
    )

    project = Path(result["module4"]["output_dir"])
    info = result["info"]
    print("\n" + "=" * 70)
    print(f"Generated project: {project}")
    print("Next steps:")
    print(f"  1) Train:   cd {project} && python -u run.py --epochs 15")
    print(f"  2) Submit:  python kaggle_submit.py {args.benchmark} \\")
    print(f"                  --project {project} --data-root {args.data_root}")
    print("=" * 70)
    print(json.dumps(result["module4"]["summary"], indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

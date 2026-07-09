"""Run a Kaggle competition benchmark through Jiaozi's own Module 3 selection.

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
    from ingestion.kaggle_loader import ingest_benchmark, build_module3_input

    info = ingest_benchmark(benchmark_key, data_root, force=force_download)
    m3_input = build_module3_input(info, priority=priority)
    print(
        f"[kaggle] Module 3 input: data_size={m3_input['data_size']} "
        f"num_classes={m3_input['num_classes']} "
        f"class_imbalance={m3_input['constraints'].get('class_imbalance')}"
    )

    from retrieval.rag_retrieval import (
        build_graph,
        build_vector_index,
        retrieve_top3_hybrid,
        build_all_task_lists,
        print_results,
    )

    graph = build_graph()
    col = build_vector_index()
    recommendations = retrieve_top3_hybrid(m3_input, graph, col)
    print_results(m3_input, recommendations, graph)

    task_lists = build_all_task_lists(recommendations, graph, fmt="nl", input_json=m3_input)

    from pipeline import derive_recommended_epochs

    # Inject the local Kaggle data + real-training settings into every candidate config.
    for task_list in task_lists:
        mc = task_list.get("model_config")
        if not isinstance(mc, dict):
            continue
        mc.setdefault("num_classes", m3_input["num_classes"])
        mc["train_csv"] = info["train_csv"]
        mc["image_dir"] = info["image_dir"]
        mc["image_column"] = info["image_column"]
        mc["label_column"] = info["label_column"]
        mc["image_path_template"] = info["image_path_template"]
        mc["image_extension"] = info["image_extension"]
        mc["evaluation_metric"] = info.get("metric") or "accuracy"
        mc["offline_smoke"] = False
        mc.setdefault(
            "recommended_epochs",
            derive_recommended_epochs(
                m3_input["data_size"],
                mc.get("finetune_strategy"),
                bool(mc.get("pretrained_hf_id")),
            ),
        )

    from pipeline import run_module4_generation

    module4 = run_module4_generation(
        task_lists,
        Path(output_dir) / "module4_code",
        skip_smoke=True,           # real-training project; local smoke is meaningless
        llm_provider=llm_provider,
    )

    from kaggle_orchestrator import write_run_manifest

    run_manifest = write_run_manifest(
        output_dir,
        benchmark_key=benchmark_key,
        info=info,
        module3_input=m3_input,
        recommendations=recommendations,
        module4=module4,
    )

    return {
        "info": info,
        "module3_input": m3_input,
        "recommendations": recommendations,
        "module4": module4,
        "run_manifest": str(run_manifest),
    }


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
    print(f"Run manifest: {result['run_manifest']}")
    print("Next steps:")
    print(f"  1) Train:   cd {project} && python -u run.py --epochs 15")
    print(f"  2) Submit:  python kaggle_submit.py {args.benchmark} \\")
    print(f"                  --project {project} --data-root {args.data_root}")
    print("=" * 70)
    print(json.dumps(result["module4"]["summary"], indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

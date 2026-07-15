"""
Integrated pipeline: Module 1 + Module 2 + Module 3.

Usage:
    python pipeline.py --dataset uoft-cs/cifar10 --query "classify images on mobile device"

Flow:
    user request ──→ Module 1 ──→ task_type / priority / constraints
    dataset ID   ──→ Module 2 ──→ data_size / class_imbalance
    merged input ──→ Module 3 ──→ top-3 model recommendations
    optional     ──→ Module 4 ──→ training/evaluation/inference code
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from recipe.tables import derive_recommended_epochs

# ═══════════════════════════════════════════════════════════════════════════════
# Module 2 to Module 3 field mapping.
# ═══════════════════════════════════════════════════════════════════════════════

# Total-image thresholds (small max, medium max). Total size is mostly a
# labeling/training-cost signal. Detection and segmentation have higher
# per-image annotation cost, so their thresholds are lower.
_TOTAL_THRESHOLDS = {
    "classification":     (3_000, 20_000),
    "feature_extraction": (3_000, 20_000),
    "object_detection":   (1_500, 10_000),
    "image_segmentation": (1_500, 10_000),
}
_DEFAULT_TOTAL_THRESHOLDS = (3_000, 20_000)

# Per-class thresholds (small max, medium max). This is used only for
# classification: 25k images over 200 classes is only 125 samples/class, so it
# behaves more like a small-data setting than a large one.
_PER_CLASS_THRESHOLDS = (100, 1_000)

_SIZE_ORDER = ["small", "medium", "large"]


def _tier(value: float, thresholds: tuple[float, float]) -> str:
    small_max, medium_max = thresholds
    if value <= small_max:
        return "small"
    if value <= medium_max:
        return "medium"
    return "large"


def derive_data_size(
    total_images: int,
    num_classes: int | None = None,
    task_type: str = "classification",
) -> str:
    """Infer data_size from image count and, for classification, class count.

    We use the more conservative of two signals:
      - total images: a cost-side signal
      - samples per class: an overfitting signal for classification
    """
    by_total = _tier(total_images, _TOTAL_THRESHOLDS.get(task_type, _DEFAULT_TOTAL_THRESHOLDS))

    if task_type == "classification" and num_classes and num_classes > 0:
        by_class = _tier(total_images / num_classes, _PER_CLASS_THRESHOLDS)
        return min(by_total, by_class, key=_SIZE_ORDER.index)

    return by_total


_IMBALANCE_RATIO_THRESHOLD = 10

def derive_class_imbalance(class_distribution: dict) -> bool:
    """Treat a class distribution as imbalanced when max/min exceeds the threshold."""
    if not class_distribution:
        return False
    counts = list(class_distribution.values())
    min_count = min(counts)
    if min_count == 0:
        return True
    return max(counts) / min_count > _IMBALANCE_RATIO_THRESHOLD


def derive_resolution_tier(stats: dict) -> str:
    """Derive low/medium/high from the average short side in Module 2 metadata."""
    try:
        avg_w = float(stats.get("avg_width", 0) or 0)
        avg_h = float(stats.get("avg_height", 0) or 0)
    except (TypeError, ValueError):
        return "medium"
    short_side = min(v for v in (avg_w, avg_h) if v > 0) if (avg_w > 0 or avg_h > 0) else 0
    if short_side <= 0:
        return "medium"
    if short_side < 256:
        return "low"
    if short_side >= 768:
        return "high"
    return "medium"


def derive_color_mode(stats: dict) -> str:
    """Derive rgb/grayscale from the dominant PIL mode in Module 2 metadata."""
    dist = stats.get("mode_distribution") or {}
    if not isinstance(dist, dict) or not dist:
        return "rgb"
    dominant = max(dist, key=dist.get)
    return "grayscale" if str(dominant).upper() in {"L", "1", "LA"} else "rgb"


def _patch_torch_metadata():
    """Patch missing torch package metadata before importing datasets."""
    import importlib.metadata
    if getattr(importlib.metadata.version, "_torch_patched", False):
        return
    _orig = importlib.metadata.version
    def _patched(name):
        v = _orig(name)
        if v is None and name == "torch":
            import torch
            return torch.__version__.split("+")[0]
        return v
    _patched._torch_patched = True
    importlib.metadata.version = _patched


def parse_dataset_id(raw: str) -> tuple[str, str | None]:
    """Parse an 'org/name:subset' string into (dataset_id, subset)."""
    if ":" in raw:
        dataset_id, subset = raw.rsplit(":", 1)
        return dataset_id, subset
    return raw, None


def run_module2_analysis(dataset_id: str, subset: str | None = None) -> dict:
    """Run the lightweight Module 2 analysis pass."""
    _patch_torch_metadata()
    from ingestion.image_loader import ImageLoader
    from analyzer.image_statistics import ImageStatisticsAnalyzer

    loader = ImageLoader()
    loaded = loader.load_dataset_by_name(dataset_id, subset=subset)
    dataset = loaded["dataset"]

    analyzer = ImageStatisticsAnalyzer()
    report = analyzer.analyze(dataset)
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Merge Module 1 and Module 2 outputs into Module 3 input.
# ═══════════════════════════════════════════════════════════════════════════════

def merge_modules(m1_output: dict, m2_report: dict) -> dict:
    """
    Merge Module 1 fields and Module 2 dataset statistics into the input format
    expected by Module 3's retrieve_top3_hybrid().

    Fields controlled by Module 2:
      - data_size: inferred from total_images and class count
      - num_classes: passed through for Module 4 head sizing
      - constraints.class_imbalance: OR of Module 1 intent and Module 2 stats
    """
    merged = dict(m1_output)
    # Copy nested constraints so the Module 1 output is not modified in place.
    merged["constraints"] = dict(m1_output.get("constraints", {}))

    class_dist = m2_report.get("class_distribution", {})
    num_classes = m2_report.get("num_classes") or len(class_dist) or None

    # Module 2 owns data_size through the total-count and per-class signals.
    total_images = m2_report.get("total_images", 0)
    merged["data_size"] = derive_data_size(
        total_images,
        num_classes=num_classes,
        task_type=merged.get("task_type", "classification"),
    )
    if num_classes:
        merged["num_classes"] = num_classes

    merged["data_stats"] = {
        "resolution_tier": derive_resolution_tier(m2_report),
        "color_mode": derive_color_mode(m2_report),
        "avg_width": m2_report.get("avg_width"),
        "avg_height": m2_report.get("avg_height"),
        "mode_distribution": m2_report.get("mode_distribution", {}),
        "format_distribution": m2_report.get("format_distribution", {}),
    }

    # Either a user requirement or the observed class distribution can enable
    # the imbalance constraint.
    m2_imbalance = derive_class_imbalance(class_dist)
    merged["constraints"]["class_imbalance"] = (
        merged["constraints"].get("class_imbalance", False) or m2_imbalance
    )

    return merged


def run_module4_generation(
    task_lists: list[dict],
    output_dir: str | Path,
    *,
    skip_smoke: bool = False,
    run_refinement: bool = False,
    timeout: int = 60,
    llm_provider: str | None = None,
) -> dict:
    """Generate Module 4 code from Module 3 task lists."""

    from module4_agent.workflow import run_workflow

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    module4_input = output_path / "module3_candidates.json"
    module4_input.write_text(
        json.dumps(task_lists, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    result = run_workflow(
        module4_input,
        output_path,
        timeout=timeout,
        skip_smoke=skip_smoke,
        run_refinement=run_refinement,
        llm_provider=llm_provider,
    )

    return {
        "input_path": str(module4_input),
        "output_dir": str(output_path),
        "summary": result.to_summary(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Full pipeline.
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    user_message: str,
    dataset_id: str,
    fmt: str = "structured",
    subset: str | None = None,
    module4_output: str | Path | None = None,
    module4_skip_smoke: bool = False,
    module4_run_refinement: bool = False,
    module4_timeout: int = 60,
    module4_llm_provider: str | None = None,
    module4_real_training: bool = False,
    use_recommender: bool = False,
    recommender_memory: str | None = None,
    use_recipe: bool = False,
) -> dict:
    """
    Main pipeline entry point.

    Returns:
      {
        "module3_input": dict,       # merged Module 3 input
        "recommendations": list,     # raw retrieve_top3_hybrid results
        "task_lists": list,          # Module 4 candidate list
        "module4": dict | None,      # optional Module 4 result
      }
    """
    # Step 1: Module 1 parses the natural-language request.
    print("[Pipeline] Module 1: Parsing user requirements...")
    from features_extraction_api import module1_pipeline

    m1_output = module1_pipeline(user_message)
    if m1_output is None:
        print("[Pipeline] Module 1 failed, cannot continue.")
        return {"module3_input": None, "recommendations": [], "task_lists": [], "module4": None}

    # Step 2: Module 2 extracts dataset-side signals.
    ds_label = f"{dataset_id}:{subset}" if subset else dataset_id
    print(f"[Pipeline] Module 2: Analyzing dataset {ds_label}...")
    m2_report = run_module2_analysis(dataset_id, subset=subset)

    # Step 3: merge the two views.
    m3_input = merge_modules(m1_output, m2_report)
    print(f"[Pipeline] Merged: task={m3_input['task_type']}  "
          f"size={m3_input['data_size']}  priority={m3_input['priority']}")

    # Step 4: Module 3 retrieves model configurations.
    print("[Pipeline] Module 3: Retrieving model configurations...")
    from retrieval.rag_retrieval import (
        build_graph, build_vector_index,
        retrieve_top3_hybrid, build_all_task_lists, print_results,
    )

    G = build_graph()
    col = build_vector_index()
    recommendations = retrieve_top3_hybrid(m3_input, G, col)

    # Print the recommendation summary.
    print_results(m3_input, recommendations, G)

    # Optional re-ranker that uses stored outcomes as memory.
    if use_recommender:
        from recommender import recommend, OutcomeMemory

        memory = OutcomeMemory(recommender_memory) if recommender_memory else OutcomeMemory()
        recommendations = recommend(recommendations, m2_report, m3_input, memory=memory)
        print("[Pipeline] Recommender re-ranked candidates:")
        for r in recommendations:
            print(f"    [{r.get('rank_basis')}] {r.get('backbone')} — {r.get('explanation')}")

    task_lists = build_all_task_lists(
        recommendations,
        G,
        fmt=fmt,
        input_json=m3_input,
        data_stats=m3_input.get("data_stats"),
    )
    module4_result = None

    if module4_output:
        print(f"[Pipeline] Module 4: Generating code to {module4_output}...")
        module4_task_lists = task_lists
        if fmt != "nl":
            module4_task_lists = build_all_task_lists(
                recommendations,
                G,
                fmt="nl",
                input_json=m3_input,
                data_stats=m3_input.get("data_stats"),
            )
        num_classes = m3_input.get("num_classes")
        for task_list in module4_task_lists:
            mc = task_list.get("model_config")
            if isinstance(mc, dict):
                if num_classes:
                    mc.setdefault("num_classes", num_classes)
                mc.setdefault("dataset_id", dataset_id)
                if subset:
                    mc.setdefault("dataset_subset", subset)
                # Module 1 picked the metric from the query (default accuracy); generated
                # train/evaluate already consume evaluation_metric.
                mc.setdefault("evaluation_metric", m3_input.get("evaluation_metric", "accuracy"))
                mc.setdefault("recommended_epochs", derive_recommended_epochs(
                    m3_input.get("data_size", "medium"),
                    mc.get("finetune_strategy"),
                    bool(mc.get("pretrained_hf_id")),
                ))
                # Legacy opt-in fallback: recipe is now produced by Module 3. Keep the
                # old recommender.recipe path only for task lists produced elsewhere.
                if use_recipe and "recipe" not in mc:
                    from recommender.recipe import recommend_recipe
                    hp = recommend_recipe(
                        backbone=mc.get("backbone"),
                        finetune_strategy=mc.get("finetune_strategy"),
                        data_size=m3_input.get("data_size", "medium"),
                        m2_report=m2_report,
                        task_type=m3_input.get("task_type", "classification"),
                    )
                    for key, value in hp.items():
                        mc.setdefault(key, value)
                if module4_real_training:
                    mc["offline_smoke"] = False
        module4_result = run_module4_generation(
            module4_task_lists,
            module4_output,
            skip_smoke=module4_skip_smoke,
            run_refinement=module4_run_refinement,
            timeout=module4_timeout,
            llm_provider=module4_llm_provider,
        )

    return {
        "module3_input":   m3_input,
        "m2_report":        m2_report,        # for the recommender fingerprint / outcome logging
        "recommendations": recommendations,
        "task_lists":       task_lists,
        "module4":          module4_result,
    }


def get_skrub_dag(
    user_message: str,
    dataset_id: str,
    subset: str | None = None,
):
    """Build the pipeline as a skrub DataOps DAG for graph visualisation.

    Returns a skrub deferred object — call ``.skb.describe_steps()`` for text
    or ``.skb.draw_graph()`` for SVG (needs graphviz binary).
    """
    from skrub_pipeline import build_pipeline

    return build_pipeline(user_message, dataset_id, subset=subset)


def main() -> int:
    parser = argparse.ArgumentParser(description="Jiaozi Pipeline: NL + Dataset → Model Recommendation")
    parser.add_argument("--query", required=True, help="Natural language task description")
    parser.add_argument("--dataset", required=True,
                        help="HuggingFace dataset ID; supports org/name:subset format")
    parser.add_argument("--subset", default=None,
                        help="Dataset config/subset name (or use --dataset org/name:subset)")
    parser.add_argument("--fmt", default="structured", choices=["structured", "nl"],
                        help="Module 4 task list format")
    parser.add_argument("--module4-output", default=None,
                        help="Optional: run Module 4 and write generated code to this directory")
    parser.add_argument("--module4-no-smoke", action="store_true",
                        help="Module 4: generate and lint only, skip local smoke tests")
    parser.add_argument("--module4-real-training", action="store_true",
                        help="Module 4: generate real training code (offline_smoke=false, auto skips smoke)")
    parser.add_argument("--module4-run-refinement", action="store_true",
                        help="Module 4: continue with refinement loop after approval")
    parser.add_argument("--module4-timeout", type=int, default=60,
                        help="Module 4: timeout per smoke command (seconds)")
    parser.add_argument("--module4-llm-provider", default=None,
                        choices=["none", "qwen", "openai", "vertex"],
                        help="Module 4 model.py provider (e.g. qwen); defaults to env var or template")
    parser.add_argument("--use-recommender", action="store_true",
                        help="Re-rank Module 3 candidates with the accumulating recommender (+ explanations)")
    parser.add_argument("--recommender-memory", default=None,
                        help="Path to the outcome-memory JSONL (default: recommender/outcomes.jsonl)")
    parser.add_argument("--use-recipe", action="store_true",
                        help="Inject recommended training hyperparameters (lr/image_size/aug/early-stop) from the recipe layer")
    args = parser.parse_args()

    dataset_id, parsed_subset = parse_dataset_id(args.dataset)
    subset = args.subset or parsed_subset

    # Real-training mode skips local synthetic smoke runs, matching run_for_testing.py.
    skip_smoke = args.module4_no_smoke or args.module4_real_training

    result = run_pipeline(
        args.query,
        dataset_id,
        fmt=args.fmt,
        subset=subset,
        module4_output=args.module4_output,
        module4_skip_smoke=skip_smoke,
        module4_run_refinement=args.module4_run_refinement,
        module4_timeout=args.module4_timeout,
        module4_llm_provider=args.module4_llm_provider,
        module4_real_training=args.module4_real_training,
        use_recommender=args.use_recommender,
        recommender_memory=args.recommender_memory,
        use_recipe=args.use_recipe,
    )

    if result["module3_input"] is None:
        print("[Pipeline] Module 1 failed, so no Module 3 or Module 4 output was produced.", file=sys.stderr)
        return 1

    print("\n═══ Module 4 Task Lists ═══")
    print(json.dumps(result["task_lists"], indent=2, ensure_ascii=False))
    if args.module4_output and not result["module4"]:
        print(
            "[Pipeline] Module 4 output was requested, but no generated summary was produced.",
            file=sys.stderr,
        )
        return 2

    if result["module4"]:
        print("\n═══ Module 4 Code Generation Summary ═══")
        print(json.dumps(result["module4"]["summary"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

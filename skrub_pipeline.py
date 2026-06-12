"""
Jiaozi pipeline expressed as a skrub DataOps DAG.

Usage — build, visualise, then run:

    from skrub_pipeline import build_pipeline

    pipeline = build_pipeline(
        user_message="classify images on a mobile device",
        dataset_id="uoft-cs/cifar10",
    )

    # Text description of the computational graph
    print(pipeline.skb.describe_steps())

    # SVG visualisation (needs graphviz system binary)
    pipeline.skb.draw_graph()

    # Execute the pipeline
    result = pipeline.skb.eval()
"""

from __future__ import annotations

import json
from pathlib import Path

import skrub


# ═══════════════════════════════════════════════════════════════════════════════
# Step functions — each becomes a node in the DAG
# ═══════════════════════════════════════════════════════════════════════════════

def module1_extract(user_message: str) -> dict:
    from features_extraction_api import module1_pipeline

    result = module1_pipeline(user_message)
    if result is None:
        raise RuntimeError("Module 1 failed — check LLM API key or rephrase the query")
    return result


def load_dataset(dataset_id: str, subset: str | None) -> dict:
    from pipeline import _patch_torch_metadata

    _patch_torch_metadata()
    from ingestion.image_loader import ImageLoader

    loader = ImageLoader()
    return loader.load_dataset_by_name(dataset_id, subset=subset)


def dataset_statistics(loaded: dict) -> dict:
    from analyzer.image_statistics import ImageStatisticsAnalyzer

    analyzer = ImageStatisticsAnalyzer()
    return analyzer._dataset_statistics(loaded["dataset"])


def metadata_statistics(loaded: dict) -> dict:
    from analyzer.image_statistics import ImageStatisticsAnalyzer

    analyzer = ImageStatisticsAnalyzer()
    return analyzer._metadata_statistics(loaded["dataset"])


def mode_distribution(loaded: dict) -> dict:
    from analyzer.image_statistics import ImageStatisticsAnalyzer

    analyzer = ImageStatisticsAnalyzer()
    return analyzer._mode_distribution(loaded["dataset"])


def format_distribution(loaded: dict) -> dict:
    from analyzer.image_statistics import ImageStatisticsAnalyzer

    analyzer = ImageStatisticsAnalyzer()
    return analyzer._format_distribution(loaded["dataset"])


def merge_analysis(stats: dict, metadata: dict, mode: dict, fmt: dict) -> dict:
    report = {}
    report.update(stats)
    report.update(metadata)
    report.update(mode)
    report.update(fmt)
    return report


def merge_m1_m2(m1_output: dict, m2_report: dict) -> dict:
    from pipeline import merge_modules

    return merge_modules(m1_output, m2_report)


def module3_retrieve(m3_input: dict) -> list:
    from retrieval.rag_retrieval import (
        build_graph,
        build_vector_index,
        retrieve_top3_hybrid,
    )

    G = build_graph()
    col = build_vector_index()
    return retrieve_top3_hybrid(m3_input, G, col)


def build_task_lists(recommendations: list) -> list:
    from retrieval.rag_retrieval import build_all_task_lists, build_graph

    G = build_graph()
    return build_all_task_lists(recommendations, G, fmt="nl")


# ═══════════════════════════════════════════════════════════════════════════════
# DAG builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_pipeline(
    user_message: str,
    dataset_id: str,
    subset: str | None = None,
):
    """
    Construct the Jiaozi pipeline as a skrub DataOps DAG.

    Returns a skrub deferred object.  Call ``.skb.describe_steps()`` for a
    text summary, ``.skb.draw_graph()`` for an SVG, or ``.skb.eval()`` to
    execute everything.
    """
    msg_var = skrub.var("user_message", value=user_message)
    ds_var = skrub.var("dataset_id", value=dataset_id)
    sub_var = skrub.var("subset", value=subset)

    # Module 1 — LLM extraction
    m1_output = skrub.deferred(module1_extract)(msg_var)

    # Module 2 — dataset analysis (branching sub-graph)
    loaded = skrub.deferred(load_dataset)(ds_var, sub_var)
    stats = skrub.deferred(dataset_statistics)(loaded)
    metadata = skrub.deferred(metadata_statistics)(loaded)
    mode_dist = skrub.deferred(mode_distribution)(loaded)
    fmt_dist = skrub.deferred(format_distribution)(loaded)
    m2_report = skrub.deferred(merge_analysis)(stats, metadata, mode_dist, fmt_dist)

    # Merge Module 1 + Module 2
    m3_input = skrub.deferred(merge_m1_m2)(m1_output, m2_report)

    # Module 3 — model retrieval
    recommendations = skrub.deferred(module3_retrieve)(m3_input)

    # Task list generation for Module 4
    task_lists = skrub.deferred(build_task_lists)(recommendations)

    return task_lists


def build_module2_pipeline(
    dataset_id: str,
    subset: str | None = None,
):
    """
    Build only the Module 2 (dataset analysis) sub-graph as a skrub DAG.

    Useful for demonstrating the branching structure of the analysis steps
    without needing an LLM API key.
    """
    ds_var = skrub.var("dataset_id", value=dataset_id)
    sub_var = skrub.var("subset", value=subset)

    loaded = skrub.deferred(load_dataset)(ds_var, sub_var)
    stats = skrub.deferred(dataset_statistics)(loaded)
    metadata = skrub.deferred(metadata_statistics)(loaded)
    mode_dist = skrub.deferred(mode_distribution)(loaded)
    fmt_dist = skrub.deferred(format_distribution)(loaded)
    m2_report = skrub.deferred(merge_analysis)(stats, metadata, mode_dist, fmt_dist)

    return m2_report


# ═══════════════════════════════════════════════════════════════════════════════
# CLI — quick demo
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Jiaozi skrub DataOps pipeline")
    parser.add_argument("--query", default=None, help="Natural language task description")
    parser.add_argument("--dataset", default=None, help="HuggingFace dataset ID")
    parser.add_argument("--subset", default=None, help="Dataset subset")
    parser.add_argument("--graph", action="store_true", help="Show computational graph (text)")
    parser.add_argument("--draw", action="store_true", help="Draw SVG graph (needs graphviz)")
    parser.add_argument("--run", action="store_true", help="Execute the pipeline")
    parser.add_argument("--module2-only", action="store_true",
                        help="Only build Module 2 sub-graph (no LLM needed)")
    args = parser.parse_args()

    if args.module2_only:
        ds = args.dataset or "uoft-cs/cifar10"
        dag = build_module2_pipeline(ds, args.subset)
        print("=== Module 2 DataOps DAG ===\n")
    else:
        query = args.query or "classify images"
        ds = args.dataset or "uoft-cs/cifar10"
        dag = build_pipeline(query, ds, args.subset)
        print("=== Full Pipeline DataOps DAG ===\n")

    if args.graph or (not args.draw and not args.run):
        print(dag.skb.describe_steps())

    if args.draw:
        drawing = dag.skb.draw_graph()
        svg_path = Path("pipeline_graph.svg")
        png_path = Path("pipeline_graph.png")
        svg_path.write_bytes(drawing.svg)
        png_path.write_bytes(drawing.png)
        print(f"\nGraph saved to {svg_path} and {png_path}")

    if args.run:
        print("\nExecuting pipeline...")
        result = dag.skb.eval()
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

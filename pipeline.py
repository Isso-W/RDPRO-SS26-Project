"""
Jiaozi 整合流水线：Module 1 + Module 2 + Module 3

用法:
    python pipeline.py --dataset uoft-cs/cifar10 --query "classify images on mobile device"

流程:
    用户自然语言 ──→ Module 1 ──→ task_type / priority / constraints
    数据集 ID    ──→ Module 2 ──→ data_size / class_imbalance
    合并         ──→ Module 3 ──→ Top 3 模型推荐
    可选         ──→ Module 4 ──→ 生成训练/评估/推理代码
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# Module 2 → Module 3 字段映射
# ═══════════════════════════════════════════════════════════════════════════════

# 总量阈值 (small 上限, medium 上限)：总量决定标注/训练成本。
# 检测和分割的单张标注成本是分类的 10-100 倍，阈值减半。
_TOTAL_THRESHOLDS = {
    "classification":     (3_000, 20_000),
    "feature_extraction": (3_000, 20_000),
    "object_detection":   (1_500, 10_000),
    "image_segmentation": (1_500, 10_000),
}
_DEFAULT_TOTAL_THRESHOLDS = (3_000, 20_000)

# 每类样本数阈值 (small 上限, medium 上限)：决定过拟合风险。
# 仅分类任务使用——25k 张图分 200 类只有 125 张/类，是小数据不是大数据。
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
    """从图片总数（+ 可选类别数）推断 data_size。

    双信号取更保守一档：
      - 总量档位：成本侧约束，总量不够大就不算大数据
      - 每类样本数档位（仅分类）：过拟合侧约束，类多样本摊薄也不算大数据
    """
    by_total = _tier(total_images, _TOTAL_THRESHOLDS.get(task_type, _DEFAULT_TOTAL_THRESHOLDS))

    if task_type == "classification" and num_classes and num_classes > 0:
        by_class = _tier(total_images / num_classes, _PER_CLASS_THRESHOLDS)
        return min(by_total, by_class, key=_SIZE_ORDER.index)

    return by_total


_RECOMMENDED_EPOCHS = {
    ("small",  "head_only"): 25,
    ("small",  "finetune"):  40,
    ("small",  "scratch"):   50,
    ("medium", "head_only"): 12,
    ("medium", "finetune"):  20,
    ("medium", "scratch"):   30,
    ("large",  "head_only"):  8,
    ("large",  "finetune"):  15,
    ("large",  "scratch"):   20,
}


def derive_recommended_epochs(
    data_size: str,
    finetune_strategy: str | None,
    use_pretrained: bool,
) -> int:
    """Recommend training epochs based on data size and training mode."""
    if not use_pretrained:
        mode = "scratch"
    elif finetune_strategy == "head_only":
        mode = "head_only"
    else:
        mode = "finetune"
    return _RECOMMENDED_EPOCHS.get((data_size, mode), 15)


_IMBALANCE_RATIO_THRESHOLD = 10

def derive_class_imbalance(class_distribution: dict) -> bool:
    """从类别分布推断是否存在类别不平衡。max/min 比值超过阈值视为不平衡。"""
    if not class_distribution:
        return False
    counts = list(class_distribution.values())
    min_count = min(counts)
    if min_count == 0:
        return True
    return max(counts) / min_count > _IMBALANCE_RATIO_THRESHOLD


def _patch_torch_metadata():
    """datasets 库在 import 时会读 torch 版本 metadata，某些 torch 安装方式下 metadata 缺失会导致崩溃。"""
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
    """解析 'org/name:subset' 格式，返回 (dataset_id, subset)。"""
    if ":" in raw:
        dataset_id, subset = raw.rsplit(":", 1)
        return dataset_id, subset
    return raw, None


def run_module2_analysis(dataset_id: str, subset: str | None = None) -> dict:
    """运行 Module 2 的轻量分析（只取统计信息，跳过标准化和特征提取）。"""
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
# 合并 Module 1 + Module 2 → Module 3 输入
# ═══════════════════════════════════════════════════════════════════════════════

def merge_modules(m1_output: dict, m2_report: dict) -> dict:
    """
    将 Module 1（LLM 提取）和 Module 2（数据集分析）的结果合并为
    Module 3 的 retrieve_top3_hybrid() 所需的输入格式。

    Module 2 覆盖的字段：
      - data_size：从 total_images + 类别数（分类任务）推断
      - num_classes：来自 class_distribution，供 Module 4 生成正确的 head 维度
      - constraints.class_imbalance：从 class_distribution 推断（与 Module 1 取 OR）
    """
    merged = dict(m1_output)
    # constraints 单独拷贝，避免原地修改 m1_output 内层 dict
    merged["constraints"] = dict(m1_output.get("constraints", {}))

    class_dist = m2_report.get("class_distribution", {})
    num_classes = m2_report.get("num_classes") or len(class_dist) or None

    # data_size 由 Module 2 决定：总量 + 每类样本数双信号
    total_images = m2_report.get("total_images", 0)
    merged["data_size"] = derive_data_size(
        total_images,
        num_classes=num_classes,
        task_type=merged.get("task_type", "classification"),
    )
    if num_classes:
        merged["num_classes"] = num_classes

    # class_imbalance: Module 1（用户说了）或 Module 2（数据显示了）任一为 True 即生效
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
# 完整流水线
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
    完整流水线入口。

    返回:
      {
        "module3_input": dict,       # 合并后的 Module 3 输入
        "recommendations": list,     # retrieve_top3_hybrid 原始结果
        "task_lists": list,          # Module 4 可消费的任务清单
        "module4": dict | None,      # 可选的 Module 4 生成结果
      }
    """
    # Step 1: Module 1 — 用户自然语言 → 结构化字段
    print("[Pipeline] Module 1: Parsing user requirements...")
    from features_extraction_api import module1_pipeline

    m1_output = module1_pipeline(user_message)
    if m1_output is None:
        print("[Pipeline] Module 1 failed, cannot continue.")
        return {"module3_input": None, "recommendations": [], "task_lists": [], "module4": None}

    # Step 2: Module 2 — 数据集分析 → data_size / class_imbalance
    ds_label = f"{dataset_id}:{subset}" if subset else dataset_id
    print(f"[Pipeline] Module 2: Analyzing dataset {ds_label}...")
    m2_report = run_module2_analysis(dataset_id, subset=subset)

    # Step 3: 合并
    m3_input = merge_modules(m1_output, m2_report)
    print(f"[Pipeline] Merged: task={m3_input['task_type']}  "
          f"size={m3_input['data_size']}  priority={m3_input['priority']}")

    # Step 4: Module 3 — 模型推荐
    print("[Pipeline] Module 3: Retrieving model configurations...")
    from retrieval.rag_retrieval import (
        build_graph, build_vector_index,
        retrieve_top3_hybrid, build_all_task_lists, print_results,
    )

    G = build_graph()
    col = build_vector_index()
    recommendations = retrieve_top3_hybrid(m3_input, G, col)

    # 输出结果
    print_results(m3_input, recommendations, G)

    # Step 4b（可选）：累积型推荐器 — 用历史结果记忆重排 + 给出解释
    if use_recommender:
        from recommender import recommend, OutcomeMemory

        memory = OutcomeMemory(recommender_memory) if recommender_memory else OutcomeMemory()
        recommendations = recommend(recommendations, m2_report, m3_input, memory=memory)
        print("[Pipeline] Recommender re-ranked candidates:")
        for r in recommendations:
            print(f"    [{r.get('rank_basis')}] {r.get('backbone')} — {r.get('explanation')}")

    task_lists = build_all_task_lists(recommendations, G, fmt=fmt)
    module4_result = None

    if module4_output:
        print(f"[Pipeline] Module 4: Generating code to {module4_output}...")
        module4_task_lists = task_lists
        if fmt != "nl":
            module4_task_lists = build_all_task_lists(recommendations, G, fmt="nl")
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
                # Recipe layer: recommended training hyperparameters (lr / image_size /
                # augmentation / early stopping). setdefault → KB/user-set values win.
                if use_recipe:
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

    # 真训练时本地 smoke 没意义（合成数据 1 步），自动跳过，和 run_for_testing.py 一致
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

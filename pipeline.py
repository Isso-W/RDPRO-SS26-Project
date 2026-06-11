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
    print("[Pipeline] Module 1: 解析用户需求...")
    from features_extraction_api import module1_pipeline

    m1_output = module1_pipeline(user_message)
    if m1_output is None:
        print("[Pipeline] Module 1 失败，无法继续。")
        return {"module3_input": None, "recommendations": [], "task_lists": [], "module4": None}

    # Step 2: Module 2 — 数据集分析 → data_size / class_imbalance
    ds_label = f"{dataset_id}:{subset}" if subset else dataset_id
    print(f"[Pipeline] Module 2: 分析数据集 {ds_label}...")
    m2_report = run_module2_analysis(dataset_id, subset=subset)

    # Step 3: 合并
    m3_input = merge_modules(m1_output, m2_report)
    print(f"[Pipeline] 合并结果: task={m3_input['task_type']}  "
          f"size={m3_input['data_size']}  priority={m3_input['priority']}")

    # Step 4: Module 3 — 模型推荐
    print("[Pipeline] Module 3: 检索模型配置...")
    from retrieval.rag_retrieval import (
        build_graph, build_vector_index,
        retrieve_top3_hybrid, build_all_task_lists, print_results,
    )

    G = build_graph()
    col = build_vector_index()
    recommendations = retrieve_top3_hybrid(m3_input, G, col)

    # 输出结果
    print_results(m3_input, recommendations, G)
    task_lists = build_all_task_lists(recommendations, G, fmt=fmt)
    module4_result = None

    if module4_output:
        print(f"[Pipeline] Module 4: 生成代码到 {module4_output}...")
        module4_task_lists = task_lists
        if fmt != "nl":
            module4_task_lists = build_all_task_lists(recommendations, G, fmt="nl")
        # 把 Module 2 探明的类别数注入 model_config，
        # 否则 Module 4 会用默认 3 类生成 head
        num_classes = m3_input.get("num_classes")
        if num_classes:
            for task_list in module4_task_lists:
                if isinstance(task_list.get("model_config"), dict):
                    task_list["model_config"].setdefault("num_classes", num_classes)
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
        "recommendations": recommendations,
        "task_lists":       task_lists,
        "module4":          module4_result,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jiaozi Pipeline: NL + Dataset → Model Recommendation")
    parser.add_argument("--query", required=True, help="用户自然语言需求描述")
    parser.add_argument("--dataset", required=True,
                        help="HuggingFace 数据集 ID，支持 org/name:subset 格式")
    parser.add_argument("--subset", default=None,
                        help="数据集子配置名（也可用 --dataset org/name:subset 简写）")
    parser.add_argument("--fmt", default="structured", choices=["structured", "nl"],
                        help="Module 4 任务清单格式")
    parser.add_argument("--module4-output", default=None,
                        help="可选：继续运行 Module 4，并把生成代码写到该目录")
    parser.add_argument("--module4-no-smoke", action="store_true",
                        help="Module 4 只生成和静态检查，不运行本地 smoke tests")
    parser.add_argument("--module4-run-refinement", action="store_true",
                        help="Module 4 通过后继续运行 refinement loop")
    parser.add_argument("--module4-timeout", type=int, default=60,
                        help="Module 4 每个 smoke command 的超时时间")
    parser.add_argument("--module4-llm-provider", default=None,
                        choices=["none", "qwen", "openai", "vertex"],
                        help="Module 4 model.py 生成 provider；例如 qwen。默认使用环境变量或模板")
    args = parser.parse_args()

    dataset_id, parsed_subset = parse_dataset_id(args.dataset)
    subset = args.subset or parsed_subset

    result = run_pipeline(
        args.query,
        dataset_id,
        fmt=args.fmt,
        subset=subset,
        module4_output=args.module4_output,
        module4_skip_smoke=args.module4_no_smoke,
        module4_run_refinement=args.module4_run_refinement,
        module4_timeout=args.module4_timeout,
        module4_llm_provider=args.module4_llm_provider,
    )

    print("\n═══ Module 4 Task Lists ═══")
    print(json.dumps(result["task_lists"], indent=2, ensure_ascii=False))
    if result["module4"]:
        print("\n═══ Module 4 Code Generation Summary ═══")
        print(json.dumps(result["module4"]["summary"], indent=2, ensure_ascii=False))

"""
Jiaozi 测试脚本 — 提供数据集位置和用户需求，一键跑完整流水线。

用法:
    python run_for_testing.py --dataset uoft-cs/cifar10 --query "classify images on mobile device"

    # 本地图片文件夹（按类别分子文件夹，即 imagefolder 布局）:
    python run_for_testing.py --dataset ./my_images --query "检测图片里的车辆"

    # 同时生成 Module 4 训练代码:
    python run_for_testing.py --dataset uoft-cs/cifar10 --query "..." --module4

结果保存在 test_runs/<时间戳>/ 下:
    run_info.json         本次运行的参数
    module3_input.json    Module 1+2 合并后的 Module 3 输入
    recommendations.json  Top 3 模型推荐（含打分明细）
    task_lists.json       Module 4 任务清单
    module4_code/         （--module4 时）生成的训练/评估/推理代码

前置条件:
    复制 .env.example 为 .env 并填入 API key（Module 1 解析需求需要 LLM）。
    本脚本会自动加载 .env，不需要手动 export。
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

# Windows 下重定向输出默认用 GBK，中文会乱码；统一成 UTF-8
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def load_env_file(path: Path) -> bool:
    """加载 .env 到环境变量（不覆盖已有变量）。项目本身不加载 .env，这里替测试同学做掉。"""
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
    """检查 Module 1 所需的 LLM 配置，返回问题列表（空 = 通过）。"""
    problems = []
    provider = os.getenv("JIAOZI_LLM_PROVIDER", os.getenv("M1_LLM_PROVIDER", "qwen")).strip().lower()
    if provider == "qwen":
        key = os.getenv("JIAOZI_DASHSCOPE_API_KEY", "")
        if not key or key.startswith("replace_with"):
            problems.append(
                "JIAOZI_DASHSCOPE_API_KEY 未设置或还是占位符。"
                "请复制 .env.example 为 .env 并填入 DashScope key。"
            )
    elif provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "")
        if not key or key.startswith("replace_with"):
            problems.append("OPENAI_API_KEY 未设置或还是占位符，请在 .env 里填入。")
    else:
        problems.append(f"未知的 JIAOZI_LLM_PROVIDER={provider!r}（支持 qwen / openai）。")
    return problems


def resolve_dataset(arg: str) -> str:
    """本地路径转为绝对路径（datasets 会按 imagefolder 加载），否则按 HuggingFace ID 处理。"""
    local = Path(arg)
    if local.exists():
        if not local.is_dir():
            sys.exit(f"[错误] {arg} 是文件不是文件夹。本地数据集请提供图片文件夹（按类别分子文件夹）。")
        print(f"[Tester] 检测到本地数据集文件夹: {local.resolve()}")
        print("[Tester] 注意: 文件夹需要 imagefolder 布局，即每个类别一个子文件夹。")
        return local.resolve().as_posix()
    if "/" not in arg:
        print(f"[Tester] 提示: {arg!r} 不是本地路径，将按 HuggingFace ID 处理"
              "（通常形如 org/name，例如 uoft-cs/cifar10）。")
    return arg


def save_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def print_summary(recommendations: list[dict]) -> None:
    print("\n" + "═" * 70)
    print("最终推荐（Top 3）")
    print("═" * 70)
    if not recommendations:
        print("没有得到任何推荐——通常说明约束太严（如 zero_shot）筛掉了所有候选。")
        return
    for i, rec in enumerate(recommendations, 1):
        detail = rec.get("score_detail", {})
        print(f"\n#{i}  {rec.get('backbone')}   总分 {rec.get('score')} "
              f"(结构化 {detail.get('structured')} / 向量 {detail.get('vector')})")
        print(f"    checkpoint: {rec.get('pretrained') or '无（从头训练）'}")
        print(f"    head: {rec.get('head')}   loss: {rec.get('loss')}   optimizer: {rec.get('optimizer')}")
        print(f"    finetune: {rec.get('finetune_strategy')}   freeze可行: {rec.get('freeze_viable')}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Jiaozi 测试入口：数据集位置 + 用户需求 → Top 3 模型推荐（可选生成代码）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset", required=True,
                        help="HuggingFace 数据集 ID（如 uoft-cs/cifar10），支持 org/name:subset 格式")
    parser.add_argument("--subset", default=None,
                        help="数据集子配置名（合集类数据集必填，也可用 --dataset org/name:subset 简写）")
    parser.add_argument("--query", required=True, help="用户自然语言需求描述")
    parser.add_argument("--output", default=None,
                        help="结果输出目录（默认 test_runs/<时间戳>）")
    parser.add_argument("--module4", action="store_true",
                        help="继续运行 Module 4 生成训练代码（较慢）")
    parser.add_argument("--no-smoke", action="store_true",
                        help="Module 4 只生成不跑 smoke test（更快）")
    parser.add_argument("--fmt", default="nl", choices=["structured", "nl"],
                        help="Module 4 任务清单格式")
    args = parser.parse_args()

    # 相对路径（chroma_db_kb、模块导入）都假定在仓库根目录
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))

    if load_env_file(REPO_ROOT / ".env"):
        print("[Tester] 已加载 .env")
    else:
        print("[Tester] 未找到 .env（将依赖已有环境变量）")

    problems = check_llm_config()
    if problems:
        print("\n[错误] LLM 配置检查未通过：")
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
    print(f"[Tester] 结果将保存到: {output_dir}")

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
        sys.exit(f"\n[错误] 数据集加载失败: {e}\n"
                 f"  - HuggingFace ID 请确认拼写（形如 org/name）且网络可达\n"
                 f"  - 本地文件夹请确认路径存在且为 imagefolder 布局")
    except Exception as e:
        traceback.print_exc()
        hint = ""
        msg = str(e).lower()
        if "401" in msg or "authentication" in msg or "api key" in msg:
            hint = "\n  看起来是 API key 问题，请检查 .env 里的 key 是否有效。"
        elif "connect" in msg or "timeout" in msg or "resolve" in msg:
            hint = "\n  看起来是网络问题（下载数据集/模型或调用 LLM 失败），请检查网络后重试。"
        sys.exit(f"\n[错误] 流水线运行失败: {e}{hint}\n  完整报错见上方 traceback，可直接截图反馈。")

    if not result.get("module3_input"):
        sys.exit("\n[错误] Module 1 解析失败（返回空），请检查 API key 或换一种描述方式重试。")

    save_json(output_dir / "module3_input.json", result["module3_input"])
    save_json(output_dir / "recommendations.json", result["recommendations"])
    save_json(output_dir / "task_lists.json", result["task_lists"])

    print_summary(result["recommendations"])

    if result.get("module4"):
        print(f"\nModule 4 代码已生成到: {result['module4']['output_dir']}")
        save_json(output_dir / "module4_summary.json", result["module4"]["summary"])

    print(f"\n所有结果文件已保存到: {output_dir}")
    print("反馈问题时请把整个输出目录打包发回。")


if __name__ == "__main__":
    main()

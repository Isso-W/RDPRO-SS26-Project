"""configs.py — 冻结的实验矩阵 + 裁决常数（唯一事实源，预注册）。

**判据先于实验写死。** 除 loss / fold_index 外，两臂一切相同（paired 5-fold）。
详见 ab_loss_imbalance_plan.md §0/§2。
"""

from __future__ import annotations

# ── 裁决常数（写死；改动 = 破坏预注册）───────────────────────────────────────
MARGIN_FLOOR = 0.005          # 平局带下限；实际带宽 = max(MARGIN_FLOOR, 2*SE)
N_FOLDS = 5
GLOBAL_SEED = 42
ARMS = ("focal_loss", "cross_entropy_loss")

# 每台的主指标 + 次级观测（次级仅记录、不进裁决）。按台设：PR-AUC 是二分类
# 概念，对多类不干净；cassava 主判据用 imbalance 敏感的 macro_f1（accuracy 对
# 不平衡盲，会让该台按构造倾向 TIE），accuracy 降为次级对照。
TESTBEDS: dict[str, dict] = {
    "siim_isic": {"metric": "roc_auc",  "image_size": 224, "epochs": 5,
                  "secondary_metrics": ["pr_auc"]},
    "cassava":   {"metric": "macro_f1", "image_size": 224, "epochs": 5,
                  "secondary_metrics": ["accuracy"]},
}

# 除 loss / fold_index 外全部冻结。pretrained 已用 Module 3 对 efficientnet 场景
# 的实际选择解析并硬编码（2026-07-05：build_graph() 中 efficientnet 的
# has_pretrained checkpoint = efficientnet_b0_imagenet，size_tier=base）——冻结，
# 不做动态解析，避免 KB 后续变动悄悄换掉实验地基。
BASE: dict = {
    "backbone": "efficientnet",
    "pretrained": "efficientnet_b0_imagenet",
    "sampler": "shuffle",     # 显式声明：无加权采样（loss×sampler 交互范围外）
    "seed": GLOBAL_SEED,
    "cv": {"n_folds": N_FOLDS, "stratified": True, "shared_across_arms": True},
}

_PLACEHOLDER_MARKERS = ("XXXX", "placeholder", "<", "TODO", "占位")


def fold_file_name(testbed: str) -> str:
    """两臂共用的折文件名（paired 的载体）。"""
    return f"folds_{testbed}.json"


def build_matrix(testbed: str) -> list[dict]:
    """某台的 10 个 run config（2 臂 × 5 折）。差异字段仅 loss / fold_index。"""
    if testbed not in TESTBEDS:
        raise KeyError(f"unknown testbed: {testbed}")
    tb = TESTBEDS[testbed]
    frozen = {
        **BASE,
        "testbed": testbed,
        "benchmark": testbed,
        "metric": tb["metric"],
        "secondary_metrics": list(tb["secondary_metrics"]),
        "image_size": tb["image_size"],
        "epochs": tb["epochs"],
        "fold_file": fold_file_name(testbed),   # 两臂同一文件 → paired
    }
    matrix = []
    for loss in ARMS:
        for fold_index in range(N_FOLDS):
            run = dict(frozen)
            run["loss"] = loss             # ← 变量 1
            run["fold_index"] = fold_index  # ← 变量 2
            matrix.append(run)
    return matrix


def has_placeholder(value: str) -> bool:
    return any(m in str(value) for m in _PLACEHOLDER_MARKERS)

"""run_ab.py — 驱动 A/B：算折 → 生成工程 → 按 (臂, 折) 顺序训练 → 落 outcomes。

真训练需 GPU + 数据下载（Kaggle）。纯逻辑（算折、指标 bundle）抽成可离线测试的
函数；编排部分调用 prepare_project + 子进程跑生成工程的 run.py。

CLI: python -m experiments.ab_loss_imbalance.run_ab --testbed cassava [--data-root ...]
     [--output ...] [--only focal_loss:3]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from experiments.ab_loss_imbalance import configs

RESULTS = Path("experiments/ab_loss_imbalance/results")
OUTCOMES = RESULTS / "outcomes.jsonl"


# ── 纯逻辑：分层折 ───────────────────────────────────────────────────────────
def compute_folds(labels: list, ids: list, n_folds: int = configs.N_FOLDS,
                  seed: int = configs.GLOBAL_SEED) -> list[list[str]]:
    """分层 5 折 → 每折的 val 样本 id 列表（按 id 存，行序无关）。"""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for i, (_, val_idx) in enumerate(skf.split(list(ids), list(labels))):
        folds[i] = [str(ids[j]) for j in val_idx]
    return folds


def fold_spec(labels, ids, n_folds=configs.N_FOLDS, seed=configs.GLOBAL_SEED) -> dict:
    return {"seed": seed, "n_folds": n_folds, "stratified": True,
            "folds": compute_folds(labels, ids, n_folds, seed)}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ── 纯逻辑：指标 bundle（从 val 预测算 macro_f1 / roc_auc / pr_auc / ...）─────
def metric_bundle(y_true: list, y_prob: list, metrics: list[str]) -> dict:
    """从 val 预测算一束指标。y_prob 为 [N, C] 概率矩阵。"""
    import numpy as np
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 cohen_kappa_score, f1_score, roc_auc_score)
    yt = np.asarray(y_true)
    yp = np.asarray(y_prob)
    preds = yp.argmax(axis=1)
    binary = yp.shape[1] == 2
    out: dict[str, float | None] = {}
    for m in metrics:
        try:
            if m == "accuracy":
                out[m] = float(accuracy_score(yt, preds))
            elif m == "macro_f1":
                out[m] = float(f1_score(yt, preds, average="macro"))
            elif m == "roc_auc":
                out[m] = float(roc_auc_score(yt, yp[:, 1]) if binary
                               else roc_auc_score(yt, yp, multi_class="ovr"))
            elif m == "pr_auc":
                # PR-AUC 是二分类概念；多类不干净 → None（次级指标，允许缺）
                out[m] = float(average_precision_score(yt, yp[:, 1])) if binary else None
            elif m == "qwk":
                out[m] = float(cohen_kappa_score(yt, preds, weights="quadratic"))
            else:
                out[m] = None
        except (ValueError, IndexError):
            out[m] = None
    return out


def metrics_for(testbed: str) -> list[str]:
    tb = configs.TESTBEDS[testbed]
    return [tb["metric"], *tb["secondary_metrics"]]


# ── 续跑：已完成的 (臂, 折) 跳过 ─────────────────────────────────────────────
def completed_pairs(outcomes_path: Path, testbed: str) -> set[tuple]:
    done = set()
    if outcomes_path.exists():
        for line in outcomes_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("benchmark") == testbed and "val_metric" in r:
                done.add((r["arm"], r["fold"]))
    return done


# ── 编排（需 GPU + 数据；此处不在离线环境执行）──────────────────────────────
def run_matrix(testbed: str, data_root: str, output_dir: str,
               only: tuple | None = None) -> None:
    from run_kaggle_benchmark import prepare_project

    RESULTS.mkdir(parents=True, exist_ok=True)
    res = prepare_project(testbed, data_root, output_dir)
    info = res["info"]
    project = Path(output_dir) / "module4_code"

    # 1) 算折（幂等：已存在则复用，绝不重切）
    import pandas as pd
    frame = pd.read_csv(info["train_csv"])
    ids = frame[info["image_column"]].tolist()
    labels = frame[info["label_column"]].tolist()
    fold_path = project / configs.fold_file_name(testbed)
    if not fold_path.exists():
        fold_path.write_text(json.dumps(fold_spec(labels, ids)), encoding="utf-8")
    fold_sha = sha256_of(fold_path)

    # 2) 冻结 base config（backbone/pretrained 来自 configs.BASE，不用 Module 3 动态选）
    base_cfg = _frozen_config(testbed, info, fold_path)
    done = completed_pairs(OUTCOMES, testbed)

    for loss in configs.ARMS:
        for fold_index in range(configs.N_FOLDS):
            if only and (loss, fold_index) != only:
                continue
            if (loss, fold_index) in done:
                print(f"[run_ab] skip done {loss}:{fold_index}")
                continue
            _run_one(testbed, project, base_cfg, loss, fold_index, fold_sha, info)


def _frozen_config(testbed, info, fold_path) -> dict:
    tb = configs.TESTBEDS[testbed]
    return {
        **configs.BASE,
        "task_type": "classification",
        "num_classes": info.get("num_classes"),
        "train_csv": info["train_csv"], "image_dir": info["image_dir"],
        "image_column": info["image_column"], "label_column": info["label_column"],
        "image_path_template": info["image_path_template"],
        "image_extension": info["image_extension"],
        "image_size": tb["image_size"], "offline_smoke": False,
        "evaluation_metric": tb["metric"],
        "fold_file": str(fold_path),
    }


def _run_one(testbed, project, base_cfg, loss, fold_index, fold_sha, info) -> None:
    tb = configs.TESTBEDS[testbed]
    preds_path = project / f"val_preds_{loss}_{fold_index}.json"
    cfg = {**base_cfg, "loss": loss, "fold_index": fold_index,
           "export_preds_path": str(preds_path)}
    cfg_path = project / f"config_{loss}_{fold_index}.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    print(f"[run_ab] train {testbed} {loss} fold {fold_index} ...", flush=True)
    subprocess.run(
        [sys.executable, "-u", "run.py", "--config", cfg_path.name,
         "--epochs", str(tb["epochs"]), "--seed", str(configs.GLOBAL_SEED)],
        cwd=str(project), check=True,
    )
    preds = json.loads(preds_path.read_text(encoding="utf-8"))
    bundle = metric_bundle(preds["y_true"], preds["y_prob"], metrics_for(testbed))
    _append_outcome(testbed, loss, fold_index, bundle, fold_sha, base_cfg)


def _append_outcome(testbed, loss, fold_index, bundle, fold_sha, base_cfg) -> None:
    import datetime
    rec = {
        "experiment": "ab_loss_imbalance", "benchmark": testbed,
        "arm": loss, "fold": fold_index, "seed": configs.GLOBAL_SEED,
        "config": {k: base_cfg[k] for k in ("backbone", "pretrained", "image_size")},
        "val_metric": bundle, "fold_file_sha256": fold_sha,
        "date": datetime.date.today().isoformat(),
    }
    with OUTCOMES.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _parse_only(s: str | None) -> tuple | None:
    if not s:
        return None
    arm, fold = s.split(":")
    return (arm, int(fold))


def _cli() -> None:
    ap = argparse.ArgumentParser(description="A/B run driver")
    ap.add_argument("--testbed", required=True, choices=list(configs.TESTBEDS))
    ap.add_argument("--data-root", default="./kaggle_data")
    ap.add_argument("--output", default="./ab_runs")
    ap.add_argument("--only", default=None, help="臂:折，如 focal_loss:3（单跑试链路）")
    args = ap.parse_args()
    run_matrix(args.testbed, args.data_root, f"{args.output}/{args.testbed}",
               only=_parse_only(args.only))


if __name__ == "__main__":
    _cli()

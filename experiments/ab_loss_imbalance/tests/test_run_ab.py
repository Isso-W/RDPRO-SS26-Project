"""test_run_ab.py — run_ab 的纯逻辑（算折 / 指标 bundle / 续跑），全离线。"""

from __future__ import annotations

import json

import pytest

from experiments.ab_loss_imbalance import run_ab


def test_compute_folds_stratified_disjoint_and_complete():
    pytest.importorskip("sklearn")
    labels = [0, 1] * 10
    ids = [f"s{i}" for i in range(20)]
    folds = run_ab.compute_folds(labels, ids, n_folds=5, seed=42)
    assert len(folds) == 5
    flat = [x for fold in folds for x in fold]
    assert set(flat) == set(ids)                 # 并集 == 全部
    assert len(flat) == len(set(flat))           # 两两不相交
    assert all(isinstance(x, str) for x in flat)  # 按 id（字符串）存


def test_compute_folds_deterministic():
    pytest.importorskip("sklearn")
    labels = [0, 1, 2] * 6
    ids = list(range(18))
    assert run_ab.compute_folds(labels, ids, 3, 42) == run_ab.compute_folds(labels, ids, 3, 42)


def test_metric_bundle_binary_perfect():
    pytest.importorskip("sklearn")
    y_true = [0, 0, 1, 1]
    y_prob = [[0.9, 0.1], [0.8, 0.2], [0.3, 0.7], [0.2, 0.8]]
    b = run_ab.metric_bundle(y_true, y_prob, ["roc_auc", "pr_auc", "macro_f1", "accuracy"])
    assert b["accuracy"] == 1.0
    assert b["macro_f1"] == 1.0
    assert b["roc_auc"] == 1.0
    assert b["pr_auc"] == 1.0


def test_metric_bundle_multiclass_pr_auc_is_none():
    pytest.importorskip("sklearn")
    y_true = [0, 1, 2]
    y_prob = [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
    b = run_ab.metric_bundle(y_true, y_prob, ["macro_f1", "pr_auc"])
    assert b["macro_f1"] == 1.0
    assert b["pr_auc"] is None            # PR-AUC 多类不干净 → None


def test_metrics_for_matches_configs():
    assert run_ab.metrics_for("cassava")[0] == "macro_f1"       # 主指标在首位
    assert "accuracy" in run_ab.metrics_for("cassava")
    assert run_ab.metrics_for("siim_isic")[0] == "roc_auc"


def test_completed_pairs_skips_finished(tmp_path):
    p = tmp_path / "outcomes.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"benchmark": "cassava", "arm": "focal_loss", "fold": 0, "val_metric": {"macro_f1": 0.7}},
        {"benchmark": "cassava", "arm": "cross_entropy_loss", "fold": 1, "val_metric": {"macro_f1": 0.7}},
        {"benchmark": "siim_isic", "arm": "focal_loss", "fold": 0, "val_metric": {"roc_auc": 0.9}},
    ]), encoding="utf-8")
    done = run_ab.completed_pairs(p, "cassava")
    assert done == {("focal_loss", 0), ("cross_entropy_loss", 1)}   # 只算本台

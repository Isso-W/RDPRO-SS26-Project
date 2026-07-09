"""test_collect.py — 裁决逻辑：台级三判例 + 边界 + 大 SE 吞噬 + 双台合并全组合。"""

from __future__ import annotations

from experiments.ab_loss_imbalance import collect


# ── 台级裁决 ────────────────────────────────────────────────────────────────
def test_ce_clear_win():
    # Δ̄≈0.02，抖动小 → 带=0.005 → CE_WINS
    r = collect.testbed_verdict([0.020, 0.021, 0.019, 0.022, 0.018])
    assert r["verdict"] == "CE_WINS"


def test_focal_clear_win():
    r = collect.testbed_verdict([-0.020, -0.021, -0.019, -0.022, -0.018])
    assert r["verdict"] == "FOCAL_WINS"


def test_small_effect_is_tie():
    r = collect.testbed_verdict([0.001, 0.0, -0.001, 0.001, 0.0])
    assert r["verdict"] == "TIE"


def test_boundary_dbar_equals_band_is_ce_win():
    # 全等 0.005 → std 0 → SE 0 → 带=max(0.005,0)=0.005；Δ̄=0.005 ≥ 带 → CE_WINS
    r = collect.testbed_verdict([0.005, 0.005, 0.005, 0.005, 0.005])
    assert r["band"] == 0.005 and r["dbar"] == 0.005
    assert r["verdict"] == "CE_WINS"


def test_large_variance_swamps_modest_effect():
    # Δ̄ 小正但折间抖动巨大 → 2·SE > Δ̄ → TIE（翻案必须跨过噪声）
    r = collect.testbed_verdict([0.02, -0.02, 0.03, -0.03, 0.02])
    assert r["band"] > abs(r["dbar"])
    assert r["verdict"] == "TIE"


def test_insufficient_folds_is_tie():
    assert collect.testbed_verdict([0.01])["verdict"] == "TIE"      # n<2 → 保守 TIE


# ── 双台合并 ────────────────────────────────────────────────────────────────
def test_merge_combinations():
    C, F, T = "CE_WINS", "FOCAL_WINS", "TIE"
    assert collect.merge_verdicts([C, T]) == "CE_WINS"      # 一台胜一台平 → 胜
    assert collect.merge_verdicts([C, C]) == "CE_WINS"
    assert collect.merge_verdicts([C, F]) == "TIE"          # 互相矛盾 → 现状赢
    assert collect.merge_verdicts([F, T]) == "FOCAL_WINS"
    assert collect.merge_verdicts([T, T]) == "TIE"
    assert collect.merge_verdicts([F, F]) == "FOCAL_WINS"


# ── 配对差提取 ──────────────────────────────────────────────────────────────
def _rec(bench, arm, fold, metric, val):
    return {"benchmark": bench, "arm": arm, "fold": fold, "val_metric": {metric: val}}


def test_paired_deltas_extracts_ce_minus_focal():
    recs = [
        _rec("siim_isic", "cross_entropy_loss", 0, "roc_auc", 0.91),
        _rec("siim_isic", "focal_loss", 0, "roc_auc", 0.90),
        _rec("siim_isic", "cross_entropy_loss", 1, "roc_auc", 0.92),
        _rec("siim_isic", "focal_loss", 1, "roc_auc", 0.905),
    ]
    d = collect.paired_deltas(recs, "siim_isic", "roc_auc")
    assert [round(x, 4) for x in d] == [0.01, 0.015]


def test_paired_deltas_skips_unpaired_fold():
    recs = [
        _rec("siim_isic", "cross_entropy_loss", 0, "roc_auc", 0.91),
        _rec("siim_isic", "focal_loss", 0, "roc_auc", 0.90),
        _rec("siim_isic", "cross_entropy_loss", 1, "roc_auc", 0.92),  # 折1缺 focal
    ]
    assert len(collect.paired_deltas(recs, "siim_isic", "roc_auc")) == 1


def test_summarize_end_to_end_tie_when_one_side_empty():
    # siim CE 全胜、cassava 无数据 → cassava TIE → 合并 CE_WINS
    recs = [
        _rec("siim_isic", "cross_entropy_loss", f, "roc_auc", 0.92) for f in range(5)
    ] + [
        _rec("siim_isic", "focal_loss", f, "roc_auc", 0.90) for f in range(5)
    ]
    s = collect.summarize(recs)
    assert s["per_testbed"]["siim_isic"]["verdict"] == "CE_WINS"
    assert s["per_testbed"]["cassava"]["verdict"] == "TIE"
    assert s["overall"] == "CE_WINS"

"""Tests for per-testbed and merged CE-vs-focal arbitration rules."""

from __future__ import annotations

from experiments.ab_loss_imbalance import collect


# Per-testbed verdicts.
def test_ce_clear_win():
    # dbar is about 0.02 with low variance, so the floor band gives CE_WINS.
    r = collect.testbed_verdict([0.020, 0.021, 0.019, 0.022, 0.018])
    assert r["verdict"] == "CE_WINS"


def test_focal_clear_win():
    r = collect.testbed_verdict([-0.020, -0.021, -0.019, -0.022, -0.018])
    assert r["verdict"] == "FOCAL_WINS"


def test_small_effect_is_tie():
    r = collect.testbed_verdict([0.001, 0.0, -0.001, 0.001, 0.0])
    assert r["verdict"] == "TIE"


def test_boundary_dbar_equals_band_is_ce_win():
    # Exactly on the positive band boundary counts as CE_WINS.
    r = collect.testbed_verdict([0.005, 0.005, 0.005, 0.005, 0.005])
    assert r["band"] == 0.005 and r["dbar"] == 0.005
    assert r["verdict"] == "CE_WINS"


def test_large_variance_swamps_modest_effect():
    # Large fold-to-fold variance can swamp a small positive effect.
    r = collect.testbed_verdict([0.02, -0.02, 0.03, -0.03, 0.02])
    assert r["band"] > abs(r["dbar"])
    assert r["verdict"] == "TIE"


def test_insufficient_folds_is_tie():
    assert collect.testbed_verdict([0.01])["verdict"] == "TIE"      # n<2 stays conservative


# Merged verdicts.
def test_merge_combinations():
    C, F, T = "CE_WINS", "FOCAL_WINS", "TIE"
    assert collect.merge_verdicts([C, T]) == "CE_WINS"
    assert collect.merge_verdicts([C, C]) == "CE_WINS"
    assert collect.merge_verdicts([C, F]) == "TIE"
    assert collect.merge_verdicts([F, T]) == "FOCAL_WINS"
    assert collect.merge_verdicts([T, T]) == "TIE"
    assert collect.merge_verdicts([F, F]) == "FOCAL_WINS"


# Paired-difference extraction.
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
        _rec("siim_isic", "cross_entropy_loss", 1, "roc_auc", 0.92),  # fold 1 has no focal pair
    ]
    assert len(collect.paired_deltas(recs, "siim_isic", "roc_auc")) == 1


def test_summarize_end_to_end_tie_when_one_side_empty():
    # SIIM-ISIC supports CE; cassava has no paired data, so the merge keeps CE_WINS.
    recs = [
        _rec("siim_isic", "cross_entropy_loss", f, "roc_auc", 0.92) for f in range(5)
    ] + [
        _rec("siim_isic", "focal_loss", f, "roc_auc", 0.90) for f in range(5)
    ]
    s = collect.summarize(recs)
    assert s["per_testbed"]["siim_isic"]["verdict"] == "CE_WINS"
    assert s["per_testbed"]["cassava"]["verdict"] == "TIE"
    assert s["overall"] == "CE_WINS"

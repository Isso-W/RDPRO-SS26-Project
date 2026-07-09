"""test_configs.py — 冻结矩阵 / paired / 无占位符 三项机器可查保证。"""

from __future__ import annotations

import pytest

from experiments.ab_loss_imbalance import configs


@pytest.mark.parametrize("testbed", list(configs.TESTBEDS))
def test_matrix_size(testbed):
    assert len(configs.build_matrix(testbed)) == 2 * configs.N_FOLDS  # 10


@pytest.mark.parametrize("testbed", list(configs.TESTBEDS))
def test_only_loss_and_fold_vary(testbed):
    """两两之间除 loss / fold_index 外所有键值相等——防手滑引入第二个变量。"""
    matrix = configs.build_matrix(testbed)
    varying = {"loss", "fold_index"}
    keys = set(matrix[0]) - varying
    for run in matrix[1:]:
        assert set(run) - varying == keys
        for k in keys:
            assert run[k] == matrix[0][k], f"{k} 在矩阵内不一致：引入了额外变量"


@pytest.mark.parametrize("testbed", list(configs.TESTBEDS))
def test_arms_share_one_fold_file(testbed):
    """两臂引用同一 fold_file（paired 的载体）。"""
    matrix = configs.build_matrix(testbed)
    assert len({run["fold_file"] for run in matrix}) == 1


@pytest.mark.parametrize("testbed", list(configs.TESTBEDS))
def test_covers_all_arms_and_folds(testbed):
    matrix = configs.build_matrix(testbed)
    pairs = {(r["loss"], r["fold_index"]) for r in matrix}
    assert pairs == {(a, f) for a in configs.ARMS for f in range(configs.N_FOLDS)}


def test_pretrained_has_no_placeholder():
    assert not configs.has_placeholder(configs.BASE["pretrained"])
    assert configs.BASE["pretrained"] == "efficientnet_b0_imagenet"


def test_cassava_primary_is_macro_f1():
    # 预注册：cassava 主判据 macro_f1（accuracy 降次级）
    assert configs.TESTBEDS["cassava"]["metric"] == "macro_f1"
    assert "accuracy" in configs.TESTBEDS["cassava"]["secondary_metrics"]
    assert configs.TESTBEDS["siim_isic"]["metric"] == "roc_auc"

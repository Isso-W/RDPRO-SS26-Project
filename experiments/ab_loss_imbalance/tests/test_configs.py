"""Machine-checkable guards for the frozen matrix and paired folds."""

from __future__ import annotations

import pytest

from experiments.ab_loss_imbalance import configs


@pytest.mark.parametrize("testbed", list(configs.TESTBEDS))
def test_matrix_size(testbed):
    assert len(configs.build_matrix(testbed)) == 2 * configs.N_FOLDS  # 10


@pytest.mark.parametrize("testbed", list(configs.TESTBEDS))
def test_only_loss_and_fold_vary(testbed):
    """Only loss and fold_index may vary across runs."""
    matrix = configs.build_matrix(testbed)
    varying = {"loss", "fold_index"}
    keys = set(matrix[0]) - varying
    for run in matrix[1:]:
        assert set(run) - varying == keys
        for k in keys:
            assert run[k] == matrix[0][k], f"{k} changed inside the matrix"


@pytest.mark.parametrize("testbed", list(configs.TESTBEDS))
def test_arms_share_one_fold_file(testbed):
    """Both arms must reference the same fold file."""
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
    assert configs.BASE["checkpoint"] == "efficientnet_b0_imagenet"
    assert configs.BASE["pretrained_hf_id"] == "google/efficientnet-b0"
    assert configs.BASE["use_pretrained"] is True


def test_base_freezes_non_loss_training_choices():
    assert configs.BASE["sampler"] == "shuffle"
    assert configs.BASE["use_class_weights"] is False
    assert configs.BASE["optimizer"] == "adamw"
    assert configs.BASE["learning_rate"] == 1.0e-4
    assert configs.BASE["finetune_strategy"] == "full"
    assert configs.BASE["freeze_backbone"] is False


def test_cassava_primary_is_macro_f1():
    # Preregistered: cassava uses macro_f1 as primary, accuracy as secondary.
    assert configs.TESTBEDS["cassava"]["metric"] == "macro_f1"
    assert "accuracy" in configs.TESTBEDS["cassava"]["secondary_metrics"]
    assert configs.TESTBEDS["siim_isic"]["metric"] == "roc_auc"

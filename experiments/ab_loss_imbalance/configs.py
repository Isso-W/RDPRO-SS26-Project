"""Frozen experiment matrix and decision constants.

The criteria are fixed before running the experiment. Apart from loss and
fold_index, both arms are identical in the paired 5-fold setup. See
docs/ab_loss_imbalance_protocol.md for the preregistered protocol.
"""

from __future__ import annotations

# Decision constants. Changing these breaks the preregistered protocol.
MARGIN_FLOOR = 0.005          # lower bound for tie band: max(MARGIN_FLOOR, 2*SE)
N_FOLDS = 5
GLOBAL_SEED = 42
ARMS = ("focal_loss", "cross_entropy_loss")

# Primary metric plus secondary observations per testbed. Secondary metrics are
# recorded but not used for arbitration. PR-AUC is binary-first, so cassava uses
# macro_f1 as the imbalance-sensitive primary metric and keeps accuracy as a
# secondary reference.
TESTBEDS: dict[str, dict] = {
    "siim_isic": {"metric": "roc_auc",  "image_size": 224, "epochs": 8,
                  "secondary_metrics": ["pr_auc"]},
    "cassava":   {"metric": "macro_f1", "image_size": 224, "epochs": 8,
                  "secondary_metrics": ["accuracy"]},
}

# Everything except loss and fold_index is frozen. The pretrained checkpoint is
# the Module 3 efficientnet selection resolved on 2026-07-05
# (efficientnet_b0_imagenet, size_tier=base). We keep it static so later KB
# changes cannot silently move the experiment baseline.
BASE: dict = {
    "backbone": "efficientnet",
    "pretrained": "efficientnet_b0_imagenet",
    "checkpoint": "efficientnet_b0_imagenet",
    "pretrained_hf_id": "google/efficientnet-b0",
    "pretrained_name": "EfficientNet-B0 / ImageNet-1k",
    "pretrain_dataset": "ImageNet-1k",
    "params_M": 5,
    "use_pretrained": True,
    "optimizer": "adamw",
    "learning_rate": 1.0e-4,
    "finetune_strategy": "full",
    "freeze_backbone": False,
    "use_class_weights": False,
    "sampler": "shuffle",     # no weighted sampling; loss x sampler is out of scope
    "seed": GLOBAL_SEED,
    "cv": {"n_folds": N_FOLDS, "stratified": True, "shared_across_arms": True},
}

_PLACEHOLDER_MARKERS = ("XXXX", "placeholder", "<", "TODO", "\u5360\u4f4d")


def fold_file_name(testbed: str) -> str:
    """Return the shared fold-file name for both paired arms."""
    return f"folds_{testbed}.json"


def build_matrix(testbed: str) -> list[dict]:
    """Return the 10 run configs for one testbed: 2 arms x 5 folds."""
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
        "fold_file": fold_file_name(testbed),   # same file for both arms
    }
    matrix = []
    for loss in ARMS:
        for fold_index in range(N_FOLDS):
            run = dict(frozen)
            run["loss"] = loss              # variable 1
            run["fold_index"] = fold_index  # variable 2
            matrix.append(run)
    return matrix


def has_placeholder(value: str) -> bool:
    return any(m in str(value) for m in _PLACEHOLDER_MARKERS)

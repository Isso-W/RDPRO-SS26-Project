"""Tests for the Colab preparation helper's per-backbone resolution snapping.

Run from this directory:
    python -m pytest test_colab_prepare.py -q
"""

from __future__ import annotations

from colab_prepare_train import _safe_image_size


def test_swin_window7_checkpoint_stays_at_224():
    item = {"backbone": "swin_transformer", "checkpoint": "microsoft/swin-base-patch4-window7-224"}
    assert _safe_image_size(item, 384) == 224


def test_dinov3_snaps_to_multiple_of_16():
    item = {"backbone": "dinov3", "checkpoint": "facebook/dinov3-vitb16-pretrain-lvd1689m"}
    assert _safe_image_size(item, 384) == 384
    # A non-multiple request snaps down to the nearest valid patch grid.
    assert _safe_image_size(item, 390) == 384


def test_dinov2_snaps_to_multiple_of_14():
    item = {"backbone": "dinov2", "checkpoint": "facebook/dinov2-base"}
    size = _safe_image_size(item, 384)
    assert size % 14 == 0
    assert size <= 384


def test_never_returns_below_a_sensible_floor():
    item = {"backbone": "dinov3", "checkpoint": "facebook/dinov3-vitb16-pretrain-lvd1689m"}
    assert _safe_image_size(item, 10) >= 16

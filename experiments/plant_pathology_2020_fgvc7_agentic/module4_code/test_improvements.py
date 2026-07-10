"""Tests for Kaggle-score improvements: multi-transform TTA, rank blending,
and high-resolution positional-encoding interpolation.

Run from this directory:
    python -m pytest test_improvements.py -q
"""

from __future__ import annotations

import numpy as np
import torch

from consumer import _apply_blend_space
from ensemble_artifacts import mean_column_auc, rank_normalize
from model_utils import _HFBackbone
from train import _apply_tta, _resolve_tta_transforms


# --------------------------------------------------------------------------
# Item 2 — multi-transform test-time augmentation
# --------------------------------------------------------------------------

def test_resolve_tta_transforms_disabled_is_empty():
    assert _resolve_tta_transforms({"tta": False}) == []
    assert _resolve_tta_transforms({}) == []
    assert _resolve_tta_transforms({"tta": {"enabled": False, "transforms": ["hflip"]}}) == []


def test_resolve_tta_transforms_bool_true_defaults_to_hflip():
    assert _resolve_tta_transforms({"tta": True}) == ["hflip"]


def test_resolve_tta_transforms_reads_dict_transform_list():
    config = {"tta": {"enabled": True, "transforms": ["hflip", "vflip", "rot90"]}}
    assert _resolve_tta_transforms(config) == ["hflip", "vflip", "rot90"]


def test_resolve_tta_transforms_ignores_unknown_names():
    config = {"tta": {"enabled": True, "transforms": ["hflip", "bogus"]}}
    assert _resolve_tta_transforms(config) == ["hflip"]


class _CornerModel(torch.nn.Module):
    """Returns logits whose first entry is the channel-0 top-left pixel value.

    Because each orientation moves a different original pixel into the
    top-left slot, averaging over orientations lets a test assert the exact
    TTA mean.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        top_left = x[:, 0, 0, 0]
        return torch.stack([top_left, torch.zeros_like(top_left)], dim=1)


def test_apply_tta_averages_over_identity_and_transforms():
    # 1x1x2x2 image with distinct corner values in channel 0.
    x = torch.zeros(1, 1, 2, 2)
    x[0, 0, 0, 0] = 1.0   # top-left
    x[0, 0, 0, 1] = 2.0   # top-right
    x[0, 0, 1, 0] = 3.0   # bottom-left
    x[0, 0, 1, 1] = 4.0   # bottom-right
    model = _CornerModel()

    ops = ["hflip", "vflip", "rot90"]
    out = _apply_tta(model, x, ops)

    # top-left pixel seen by each view:
    #   identity -> 1 ; hflip(width) -> 2 ; vflip(height) -> 3 ;
    #   rot90(dims 2,3) -> old[0, W-1] = 2
    expected = np.mean([1.0, 2.0, 3.0, 2.0])
    assert out.shape == (1, 2)
    assert out[0, 0].item() == expected


def test_apply_tta_identity_only_when_no_ops():
    x = torch.zeros(1, 1, 2, 2)
    x[0, 0, 0, 0] = 5.0
    out = _apply_tta(_CornerModel(), x, [])
    assert out[0, 0].item() == 5.0


# --------------------------------------------------------------------------
# Item 4 — rank-space blending for a pure-ranking (ROC AUC) metric
# --------------------------------------------------------------------------

def test_rank_normalize_is_monotonic_and_unit_ranged():
    arr = np.array([[0.10], [0.90], [0.30], [0.30]], dtype=float)
    ranks = rank_normalize(arr)
    assert ranks.min() >= 0.0 and ranks.max() <= 1.0
    # smallest value gets smallest rank, largest gets largest
    assert ranks[0, 0] < ranks[2, 0] < ranks[1, 0]
    # ties share the same (average) rank
    assert ranks[2, 0] == ranks[3, 0]


def test_rank_normalize_preserves_per_column_auc():
    y_true = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=float)
    y_prob = np.array(
        [[0.9, 0.2], [0.1, 0.8], [0.6, 0.3], [0.2, 0.7]], dtype=float
    )
    labels = ["a", "b"]
    prob_auc = mean_column_auc(y_true, y_prob, labels)["metric_value"]
    rank_auc = mean_column_auc(y_true, rank_normalize(y_prob), labels)["metric_value"]
    assert abs(prob_auc - rank_auc) < 1e-9


def test_rank_normalize_rescales_compressed_scores():
    # A model whose probabilities are compressed into a tiny band still gets
    # a full 0..1 spread after rank-normalisation, so it is not drowned out
    # when blended with a wide-range model.
    compressed = np.array([[0.500], [0.501], [0.499]], dtype=float)
    ranks = rank_normalize(compressed)
    assert ranks.max() - ranks.min() == 1.0


def test_apply_blend_space_prob_is_identity():
    arrays = [np.array([[0.2, 0.8]]), np.array([[0.5, 0.5]])]
    out = _apply_blend_space(arrays, "prob")
    assert np.array_equal(out[0], arrays[0])
    assert np.array_equal(out[1], arrays[1])


def test_apply_blend_space_rank_transforms_each_array():
    arrays = [np.array([[0.10], [0.90], [0.30]])]
    out = _apply_blend_space(arrays, "rank")
    assert np.array_equal(out[0], rank_normalize(arrays[0]))


# --------------------------------------------------------------------------
# Item 1 — high-resolution inference via positional-encoding interpolation
# --------------------------------------------------------------------------

class _RecordingHF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.captured_kwargs = None

    def forward(self, pixel_values=None, **kwargs):
        self.captured_kwargs = kwargs

        class _Out:
            pooler_output = None
            last_hidden_state = torch.zeros(pixel_values.shape[0], 1, 4)

        return _Out()


def test_hf_backbone_requests_pos_encoding_interpolation():
    inner = _RecordingHF()
    backbone = _HFBackbone(inner)
    backbone(torch.zeros(2, 3, 384, 384))
    assert inner.captured_kwargs.get("interpolate_pos_encoding") is True

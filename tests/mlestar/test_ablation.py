"""Tests for component impact selection and source-isolated refinement."""

from __future__ import annotations

from mlestar.ablation import make_ablation_source, rank_component_impact
from mlestar.refinement import apply_component_patch, component_body


SOURCE = """# MLESTAR_COMPONENT:data_loading:START
load
# MLESTAR_COMPONENT:data_loading:END
# MLESTAR_COMPONENT:data_preparation:START
prepare
# MLESTAR_COMPONENT:data_preparation:END
# MLESTAR_COMPONENT:model:START
model
# MLESTAR_COMPONENT:model:END
# MLESTAR_COMPONENT:training:START
train
# MLESTAR_COMPONENT:training:END
# MLESTAR_COMPONENT:prediction:START
predict
# MLESTAR_COMPONENT:prediction:END
"""


def test_largest_metric_drop_selects_component_with_metric_direction() -> None:
    scores = {"data_loading": 0.80, "data_preparation": 0.71, "model": 0.78, "training": 0.76, "prediction": 0.79}
    assert rank_component_impact(0.82, scores, greater_is_better=True) == "data_preparation"
    assert rank_component_impact(0.20, {"model": 0.35, "training": 0.24}, greater_is_better=False) == "model"


def test_refinement_replaces_only_selected_marker_body() -> None:
    after = apply_component_patch(SOURCE, "training", "scheduler = cosine_scheduler(optimizer)")
    assert component_body(after, "model") == component_body(SOURCE, "model")
    assert "cosine_scheduler" in component_body(after, "training")
    assert component_body(make_ablation_source(SOURCE, "model", "baseline_model"), "model").strip() == "baseline_model"

"""test_integration.py — recipe 接进 build_task_list（从仓库根跑，recipe 可导入）。"""

from __future__ import annotations

import pytest

rag = pytest.importorskip("retrieval.rag_retrieval")


def _result():
    return {
        "backbone": "efficientnet", "pretrained": "efficientnet_b0_imagenet",
        "head": "classification_head", "loss": "cross_entropy_loss",
        "optimizer": "adamw", "finetune_strategy": "full",
        "freeze_viable": False, "scratch_viable": True, "alt_backbones": [], "score": 0.7,
    }


def test_recipe_injected_when_input_json_given():
    g = rag.build_graph()
    inp = {"task_type": "classification", "data_size": "small", "priority": "balanced",
           "constraints": {"fine_grained": True},
           "data_stats": {"resolution_tier": "high", "color_mode": "rgb"}}
    tl = rag.build_task_list(_result(), g, fmt="nl", input_json=inp)
    mc = tl["model_config"]
    assert "recipe" in mc and "recipe_provenance" in mc
    assert set(mc["recipe"]) == {
        "epochs", "image_size", "learning_rate", "augmentation", "early_stopping_patience"
    }
    assert mc["recipe"]["image_size"] == 384          # fine_grained + high_res 上调
    # image_size/lr/epochs/early stop 也提到顶层，Module 4 模板直接生效
    assert mc["image_size"] == 384
    assert mc["learning_rate"] == mc["recipe"]["learning_rate"]
    assert mc["recommended_epochs"] == mc["recipe"]["epochs"]
    assert mc["early_stopping_patience"] == mc["recipe"]["early_stopping_patience"]


def test_no_recipe_without_input_json():
    # 向后兼容：不传 input_json → 无 recipe 字段（现有调用方不受影响）
    g = rag.build_graph()
    tl = rag.build_task_list(_result(), g, fmt="nl")
    assert "recipe" not in tl["model_config"]

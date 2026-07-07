"""test_recipe.py — recipe 层：子决策 + 不变量 + golden 端到端，全离线。"""

from __future__ import annotations

import itertools

import pytest

from recipe import build_recipe, tables


def _cfg(backbone="resnet", strategy="full", pretrained=True):
    c = {"backbone": backbone, "finetune_strategy": strategy}
    if pretrained:
        c["pretrained_hf_id"] = "hf/x"
    return c


def _inp(data_size="medium", priority="balanced", **constraints):
    return {"data_size": data_size, "priority": priority, "constraints": constraints}


# ── 子决策 ──────────────────────────────────────────────────────────────────
def test_epochs_migration_unchanged():
    # 收编后数值必须与原 pipeline 表一致（回归）
    assert tables.derive_recommended_epochs("small", "finetune", True) == 40
    assert tables.derive_recommended_epochs("large", "head_only", True) == 8
    assert tables.derive_recommended_epochs("medium", None, False) == 30   # scratch
    assert tables.derive_recommended_epochs("weird", "x", True) == 15      # 缺省


def test_lr_table_hit():
    r, _ = build_recipe(_cfg("vit", "full"), _inp(), {}, None)
    assert r["learning_rate"] == 3e-5          # transformer finetune
    r2, _ = build_recipe(_cfg("resnet", "head_only"), _inp(), {}, None)
    assert r2["learning_rate"] == 1e-3         # cnn head_only


def test_image_size_bump_and_snap():
    # ViT + fine_grained + high_res → 224→384，且 /16 合法
    r, prov = build_recipe(_cfg("vit"), _inp("small", fine_grained=True), {},
                           {"resolution_tier": "high", "color_mode": "rgb"})
    assert r["image_size"] == 384
    # speed 优先 → 不上调
    r2, _ = build_recipe(_cfg("vit"), _inp("small", "speed", fine_grained=True), {},
                         {"resolution_tier": "high"})
    assert r2["image_size"] == 224


def test_checkpoint_resolution_overrides_family_default():
    # backbone_facts 带 image_size 时优先用它
    r, prov = build_recipe(_cfg("resnet"), _inp(), {"image_size": 260}, None)
    assert r["image_size"] == 260 and "ckpt_default=260" in prov["image_size"]


# ── 不变量（对所有构造输入必成立）─────────────────────────────────────────────
_BACKBONES = ["resnet", "efficientnet", "vit", "swin_transformer", "dinov2", "convnext"]
_SIZES = ["small", "medium", "large"]
_STRAT = ["full", "head_only"]


@pytest.mark.parametrize("backbone,size,strat", itertools.product(_BACKBONES, _SIZES, _STRAT))
def test_invariant_image_size_divisible(backbone, size, strat):
    r, _ = build_recipe(_cfg(backbone, strat), _inp(size, fine_grained=True), {},
                        {"resolution_tier": "high", "color_mode": "rgb"})
    div = tables.image_divisor(backbone)
    if div:
        assert r["image_size"] % div == 0, f"{backbone} size {r['image_size']} 不满足 /{div}"


@pytest.mark.parametrize("size,strat", itertools.product(_SIZES, _STRAT))
def test_invariant_head_only_never_heavy(size, strat):
    r, _ = build_recipe(_cfg("resnet", strat), _inp(size), {}, None)
    if strat == "head_only":
        assert r["augmentation"]["tier"] != "heavy"


def test_invariant_grayscale_forces_color_off():
    for size in _SIZES:
        r, _ = build_recipe(_cfg("resnet"), _inp(size), {},
                            {"color_mode": "grayscale"})
        assert r["augmentation"]["invariance"]["color"] is False


def test_invariant_fine_grained_crop_floor():
    for size in _SIZES:
        r, _ = build_recipe(_cfg("resnet"), _inp(size, fine_grained=True), {}, None)
        assert r["augmentation"]["invariance"]["crop_scale_min"] >= 0.5


def test_invariant_no_data_stats_degrades_gracefully():
    # 缺 data_stats 不崩，image_size 退化为默认，信号缺失被标注
    r, prov = build_recipe(_cfg("dinov2"), _inp("large", "speed"), {}, None)
    assert r["image_size"] % 14 == 0                    # /14 吸附仍执行
    assert "res_signal_missing" in prov["image_size"]
    assert "color_signal_missing" in prov["augmentation"]


# ── golden 端到端 ───────────────────────────────────────────────────────────
def test_golden_few_shot_forces_heavy():
    r, _ = build_recipe(_cfg("vit"), _inp("large", few_shot=True), {}, None)
    assert r["augmentation"]["tier"] == "heavy"          # few_shot 覆盖 large→light


def test_golden_recipe_has_all_fields():
    r, prov = build_recipe(_cfg("efficientnet"), _inp("medium"), {},
                           {"resolution_tier": "medium", "color_mode": "rgb"})
    assert set(r) == {"epochs", "image_size", "learning_rate", "augmentation"}
    assert set(prov) == {"epochs", "image_size", "learning_rate", "augmentation"}
    assert set(r["augmentation"]) == {"tier", "invariance", "schedule"}

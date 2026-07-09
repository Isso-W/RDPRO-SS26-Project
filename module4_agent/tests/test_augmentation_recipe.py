"""test_augmentation_recipe.py — 生成的 train.py 消费 recipe 增广（维度 A）。

真导入生成的 train.py，驱动 _build_image_transform，逐 op 校验不变性掩码是否
被尊重（灰度不抖色、文档不翻转、细粒度裁剪有下限、tier=none 不增广）。
"""

from __future__ import annotations

import importlib
import sys

import pytest

from module4_agent.code_generator import generate_files
from module4_agent.tests.test_code_generator import _specs


def _import_generated(tmp_path, monkeypatch, module_name):
    generated = generate_files(_specs(), llm_provider="none")
    for name, content in generated.files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    for mod in ("train", "evaluate", "model", "smoke_data", "utils"):
        sys.modules.pop(mod, None)
    return importlib.import_module(module_name)


def _op_names(transform):
    return [type(op).__name__ for op in transform.transforms]


def _recipe_config(tier, invariance, schedule="constant", image_size=224):
    return {
        "task_type": "classification",
        "image_size": image_size,
        "recipe": {
            "augmentation": {"tier": tier, "invariance": invariance, "schedule": schedule},
        },
    }


_FULL = {"hflip": True, "vflip": True, "rot90": True, "color": True, "crop_scale_min": 0.4}


# ── 结构化增广被消费 ─────────────────────────────────────────────────────────
def test_structured_recipe_builds_all_ops(tmp_path, monkeypatch):
    pytest.importorskip("torchvision")
    train = _import_generated(tmp_path, monkeypatch, "train")
    t = train._build_image_transform(_recipe_config("heavy", dict(_FULL)), "train")
    names = _op_names(t)
    assert "RandomResizedCrop" in names
    assert "RandomHorizontalFlip" in names and "RandomVerticalFlip" in names
    assert "RandomRotation" in names          # rot90 → 大角度旋转
    assert "ColorJitter" in names
    assert "RandomErasing" in names           # heavy 档
    # crop 下限透传进 RandomResizedCrop.scale
    crop = next(op for op in t.transforms if type(op).__name__ == "RandomResizedCrop")
    assert abs(crop.scale[0] - 0.4) < 1e-6


def test_grayscale_veto_drops_color(tmp_path, monkeypatch):
    pytest.importorskip("torchvision")
    train = _import_generated(tmp_path, monkeypatch, "train")
    inv = dict(_FULL, color=False)            # 灰度 veto 由 recipe 层置 color=False
    t = train._build_image_transform(_recipe_config("medium", inv), "train")
    assert "ColorJitter" not in _op_names(t)


def test_document_veto_drops_flips(tmp_path, monkeypatch):
    pytest.importorskip("torchvision")
    train = _import_generated(tmp_path, monkeypatch, "train")
    inv = {"hflip": False, "vflip": False, "rot90": False, "color": True, "crop_scale_min": 0.6}
    names = _op_names(train._build_image_transform(_recipe_config("medium", inv), "train"))
    assert "RandomHorizontalFlip" not in names and "RandomVerticalFlip" not in names
    assert "RandomRotation" not in names


def test_fine_grained_crop_floor_respected(tmp_path, monkeypatch):
    pytest.importorskip("torchvision")
    train = _import_generated(tmp_path, monkeypatch, "train")
    # recipe 层对 fine_grained 强制 crop_scale_min≥0.5；transform 必须照搬
    inv = dict(_FULL, crop_scale_min=0.5)
    t = train._build_image_transform(_recipe_config("heavy", inv), "train")
    crop = next(op for op in t.transforms if type(op).__name__ == "RandomResizedCrop")
    assert crop.scale[0] >= 0.5


def test_tier_none_disables_augmentation(tmp_path, monkeypatch):
    pytest.importorskip("torchvision")
    train = _import_generated(tmp_path, monkeypatch, "train")
    inv = {"hflip": False, "vflip": False, "rot90": False, "color": False, "crop_scale_min": 1.0}
    names = _op_names(train._build_image_transform(_recipe_config("none", inv), "train"))
    assert names == ["Resize", "ToTensor", "Normalize"]


def test_non_train_split_is_deterministic(tmp_path, monkeypatch):
    pytest.importorskip("torchvision")
    train = _import_generated(tmp_path, monkeypatch, "train")
    names = _op_names(train._build_image_transform(_recipe_config("heavy", dict(_FULL)), "test"))
    assert names == ["Resize", "CenterCrop", "ToTensor", "Normalize"]   # 无随机 op


# ── 向后兼容：旧字符串路径不受影响 ───────────────────────────────────────────
def test_legacy_string_augmentation_still_works(tmp_path, monkeypatch):
    pytest.importorskip("torchvision")
    train = _import_generated(tmp_path, monkeypatch, "train")
    strong = train._build_image_transform(
        {"task_type": "classification", "image_size": 224, "augmentation": "strong"}, "train"
    )
    assert "RandomErasing" in _op_names(strong)
    basic = train._build_image_transform(
        {"task_type": "classification", "image_size": 224, "augmentation": "basic"}, "train"
    )
    assert _op_names(basic) == ["Resize", "RandomHorizontalFlip", "ToTensor", "Normalize"]


# ── schedule helper ─────────────────────────────────────────────────────────
def test_schedule_helper_reads_taper(tmp_path, monkeypatch):
    train = _import_generated(tmp_path, monkeypatch, "train")
    cfg = _recipe_config("medium", dict(_FULL), schedule="taper_last_20pct")
    assert train._augmentation_schedule(cfg) == "taper_last_20pct"
    assert train._augmentation_schedule({"augmentation": "strong"}) == ""

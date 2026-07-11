"""test_backbone_fallback.py — real training must fail loud, not silently train a toy.

When use_pretrained=true but no real backbone can be loaded (e.g. a gated DINOv3
checkpoint with no HF token), load_backbone must raise instead of silently
returning a random TinyBackbone. Smoke mode and the explicit override still fall back.
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
    for mod in ("train", "model", "model_utils", "smoke_data", "utils"):
        sys.modules.pop(mod, None)
    return importlib.import_module(module_name)


def _cfg(**over):
    # dinov3 has no torchvision fallback; empty hf_id skips the (network) HF attempt,
    # so this exercises the "nothing real could load" path deterministically offline.
    base = {
        "backbone": "dinov3",
        "use_pretrained": True,
        "pretrained_hf_id": "",
        "offline_smoke": False,
        "image_size": 32,
    }
    base.update(over)
    return base


def test_real_pretrained_load_failure_raises(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    mu = _import_generated(tmp_path, monkeypatch, "model_utils")
    with pytest.raises(RuntimeError, match="Refusing to silently fall back"):
        mu.load_backbone(_cfg())


def test_override_flag_allows_fallback(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    mu = _import_generated(tmp_path, monkeypatch, "model_utils")
    backbone, channels = mu.load_backbone(_cfg(allow_backbone_fallback=True))
    assert type(backbone).__name__ == "TinyBackbone" and channels > 0


def test_smoke_mode_still_falls_back(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    mu = _import_generated(tmp_path, monkeypatch, "model_utils")
    # offline_smoke forces pretrained off → no real backbone expected → no raise
    backbone, _ = mu.load_backbone(_cfg(offline_smoke=True))
    assert type(backbone).__name__ == "TinyBackbone"


def test_non_pretrained_does_not_raise(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    mu = _import_generated(tmp_path, monkeypatch, "model_utils")
    backbone, _ = mu.load_backbone(_cfg(use_pretrained=False))
    assert type(backbone).__name__ == "TinyBackbone"


def test_backbone_load_info_reports_source_and_params(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    mu = _import_generated(tmp_path, monkeypatch, "model_utils")
    bb, ch = mu.load_backbone(_cfg(allow_backbone_fallback=True))   # -> tiny_fallback

    class _M(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
            self.head = nn.Linear(ch, 2)

    info = mu.backbone_load_info(_M(bb), _cfg(allow_backbone_fallback=True))
    assert info["source"] == "tiny_fallback"
    assert info["actual_model"] == "TinyBackbone"
    assert info["total_params"] > 0 and info["trainable_params"] >= 0
    assert info["fallback_reason"]   # non-empty reason recorded

"""augment.py — 三维增广解析：强度档 ⊗ 不变性掩码 ⊗ 日程。

规格见 recipe_layer_plan.md §4。输出结构化配置，Module 4 翻成 torchvision v2
transform（recipe 只产配置、不产代码）。
"""

from __future__ import annotations

from recipe.tables import training_mode

_TIER_ORDER = ["none", "light", "medium", "heavy"]

# 每档的不变性默认（"能加哪些"的开关），之后被安全 veto 覆盖
_TIER_INVARIANCE = {
    "none":   {"hflip": False, "vflip": False, "rot90": False, "color": False, "crop_scale_min": 1.0},
    "light":  {"hflip": True,  "vflip": False, "rot90": False, "color": False, "crop_scale_min": 0.8},
    "medium": {"hflip": True,  "vflip": False, "rot90": False, "color": True,  "crop_scale_min": 0.6},
    "heavy":  {"hflip": True,  "vflip": False, "rot90": False, "color": True,  "crop_scale_min": 0.4},
}

# domain → 不变性 veto（v0：Module 1 不抽 domain，故均为骨架，实际不触发）
_DOMAIN_VETO = {
    "satellite":  {"vflip": True, "rot90": True},   # 无固定方位
    "aerial":     {"vflip": True, "rot90": True},
    "pathology":  {"vflip": True, "rot90": True},
    "microscopy": {"vflip": True, "rot90": True},
    "document":   {"hflip": False, "vflip": False, "rot90": False},  # 翻转改标签
    "digit":      {"hflip": False, "vflip": False, "rot90": False},
    "ocr":        {"hflip": False, "vflip": False, "rot90": False},
}

FINE_GRAINED_CROP_FLOOR = 0.5   # 细粒度：裁剪不得低于此（激进裁剪会裁掉判别特征）


def _demote(tier: str) -> str:
    i = _TIER_ORDER.index(tier)
    return _TIER_ORDER[max(0, i - 1)]


def _strength_tier(data_size: str, mode: str, few_shot: bool) -> tuple[str, str]:
    """→ (tier, provenance)。"""
    base = {"small": "heavy", "medium": "medium", "large": "light"}.get(data_size, "medium")
    prov = f"data_size={data_size}→{base}"
    if few_shot:                       # few_shot 强制 heavy（含 RandAugment）
        return "heavy", prov + " | few_shot→heavy"
    if mode == "head_only":            # 冻结骨干适应不了强畸变 → 压一档
        demoted = _demote(base)
        return demoted, prov + f" | head_only→{demoted}"
    return base, prov


def build_augment(config: dict, input_json: dict, data_stats: dict | None) -> tuple[dict, str]:
    """→ ({tier, invariance, schedule}, provenance)。"""
    c = input_json.get("constraints", {})
    data_size = input_json.get("data_size", "medium")
    use_pretrained = bool(config.get("pretrained_hf_id"))
    mode = training_mode(config.get("finetune_strategy"), use_pretrained)

    tier, tier_prov = _strength_tier(data_size, mode, bool(c.get("few_shot")))
    invariance = dict(_TIER_INVARIANCE[tier])

    # ── 不变性 veto（硬规则，优先级最高）─────────────────────────────────
    veto_notes = []
    color_mode = (data_stats or {}).get("color_mode")
    if color_mode == "grayscale":
        invariance["color"] = False
        veto_notes.append("grayscale→color=False")
    elif color_mode is None:
        veto_notes.append("color_signal_missing")

    domain = input_json.get("domain") or c.get("domain")   # v0 恒 None
    if domain in _DOMAIN_VETO:
        invariance.update(_DOMAIN_VETO[domain])
        veto_notes.append(f"domain={domain}→veto")
    else:
        veto_notes.append("domain_signal_missing")

    if c.get("fine_grained"):
        invariance["crop_scale_min"] = max(invariance["crop_scale_min"], FINE_GRAINED_CROP_FLOOR)
        veto_notes.append(f"fine_grained→crop_scale_min≥{FINE_GRAINED_CROP_FLOOR}")

    # ── 日程 ─────────────────────────────────────────────────────────────
    schedule = "taper_last_20pct" if data_size in ("small", "medium") else "constant"

    provenance = f"tier[{tier_prov}]; invariance[{', '.join(veto_notes)}]; schedule[{data_size}→{schedule}]"
    return {"tier": tier, "invariance": invariance, "schedule": schedule}, provenance

"""Augmentation recipe resolver.

The recipe output is declarative: Module 4 translates this structure into
torchvision transforms while keeping the old string-based augmentation path.
"""

from __future__ import annotations

from copy import deepcopy


_TIER_ORDER = ["none", "light", "medium", "heavy"]

_DEFAULT_INVARIANCE = {
    "none": {
        "hflip": False,
        "vflip": False,
        "rot90": False,
        "color": False,
        "crop_scale_min": 1.0,
        "randaugment": False,
        "mixup_cutmix": False,
        "random_erasing": False,
    },
    "light": {
        "hflip": True,
        "vflip": False,
        "rot90": False,
        "color": False,
        "crop_scale_min": 0.8,
        "randaugment": False,
        "mixup_cutmix": False,
        "random_erasing": False,
    },
    "medium": {
        "hflip": True,
        "vflip": False,
        "rot90": False,
        "color": True,
        "crop_scale_min": 0.75,
        "randaugment": False,
        "mixup_cutmix": False,
        "random_erasing": True,
    },
    "heavy": {
        "hflip": True,
        "vflip": False,
        "rot90": False,
        "color": True,
        "crop_scale_min": 0.65,
        "randaugment": True,
        "mixup_cutmix": True,
        "random_erasing": True,
    },
}


def _downshift(tier: str) -> str:
    idx = max(0, _TIER_ORDER.index(tier) - 1)
    return _TIER_ORDER[idx]


def resolve_augmentation(
    *,
    data_size: str,
    finetune_strategy: str | None,
    constraints: dict | None = None,
    data_stats: dict | None = None,
) -> tuple[dict, str]:
    """Return (augmentation_recipe, provenance)."""
    constraints = constraints or {}
    data_stats = data_stats or {}
    notes: list[str] = []

    tier = {"small": "heavy", "medium": "medium", "large": "light"}.get(
        str(data_size or "medium").lower(),
        "medium",
    )
    notes.append(f"tier_by_data_size={tier}")

    few_shot = bool(constraints.get("few_shot"))
    if few_shot:
        tier = "heavy"
        notes.append("few_shot→heavy+RandAugment")

    if finetune_strategy == "head_only" and tier == "heavy":
        tier = "medium"
        notes.append("head_only cap: heavy→medium")
    elif finetune_strategy == "head_only":
        shifted = _downshift(tier)
        if shifted != tier:
            notes.append(f"head_only downshift: {tier}→{shifted}")
            tier = shifted

    invariance = deepcopy(_DEFAULT_INVARIANCE[tier])
    if few_shot:
        invariance["randaugment"] = True

    if data_stats.get("color_mode") == "grayscale":
        invariance["color"] = False
        notes.append("grayscale veto: color=False")
    elif "color_mode" not in data_stats:
        notes.append("signal_missing: color_mode")

    domain = data_stats.get("domain") or constraints.get("domain")
    if domain:
        d = str(domain).lower()
        if d in {"satellite", "aerial", "pathology", "microscopy"}:
            invariance["vflip"] = True
            invariance["rot90"] = True
            notes.append(f"domain={d}: vflip+rot90")
        elif d in {"document", "digit", "ocr"}:
            invariance["hflip"] = False
            invariance["vflip"] = False
            invariance["rot90"] = False
            notes.append(f"domain={d}: flip/rot veto")
    else:
        notes.append("domain_signal_missing")

    if constraints.get("medical") and str(domain or "").lower() in {"xray", "chest_xray"}:
        invariance["hflip"] = False
        notes.append("medical orientation veto: hflip=False")

    if constraints.get("fine_grained"):
        old = float(invariance.get("crop_scale_min", 0.8))
        invariance["crop_scale_min"] = max(old, 0.5)
        notes.append(f"fine_grained crop floor: {old:g}→{invariance['crop_scale_min']:g}")

    schedule = "constant" if str(data_size or "medium").lower() == "large" else "taper_last_20pct"
    notes.append(f"schedule={schedule}")
    return {"tier": tier, "invariance": invariance, "schedule": schedule}, " | ".join(notes)

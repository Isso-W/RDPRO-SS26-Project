"""Dataset fingerprint — semantic signals used to find similar past tasks.

Derived from the Module 2 analysis report (+ optional Module 3 input), reusing the
image statistics Module 2 already computes (resolution, colour mode) that the
current selection drops. Two fingerprints are compared by `fingerprint_distance`;
task_type is a hard gate (different tasks are never "similar").
"""

from __future__ import annotations

import math

_SIZE_ORDER = ["small", "medium", "large"]
_RES_ORDER = ["low", "medium", "high"]


def _resolution_tier(m2_report: dict) -> str:
    """low (<160) / medium (<384) / high, from the average of avg width & height."""
    w = float(m2_report.get("avg_width", 0) or 0)
    h = float(m2_report.get("avg_height", 0) or 0)
    avg = (w + h) / 2 if (w or h) else 0
    if avg <= 0:
        return "medium"
    if avg < 160:
        return "low"
    if avg < 384:
        return "medium"
    return "high"


def _color_mode(m2_report: dict) -> str:
    """rgb / grayscale, from the dominant PIL mode in mode_distribution."""
    dist = m2_report.get("mode_distribution") or {}
    if not dist:
        return "rgb"
    dominant = max(dist, key=dist.get)
    return "grayscale" if str(dominant).upper() in {"L", "1", "LA"} else "rgb"


def dataset_fingerprint(m2_report: dict, m3_input: dict | None = None) -> dict:
    """Build a dataset fingerprint from the Module 2 report (+ optional M3 input)."""
    m3_input = m3_input or {}
    constraints = m3_input.get("constraints", {})

    class_dist = m2_report.get("class_distribution") or {}
    num_classes = m2_report.get("num_classes") or len(class_dist) or 0
    imbalance = bool(constraints.get("class_imbalance", False))

    return {
        "task_type": m3_input.get("task_type") or m2_report.get("annotation_format") or "classification",
        "num_classes": int(num_classes),
        "data_size": m3_input.get("data_size", "medium"),
        "total_images": int(m2_report.get("total_images", 0) or 0),
        "class_imbalance": imbalance,
        "resolution_tier": _resolution_tier(m2_report),
        "color_mode": _color_mode(m2_report),
    }


def _ordinal_dist(a: str, b: str, order: list[str]) -> float:
    if a in order and b in order:
        return abs(order.index(a) - order.index(b)) / max(len(order) - 1, 1)
    return 0.0 if a == b else 1.0


def fingerprint_distance(a: dict, b: dict) -> float:
    """Distance in [0, inf]; lower = more similar. Different task_type → inf."""
    if a.get("task_type") != b.get("task_type"):
        return math.inf

    # log-scaled class-count difference, normalised softly
    ca, cb = max(int(a.get("num_classes", 0)), 1), max(int(b.get("num_classes", 0)), 1)
    class_d = abs(math.log10(ca) - math.log10(cb)) / 2.0

    size_d = _ordinal_dist(a.get("data_size", "medium"), b.get("data_size", "medium"), _SIZE_ORDER)
    res_d = _ordinal_dist(a.get("resolution_tier", "medium"), b.get("resolution_tier", "medium"), _RES_ORDER)
    imb_d = 0.0 if a.get("class_imbalance") == b.get("class_imbalance") else 1.0
    color_d = 0.0 if a.get("color_mode") == b.get("color_mode") else 1.0

    # weighted sum — class count & data size dominate, the rest are nudges
    return 2.0 * class_d + 1.5 * size_d + 0.5 * res_d + 0.5 * imb_d + 0.5 * color_d

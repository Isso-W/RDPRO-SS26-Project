"""Data catalog for kb_mining, plus a read-only candidate listing helper.

Data groups:
  1. COMPETITIONS   - competition catalog and feature cards
  2. FAMILY_RELEASE - backbone-family release dates for coexistence filtering
  3. MODEL_ALIASES / LOSS_ALIASES - ordered raw-string to KB-id regex maps
  4. list_recent_cv_candidates() - candidate listing from Competitions.csv

Competition start/end/team counts were checked against the local Meta Kaggle
dump on 2026-07-03. Traits with traits_verified=False are preliminary and must
be reviewed before being treated as final evidence.
"""

from __future__ import annotations

import re
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Competition catalog and feature cards.
# ═══════════════════════════════════════════════════════════════════════════════
#
# Trait keys mirror condition keys without the "=True" suffix. fine_grained is
# not yet a valid runtime condition, so decide.py routes it to schema-ext.
# data_size is preliminary and is used for archetype queries and field-fix
# evidence, not as an independently mined trait.
#
# Any traits_verified=False entry still needs manual review against competition
# overview/data pages and write-ups.

COMPETITIONS: dict[str, dict] = {
    "cassava-leaf-disease-classification": {
        "slug": "cassava-leaf-disease-classification",
        "title": "Cassava Leaf Disease Classification",
        "start": "2020-11", "end": "2021-02", "task_type": "classification",
        "traits": {"fine_grained": True, "class_imbalance": True, "medical": False,
                   "data_size": "medium"},
        "traits_verified": True,
        "notes": "Five-class fine-grained leaf disease task with mild imbalance; enough forum evidence.",
    },
    "plant-pathology-2021-fgvc8": {
        "slug": "plant-pathology-2021-fgvc8",
        "title": "Plant Pathology 2021 - FGVC8",
        "start": "2021-03", "end": "2021-05", "task_type": "classification",
        "traits": {"fine_grained": True, "class_imbalance": False, "medical": False,
                   "multi_label": True, "data_size": "medium"},
        "traits_verified": True,
        "notes": "Multi-label plant disease task; loss evidence is kept in a separate multi_label cell.",
    },
    "herbarium-2022-fgvc9": {
        "slug": "herbarium-2022-fgvc9",
        "title": "Herbarium 2022 - FGVC9",
        "start": "2022-02", "end": "2022-05", "task_type": "classification",
        "traits": {"fine_grained": True, "class_imbalance": True, "medical": False,
                   "data_size": "large"},
        "traits_verified": True,
        "notes": "15k+ species with a heavy long tail; write-up count may be low.",
    },
    "sorghum-id-fgvc-9": {
        "slug": "sorghum-id-fgvc-9",
        "title": "Sorghum -100 Cultivar Identification - FGVC 9",
        "start": "2022-03", "end": "2022-05", "task_type": "classification",
        "traits": {"fine_grained": True, "class_imbalance": False, "medical": False,
                   "data_size": "medium"},
        "traits_verified": True,
        "notes": "Fine-grained 100-cultivar task; check write-up coverage.",
    },
    "paddy-disease-classification": {
        "slug": "paddy-disease-classification",
        "title": "Paddy Doctor: Paddy Disease Classification",
        "start": "2022-04", "end": "2022-08", "task_type": "classification",
        "traits": {"fine_grained": True, "class_imbalance": False, "medical": False,
                   "data_size": "small"},
        "traits_verified": True,
        "notes": "~10k images and 10 classes.",
    },
    "happy-whale-and-dolphin": {
        "slug": "happy-whale-and-dolphin",
        "title": "Happywhale - Whale and Dolphin Identification",
        "start": "2022-02", "end": "2022-04", "task_type": "classification",
        "traits": {"fine_grained": True, "class_imbalance": True, "medical": False,
                   "data_size": "large"},
        "traits_verified": True,
        "loss_voting": False,  # metric-learning-heavy; loss signal is not useful for classifier choice
        "notes": "Metric-learning-heavy individual re-id task; keep backbone votes but exclude loss voting.",
    },
    "mayo-clinic-strip-ai": {
        "slug": "mayo-clinic-strip-ai",
        "title": "Mayo Clinic - STRIP AI",
        "start": "2022-07", "end": "2022-10", "task_type": "classification",
        "traits": {"fine_grained": False, "class_imbalance": False, "medical": True,
                   "data_size": "small"},
        "traits_verified": True,
        "notes": "Small-sample thrombosis pathology WSI task.",
    },
    "rsna-breast-cancer-detection": {
        "slug": "rsna-breast-cancer-detection",
        "title": "RSNA Screening Mammography Breast Cancer Detection",
        "start": "2022-11", "end": "2023-02", "task_type": "classification",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": True,
                   "data_size": "large"},
        "traits_verified": True,
        "notes": "Mammography task with extreme positive-class imbalance.",
    },
    "UBC-OCEAN": {
        "slug": "UBC-OCEAN",  # slug is case-sensitive in the dump
        "title": "UBC Ovarian Cancer Subtype Classification and Outlier Detection",
        "start": "2023-10", "end": "2024-01", "task_type": "classification",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": True,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "Ovarian cancer subtype pathology WSI task with outlier detection.",
    },
    "isic-2024-challenge": {
        "slug": "isic-2024-challenge",
        "title": "ISIC 2024 - Skin Cancer Detection with 3D-TBP",
        "start": "2024-06", "end": "2024-09", "task_type": "classification",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": True,
                   "data_size": "large"},
        "traits_verified": False,
        "notes": "Skin-cancer detection with extreme class imbalance.",
    },
    "hms-harmful-brain-activity-classification": {
        "slug": "hms-harmful-brain-activity-classification",
        "title": "HMS - Harmful Brain Activity Classification",
        "start": "2024-01", "end": "2024-04", "task_type": "classification",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": True,
                   "data_size": "large"},
        "traits_verified": False,
        "notes": "EEG spectrogram images rather than natural images; transfer to natural-image tasks is uncertain.",
    },
    "rsna-2024-lumbar-spine-degenerative-classification": {
        "slug": "rsna-2024-lumbar-spine-degenerative-classification",
        "title": "RSNA 2024 Lumbar Spine Degenerative Classification",
        "start": "2024-05", "end": "2024-10", "task_type": "classification",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": True,
                   "data_size": "large"},
        "traits_verified": False,
        "notes": "Lumbar MRI grading with multiple outputs and localization subtasks.",
    },
    "fathomnet-2025": {
        "slug": "fathomnet-2025",
        "title": "FathomNet 2025",
        "start": "2025-03", "end": "2025-05", "task_type": "classification",
        "traits": {"fine_grained": True, "class_imbalance": False, "medical": False,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "Marine-species hierarchical classification; very low team count, likely low write-up coverage.",
    },
    "rsna-intracranial-aneurysm-detection": {
        "slug": "rsna-intracranial-aneurysm-detection",
        "title": "RSNA Intracranial Aneurysm Detection",
        "start": "2025-07", "end": "2025-10", "task_type": "classification",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": True,
                   "data_size": "large"},
        "traits_verified": True,
        "notes": "CT/MR intracranial aneurysm task with localization subtasks; enough write-up coverage.",
    },

    # Extension batch (2026-07-04): reduce medical skew and cover detection/segmentation.
    # Traits are preliminary; data_size and class_imbalance especially need review.

    # Non-medical classification: BirdCLEF spectrogram tasks with long-tail labels.
    "birdclef-2021": {
        "slug": "birdclef-2021", "title": "BirdCLEF 2021 - Birdcall Identification",
        "start": "2021-04", "end": "2021-06", "task_type": "classification",
        "traits": {"fine_grained": True, "class_imbalance": True, "medical": False,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "Bird-call mel-spectrogram classification; non-natural images and long-tail species.",
    },
    "birdclef-2022": {
        "slug": "birdclef-2022", "title": "BirdCLEF 2022",
        "start": "2022-02", "end": "2022-05", "task_type": "classification",
        "traits": {"fine_grained": True, "class_imbalance": True, "medical": False,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "Bird-call spectrogram classification with long-tail species.",
    },
    "birdclef-2023": {
        "slug": "birdclef-2023", "title": "BirdCLEF 2023",
        "start": "2023-03", "end": "2023-05", "task_type": "classification",
        "traits": {"fine_grained": True, "class_imbalance": True, "medical": False,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "Bird-call spectrogram classification with long-tail species.",
    },
    "birdclef-2024": {
        "slug": "birdclef-2024", "title": "BirdCLEF 2024",
        "start": "2024-04", "end": "2024-06", "task_type": "classification",
        "traits": {"fine_grained": True, "class_imbalance": True, "medical": False,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "Bird-call spectrogram classification with long-tail species.",
    },
    "birdclef-2025": {
        "slug": "birdclef-2025", "title": "BirdCLEF+ 2025",
        "start": "2025-03", "end": "2025-06", "task_type": "classification",
        "traits": {"fine_grained": True, "class_imbalance": True, "medical": False,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "Bird-call spectrogram classification with long-tail species.",
    },
    "landmark-recognition-2021": {
        "slug": "landmark-recognition-2021",
        "title": "Google Landmark Recognition 2021",
        "start": "2021-08", "end": "2021-10", "task_type": "classification",
        "traits": {"fine_grained": True, "class_imbalance": True, "medical": False,
                   "data_size": "large"},
        "traits_verified": False,
        "notes": "100k+ landmark recognition task; retrieval/metric-learning heavy.",
    },

    # Non-medical detection.
    "tensorflow-great-barrier-reef": {
        "slug": "tensorflow-great-barrier-reef",
        "title": "TensorFlow - Help Protect the Great Barrier Reef",
        "start": "2021-11", "end": "2022-02", "task_type": "object_detection",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": False,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "Underwater video starfish detection with sparse targets.",
    },
    "czii-cryo-et-object-identification": {
        "slug": "czii-cryo-et-object-identification",
        "title": "CZII - CryoET Object Identification",
        "start": "2024-11", "end": "2025-02", "task_type": "object_detection",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": False,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "3D cryo-ET particle localization; biological but non-medical.",
    },
    "byu-locating-bacterial-flagellar-motors-2025": {
        "slug": "byu-locating-bacterial-flagellar-motors-2025",
        "title": "BYU - Locating Bacterial Flagellar Motors 2025",
        "start": "2025-03", "end": "2025-06", "task_type": "object_detection",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": False,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "Cryo-ET tomogram bacterial flagellar motor localization.",
    },

    # Medical detection, used as evidence for DETR/YOLO/RT-DETR.
    "vinbigdata-chest-xray-abnormalities-detection": {
        "slug": "vinbigdata-chest-xray-abnormalities-detection",
        "title": "VinBigData Chest X-ray Abnormalities Detection",
        "start": "2020-12", "end": "2021-03", "task_type": "object_detection",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": True,
                   "data_size": "large"},
        "traits_verified": False,
        "notes": "Chest X-ray abnormality detection with 14 classes and imbalance.",
    },
    "siim-covid19-detection": {
        "slug": "siim-covid19-detection",
        "title": "SIIM-FISABIO-RSNA COVID-19 Detection",
        "start": "2021-05", "end": "2021-08", "task_type": "object_detection",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": True,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "Chest X-ray COVID detection plus image-level classification.",
    },

    # Non-medical segmentation.
    "vesuvius-challenge-ink-detection": {
        "slug": "vesuvius-challenge-ink-detection",
        "title": "Vesuvius Challenge - Ink Detection",
        "start": "2023-03", "end": "2023-06", "task_type": "image_segmentation",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": False,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "3D CT ink segmentation on carbonized scrolls; sparse positives.",
    },

    # Medical segmentation, used as evidence for U-Net/Mask2Former/SegFormer.
    "sartorius-cell-instance-segmentation": {
        "slug": "sartorius-cell-instance-segmentation",
        "title": "Sartorius - Cell Instance Segmentation",
        "start": "2021-10", "end": "2021-12", "task_type": "image_segmentation",
        "traits": {"fine_grained": False, "class_imbalance": False, "medical": True,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "Microscopy neuron cell instance segmentation.",
    },
    "uw-madison-gi-tract-image-segmentation": {
        "slug": "uw-madison-gi-tract-image-segmentation",
        "title": "UW-Madison GI Tract Image Segmentation",
        "start": "2022-04", "end": "2022-07", "task_type": "image_segmentation",
        "traits": {"fine_grained": False, "class_imbalance": False, "medical": True,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "MRI gastrointestinal-organ segmentation.",
    },
    "hubmap-organ-segmentation": {
        "slug": "hubmap-organ-segmentation",
        "title": "HuBMAP + HPA - Hacking the Human Body",
        "start": "2022-06", "end": "2022-09", "task_type": "image_segmentation",
        "traits": {"fine_grained": False, "class_imbalance": False, "medical": True,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "Multi-organ tissue functional-unit segmentation.",
    },
    "hubmap-hacking-the-human-vasculature": {
        "slug": "hubmap-hacking-the-human-vasculature",
        "title": "HuBMAP - Hacking the Human Vasculature",
        "start": "2023-05", "end": "2023-07", "task_type": "image_segmentation",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": True,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "Microscopy blood-vessel instance segmentation.",
    },
    "blood-vessel-segmentation": {
        "slug": "blood-vessel-segmentation",
        "title": "SenNet + HOA - Hacking the Human Vasculature in 3D",
        "start": "2023-11", "end": "2024-02", "task_type": "image_segmentation",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": True,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "3D kidney blood-vessel segmentation.",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Architecture release dates for coexistence filtering.
# ═══════════════════════════════════════════════════════════════════════════════
#
# Keys match retrieval/rag_retrieval.py backbone ids. Values are paper/release
# months. aggregate.py uses them to exclude evidence from competitions that
# started before an architecture existed.

FAMILY_RELEASE: dict[str, str] = {   # family_id -> "YYYY-MM"
    "resnet": "2015-12",  "efficientnet": "2019-05", "mobilenet_v3": "2019-05",
    "vit": "2020-10",     "swin_transformer": "2021-03", "convnext": "2022-01",
    "yolov8": "2023-01",  "detr": "2020-05",  "rt_detr": "2023-04",
    "segformer": "2021-05", "mask2former": "2021-12", "unet": "2015-05",
    "dinov2": "2023-04",  "clip_vit": "2021-01",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Component alias tables, raw strings to KB ids.
# ═══════════════════════════════════════════════════════════════════════════════
#
# Rules are ordered: earlier patterns win. Unmatched values become "unknown",
# which feeds the unknown_components side table and future node candidates. The
# mapping is code-owned rather than LLM-owned.

MODEL_ALIASES: list[tuple[str, str]] = [
    # EfficientNet and common abbreviations.
    (r"(tf_)?eff(icient)?[-_ ]?net(v2)?|effv2|efn[-_ ]?b?\d|efficient[-_ ]?b\d", "efficientnet"),
    (r"convnext",                "convnext"),
    (r"swin",                    "swin_transformer"),
    (r"(deit|beit|^vit|_vit|vit_|vision[-_ ]?transformer)", "vit"),
    (r"dinov2",                  "dinov2"),
    (r"clip",                    "clip_vit"),
    (r"(resnet|resnext|resnest|se_?resnext)", "resnet"),
    (r"mobilenet",               "mobilenet_v3"),
    # Detection/segmentation models. Order matters for RT-DETR and Mask2Former.
    (r"rt.?detr|rtdetr",         "rt_detr"),
    (r"\bdetr\b|deformable.?detr", "detr"),
    (r"yolo",                    "yolov8"),        # one YOLO node covers versions
    (r"mask2?former",            "mask2former"),
    (r"segformer",               "segformer"),
    (r"u[-_ ]?net(\+\+)?|unet",  "unet"),
    # EfficientDet, Faster R-CNN, Mask R-CNN, RetinaNet, SAM, Detectron, etc.
    # remain unknown until corresponding KB nodes are added.
]


# Family role: frame architecture vs encoder/backbone engine. Detection and
# segmentation write-ups often name both, e.g. U-Net + EfficientNet encoder.
# aggregate.py keeps the roles in separate voting pools.
#   frame  - whole detection/segmentation meta-architecture
#   engine - classifier-style backbone used as an encoder
FAMILY_ROLE: dict[str, str] = {
    # Frame architectures.
    "yolov8": "frame", "detr": "frame", "rt_detr": "frame",
    "segformer": "frame", "mask2former": "frame", "unet": "frame",
    # Encoder engines.
    "resnet": "engine", "efficientnet": "engine", "mobilenet_v3": "engine",
    "vit": "engine", "swin_transformer": "engine", "convnext": "engine",
    "dinov2": "engine", "clip_vit": "engine",
}


def family_role(family: str) -> str | None:
    """Return "frame" or "engine" for a known family."""
    return FAMILY_ROLE.get(family)

# Loss merge discipline. Loss consensus feeds Phase B, so noisy merges are
# costly:
#   - ArcFace/CosFace/triplet/metric-learning losses stay unknown rather than
#     inflating InfoNCE support.
#   - weighted CE maps to focal because the KB has no weighted-CE node.
LOSS_ALIASES: list[tuple[str, str]] = [
    (r"focal",                                          "focal_loss"),
    (r"(weighted|class.?weight).*(ce|cross.?entropy)",  "focal_loss"),
    # bce_dice must precede \bdice\b and cross_entropy.
    (r"(?=.*bce)(?=.*dice)",                            "bce_dice_loss"),
    (r"cross.?entropy|\bce\b|\bbce\b|label.?smooth",    "cross_entropy_loss"),
    (r"\bdice\b",                                        "dice_loss"),
    (r"infonce|(?<!arc)contrastive",                     "infonce_loss"),
    (r"arcface|cosface|triplet|metric.?learning",        "unknown"),
    (r"hungarian|matching",                              "hungarian_matching_loss"),
]

# Raw metric-learning losses are tagged in the unknown side table.
_METRIC_LEARNING_RE = re.compile(r"arcface|cosface|triplet|metric.?learning", re.I)


def _match_alias(raw: str | None, table: list[tuple[str, str]]) -> str:
    """Match raw text to a KB id using ordered, case-insensitive regexes."""
    if not raw:
        return "unknown"
    for pattern, kb_id in table:
        if re.search(pattern, raw, re.I):
            return kb_id
    return "unknown"


def map_model(raw: str | None) -> str:
    """Map a raw model name to a backbone family id."""
    return _match_alias(raw, MODEL_ALIASES)


# Literal loss-name patterns for hybrid counting. These do not include the
# weighted-CE -> focal re-bucket rule or the bce_dice aggregate node.
_LOSS_FAMILY_PATTERNS: list[tuple[str, str]] = [
    (r"focal",                                       "focal_loss"),
    (r"cross.?entropy|\bce\b|\bbce\b|label.?smooth", "cross_entropy_loss"),
    (r"\bdice\b",                                    "dice_loss"),
    (r"infonce|(?<!arc)contrastive",                 "infonce_loss"),
    (r"hungarian|matching",                          "hungarian_matching_loss"),
]


def loss_families_in(raw: str | None) -> set[str]:
    """Return distinct KB loss families mentioned by raw text."""
    if not raw:
        return set()
    return {kb for pat, kb in _LOSS_FAMILY_PATTERNS if re.search(pat, raw, re.I)}


def is_hybrid_loss(raw: str | None) -> bool:
    """Return whether raw text combines two or more loss families.

    bce_dice is an accepted KB node and is not treated as a noisy hybrid.
    Detection/segmentation systems often combine classification, segmentation,
    and regression losses; those are kept unknown when flattened evidence would
    be misleading.
    """
    if not raw:
        return False
    if _match_alias(raw, LOSS_ALIASES) == "bce_dice_loss":
        return False
    return len(loss_families_in(raw)) >= 2


def map_loss(raw: str | None) -> str:
    """Map a raw loss name to a KB loss id."""
    if is_hybrid_loss(raw):
        return "unknown"
    return _match_alias(raw, LOSS_ALIASES)


def is_metric_learning_loss(raw: str | None) -> bool:
    """Return whether raw text names a metric-learning-style loss."""
    return bool(raw) and bool(_METRIC_LEARNING_RE.search(raw))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Mechanical listing of recent candidate competitions.
# ═══════════════════════════════════════════════════════════════════════════════

_CV_TITLE_RE = re.compile(
    r"classif|detect|segment|recogn|image|vision|disease|cancer|lesion|"
    r"species|grading|diagnos",
    re.I,
)


def list_recent_cv_candidates(
    dump_dir: str | Path,
    since: str = "2025-01",
    min_teams: int = 300,
) -> list[dict]:
    """List recent CV-like competitions from Meta Kaggle Competitions.csv.

    Filters: DeadlineDate >= since, title contains a CV signal, and TotalTeams
    >= min_teams. The output is for manual review only; it does not add entries
    to COMPETITIONS.
    """
    import pandas as pd

    path = Path(dump_dir) / "Competitions.csv"
    df = pd.read_csv(
        path,
        usecols=["Slug", "Title", "EnabledDate", "DeadlineDate", "TotalTeams",
                 "HostSegmentTitle"],
    )
    df["DeadlineDate"] = pd.to_datetime(df["DeadlineDate"], errors="coerce")
    since_ts = pd.Timestamp(since + "-01")
    known = set(COMPETITIONS)

    out: list[dict] = []
    for _, r in df.iterrows():
        if pd.isna(r.DeadlineDate) or r.DeadlineDate < since_ts:
            continue
        if pd.isna(r.TotalTeams) or r.TotalTeams < min_teams:
            continue
        title = str(r.Title) if not pd.isna(r.Title) else ""
        if not _CV_TITLE_RE.search(title):
            continue
        if r.Slug in known:
            continue
        out.append({
            "slug": r.Slug,
            "title": title,
            "start": pd.to_datetime(r.EnabledDate, errors="coerce").strftime("%Y-%m")
                     if not pd.isna(r.EnabledDate) else None,
            "end": r.DeadlineDate.strftime("%Y-%m"),
            "teams": int(r.TotalTeams),
            "segment": r.HostSegmentTitle,
        })
    out.sort(key=lambda d: d["teams"], reverse=True)
    return out

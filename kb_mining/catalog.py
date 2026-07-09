"""catalog.py — kb_mining 的纯数据层（+ 一个只读枚举辅助函数）。

四类数据：
  1. COMPETITIONS   —— 竞赛清单 + 特征卡（挖掘语料的范围与标签）
  2. FAMILY_RELEASE —— 14 个 backbone 家族的发布时间（共存性过滤用）
  3. MODEL_ALIASES / LOSS_ALIASES —— 原始字符串 → KB id 的按序正则映射
  4. list_recent_cv_candidates() —— 扫 Competitions.csv 枚举新竞赛候选（人工挑选）

竞赛的 start/end/TotalTeams 已用本地 Meta Kaggle dump 的真实值填入（2026-07-03
核对）；traits 是初判，traits_verified=False —— 必须人工过目后置 True，
aggregate 对未核对的竞赛在 consensus.md 里打 ⚠。
"""

from __future__ import annotations

import re
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 竞赛清单 + 特征卡
# ═══════════════════════════════════════════════════════════════════════════════
#
# traits 键 = 合法 condition 键去掉 "=True"（fine_grained 目前不是合法 condition
# 键，decide 会把它走档 4 schema-ext）。data_size 是竞赛数据量的初判，不作为
# 独立挖掘 trait（见 aggregate §4.3），只用于原型查询填充与档 1 field-fix 证据。
#
# ⚠ 未决数据：所有 traits_verified=False。data_size 初判尤其粗糙，需人工按
#   竞赛 Overview/Data 页 + write-up 交叉核对后再置 True。

COMPETITIONS: dict[str, dict] = {
    "cassava-leaf-disease-classification": {
        "slug": "cassava-leaf-disease-classification",
        "title": "Cassava Leaf Disease Classification",
        "start": "2020-11", "end": "2021-02", "task_type": "classification",
        "traits": {"fine_grained": True, "class_imbalance": True, "medical": False,
                   "data_size": "medium"},
        "traits_verified": True,
        "notes": "5 类叶病细粒度；轻度类不平衡（cmi 主导类占比高）。3900 队，语料充足。",
    },
    "plant-pathology-2021-fgvc8": {
        "slug": "plant-pathology-2021-fgvc8",
        "title": "Plant Pathology 2021 - FGVC8",
        "start": "2021-03", "end": "2021-05", "task_type": "classification",
        "traits": {"fine_grained": True, "class_imbalance": False, "medical": False,
                   "multi_label": True, "data_size": "medium"},
        "traits_verified": True,
        "notes": "多标签（一图多病）——loss 偏 BCE；multi_label=True 隔离到自己的格子。625 队。",
    },
    "herbarium-2022-fgvc9": {
        "slug": "herbarium-2022-fgvc9",
        "title": "Herbarium 2022 - FGVC9",
        "start": "2022-02", "end": "2022-05", "task_type": "classification",
        "traits": {"fine_grained": True, "class_imbalance": True, "medical": False,
                   "data_size": "large"},
        "traits_verified": True,
        "notes": "15k+ 物种、极端长尾。仅 134 队——write-up 可能不足 5，harvest 召回政策定去留。",
    },
    "sorghum-id-fgvc-9": {
        "slug": "sorghum-id-fgvc-9",
        "title": "Sorghum -100 Cultivar Identification - FGVC 9",
        "start": "2022-03", "end": "2022-05", "task_type": "classification",
        "traits": {"fine_grained": True, "class_imbalance": False, "medical": False,
                   "data_size": "medium"},
        "traits_verified": True,
        "notes": "100 品种细粒度。252 队——偏小，核实 write-up 数量。",
    },
    "paddy-disease-classification": {
        "slug": "paddy-disease-classification",
        "title": "Paddy Doctor: Paddy Disease Classification",
        "start": "2022-04", "end": "2022-08", "task_type": "classification",
        "traits": {"fine_grained": True, "class_imbalance": False, "medical": False,
                   "data_size": "small"},
        "traits_verified": True,
        "notes": "~10k 图、10 类。657 队。",
    },
    "happy-whale-and-dolphin": {
        "slug": "happy-whale-and-dolphin",
        "title": "Happywhale - Whale and Dolphin Identification",
        "start": "2022-02", "end": "2022-04", "task_type": "classification",
        "traits": {"fine_grained": True, "class_imbalance": True, "medical": False,
                   "data_size": "large"},
        "traits_verified": True,
        "loss_voting": False,  # metric-learning 味极重（arcface 主导），loss 信号不适用分类推荐
        "notes": "metric-learning 味极重（个体 re-id，arcface 主导）——loss 投票整场"
                 "排除（loss_voting=False）；backbone 投票保留。1588 队。",
    },
    "mayo-clinic-strip-ai": {
        "slug": "mayo-clinic-strip-ai",
        "title": "Mayo Clinic - STRIP AI",
        "start": "2022-07", "end": "2022-10", "task_type": "classification",
        "traits": {"fine_grained": False, "class_imbalance": False, "medical": True,
                   "data_size": "small"},
        "traits_verified": True,
        "notes": "血栓病理 WSI，样本少。888 队。",
    },
    "rsna-breast-cancer-detection": {
        "slug": "rsna-breast-cancer-detection",
        "title": "RSNA Screening Mammography Breast Cancer Detection",
        "start": "2022-11", "end": "2023-02", "task_type": "classification",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": True,
                   "data_size": "large"},
        "traits_verified": True,
        "notes": "乳腺钼靶，极端不平衡（阳性 ~2%）。1687 队。",
    },
    "UBC-OCEAN": {
        "slug": "UBC-OCEAN",  # ⚠ slug 大小写敏感，dump 里是全大写
        "title": "UBC Ovarian Cancer Subtype Classification and Outlier Detection",
        "start": "2023-10", "end": "2024-01", "task_type": "classification",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": True,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "卵巢癌亚型病理 WSI + 离群检测。1326 队。",
    },
    "isic-2024-challenge": {
        "slug": "isic-2024-challenge",
        "title": "ISIC 2024 - Skin Cancer Detection with 3D-TBP",
        "start": "2024-06", "end": "2024-09", "task_type": "classification",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": True,
                   "data_size": "large"},
        "traits_verified": False,
        "notes": "皮肤癌，极端不平衡。2739 队。",
    },
    "hms-harmful-brain-activity-classification": {
        "slug": "hms-harmful-brain-activity-classification",
        "title": "HMS - Harmful Brain Activity Classification",
        "start": "2024-01", "end": "2024-04", "task_type": "classification",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": True,
                   "data_size": "large"},
        "traits_verified": False,
        "notes": "输入为脑电（EEG）频谱图渲染的图像，非自然图像——backbone 结论对"
                 "自然图像任务的迁移性存疑，decide 时留意。2767 队。",
    },
    "rsna-2024-lumbar-spine-degenerative-classification": {
        "slug": "rsna-2024-lumbar-spine-degenerative-classification",
        "title": "RSNA 2024 Lumbar Spine Degenerative Classification",
        "start": "2024-05", "end": "2024-10", "task_type": "classification",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": True,
                   "data_size": "large"},
        "traits_verified": False,
        "notes": "腰椎 MRI 多部位分级，多输出——含检测/定位子任务，方案未必纯分类。1874 队。",
    },
    "fathomnet-2025": {
        "slug": "fathomnet-2025",
        "title": "FathomNet 2025",
        "start": "2025-03", "end": "2025-05", "task_type": "classification",
        "traits": {"fine_grained": True, "class_imbalance": False, "medical": False,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "海洋物种层级分类。⚠ 仅 79 队——几乎肯定不足 5 篇 write-up，"
                 "harvest 召回政策大概率剔除；fine_grained 已由其它竞赛充分覆盖，可弃。",
    },
    "rsna-intracranial-aneurysm-detection": {
        "slug": "rsna-intracranial-aneurysm-detection",
        "title": "RSNA Intracranial Aneurysm Detection",
        "start": "2025-07", "end": "2025-10", "task_type": "classification",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": True,
                   "data_size": "large"},
        "traits_verified": True,
        "notes": "CT/MR 颅内动脉瘤，含定位子任务。1147 队，write-up 充足。",
    },

    # ═══ 扩展批次（2026-07-04）：去医疗偏斜 + 检测/分割任务覆盖 ═══════════════
    # traits 全为初判，traits_verified=False。data_size / class_imbalance 尤其待核。

    # ── 分类·非医疗（birdclef 系列：鸟鸣频谱图，非自然图像，细粒度长尾）──────
    "birdclef-2021": {
        "slug": "birdclef-2021", "title": "BirdCLEF 2021 - Birdcall Identification",
        "start": "2021-04", "end": "2021-06", "task_type": "classification",
        "traits": {"fine_grained": True, "class_imbalance": True, "medical": False,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "鸟鸣 mel 频谱图分类（非自然图像，同 hms 的争议）；长尾物种。816 队。",
    },
    "birdclef-2022": {
        "slug": "birdclef-2022", "title": "BirdCLEF 2022",
        "start": "2022-02", "end": "2022-05", "task_type": "classification",
        "traits": {"fine_grained": True, "class_imbalance": True, "medical": False,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "鸟鸣频谱图，长尾。801 队。",
    },
    "birdclef-2023": {
        "slug": "birdclef-2023", "title": "BirdCLEF 2023",
        "start": "2023-03", "end": "2023-05", "task_type": "classification",
        "traits": {"fine_grained": True, "class_imbalance": True, "medical": False,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "鸟鸣频谱图，长尾。1189 队。",
    },
    "birdclef-2024": {
        "slug": "birdclef-2024", "title": "BirdCLEF 2024",
        "start": "2024-04", "end": "2024-06", "task_type": "classification",
        "traits": {"fine_grained": True, "class_imbalance": True, "medical": False,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "鸟鸣频谱图，长尾。974 队。",
    },
    "birdclef-2025": {
        "slug": "birdclef-2025", "title": "BirdCLEF+ 2025",
        "start": "2025-03", "end": "2025-06", "task_type": "classification",
        "traits": {"fine_grained": True, "class_imbalance": True, "medical": False,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "鸟鸣频谱图，长尾。2031 队。",
    },
    "landmark-recognition-2021": {
        "slug": "landmark-recognition-2021",
        "title": "Google Landmark Recognition 2021",
        "start": "2021-08", "end": "2021-10", "task_type": "classification",
        "traits": {"fine_grained": True, "class_imbalance": True, "medical": False,
                   "data_size": "large"},
        "traits_verified": False,
        "notes": "10 万+ 地标细粒度识别（本质检索/metric-learning 味重）。383 队。",
    },

    # ── 检测·非医疗 ────────────────────────────────────────────────────────
    "tensorflow-great-barrier-reef": {
        "slug": "tensorflow-great-barrier-reef",
        "title": "TensorFlow - Help Protect the Great Barrier Reef",
        "start": "2021-11", "end": "2022-02", "task_type": "object_detection",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": False,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "水下视频海星检测，目标稀疏（不平衡）。2025 队。",
    },
    "czii-cryo-et-object-identification": {
        "slug": "czii-cryo-et-object-identification",
        "title": "CZII - CryoET Object Identification",
        "start": "2024-11", "end": "2025-02", "task_type": "object_detection",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": False,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "3D cryo-ET 颗粒定位（生物，非医疗）。931 队。",
    },
    "byu-locating-bacterial-flagellar-motors-2025": {
        "slug": "byu-locating-bacterial-flagellar-motors-2025",
        "title": "BYU - Locating Bacterial Flagellar Motors 2025",
        "start": "2025-03", "end": "2025-06", "task_type": "object_detection",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": False,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "cryo-ET 断层图鞭毛马达定位（生物）。1136 队。",
    },

    # ── 检测·医疗（为 DETR/YOLO/rt_detr 供证据）────────────────────────────
    "vinbigdata-chest-xray-abnormalities-detection": {
        "slug": "vinbigdata-chest-xray-abnormalities-detection",
        "title": "VinBigData Chest X-ray Abnormalities Detection",
        "start": "2020-12", "end": "2021-03", "task_type": "object_detection",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": True,
                   "data_size": "large"},
        "traits_verified": False,
        "notes": "胸片 14 类异常检测，类不平衡。1275 队。",
    },
    "siim-covid19-detection": {
        "slug": "siim-covid19-detection",
        "title": "SIIM-FISABIO-RSNA COVID-19 Detection",
        "start": "2021-05", "end": "2021-08", "task_type": "object_detection",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": True,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "胸片 COVID 检测 + 图像级分类。1305 队。",
    },

    # ── 分割·非医疗 ────────────────────────────────────────────────────────
    "vesuvius-challenge-ink-detection": {
        "slug": "vesuvius-challenge-ink-detection",
        "title": "Vesuvius Challenge - Ink Detection",
        "start": "2023-03", "end": "2023-06", "task_type": "image_segmentation",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": False,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "碳化卷轴 3D CT 墨迹分割（非医疗）；正样本稀疏。1249 队。",
    },

    # ── 分割·医疗（为 U-Net/Mask2Former/SegFormer 供证据）──────────────────
    "sartorius-cell-instance-segmentation": {
        "slug": "sartorius-cell-instance-segmentation",
        "title": "Sartorius - Cell Instance Segmentation",
        "start": "2021-10", "end": "2021-12", "task_type": "image_segmentation",
        "traits": {"fine_grained": False, "class_imbalance": False, "medical": True,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "显微镜神经元细胞实例分割。1505 队。",
    },
    "uw-madison-gi-tract-image-segmentation": {
        "slug": "uw-madison-gi-tract-image-segmentation",
        "title": "UW-Madison GI Tract Image Segmentation",
        "start": "2022-04", "end": "2022-07", "task_type": "image_segmentation",
        "traits": {"fine_grained": False, "class_imbalance": False, "medical": True,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "MRI 胃肠道器官分割。1548 队。",
    },
    "hubmap-organ-segmentation": {
        "slug": "hubmap-organ-segmentation",
        "title": "HuBMAP + HPA - Hacking the Human Body",
        "start": "2022-06", "end": "2022-09", "task_type": "image_segmentation",
        "traits": {"fine_grained": False, "class_imbalance": False, "medical": True,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "多器官组织功能单元分割。1174 队。",
    },
    "hubmap-hacking-the-human-vasculature": {
        "slug": "hubmap-hacking-the-human-vasculature",
        "title": "HuBMAP - Hacking the Human Vasculature",
        "start": "2023-05", "end": "2023-07", "task_type": "image_segmentation",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": True,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "显微镜血管实例分割。1021 队。",
    },
    "blood-vessel-segmentation": {
        "slug": "blood-vessel-segmentation",
        "title": "SenNet + HOA - Hacking the Human Vasculature in 3D",
        "start": "2023-11", "end": "2024-02", "task_type": "image_segmentation",
        "traits": {"fine_grained": False, "class_imbalance": True, "medical": True,
                   "data_size": "medium"},
        "traits_verified": False,
        "notes": "肾脏 3D 血管分割。1149 队。",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 架构发布时间表（共存性过滤）
# ═══════════════════════════════════════════════════════════════════════════════
#
# 键与 retrieval/rag_retrieval.py 的 COMPONENTS backbone id 完全一致（已核对 14 个）。
# 值为论文/发布月份（arXiv 首发，月粒度）。aggregate 用它排除"竞赛开赛时该架构
# 尚不存在"的伪证据：FAMILY_RELEASE[A] < 竞赛 start 才计票。

FAMILY_RELEASE: dict[str, str] = {   # family_id -> "YYYY-MM"
    "resnet": "2015-12",  "efficientnet": "2019-05", "mobilenet_v3": "2019-05",
    "vit": "2020-10",     "swin_transformer": "2021-03", "convnext": "2022-01",
    "yolov8": "2023-01",  "detr": "2020-05",  "rt_detr": "2023-04",
    "segformer": "2021-05", "mask2former": "2021-12", "unet": "2015-05",
    "dinov2": "2023-04",  "clip_vit": "2021-01",
}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 组件别名表（原始字符串 → KB id，按序匹配，全部不区分大小写）
# ═══════════════════════════════════════════════════════════════════════════════
#
# 按序匹配：列表靠前的规则先命中。无一命中 → "unknown"（进 unknown_components
# 侧表，未来建节点候选池）。映射由代码做、不让 LLM 直接输出 KB id，降低幻觉面。

MODEL_ALIASES: list[tuple[str, str]] = [
    # EfficientNet 及常见简写：efficientnet / effnet / eff net / efn-b4 / effv2s /
    # efficient-b6 / EFFNet B5 等（B\d 单独出现太歧义，不收）
    (r"(tf_)?eff(icient)?[-_ ]?net(v2)?|effv2|efn[-_ ]?b?\d|efficient[-_ ]?b\d", "efficientnet"),
    (r"convnext",                "convnext"),
    (r"swin",                    "swin_transformer"),
    (r"(deit|beit|^vit|_vit|vit_|vision[-_ ]?transformer)", "vit"),
    (r"dinov2",                  "dinov2"),
    (r"clip",                    "clip_vit"),
    (r"(resnet|resnext|resnest|se_?resnext)", "resnet"),
    (r"mobilenet",               "mobilenet_v3"),
    # ── 检测/分割模型（rt_detr 必须在 detr 之前；mask2former 在 segformer/former 之前）──
    (r"rt.?detr|rtdetr",         "rt_detr"),
    (r"\bdetr\b|deformable.?detr", "detr"),
    (r"yolo",                    "yolov8"),        # KB 只有一个 YOLO 节点，各版本归一
    (r"mask2?former",            "mask2former"),
    (r"segformer",               "segformer"),
    (r"u[-_ ]?net(\+\+)?|unet",  "unet"),
    # efficientdet / faster-rcnn / mask-rcnn / retinanet / sam / detectron 等
    # KB 无对应节点 → 落 unknown（进侧表，建新节点候选）
]


# ── 家族角色：车架（元架构） vs 发动机（编码器 backbone）───────────────────────
# 检测/分割方案常是"车架 + 发动机"（如 U-Net + EfficientNet encoder），一个方案
# 同时点名两者。aggregate 按角色分两组各比各的，避免编码器票稀释元架构票。
#   frame  车架：整体检测/分割元架构（YOLO/DETR/U-Net/SegFormer/Mask2Former…）
#   engine 发动机：作为编码器/主干的分类 backbone（ResNet/EfficientNet/Swin…）
FAMILY_ROLE: dict[str, str] = {
    # 车架 frame
    "yolov8": "frame", "detr": "frame", "rt_detr": "frame",
    "segformer": "frame", "mask2former": "frame", "unet": "frame",
    # 发动机 engine
    "resnet": "engine", "efficientnet": "engine", "mobilenet_v3": "engine",
    "vit": "engine", "swin_transformer": "engine", "convnext": "engine",
    "dinov2": "engine", "clip_vit": "engine",
}


def family_role(family: str) -> str | None:
    """family → "frame" / "engine"；未知 family → None。"""
    return FAMILY_ROLE.get(family)

# loss 归并纪律（loss 共识是 Phase B 的消费对象，污染代价最高）：
#   - arcface/cosface/triplet/metric-learning 损失 → "unknown"（不归并到 infonce，
#     否则虚增 infonce 的 support）；带 metric-learning 标签进侧表。
#   - 加权 CE → focal（KB 无加权 CE 节点，两者对应同一条边）；consensus.md 须
#     拆分显示 raw 计数，加权 CE 占比 > 50% 时该行加 ⚠。
LOSS_ALIASES: list[tuple[str, str]] = [
    (r"focal",                                          "focal_loss"),
    (r"(weighted|class.?weight).*(ce|cross.?entropy)",  "focal_loss"),
    # bce_dice 必须在 \bdice\b / cross_entropy 之前。双 lookahead 顺序无关，
    # 接住 "BCE + Dice" / "Dice+BCE" / "bce_dice" 等任意分隔与写序。
    (r"(?=.*bce)(?=.*dice)",                            "bce_dice_loss"),
    (r"cross.?entropy|\bce\b|\bbce\b|label.?smooth",    "cross_entropy_loss"),
    (r"\bdice\b",                                        "dice_loss"),
    (r"infonce|(?<!arc)contrastive",                     "infonce_loss"),
    (r"arcface|cosface|triplet|metric.?learning",        "unknown"),
    (r"hungarian|matching",                              "hungarian_matching_loss"),
]

# metric-learning 类损失的 raw（映射为 unknown 后，供侧表打标签用）
_METRIC_LEARNING_RE = re.compile(r"arcface|cosface|triplet|metric.?learning", re.I)


def _match_alias(raw: str | None, table: list[tuple[str, str]]) -> str:
    """按序正则匹配 raw → KB id；raw 为空或无一命中 → "unknown"。不区分大小写。"""
    if not raw:
        return "unknown"
    for pattern, kb_id in table:
        if re.search(pattern, raw, re.I):
            return kb_id
    return "unknown"


def map_model(raw: str | None) -> str:
    """原始模型名 → backbone family id（无命中 → "unknown"）。"""
    return _match_alias(raw, MODEL_ALIASES)


# 损失"原名"模式（用于 hybrid 计数）——只认损失字面名，**不含** weighted-CE→focal
# 那条 re-bucket 规则（"weighted cross entropy" 是单个损失，非组合），也不含 bce_dice。
_LOSS_FAMILY_PATTERNS: list[tuple[str, str]] = [
    (r"focal",                                       "focal_loss"),
    (r"cross.?entropy|\bce\b|\bbce\b|label.?smooth", "cross_entropy_loss"),
    (r"\bdice\b",                                    "dice_loss"),
    (r"infonce|(?<!arc)contrastive",                 "infonce_loss"),
    (r"hungarian|matching",                          "hungarian_matching_loss"),
]


def loss_families_in(raw: str | None) -> set[str]:
    """raw 命中的不同 KB loss 家族（按原名，用于判组合损失）。"""
    if not raw:
        return set()
    return {kb for pat, kb in _LOSS_FAMILY_PATTERNS if re.search(pat, raw, re.I)}


def is_hybrid_loss(raw: str | None) -> bool:
    """raw 是否为组合损失（含 ≥2 个不同 loss 家族，如 "BCE + Focal"）。

    bce_dice 是 KB 认可的组合损失节点，不算 hybrid。检测/分割方案常把
    分类头损失 + 分割/回归损失加权求和，压平成单票是伪象——这类归 unknown。
    """
    if not raw:
        return False
    if _match_alias(raw, LOSS_ALIASES) == "bce_dice_loss":
        return False   # 已有专属节点的认可组合
    return len(loss_families_in(raw)) >= 2


def map_loss(raw: str | None) -> str:
    """原始 loss 名 → loss KB id。组合损失（≥2 家族）→ unknown；无命中 → unknown。"""
    if is_hybrid_loss(raw):
        return "unknown"
    return _match_alias(raw, LOSS_ALIASES)


def is_metric_learning_loss(raw: str | None) -> bool:
    """raw 是否为 metric-learning 类损失（供 unknown 侧表打标签）。"""
    return bool(raw) and bool(_METRIC_LEARNING_RE.search(raw))


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 新竞赛机械枚举（弥补手工清单的时效盲区；只枚举、不自动入清单）
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
    """扫 Meta Kaggle Competitions.csv，枚举可能相关的近期 CV 竞赛候选。

    过滤：①DeadlineDate >= since ②标题含 CV 信号 ③TotalTeams >= min_teams。
    只输出候选行供人工挑选后手动补进 COMPETITIONS——不自动入清单、不填特征卡。
    harvest 的 --list-recent 开关调用它。
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

"""
Module 3 知识库 — CV任务专用
任务: Image Classification | Object Detection | Image Segmentation | Feature Extraction
结构: 组件图（NetworkX）+ 语义向量索引（ChromaDB）

节点类型:
  backbone        — 模型架构（粗粒度）
  pretrained_model — HuggingFace 具体 checkpoint，通过 has_pretrained 边关联到 backbone
  head            — 输出头
  loss            — 损失函数
  optimizer       — 优化器

边类型:
  compatible_with  — 可以搭配使用
  has_pretrained   — backbone → 对应的预训练 checkpoint
  alternative_to   — 可以互相替换
  preferred_when   — 某条件下更优（edge attr: condition）
"""

import hashlib
import json
import os
import re
import networkx as nx
import numpy as np
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# ═══════════════════════════════════════════════════════════════════════════════
# 1. 组件节点
# ═══════════════════════════════════════════════════════════════════════════════

COMPONENTS = [

    # ── Backbones ──────────────────────────────────────────────────────────────

    {
        "id": "resnet",
        "name": "ResNet",
        "component_type": "backbone",
        "task_type": ["classification", "object_detection", "feature_extraction"],
        "data_size": ["medium", "large"],
        "complexity": "medium",
        "finetune_recommended": False,
        "scratch_viable_from": "medium",
        "domain_transfer": "moderate",
        "tier": {
            "classification":     "special_case",
            "object_detection":   "special_case",
            "feature_extraction": "special_case",
        },
        "description": (
            "Classic deep residual CNN. Strong baseline for image classification and "
            "as a detection backbone (e.g., Faster R-CNN). Training from scratch feasible "
            "on medium to large datasets."
        ),
    },
    {
        "id": "efficientnet",
        "name": "EfficientNet",
        "component_type": "backbone",
        "task_type": ["classification", "feature_extraction"],
        "data_size": ["small", "medium"],
        "complexity": "low",
        "finetune_recommended": False,
        "scratch_viable_from": "medium",
        "domain_transfer": "moderate",
        "tier": {
            "classification":     "default",
            "feature_extraction": "special_case",
        },
        "description": (
            "Compound-scaled CNN with high accuracy-efficiency tradeoff. Good for image "
            "classification on small to medium datasets. Pretrained weights widely available."
        ),
    },
    {
        "id": "mobilenet_v3",
        "name": "MobileNetV3",
        "component_type": "backbone",
        "task_type": ["classification"],
        "data_size": ["small"],
        "complexity": "low",
        "finetune_recommended": False,
        "scratch_viable_from": "small",
        "domain_transfer": "moderate",
        "tier": {
            "classification": "special_case",
        },
        "description": (
            "Ultra-lightweight CNN for edge and mobile deployment. Best when inference "
            "speed and compute are the primary constraints."
        ),
    },
    {
        "id": "vit",
        "name": "Vision Transformer (ViT)",
        "component_type": "backbone",
        "task_type": ["classification", "feature_extraction"],
        "data_size": ["large"],
        "complexity": "high",
        "finetune_recommended": True,
        "scratch_viable_from": None,
        "domain_transfer": "moderate",
        "tier": {
            "classification":     "accuracy_upgrade",
            "feature_extraction": "special_case",
        },
        "description": (
            "Pure transformer applied to image patches. Requires large datasets or "
            "pretrained weights. Finetuning from ImageNet-21k checkpoint recommended. "
            "Strong global feature representations."
        ),
    },
    {
        "id": "swin_transformer",
        "name": "Swin Transformer",
        "component_type": "backbone",
        "task_type": ["classification", "object_detection", "image_segmentation", "feature_extraction"],
        "data_size": ["medium", "large"],
        "complexity": "high",
        "finetune_recommended": True,
        "scratch_viable_from": "large",
        "domain_transfer": "moderate",
        "tier": {
            "classification":     "accuracy_upgrade",
            "object_detection":   "accuracy_upgrade",
            "image_segmentation": "accuracy_upgrade",
            "feature_extraction": "special_case",
        },
        "description": (
            "Hierarchical transformer with shifted windows. Most versatile CV backbone — "
            "works across classification, detection, and segmentation. Better than vanilla "
            "ViT for dense prediction tasks. Finetuning from ImageNet-22k is standard."
        ),
    },
    {
        "id": "convnext",
        "name": "ConvNeXt",
        "component_type": "backbone",
        "task_type": ["classification", "object_detection", "image_segmentation"],
        "data_size": ["medium", "large"],
        "complexity": "medium",
        "finetune_recommended": True,
        "scratch_viable_from": "large",
        "domain_transfer": "moderate",
        "tier": {
            "classification":     "accuracy_upgrade",
            "object_detection":   "accuracy_upgrade",
            "image_segmentation": "accuracy_upgrade",
        },
        "description": (
            "Modernized pure CNN that matches transformer performance. Good drop-in "
            "replacement for Swin when convolutional inductive bias is preferred. "
            "Finetuning from ImageNet-22k recommended."
        ),
    },
    {
        "id": "yolov8",
        "name": "YOLOv8",
        "component_type": "backbone",
        "task_type": ["object_detection", "image_segmentation"],
        "data_size": ["small", "medium", "large"],
        "complexity": "medium",
        "finetune_recommended": True,
        "scratch_viable_from": "medium",
        "domain_transfer": "moderate",
        "tier": {
            "object_detection":   "default",
            "image_segmentation": "special_case",
        },
        "description": (
            "State-of-the-art single-stage detector and instance segmenter. Fast inference, "
            "easy to fine-tune. Multiple sizes (nano to extra-large). Best for real-time "
            "or resource-constrained object detection and instance segmentation."
        ),
    },
    {
        "id": "detr",
        "name": "DETR",
        "component_type": "backbone",
        "task_type": ["object_detection"],
        "data_size": ["large"],
        "complexity": "high",
        "finetune_recommended": True,
        "scratch_viable_from": "large",
        "domain_transfer": "moderate",
        "tier": {
            "object_detection": "accuracy_upgrade",
        },
        "description": (
            "End-to-end transformer detector. No anchor boxes or NMS. Slower to train "
            "than YOLO but strong on complex scenes. Finetuning from COCO pretrained "
            "checkpoint is standard."
        ),
    },
    {
        "id": "rt_detr",
        "name": "RT-DETR",
        "component_type": "backbone",
        "task_type": ["object_detection"],
        "data_size": ["medium", "large"],
        "complexity": "high",
        "finetune_recommended": True,
        "scratch_viable_from": "large",
        "domain_transfer": "moderate",
        "tier": {
            "object_detection": "accuracy_upgrade",
        },
        "description": (
            "Real-time DETR: transformer detector that matches YOLO speed with DETR-level "
            "accuracy. Good balance between end-to-end design and practical inference speed. "
            "Finetuning from COCO pretrained checkpoint recommended."
        ),
    },
    {
        "id": "segformer",
        "name": "SegFormer",
        "component_type": "backbone",
        "task_type": ["image_segmentation"],
        "data_size": ["medium", "large"],
        "complexity": "medium",
        "finetune_recommended": True,
        "scratch_viable_from": "large",
        "domain_transfer": "moderate",
        "tier": {
            "image_segmentation": "default",
        },
        "description": (
            "Efficient hierarchical transformer for semantic segmentation with a lightweight "
            "MLP decoder. Multiple sizes (B0–B5). Finetuning from ADE20k or Cityscapes "
            "checkpoint is common. Better speed-accuracy than DeepLab."
        ),
    },
    {
        "id": "mask2former",
        "name": "Mask2Former",
        "component_type": "backbone",
        "task_type": ["image_segmentation"],
        "data_size": ["large"],
        "complexity": "high",
        "finetune_recommended": True,
        "scratch_viable_from": "large",
        "domain_transfer": "moderate",
        "tier": {
            "image_segmentation": "accuracy_upgrade",
        },
        "description": (
            "Universal segmentation model: handles semantic, instance, and panoptic "
            "segmentation with the same architecture. Uses masked cross-attention. "
            "Best accuracy for complex segmentation. Finetuning from COCO panoptic checkpoint."
        ),
    },
    {
        "id": "unet",
        "name": "U-Net",
        "component_type": "backbone",
        "task_type": ["image_segmentation"],
        "data_size": ["small", "medium"],
        "complexity": "medium",
        "finetune_recommended": False,
        "scratch_viable_from": "small",
        "domain_transfer": "weak",
        "tier": {
            "image_segmentation": "special_case",
        },
        "description": (
            "Encoder-decoder with skip connections. Classic choice for semantic segmentation, "
            "especially for medical imaging or small annotated datasets. "
            "Training from scratch is feasible."
        ),
    },
    {
        "id": "dinov2",
        "name": "DINOv2",
        "component_type": "backbone",
        "task_type": ["feature_extraction", "classification", "image_segmentation"],
        "data_size": ["small", "medium", "large"],
        "complexity": "high",
        "finetune_recommended": True,
        "scratch_viable_from": None,
        "domain_transfer": "strong",
        "capabilities": ["zero_shot", "few_shot", "open_vocabulary"],
        "tier": {
            "feature_extraction": "default",
            "classification":     "special_case",
            "image_segmentation": "special_case",
        },
        "description": (
            "Self-supervised ViT pretrained on 142M curated images. Best general-purpose "
            "visual features without task-specific pretraining. Strong zero-shot and few-shot "
            "performance. Freeze the backbone and attach a lightweight head."
        ),
    },
    {
        "id": "clip_vit",
        "name": "CLIP ViT",
        "component_type": "backbone",
        "task_type": ["feature_extraction", "classification"],
        "data_size": ["small", "medium", "large"],
        "complexity": "high",
        "finetune_recommended": True,
        "scratch_viable_from": None,
        "domain_transfer": "strong",
        "capabilities": ["zero_shot", "few_shot", "open_vocabulary", "cross_modal"],
        "tier": {
            "feature_extraction": "special_case",
            "classification":     "special_case",
        },
        "description": (
            "Vision encoder from CLIP, trained on 400M image-text pairs. Language-aligned "
            "visual features. Good for cross-modal retrieval and open-vocabulary classification."
        ),
    },

    # ── Pretrained Model Checkpoints（HuggingFace）────────────────────────────
    # freeze_viable: 冻结 backbone 只训 head 是否可行
    # finetune_strategy: "full" 全量更新 | "head_only" 推荐冻结骨干 | "either" 两种均可

    {
        "id": "resnet50_imagenet",
        "name": "ResNet-50 / ImageNet-1k",
        "component_type": "pretrained_model",
        "hf_id": "microsoft/resnet-50",
        "finetune_base": "resnet",
        "pretrain_dataset": "ImageNet-1k",
        "params_M": 25,
        "task_type": ["classification", "object_detection", "feature_extraction"],
        "size_tier": "base",
        "recommended_when": {},
        "freeze_viable": True,
        "finetune_strategy": "either",
        "description": "ResNet-50 pretrained on ImageNet-1k. Classic baseline; full finetune or freeze+head for feature extraction.",
    },
    {
        "id": "efficientnet_b0_imagenet",
        "name": "EfficientNet-B0 / ImageNet-1k",
        "component_type": "pretrained_model",
        "hf_id": "google/efficientnet-b0",
        "finetune_base": "efficientnet",
        "pretrain_dataset": "ImageNet-1k",
        "params_M": 5,
        "task_type": ["classification", "feature_extraction"],
        "size_tier": "base",
        "recommended_when": {},
        "freeze_viable": True,
        "finetune_strategy": "either",
        "description": "EfficientNet-B0 pretrained on ImageNet-1k. Compact baseline; full finetune or freeze+head.",
    },
    {
        "id": "swin_base_in22k",
        "name": "Swin-Base / ImageNet-22k",
        "component_type": "pretrained_model",
        "hf_id": "microsoft/swin-base-patch4-window7-224",
        "finetune_base": "swin_transformer",
        "pretrain_dataset": "ImageNet-22k",
        "params_M": 88,
        "task_type": ["classification", "object_detection", "image_segmentation"],
        "size_tier": "base",
        "recommended_when": {},
        "freeze_viable": True,
        "finetune_strategy": "full",
        "description": "Swin-Base pretrained on ImageNet-22k. Standard starting point for most Swin finetuning tasks.",
    },
    {
        "id": "swin_large_in22k",
        "name": "Swin-Large / ImageNet-22k",
        "component_type": "pretrained_model",
        "hf_id": "microsoft/swin-large-patch4-window7-224",
        "finetune_base": "swin_transformer",
        "pretrain_dataset": "ImageNet-22k",
        "params_M": 197,
        "task_type": ["classification", "object_detection", "image_segmentation"],
        "size_tier": "large",
        "recommended_when": {"data_size": "large", "priority": "accuracy"},
        "freeze_viable": True,
        "finetune_strategy": "full",
        "description": "Swin-Large pretrained on ImageNet-22k. Higher capacity; use when Base underfits.",
    },
    {
        "id": "vit_base_in21k",
        "name": "ViT-Base/16 / ImageNet-21k",
        "component_type": "pretrained_model",
        "hf_id": "google/vit-base-patch16-224",
        "finetune_base": "vit",
        "pretrain_dataset": "ImageNet-21k",
        "params_M": 86,
        "task_type": ["classification", "feature_extraction"],
        "size_tier": "base",
        "recommended_when": {},
        "freeze_viable": True,
        "finetune_strategy": "either",
        "description": "ViT-Base/16 pretrained on ImageNet-21k. Standard ViT finetuning checkpoint.",
    },
    {
        "id": "vit_large_in21k",
        "name": "ViT-Large/16 / ImageNet-21k",
        "component_type": "pretrained_model",
        "hf_id": "google/vit-large-patch16-224",
        "finetune_base": "vit",
        "pretrain_dataset": "ImageNet-21k",
        "params_M": 307,
        "task_type": ["classification", "feature_extraction"],
        "size_tier": "large",
        "recommended_when": {"data_size": "large", "priority": "accuracy"},
        "freeze_viable": True,
        "finetune_strategy": "either",
        "description": "ViT-Large/16 pretrained on ImageNet-21k. Use when Base underfits on complex tasks.",
    },
    {
        "id": "convnext_base_in22k",
        "name": "ConvNeXt-Base / ImageNet-22k",
        "component_type": "pretrained_model",
        "hf_id": "facebook/convnext-base-224-22k",
        "finetune_base": "convnext",
        "pretrain_dataset": "ImageNet-22k",
        "params_M": 89,
        "task_type": ["classification", "object_detection", "image_segmentation"],
        "size_tier": "base",
        "recommended_when": {},
        "freeze_viable": True,
        "finetune_strategy": "full",
        "description": "ConvNeXt-Base pretrained on ImageNet-22k. Alternative to Swin-Base with CNN inductive bias.",
    },
    {
        "id": "detr_resnet50_coco",
        "name": "DETR-ResNet50 / COCO",
        "component_type": "pretrained_model",
        "hf_id": "facebook/detr-resnet-50",
        "finetune_base": "detr",
        "pretrain_dataset": "COCO",
        "params_M": 41,
        "task_type": ["object_detection"],
        "size_tier": "base",
        "recommended_when": {},
        "freeze_viable": False,
        "finetune_strategy": "full",
        "description": "DETR with ResNet-50 backbone pretrained on COCO. Standard DETR finetuning checkpoint.",
    },
    {
        "id": "rt_detr_r50_coco",
        "name": "RT-DETR-R50 / COCO",
        "component_type": "pretrained_model",
        "hf_id": "PekingU/rtdetr_r50vd",
        "finetune_base": "rt_detr",
        "pretrain_dataset": "COCO",
        "params_M": 42,
        "task_type": ["object_detection"],
        "size_tier": "base",
        "recommended_when": {},
        "freeze_viable": False,
        "finetune_strategy": "full",
        "description": "RT-DETR with ResNet-50vd pretrained on COCO. Faster than DETR; good for real-time finetuning.",
    },
    {
        "id": "yolov8m_coco",
        "name": "YOLOv8-Medium / COCO",
        "component_type": "pretrained_model",
        "hf_id": "ultralytics/assets",
        "finetune_base": "yolov8",
        "pretrain_dataset": "COCO",
        "params_M": 26,
        "task_type": ["object_detection", "image_segmentation"],
        "size_tier": "base",
        "recommended_when": {},
        "freeze_viable": False,
        "finetune_strategy": "full",
        "description": "YOLOv8-Medium pretrained on COCO. Balanced speed/accuracy; good default for detection finetuning.",
    },
    {
        "id": "segformer_b2_ade",
        "name": "SegFormer-B2 / ADE20k",
        "component_type": "pretrained_model",
        "hf_id": "nvidia/segformer-b2-finetuned-ade-512-512",
        "finetune_base": "segformer",
        "pretrain_dataset": "ADE20k",
        "params_M": 25,
        "task_type": ["image_segmentation"],
        "size_tier": "base",
        "recommended_when": {},
        "freeze_viable": False,
        "finetune_strategy": "full",
        "description": "SegFormer-B2 finetuned on ADE20k (150 categories). Good mid-size segmentation checkpoint.",
    },
    {
        "id": "mask2former_swin_coco",
        "name": "Mask2Former-Swin-Base / COCO Panoptic",
        "component_type": "pretrained_model",
        "hf_id": "facebook/mask2former-swin-base-coco-panoptic",
        "finetune_base": "mask2former",
        "pretrain_dataset": "COCO panoptic",
        "params_M": 101,
        "task_type": ["image_segmentation"],
        "size_tier": "base",
        "recommended_when": {},
        "freeze_viable": False,
        "finetune_strategy": "full",
        "description": "Mask2Former with Swin-Base backbone on COCO panoptic. Handles semantic/instance/panoptic.",
    },
    {
        "id": "dinov2_base",
        "name": "DINOv2-Base",
        "component_type": "pretrained_model",
        "hf_id": "facebook/dinov2-base",
        "finetune_base": "dinov2",
        "pretrain_dataset": "LVD-142M (self-supervised)",
        "params_M": 86,
        "task_type": ["feature_extraction", "classification", "image_segmentation"],
        "size_tier": "base",
        "recommended_when": {},
        "freeze_viable": True,
        "finetune_strategy": "head_only",
        "description": "DINOv2-Base ViT. Strong general-purpose features. Use frozen for extraction or finetune the head.",
    },
    {
        "id": "dinov2_large",
        "name": "DINOv2-Large",
        "component_type": "pretrained_model",
        "hf_id": "facebook/dinov2-large",
        "finetune_base": "dinov2",
        "pretrain_dataset": "LVD-142M (self-supervised)",
        "params_M": 307,
        "task_type": ["feature_extraction", "classification"],
        "size_tier": "large",
        "recommended_when": {"data_size": "large", "priority": "accuracy"},
        "freeze_viable": True,
        "finetune_strategy": "head_only",
        "description": "DINOv2-Large. Best feature quality at higher compute. Use when Base-level features are insufficient.",
    },
    {
        "id": "clip_vit_base_32",
        "name": "CLIP ViT-B/32",
        "component_type": "pretrained_model",
        "hf_id": "openai/clip-vit-base-patch32",
        "finetune_base": "clip_vit",
        "pretrain_dataset": "WIT-400M (image-text)",
        "params_M": 86,
        "task_type": ["feature_extraction", "classification"],
        "size_tier": "base",
        "recommended_when": {},
        "freeze_viable": True,
        "finetune_strategy": "head_only",
        "description": "CLIP ViT-B/32 vision encoder. Fast, language-aligned features. Good for cross-modal retrieval.",
    },
    {
        "id": "clip_vit_large_14",
        "name": "CLIP ViT-L/14",
        "component_type": "pretrained_model",
        "hf_id": "openai/clip-vit-large-patch14",
        "finetune_base": "clip_vit",
        "pretrain_dataset": "WIT-400M (image-text)",
        "params_M": 307,
        "task_type": ["feature_extraction", "classification"],
        "size_tier": "large",
        "recommended_when": {"priority": "accuracy"},
        "freeze_viable": True,
        "finetune_strategy": "head_only",
        "description": "CLIP ViT-L/14. Higher quality features than ViT-B/32. Use when feature quality matters more than speed.",
    },

    {
        "id": "mobilenet_v3_imagenet",
        "name": "MobileNetV3-Small / ImageNet-1k",
        "component_type": "pretrained_model",
        "hf_id": "google/mobilenet_v3_small_1.0_224",
        "finetune_base": "mobilenet_v3",
        "pretrain_dataset": "ImageNet-1k",
        "params_M": 2.5,
        "task_type": ["classification"],
        "size_tier": "nano",
        "recommended_when": {},
        "freeze_viable": True,
        "finetune_strategy": "either",
        "description": "MobileNetV3-Small pretrained on ImageNet-1k. Ultra-lightweight; finetune or freeze for edge classification.",
    },
    {
        "id": "yolov8n_coco",
        "name": "YOLOv8-Nano / COCO",
        "component_type": "pretrained_model",
        "hf_id": "ultralytics/assets",
        "finetune_base": "yolov8",
        "pretrain_dataset": "COCO",
        "params_M": 3.2,
        "task_type": ["object_detection", "image_segmentation"],
        "size_tier": "nano",
        "recommended_when": {},
        "freeze_viable": False,
        "finetune_strategy": "full",
        "description": "YOLOv8-Nano pretrained on COCO. Minimal compute; use for edge or mobile detection.",
    },
    {
        "id": "yolov8l_coco",
        "name": "YOLOv8-Large / COCO",
        "component_type": "pretrained_model",
        "hf_id": "ultralytics/assets",
        "finetune_base": "yolov8",
        "pretrain_dataset": "COCO",
        "params_M": 43.7,
        "task_type": ["object_detection", "image_segmentation"],
        "size_tier": "large",
        "recommended_when": {"priority": "accuracy"},
        "freeze_viable": False,
        "finetune_strategy": "full",
        "description": "YOLOv8-Large pretrained on COCO. Higher accuracy at the cost of speed; use when mAP matters.",
    },
    {
        "id": "segformer_b0_ade",
        "name": "SegFormer-B0 / ADE20k",
        "component_type": "pretrained_model",
        "hf_id": "nvidia/segformer-b0-finetuned-ade-512-512",
        "finetune_base": "segformer",
        "pretrain_dataset": "ADE20k",
        "params_M": 3.7,
        "task_type": ["image_segmentation"],
        "size_tier": "nano",
        "recommended_when": {},
        "freeze_viable": False,
        "finetune_strategy": "full",
        "description": "SegFormer-B0 finetuned on ADE20k. Lightest SegFormer variant; use for speed-constrained segmentation.",
    },
    {
        "id": "segformer_b5_ade",
        "name": "SegFormer-B5 / ADE20k",
        "component_type": "pretrained_model",
        "hf_id": "nvidia/segformer-b5-finetuned-ade-640-640",
        "finetune_base": "segformer",
        "pretrain_dataset": "ADE20k",
        "params_M": 84.7,
        "task_type": ["image_segmentation"],
        "size_tier": "large",
        "recommended_when": {"priority": "accuracy"},
        "freeze_viable": False,
        "finetune_strategy": "full",
        "description": "SegFormer-B5 finetuned on ADE20k. Highest accuracy SegFormer variant; use when mIoU is the priority.",
    },
    {
        "id": "convnext_large_in22k",
        "name": "ConvNeXt-Large / ImageNet-22k",
        "component_type": "pretrained_model",
        "hf_id": "facebook/convnext-large-224-22k",
        "finetune_base": "convnext",
        "pretrain_dataset": "ImageNet-22k",
        "params_M": 198,
        "task_type": ["classification", "object_detection", "image_segmentation"],
        "size_tier": "large",
        "recommended_when": {"data_size": "large", "priority": "accuracy"},
        "freeze_viable": True,
        "finetune_strategy": "full",
        "description": "ConvNeXt-Large pretrained on ImageNet-22k. Higher capacity than Base; use when accuracy is the priority.",
    },

    # ── Heads ─────────────────────────────────────────────────────────────────
    # params_scale: head 自身的参数量级
    #   "none"     — 无可训练参数（纯池化）
    #   "minimal"  — 单线性层或轻量 MLP，数据量要求低
    #   "moderate" — 多层 CNN/MLP head，需要一定数据
    #   "heavy"    — transformer decoder 级别，数据量要求高

    {
        "id": "classification_head",
        "name": "Classification Head",
        "component_type": "head",
        "task_type": ["classification"],
        "params_scale": "minimal",
        "description": "Linear + softmax layer for multi-class image classification.",
    },
    {
        "id": "detection_head_anchor_free",
        "name": "Anchor-Free Detection Head",
        "component_type": "head",
        "task_type": ["object_detection"],
        "params_scale": "moderate",
        "description": "Decoupled anchor-free head for single-stage detectors (YOLO-style).",
    },
    {
        "id": "detection_head_transformer",
        "name": "Transformer Detection Head",
        "component_type": "head",
        "task_type": ["object_detection"],
        "params_scale": "heavy",
        "description": "Cross-attention decoder for DETR-style end-to-end detection. Uses Hungarian matching.",
    },
    {
        "id": "semantic_seg_head",
        "name": "Semantic Segmentation Head",
        "component_type": "head",
        "task_type": ["image_segmentation"],
        "params_scale": "minimal",
        "description": "Per-pixel classification head (MLP or ASPP-based). Dense class predictions.",
    },
    {
        "id": "panoptic_seg_head",
        "name": "Panoptic Segmentation Head",
        "component_type": "head",
        "task_type": ["image_segmentation"],
        "params_scale": "heavy",
        "description": "Unified head for panoptic segmentation (things + stuff). Used in Mask2Former.",
    },
    {
        "id": "feature_pooling_head",
        "name": "Feature Pooling Head",
        "component_type": "head",
        "task_type": ["feature_extraction"],
        "params_scale": "none",
        "description": "Global average pooling or CLS token extraction. Produces a fixed-size feature vector.",
    },
    {
        "id": "projection_head",
        "name": "Projection Head",
        "component_type": "head",
        "task_type": ["feature_extraction"],
        "params_scale": "minimal",
        "description": "MLP projection onto a lower-dimensional embedding space. Used in contrastive learning.",
    },

    # ── Loss Functions ────────────────────────────────────────────────────────

    {
        "id": "cross_entropy_loss",
        "name": "CrossEntropyLoss",
        "component_type": "loss",
        "task_type": ["classification", "image_segmentation"],
        "description": "Standard classification loss. For segmentation: applied per-pixel. Assumes balanced classes.",
    },
    {
        "id": "focal_loss",
        "name": "FocalLoss",
        "component_type": "loss",
        "task_type": ["object_detection", "classification"],
        "description": "Down-weights easy examples. Better than CE for class-imbalanced detection and classification.",
    },
    {
        "id": "hungarian_matching_loss",
        "name": "Hungarian Matching Loss",
        "component_type": "loss",
        "task_type": ["object_detection"],
        "description": "Bipartite matching loss for DETR-style detectors. Combines classification + L1 + GIoU.",
    },
    {
        "id": "dice_loss",
        "name": "Dice Loss",
        "component_type": "loss",
        "task_type": ["image_segmentation"],
        "description": "Overlap-based loss for segmentation. Robust to class imbalance between foreground/background.",
    },
    {
        "id": "bce_dice_loss",
        "name": "BCE + Dice Loss",
        "component_type": "loss",
        "task_type": ["image_segmentation"],
        "description": "Combined BCE and Dice. Common in binary mask prediction and medical segmentation.",
    },
    {
        "id": "infonce_loss",
        "name": "InfoNCE Loss",
        "component_type": "loss",
        "task_type": ["feature_extraction"],
        "description": "Contrastive loss that pulls positive pairs together and pushes negatives apart.",
    },

    # ── Optimizers ────────────────────────────────────────────────────────────

    {
        "id": "adamw",
        "name": "AdamW",
        "component_type": "optimizer",
        "task_type": ["classification", "object_detection", "image_segmentation", "feature_extraction"],
        "description": "Adam with decoupled weight decay. Standard optimizer for finetuning transformer-based models.",
    },
    {
        "id": "adam",
        "name": "Adam",
        "component_type": "optimizer",
        "task_type": ["classification", "image_segmentation", "feature_extraction"],
        "description": "Adaptive optimizer. Good default for CNN-based models and from-scratch training.",
    },
    {
        "id": "sgd_momentum",
        "name": "SGD with Momentum",
        "component_type": "optimizer",
        "task_type": ["classification", "object_detection"],
        "description": "Classic optimizer for training CNN backbones from scratch on large image datasets.",
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# 2. 边定义
# ═══════════════════════════════════════════════════════════════════════════════

EDGES = [
    # resnet
    ("resnet", "resnet50_imagenet",           "has_pretrained"),
    ("resnet", "classification_head",         "compatible_with"),
    ("resnet", "detection_head_anchor_free",   "compatible_with"),
    ("resnet", "feature_pooling_head",         "compatible_with"),
    ("resnet", "cross_entropy_loss",           "compatible_with"),
    ("resnet", "focal_loss",                   "compatible_with"),
    ("resnet", "adam",                         "compatible_with"),
    ("resnet", "sgd_momentum",                 "compatible_with"),

    # efficientnet
    ("efficientnet", "efficientnet_b0_imagenet", "has_pretrained"),
    ("efficientnet", "classification_head",    "compatible_with"),
    ("efficientnet", "feature_pooling_head",   "compatible_with"),
    ("efficientnet", "cross_entropy_loss",     "compatible_with"),
    ("efficientnet", "focal_loss",             "compatible_with"),
    ("efficientnet", "adam",                   "compatible_with"),

    # mobilenet_v3
    ("mobilenet_v3", "mobilenet_v3_imagenet",  "has_pretrained"),
    ("mobilenet_v3", "classification_head",    "compatible_with"),
    ("mobilenet_v3", "cross_entropy_loss",     "compatible_with"),
    ("mobilenet_v3", "adam",                   "compatible_with"),

    # vit
    ("vit", "classification_head",             "compatible_with"),
    ("vit", "feature_pooling_head",            "compatible_with"),
    ("vit", "cross_entropy_loss",              "compatible_with"),
    ("vit", "adamw",                           "compatible_with"),
    ("vit", "vit_base_in21k",                 "has_pretrained"),
    ("vit", "vit_large_in21k",                "has_pretrained"),

    # swin_transformer
    ("swin_transformer", "classification_head",          "compatible_with"),
    ("swin_transformer", "detection_head_anchor_free",   "compatible_with"),
    ("swin_transformer", "detection_head_transformer",   "compatible_with"),
    ("swin_transformer", "semantic_seg_head",            "compatible_with"),
    ("swin_transformer", "panoptic_seg_head",            "compatible_with"),
    ("swin_transformer", "feature_pooling_head",         "compatible_with"),
    ("swin_transformer", "cross_entropy_loss",           "compatible_with"),
    ("swin_transformer", "focal_loss",                   "compatible_with"),
    ("swin_transformer", "dice_loss",                    "compatible_with"),
    ("swin_transformer", "adamw",                        "compatible_with"),
    ("swin_transformer", "swin_base_in22k",              "has_pretrained"),
    ("swin_transformer", "swin_large_in22k",             "has_pretrained"),

    # convnext
    ("convnext", "convnext_large_in22k",      "has_pretrained"),
    ("convnext", "classification_head",        "compatible_with"),
    ("convnext", "detection_head_anchor_free", "compatible_with"),
    ("convnext", "semantic_seg_head",          "compatible_with"),
    ("convnext", "cross_entropy_loss",         "compatible_with"),
    ("convnext", "focal_loss",                 "compatible_with"),
    ("convnext", "dice_loss",                  "compatible_with"),
    ("convnext", "adamw",                      "compatible_with"),
    ("convnext", "convnext_base_in22k",        "has_pretrained"),

    # yolov8
    ("yolov8", "yolov8n_coco",               "has_pretrained"),
    ("yolov8", "yolov8l_coco",               "has_pretrained"),
    ("yolov8", "detection_head_anchor_free",   "compatible_with"),
    ("yolov8", "semantic_seg_head",            "compatible_with"),
    ("yolov8", "focal_loss",                   "compatible_with"),
    ("yolov8", "bce_dice_loss",               "compatible_with"),
    ("yolov8", "adam",                         "compatible_with"),
    ("yolov8", "sgd_momentum",                 "compatible_with"),
    ("yolov8", "yolov8m_coco",               "has_pretrained"),

    # detr（head 和 loss 固定，不可替换）
    ("detr", "detection_head_transformer",     "requires"),
    ("detr", "hungarian_matching_loss",        "requires"),
    ("detr", "adamw",                          "compatible_with"),
    ("detr", "detr_resnet50_coco",            "has_pretrained"),

    # rt_detr（head 和 loss 固定，不可替换）
    ("rt_detr", "detection_head_transformer",  "requires"),
    ("rt_detr", "hungarian_matching_loss",     "requires"),
    ("rt_detr", "adamw",                       "compatible_with"),
    ("rt_detr", "rt_detr_r50_coco",          "has_pretrained"),

    # segformer
    ("segformer", "segformer_b0_ade",         "has_pretrained"),
    ("segformer", "segformer_b5_ade",         "has_pretrained"),
    ("segformer", "semantic_seg_head",         "compatible_with"),
    ("segformer", "cross_entropy_loss",        "compatible_with"),
    ("segformer", "dice_loss",                 "compatible_with"),
    ("segformer", "adamw",                     "compatible_with"),
    ("segformer", "segformer_b2_ade",         "has_pretrained"),

    # mask2former
    ("mask2former", "panoptic_seg_head",       "compatible_with"),
    ("mask2former", "semantic_seg_head",       "compatible_with"),
    ("mask2former", "dice_loss",               "compatible_with"),
    ("mask2former", "bce_dice_loss",          "compatible_with"),
    ("mask2former", "adamw",                   "compatible_with"),
    ("mask2former", "mask2former_swin_coco", "has_pretrained"),

    # unet
    ("unet", "semantic_seg_head",              "compatible_with"),
    ("unet", "bce_dice_loss",                 "compatible_with"),
    ("unet", "dice_loss",                     "compatible_with"),
    ("unet", "adam",                           "compatible_with"),

    # dinov2
    ("dinov2", "feature_pooling_head",         "compatible_with"),
    ("dinov2", "projection_head",              "compatible_with"),
    ("dinov2", "classification_head",          "compatible_with"),
    ("dinov2", "semantic_seg_head",            "compatible_with"),
    ("dinov2", "infonce_loss",                 "compatible_with"),
    ("dinov2", "cross_entropy_loss",           "compatible_with"),
    ("dinov2", "adamw",                        "compatible_with"),
    ("dinov2", "dinov2_base",                 "has_pretrained"),
    ("dinov2", "dinov2_large",               "has_pretrained"),

    # clip_vit
    ("clip_vit", "feature_pooling_head",       "compatible_with"),
    ("clip_vit", "projection_head",            "compatible_with"),
    ("clip_vit", "classification_head",        "compatible_with"),
    ("clip_vit", "infonce_loss",               "compatible_with"),
    ("clip_vit", "cross_entropy_loss",         "compatible_with"),
    ("clip_vit", "adamw",                      "compatible_with"),
    ("clip_vit", "clip_vit_base_32",          "has_pretrained"),
    ("clip_vit", "clip_vit_large_14",        "has_pretrained"),

    # ── alternative_to（双向）─────────────────────────────────────────────────
    ("swin_transformer", "convnext",           "alternative_to"),
    ("convnext",         "swin_transformer",   "alternative_to"),
    ("swin_transformer", "vit",               "alternative_to"),
    ("vit",              "swin_transformer",   "alternative_to"),
    ("resnet",           "efficientnet",       "alternative_to"),
    ("efficientnet",     "resnet",             "alternative_to"),
    ("efficientnet",     "mobilenet_v3",       "alternative_to"),
    ("mobilenet_v3",     "efficientnet",       "alternative_to"),
    ("yolov8",           "rt_detr",            "alternative_to"),
    ("rt_detr",          "yolov8",             "alternative_to"),
    ("detr",             "rt_detr",            "alternative_to"),
    ("rt_detr",          "detr",               "alternative_to"),
    ("segformer",        "mask2former",        "alternative_to"),
    ("mask2former",      "segformer",          "alternative_to"),
    ("segformer",        "unet",               "alternative_to"),
    ("unet",             "segformer",          "alternative_to"),
    ("dinov2",           "clip_vit",           "alternative_to"),
    ("clip_vit",         "dinov2",             "alternative_to"),
    ("cross_entropy_loss","focal_loss",         "alternative_to"),
    ("focal_loss",       "cross_entropy_loss", "alternative_to"),
    ("dice_loss",        "bce_dice_loss",      "alternative_to"),
    ("bce_dice_loss",    "dice_loss",          "alternative_to"),
    ("swin_base_in22k",  "swin_large_in22k",   "alternative_to"),
    ("swin_large_in22k", "swin_base_in22k",    "alternative_to"),
    ("dinov2_base",      "dinov2_large",       "alternative_to"),
    ("dinov2_large",     "dinov2_base",        "alternative_to"),
    ("clip_vit_base_32", "clip_vit_large_14",  "alternative_to"),
    ("clip_vit_large_14","clip_vit_base_32",   "alternative_to"),

    # ── preferred_when ────────────────────────────────────────────────────────
    ("yolov8",          "rt_detr",             "preferred_when"),
    ("unet",            "segformer",           "preferred_when"),
    ("focal_loss",      "cross_entropy_loss",  "preferred_when"),
    ("mobilenet_v3",    "efficientnet",        "preferred_when"),
    ("dinov2",          "clip_vit",            "preferred_when"),
    ("clip_vit",        "dinov2",              "preferred_when"),
    ("swin_large_in22k","swin_base_in22k",     "preferred_when"),
    ("dinov2_large",    "dinov2_base",         "preferred_when"),
]

EDGE_CONDITIONS = {
    ("yolov8",           "rt_detr"):          {"condition": {"all": ["real_time=True"]}},
    ("unet",             "segformer"):        {"condition": {"any": ["data_size=small", "medical=True"]}},
    ("focal_loss",       "cross_entropy_loss"):{"condition": {"all": ["class_imbalance=True"]}},
    ("mobilenet_v3",     "efficientnet"):     {"condition": {"all": ["edge_deployment=True"]}},
    ("dinov2",           "clip_vit"):         {"condition": {"all": ["no_text_modality=True"]}},
    ("clip_vit",         "dinov2"):           {"condition": {"all": ["cross_modal=True"]}},
    ("swin_large_in22k", "swin_base_in22k"):  {"condition": {"all": ["large_data=True", "high_accuracy_priority=True"]}},
    ("dinov2_large",     "dinov2_base"):      {"condition": {"all": ["feature_quality_priority=True"]}},
}


# ═══════════════════════════════════════════════════════════════════════════════
# 3. 模拟 Module 1 输出（实际由 Module 1 提供）
# ═══════════════════════════════════════════════════════════════════════════════

# Module 1 输出 schema：
# {
#   "task_type":   str   — classification | object_detection | image_segmentation | feature_extraction
#   "data_size":   str   — small | medium | large
#   "priority":    str   — speed | accuracy | balanced
#   "constraints": dict  — 布尔标志位
#   "description": str   — 自由文本，用于向量检索
# }

MODULE1_EXAMPLES = {
    "real_time_detection": {
        "task_type": "object_detection",
        "data_size": "medium",
        "priority": "speed",
        "constraints": {
            "real_time":       True,
            "edge_deployment": False,
            "class_imbalance": False,
            "cross_modal":     False,
            "medical":         False,
        },
        "description": "Detect vehicles and pedestrians from traffic camera footage, must run at 30fps",
    },
    "accurate_segmentation": {
        "task_type": "image_segmentation",
        "data_size": "large",
        "priority": "accuracy",
        "constraints": {
            "real_time":       False,
            "edge_deployment": False,
            "class_imbalance": False,
            "cross_modal":     False,
            "medical":         False,
        },
        "description": "Panoptic segmentation on a large annotated urban driving dataset",
    },
    "small_dataset_classification": {
        "task_type": "classification",
        "data_size": "small",
        "priority": "balanced",
        "constraints": {
            "real_time":       False,
            "edge_deployment": True,
            "class_imbalance": True,
            "cross_modal":     False,
            "medical":         False,
        },
        "description": "Classify plant diseases from a small dataset, deploy on mobile device, classes are imbalanced",
    },
    "cross_modal_retrieval": {
        "task_type": "feature_extraction",
        "data_size": "medium",
        "priority": "accuracy",
        "constraints": {
            "real_time":       False,
            "edge_deployment": False,
            "class_imbalance": False,
            "cross_modal":     True,
            "medical":         False,
        },
        "description": "Extract visual features aligned with text for image-text cross-modal retrieval",
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# 4. 图 + 向量索引构建
# ═══════════════════════════════════════════════════════════════════════════════

class LocalHashEmbeddingFunction:
    """Offline deterministic embedding fallback for local tests.

    The real SentenceTransformer embedding can still be enabled with
    CV_AUTODL_EMBEDDINGS=hf, but tests should not require HuggingFace network
    access just to exercise the retrieval logic.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    @staticmethod
    def name() -> str:
        return "local_hash"

    @staticmethod
    def build_from_config(config: dict) -> "LocalHashEmbeddingFunction":
        return LocalHashEmbeddingFunction(dimension=int(config.get("dimension", 384)))

    def get_config(self) -> dict:
        return {"dimension": self.dimension}

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]

    def is_legacy(self) -> bool:
        return False

    def embed_query(self, input):
        return self.__call__(input)

    def __call__(self, input):
        documents = input if isinstance(input, list) else [input]
        return [self._embed(str(document)) for document in documents]

    def _embed(self, text: str):
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector


def _make_embedding_function():
    mode = os.environ.get("CV_AUTODL_EMBEDDINGS", "local").strip().lower()
    if mode in {"hf", "huggingface", "sentence_transformer", "sentence-transformer"}:
        try:
            return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2"), "cv_backbones"
        except Exception as exc:
            print(f"[Index] Falling back to local embeddings: {exc}")
    return LocalHashEmbeddingFunction(), "cv_backbones_local"

def build_graph() -> nx.DiGraph:
    G = nx.DiGraph()
    for c in COMPONENTS:
        G.add_node(c["id"], **c)
    for src, dst, rel in EDGES:
        attrs = {"relation": rel}
        attrs.update(EDGE_CONDITIONS.get((src, dst), {}))
        G.add_edge(src, dst, **attrs)
    return G


def build_vector_index(persist_path: str = "./chroma_db_kb") -> chromadb.Collection:
    """向量索引只对 backbone 建立；pretrained_model 通过图遍历关联。"""
    client = chromadb.PersistentClient(path=persist_path)
    ef, collection_name = _make_embedding_function()
    collection = client.get_or_create_collection(collection_name, embedding_function=ef)

    backbones = [c for c in COMPONENTS if c["component_type"] == "backbone"]
    existing_ids = set(collection.get()["ids"])
    new = [b for b in backbones if b["id"] not in existing_ids]

    if new:
        collection.add(
            ids=[b["id"] for b in new],
            documents=[b["description"] for b in new],
            metadatas=[
                {
                    "task_type":            json.dumps(b["task_type"]),
                    "data_size":            json.dumps(b["data_size"]),
                    "complexity":           b["complexity"],
                    "finetune_recommended": str(b["finetune_recommended"]),
                }
                for b in new
            ],
        )
        print(f"[Index] Added {len(new)} backbone entries.")
    else:
        print("[Index] Already up to date.")

    return collection


# ═══════════════════════════════════════════════════════════════════════════════
# 5. 混合检索（方案 C）
# ═══════════════════════════════════════════════════════════════════════════════

def _matches_condition(condition: dict, input_json: dict) -> bool:
    """判断 preferred_when 边的条件是否与输入匹配。condition 格式：{"all": [...]} 或 {"any": [...]}"""
    c = input_json.get("constraints", {})
    checks = {
        "real_time=True":                c.get("real_time", False),
        "edge_deployment=True":          c.get("edge_deployment", False),
        "class_imbalance=True":          c.get("class_imbalance", False),
        "cross_modal=True":              c.get("cross_modal", False),
        "no_text_modality=True":         not c.get("cross_modal", False),
        "medical=True":                  c.get("medical", False),
        "zero_shot=True":                c.get("zero_shot", False),
        "few_shot=True":                 c.get("few_shot", False),
        "data_size=small":               input_json.get("data_size") == "small",
        "large_data=True":               input_json.get("data_size") == "large",
        "high_accuracy_priority=True":   input_json.get("priority") == "accuracy",
        "feature_quality_priority=True": input_json.get("priority") == "accuracy",
    }
    if "all" in condition:
        return all(checks.get(k, False) for k in condition["all"])
    if "any" in condition:
        return any(checks.get(k, False) for k in condition["any"])
    return False


def _input_to_query_text(input_json: dict) -> str:
    """把结构化输入转成自然语言，用于向量检索。"""
    task = input_json["task_type"].replace("_", " ")
    size = input_json.get("data_size", "medium")
    priority = input_json.get("priority", "balanced")
    c = input_json.get("constraints", {})
    desc = input_json.get("description", "")

    parts = [f"{task} task on a {size} dataset"]

    if priority == "speed":
        parts.append("speed and efficiency are the priority")
    elif priority == "accuracy":
        parts.append("high accuracy is the priority")

    flags = []
    if c.get("real_time"):        flags.append("real-time inference required")
    if c.get("edge_deployment"):  flags.append("edge or mobile deployment")
    if c.get("class_imbalance"):  flags.append("class-imbalanced data")
    if c.get("cross_modal"):      flags.append("cross-modal language-aligned features")
    if c.get("medical"):          flags.append("medical imaging domain")
    if flags:
        parts.append(", ".join(flags))

    if desc:
        parts.append(desc)

    return ". ".join(parts)


def _score_backbone(backbone_id: str, input_json: dict, graph: nx.DiGraph) -> float:
    """
    结构化打分（0–6 分制，归一化前）:
      data_size 匹配     0–2 分
      priority vs complexity  0–2 分
      preferred_when 命中  每条 +1.5 分
    """
    node = graph.nodes[backbone_id]
    score = 0.0

    # ── data_size 匹配 ─────────────────────────────────────────────
    size_order = ["small", "medium", "large"]
    req = input_json.get("data_size", "medium")
    backbone_sizes = node.get("data_size", [])

    if req in backbone_sizes:
        score += 2.0
    else:
        req_i = size_order.index(req) if req in size_order else 1
        for s in backbone_sizes:
            if s in size_order and abs(size_order.index(s) - req_i) == 1:
                score += 1.0
                break

    # ── priority vs complexity ─────────────────────────────────────
    priority = input_json.get("priority", "balanced")
    c_val = {"low": 1, "medium": 2, "high": 3}.get(node.get("complexity", "medium"), 2)

    if priority == "speed":
        score += (4 - c_val) / 3 * 2      # low→2, medium→1.33, high→0.67
    elif priority == "accuracy":
        score += c_val / 3 * 2             # high→2, medium→1.33, low→0.67
    else:
        score += (1 - abs(c_val - 2) / 2)  # medium→1, low/high→0.5

    # ── preferred_when 命中加分 ────────────────────────────────────
    for succ in graph.successors(backbone_id):
        edge = graph[backbone_id][succ]
        if edge.get("relation") == "preferred_when":
            if _matches_condition(edge.get("condition", {}), input_json):
                score += 1.5

    # ── capabilities 软加分（few_shot）────────────────────────────
    # zero_shot 由 _filter_by_tier 硬过滤；few_shot 在此软加分
    caps = node.get("capabilities", [])
    if input_json.get("constraints", {}).get("few_shot") and "few_shot" in caps:
        score += 1.5

    return score


def _select_components(
    backbone_id: str,
    task_type: str,
    input_json: dict,
    graph: nx.DiGraph,
) -> dict[str, str | None]:
    """
    对 head / loss / optimizer 各选一个最优项，考虑 constraints。
    规则优先级高于 compatible_with 边的顺序。
    """
    neighbors = list(graph.successors(backbone_id))
    c = input_json.get("constraints", {})
    result: dict[str, str | None] = {}

    for ctype in ("head", "loss", "optimizer"):
        # requires 边优先：集成架构（DETR 等）的固定组件
        required = [
            n for n in neighbors
            if graph[backbone_id][n]["relation"] == "requires"
            and graph.nodes[n]["component_type"] == ctype
        ]
        if required:
            result[ctype] = required[0]
            continue

        candidates = [
            n for n in neighbors
            if graph[backbone_id][n]["relation"] == "compatible_with"
            and graph.nodes[n]["component_type"] == ctype
            and task_type in graph.nodes[n].get("task_type", [])
        ]
        if not candidates:
            result[ctype] = None
            continue

        chosen = candidates[0]  # default: 第一个兼容项

        if ctype == "loss":
            # 类别不平衡 → 优先 focal_loss
            if c.get("class_imbalance") and "focal_loss" in candidates:
                chosen = "focal_loss"
            # 分割任务 → 优先 dice_loss，二值场景用 bce_dice
            elif task_type == "image_segmentation":
                if c.get("class_imbalance") and "bce_dice_loss" in candidates:
                    chosen = "bce_dice_loss"
                elif "dice_loss" in candidates:
                    chosen = "dice_loss"
            # DETR 系 → hungarian_matching_loss
            elif backbone_id in ("detr", "rt_detr") and "hungarian_matching_loss" in candidates:
                chosen = "hungarian_matching_loss"

        result[ctype] = chosen

    return result


_SIZE_TIER_ORDER = ["nano", "small", "base", "large", "xlarge"]

# special_case backbone → 触发它出现所需的 constraint 字段
# 不在此表里的 special_case backbone 视为"宽松特殊"，无约束也保留
_SPECIAL_CASE_REQUIRES: dict[str, str] = {
    "mobilenet_v3": "edge_deployment",
    "unet":         "medical",
    "clip_vit":     "cross_modal",
}


def _select_checkpoint(
    backbone_id: str,
    task_type: str,
    input_json: dict,
    graph: nx.DiGraph,
    scale_band: list[str] | None = None,
) -> str | None:
    """
    从 has_pretrained 边中选出最合适的 checkpoint。
    若指定 scale_band，只在该尺寸范围内选；
    根据 priority / constraints 确定目标 size_tier，选最接近的变体。
    """
    neighbors = list(graph.successors(backbone_id))
    candidates = [
        n for n in neighbors
        if graph[backbone_id][n]["relation"] == "has_pretrained"
        and task_type in graph.nodes[n].get("task_type", [])
        and (scale_band is None or graph.nodes[n].get("size_tier") in scale_band)
    ]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    priority  = input_json.get("priority", "balanced")
    data_size = input_json.get("data_size", "medium")
    c         = input_json.get("constraints", {})

    if c.get("edge_deployment"):
        target = "nano"
    elif priority == "speed":
        target = "small"
    elif priority == "accuracy" and data_size == "large":
        target = "large"
    else:
        target = "base"

    target_idx = _SIZE_TIER_ORDER.index(target) if target in _SIZE_TIER_ORDER else 2

    def _tier_dist(cid: str) -> int:
        tier = graph.nodes[cid].get("size_tier", "base")
        idx  = _SIZE_TIER_ORDER.index(tier) if tier in _SIZE_TIER_ORDER else 2
        return abs(idx - target_idx)

    return min(candidates, key=_tier_dist)


def _determine_scale_band(input_json: dict) -> list[str]:
    """
    根据输入信号确定可接受的 checkpoint 尺寸范围。
      edge/real_time  → 只允许 nano / small（硬约束）
      data=small      → 最多 base（large 模型小数据过拟合风险高）
      large+accuracy  → 优先 base / large
      其余            → 不限制
    """
    priority  = input_json.get("priority", "balanced")
    data_size = input_json.get("data_size", "medium")
    c         = input_json.get("constraints", {})

    if c.get("edge_deployment") or c.get("real_time"):
        return ["nano", "small"]
    if data_size == "small":
        return ["nano", "small", "base"]
    if data_size == "large" and priority == "accuracy":
        return ["base", "large"]
    return list(_SIZE_TIER_ORDER)


def _get_eligible_pairs(
    task_type: str,
    scale_band: list[str],
    input_json: dict,
    graph: nx.DiGraph,
) -> list[tuple[str, str | None]]:
    """
    返回 (backbone_id, checkpoint_id | None) 对。
    backbone 进入候选的条件：
      - 支持 task_type，且
      - 在 scale_band 内有 checkpoint，或
      - scratch_viable_from 允许当前 data_size（checkpoint=None）
    """
    size_order = ["small", "medium", "large"]
    data_size  = input_json.get("data_size", "medium")
    data_idx   = size_order.index(data_size) if data_size in size_order else 1

    pairs: list[tuple[str, str | None]] = []
    for node_id, node_data in graph.nodes(data=True):
        if node_data.get("component_type") != "backbone":
            continue
        if task_type not in node_data.get("task_type", []):
            continue

        cps_in_band = [
            n for n in graph.successors(node_id)
            if graph[node_id][n]["relation"] == "has_pretrained"
            and task_type in graph.nodes[n].get("task_type", [])
            and graph.nodes[n].get("size_tier") in scale_band
        ]

        if cps_in_band:
            cp = _select_checkpoint(node_id, task_type, input_json, graph, scale_band)
            pairs.append((node_id, cp))
        else:
            scratch_from = node_data.get("scratch_viable_from")
            if scratch_from is None:
                continue
            scratch_idx = size_order.index(scratch_from) if scratch_from in size_order else 99
            if data_idx >= scratch_idx:
                pairs.append((node_id, None))

    return pairs


def _filter_by_tier(
    pairs: list[tuple[str, str | None]],
    task_type: str,
    input_json: dict,
    graph: nx.DiGraph,
) -> list[tuple[str, str | None]]:
    """
    按 tier 过滤候选：
      default          → 始终保留
      accuracy_upgrade → 仅 priority=accuracy 时保留
      special_case     → 需要对应 constraint 激活（_SPECIAL_CASE_REQUIRES），
                         不在表里的 special_case 无条件保留（宽松特殊）
                         特例：yolov8 做 image_segmentation 需要 real_time 或 edge_deployment
    """
    priority = input_json.get("priority", "balanced")
    c        = input_json.get("constraints", {})
    zero_shot = c.get("zero_shot", False)

    result = []
    for backbone_id, checkpoint_id in pairs:
        # zero_shot 硬过滤：必须有 zero_shot capability
        if zero_shot and "zero_shot" not in graph.nodes[backbone_id].get("capabilities", []):
            continue

        tier = graph.nodes[backbone_id].get("tier", {}).get(task_type, "default")

        if tier == "default":
            result.append((backbone_id, checkpoint_id))

        elif tier == "accuracy_upgrade":
            if priority == "accuracy":
                result.append((backbone_id, checkpoint_id))

        elif tier == "special_case":
            required = _SPECIAL_CASE_REQUIRES.get(backbone_id)
            if required:
                if c.get(required):
                    result.append((backbone_id, checkpoint_id))
            elif backbone_id == "yolov8" and task_type == "image_segmentation":
                if c.get("real_time") or c.get("edge_deployment"):
                    result.append((backbone_id, checkpoint_id))
            else:
                # 宽松特殊：无严格约束要求，保留为备选
                result.append((backbone_id, checkpoint_id))

    return result


def _recommend_training(
    backbone_id: str,
    checkpoint_id: str | None,
    input_json: dict,
    graph: nx.DiGraph,
) -> dict:
    """
    返回训练策略建议：
      scratch_viable    — 当前数据量下从头训练是否可行
      finetune_strategy — "full" | "head_only" | "either"（来自 checkpoint 节点）
      freeze_viable     — 是否可以冻结骨干只训 head
    """
    node = graph.nodes[backbone_id]
    data_size = input_json.get("data_size", "medium")
    size_order = ["small", "medium", "large"]
    scratch_from = node.get("scratch_viable_from")

    scratch_viable = False
    if scratch_from is not None and scratch_from in size_order and data_size in size_order:
        scratch_viable = size_order.index(data_size) >= size_order.index(scratch_from)

    finetune_strategy = None
    freeze_viable = False
    if checkpoint_id:
        cp = graph.nodes[checkpoint_id]
        finetune_strategy = cp.get("finetune_strategy")
        freeze_viable = cp.get("freeze_viable", False)

    return {
        "scratch_viable":   scratch_viable,
        "finetune_strategy": finetune_strategy,
        "freeze_viable":    freeze_viable,
    }


def retrieve_top3_hybrid(
    input_json: dict,
    graph: nx.DiGraph,
    collection: chromadb.Collection,
    w_vector: float = 0.4,
    w_structured: float = 0.6,
) -> list[dict]:
    """
    混合检索流程：
      Step 1  规模带过滤 — scale_band → eligible pairs → tier filter
              (backbone, checkpoint) 作为原子单元进入候选
      Step 2  结构化打分 — data_size、priority、preferred_when
      Step 3  向量打分 — 输入转自然语言后与 backbone description 做相似度
      Step 4  加权合并排序 → Top 3
      Step 5  图遍历拼装 head / loss / optimizer / pretrained
    """
    task_type = input_json["task_type"]

    # Step 1: Scale band → eligible pairs → tier filter
    scale_band = _determine_scale_band(input_json)
    pairs      = _get_eligible_pairs(task_type, scale_band, input_json, graph)
    pairs      = _filter_by_tier(pairs, task_type, input_json, graph)
    if not pairs:
        return []
    eligible  = [bb for bb, _ in pairs]
    pair_map  = dict(pairs)

    # Step 2: 结构化分数（归一化到 [0, 1]）
    raw_struct = {bid: _score_backbone(bid, input_json, graph) for bid in eligible}
    max_s = max(raw_struct.values()) or 1.0
    struct_scores = {k: v / max_s for k, v in raw_struct.items()}

    # Step 3: 向量分数（归一化到 [0, 1]，距离越小分越高）
    query_text = _input_to_query_text(input_json)
    vr = collection.query(query_texts=[query_text], n_results=collection.count())
    raw_vec = dict(zip(vr["ids"][0], vr["distances"][0]))

    # 只保留 eligible 里的结果
    raw_vec = {k: v for k, v in raw_vec.items() if k in eligible}
    if raw_vec:
        min_d, max_d = min(raw_vec.values()), max(raw_vec.values())
        span = max_d - min_d if max_d > min_d else 1.0
        # 距离越小 → 分数越高（线性翻转）
        vec_scores = {k: 1.0 - (v - min_d) / span for k, v in raw_vec.items()}
    else:
        vec_scores = {}

    # Step 4: 加权合并
    final_scores = {
        bid: w_structured * struct_scores.get(bid, 0) + w_vector * vec_scores.get(bid, 0)
        for bid in eligible
    }
    top3 = sorted(final_scores, key=lambda x: final_scores[x], reverse=True)[:3]

    # Step 5: 图遍历拼装
    configurations = []
    for backbone_id in top3:
        neighbors = list(graph.successors(backbone_id))

        components = _select_components(backbone_id, task_type, input_json, graph)
        checkpoint  = pair_map[backbone_id]
        training    = _recommend_training(backbone_id, checkpoint, input_json, graph)

        alt_backbones = [
            n for n in neighbors if graph[backbone_id][n]["relation"] == "alternative_to"
        ]

        configurations.append({
            "backbone":          backbone_id,
            "head":              components.get("head"),
            "loss":              components.get("loss"),
            "optimizer":         components.get("optimizer"),
            "pretrained":        checkpoint,
            "scratch_viable":    training["scratch_viable"],
            "finetune_strategy": training["finetune_strategy"],
            "freeze_viable":     training["freeze_viable"],
            "alt_backbones":     alt_backbones,
            "score": round(final_scores[backbone_id], 3),
            "score_detail": {
                "structured": round(struct_scores.get(backbone_id, 0), 3),
                "vector":     round(vec_scores.get(backbone_id, 0), 3),
            },
        })

    return configurations


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Module 4 接口 — 任务清单生成
# ═══════════════════════════════════════════════════════════════════════════════

def build_task_list(result: dict, graph: nx.DiGraph, fmt: str = "structured") -> dict:
    """
    将单条 retrieve_top3_hybrid 结果转换为 Module 4 可消费的任务清单。

    fmt="structured" — 结构化 JSON，适合确定性代码模板填充
    fmt="nl"         — 自然语言任务列表 + 元数据，适合 LLM agent prompt
    """
    backbone_id   = result["backbone"]
    checkpoint_id = result.get("pretrained")
    head_id       = result.get("head")
    loss_id       = result.get("loss")
    optimizer_id  = result.get("optimizer")
    strategy      = result.get("finetune_strategy")
    freeze        = result.get("freeze_viable", False)
    scratch       = result.get("scratch_viable", False)
    alternatives  = result.get("alt_backbones", [])

    def _name(nid):
        return graph.nodes[nid]["name"] if nid else None

    if fmt == "structured":
        tasks = []

        if checkpoint_id:
            cp = graph.nodes[checkpoint_id]
            tasks.append({
                "id":           "load_model",
                "action":       "load_pretrained",
                "hf_id":        cp["hf_id"],
                "model_name":   cp["name"],
                "params_M":     cp["params_M"],
                "finetune_base": cp["finetune_base"],
            })
        else:
            tasks.append({
                "id":      "load_model",
                "action":  "train_from_scratch",
                "backbone": backbone_id,
            })

        tasks.append({
            "id":              "train_strategy",
            "action":          "set_finetune_strategy",
            "strategy":        strategy,
            "freeze_backbone": not freeze or strategy == "full",
            "scratch_viable":  scratch,
        })

        if head_id:
            tasks.append({
                "id":     "head",
                "action": "configure_head",
                "type":   head_id,
                "name":   _name(head_id),
            })

        if loss_id:
            tasks.append({
                "id":     "loss",
                "action": "configure_loss",
                "type":   loss_id,
                "name":   _name(loss_id),
            })

        if optimizer_id:
            tasks.append({
                "id":     "optimizer",
                "action": "configure_optimizer",
                "type":   optimizer_id,
                "name":   _name(optimizer_id),
            })

        return {
            "format":       "structured",
            "backbone":     backbone_id,
            "backbone_name": _name(backbone_id),
            "tasks":        tasks,
            "alternatives": alternatives,
        }

    elif fmt == "nl":
        nl_tasks = []

        if checkpoint_id:
            cp = graph.nodes[checkpoint_id]
            nl_tasks.append(
                f"Load {cp['name']} from {cp['hf_id']} "
                f"({cp['params_M']}M params, pretrained on {cp['pretrain_dataset']})"
            )
        else:
            nl_tasks.append(
                f"Train {_name(backbone_id)} from scratch on your dataset"
            )

        strategy_desc = {
            "full":      "Full finetune: update all backbone and head weights",
            "head_only": "Head-only finetune: freeze backbone, train head only",
            "either":    "Either full finetune or freeze backbone + train head is viable",
        }.get(strategy, f"Finetune strategy: {strategy}")
        nl_tasks.append(strategy_desc)

        if head_id:
            nl_tasks.append(f"Use {_name(head_id)} as the output head")
        if loss_id:
            nl_tasks.append(f"Use {_name(loss_id)} as the training loss")
        if optimizer_id:
            nl_tasks.append(f"Use {_name(optimizer_id)} as the optimizer")

        model_config: dict = {"backbone": backbone_id}
        if checkpoint_id:
            cp = graph.nodes[checkpoint_id]
            model_config.update({
                "pretrained_hf_id":   cp["hf_id"],
                "pretrained_name":    cp["name"],
                "pretrain_dataset":   cp["pretrain_dataset"],
                "params_M":           cp["params_M"],
            })
        model_config.update({
            "head":              head_id,
            "loss":              loss_id,
            "optimizer":         optimizer_id,
            "finetune_strategy": strategy,
            "freeze_backbone":   not freeze or strategy == "full",
            "scratch_viable":    scratch,
        })

        return {
            "format":       "nl",
            "model_config": model_config,
            "tasks":        nl_tasks,
            "alternatives": alternatives,
        }

    else:
        raise ValueError(f"Unknown fmt '{fmt}'. Use 'structured' or 'nl'.")


def build_all_task_lists(
    results: list[dict],
    graph: nx.DiGraph,
    fmt: str = "structured",
) -> list[dict]:
    """Top 3 结果全部转换为任务清单，rank 字段标注排名。"""
    out = []
    for rank, result in enumerate(results, 1):
        tl = build_task_list(result, graph, fmt=fmt)
        tl["rank"]  = rank
        tl["score"] = result.get("score")
        out.append(tl)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# 7. 输出格式化
# ═══════════════════════════════════════════════════════════════════════════════

def print_results(input_json: dict, results: list[dict], graph: nx.DiGraph) -> None:
    print(f"\n{'─'*70}")
    print(f"Task   : {input_json['task_type']}")
    print(f"Input  : size={input_json['data_size']}  priority={input_json['priority']}")
    constraints_on = [k for k, v in input_json.get("constraints", {}).items() if v]
    if constraints_on:
        print(f"Flags  : {', '.join(constraints_on)}")
    print(f"Desc   : {input_json.get('description', '')}")
    print(f"{'─'*70}")

    for i, r in enumerate(results, 1):
        detail = r["score_detail"]
        print(f"\nTop {i}  [score={r['score']}  struct={detail['structured']}  vec={detail['vector']}]")
        for role in ("backbone", "head", "loss", "optimizer"):
            cid = r.get(role)
            if cid:
                print(f"  {role:10s}: {graph.nodes[cid]['name']}")

        pid = r.get("pretrained")
        if pid:
            p = graph.nodes[pid]
            strategy_label = {
                "full":      "全量 finetune",
                "head_only": "冻结骨干，只训 head",
                "either":    "全量 finetune 或冻结骨干均可",
            }.get(r.get("finetune_strategy", ""), r.get("finetune_strategy", ""))
            print(f"  {'pretrained':10s}: {p['name']}")
            print(f"               {p['hf_id']}  ({p['pretrain_dataset']}, {p['params_M']}M)")
            print(f"               策略: {strategy_label}")
        else:
            print(f"  {'pretrained':10s}: 无（从头训练）")

        scratch = r.get("scratch_viable", False)
        if not pid:
            print(f"  {'training':10s}: 从头训练")
        elif scratch:
            print(f"  {'training':10s}: 建议 finetune，数据量足够时从头训练也可行")
        else:
            print(f"  {'training':10s}: 必须使用预训练权重（数据量不足以从头训练）")

        if r["alt_backbones"]:
            alts = [graph.nodes[a]["name"] for a in r["alt_backbones"]]
            print(f"  {'alt':10s}: {', '.join(alts)}")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# 7. 运行示例
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    G = build_graph()
    col = build_vector_index()

    for name, input_json in MODULE1_EXAMPLES.items():
        results = retrieve_top3_hybrid(input_json, G, col)
        print_results(input_json, results, G)

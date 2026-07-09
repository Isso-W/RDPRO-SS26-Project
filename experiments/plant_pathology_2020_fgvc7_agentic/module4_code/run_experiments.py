"""Sweep all Module 3 candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate import evaluate
from train import train_model
from utils import as_bool, as_int, compact_config_summary, get_value, load_configs, set_seed


DEFAULT_CONFIGS = json.loads('[\n  {\n    "alternatives": [],\n    "augmentation": "basic",\n    "backbone": "dinov3",\n    "checkpoint": "facebook/dinov3-vitb16-pretrain-lvd1689m",\n    "class_imbalance": true,\n    "data_size": "medium",\n    "embedding_dim": 32,\n    "finetune_strategy": "partial",\n    "freeze_backbone": false,\n    "head": "classification_head",\n    "image_size": 224,\n    "learning_rate": 0.0001,\n    "loss": "cross_entropy_loss",\n    "model_config": {\n      "backbone": "dinov3",\n      "batch_size": 8,\n      "class_imbalance": true,\n      "class_weight_power": 0.5,\n      "early_stopping_patience": 4,\n      "eval_batch_size": 16,\n      "evaluation_metric": "roc_auc",\n      "export_preds_path": "outputs/val_predictions_candidate_1.csv",\n      "finetune_strategy": "partial",\n      "fold_file": "/content/kaggle_data/plant-pathology-2020-fgvc7/folds.json",\n      "fold_index": 0,\n      "freeze_backbone": false,\n      "head": "classification_head",\n      "image_column": "image_id",\n      "image_dir": "/content/kaggle_data/plant-pathology-2020-fgvc7/images",\n      "image_extension": ".jpg",\n      "image_path_template": "{image}",\n      "image_size": 224,\n      "label_column": "__jiaozi_label",\n      "label_columns": [\n        "healthy",\n        "multiple_diseases",\n        "rust",\n        "scab"\n      ],\n      "learning_rate": 0.0001,\n      "loss": "cross_entropy_loss",\n      "mixed_precision": true,\n      "num_classes": 4,\n      "offline_smoke": false,\n      "optimizer": "adamw",\n      "params_M": 86,\n      "pretrain_dataset": "LVD-1689M (self-supervised)",\n      "pretrained_hf_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",\n      "pretrained_name": "DINOv3-B/16 / LVD-1689M",\n      "recipe": {\n        "augmentation": {\n          "invariance": {\n            "color": true,\n            "crop_scale_min": 0.4,\n            "hflip": true,\n            "rot90": false,\n            "vflip": false\n          },\n          "schedule": "taper_last_20pct",\n          "tier": "heavy"\n        },\n        "epochs": 40,\n        "image_size": 224,\n        "learning_rate": 0.0001\n      },\n      "recipe_provenance": {\n        "augmentation": "tier[data_size=small\\u2192heavy]; invariance[domain_signal_missing]; schedule[small\\u2192taper_last_20pct]",\n        "epochs": "epochs_table[small,finetune]",\n        "image_size": "family_default=224",\n        "learning_rate": "lr_base[cnn,finetune]"\n      },\n      "recommended_epochs": 40,\n      "scratch_viable": false,\n      "strategy_ablation_group": "rank1_dinov3_strategy",\n      "strategy_ablation_variant": "partial_last2",\n      "train_csv": "/content/kaggle_data/plant-pathology-2020-fgvc7/train__jiaozi_labels.csv",\n      "train_norm_layers": true,\n      "tta": {\n        "enabled": true,\n        "num_augments": 2,\n        "transforms": [\n          "hflip"\n        ]\n      },\n      "unfreeze_last_n_blocks": 2,\n      "use_class_weights": true\n    },\n    "num_classes": 4,\n    "offline_smoke": false,\n    "optimizer": "adamw",\n    "params_M": 86.0,\n    "pretrained_hf_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",\n    "pretrained_name": "DINOv3-B/16 / LVD-1689M",\n    "rank": 1,\n    "score": 1.0,\n    "scratch_viable": false,\n    "strategy_ablation_group": "rank1_dinov3_strategy",\n    "strategy_ablation_variant": "partial_last2",\n    "task_type": "classification",\n    "tasks": [\n      "Load DINOv3-B/16 / LVD-1689M from facebook/dinov3-vitb16-pretrain-lvd1689m (86M params, pretrained on LVD-1689M (self-supervised))",\n      "Partial finetune: update head, norm layers, and the last 2 backbone blocks",\n      "Use Classification Head as the output head",\n      "Use CrossEntropyLoss as the training loss",\n      "Use AdamW as the optimizer"\n    ],\n    "train_norm_layers": true,\n    "tta": false,\n    "unfreeze_last_n_blocks": 2,\n    "use_pretrained": true\n  },\n  {\n    "alternatives": [],\n    "augmentation": "basic",\n    "backbone": "dinov3",\n    "checkpoint": "facebook/dinov3-vitb16-pretrain-lvd1689m",\n    "class_imbalance": true,\n    "data_size": "medium",\n    "embedding_dim": 32,\n    "finetune_strategy": "partial",\n    "freeze_backbone": false,\n    "head": "classification_head",\n    "image_size": 224,\n    "learning_rate": 0.0001,\n    "loss": "cross_entropy_loss",\n    "model_config": {\n      "backbone": "dinov3",\n      "batch_size": 8,\n      "class_imbalance": true,\n      "class_weight_power": 0.5,\n      "early_stopping_patience": 4,\n      "eval_batch_size": 16,\n      "evaluation_metric": "roc_auc",\n      "export_preds_path": "outputs/val_predictions_candidate_1.csv",\n      "finetune_strategy": "partial",\n      "fold_file": "/content/kaggle_data/plant-pathology-2020-fgvc7/folds.json",\n      "fold_index": 0,\n      "freeze_backbone": false,\n      "head": "classification_head",\n      "image_column": "image_id",\n      "image_dir": "/content/kaggle_data/plant-pathology-2020-fgvc7/images",\n      "image_extension": ".jpg",\n      "image_path_template": "{image}",\n      "image_size": 224,\n      "label_column": "__jiaozi_label",\n      "label_columns": [\n        "healthy",\n        "multiple_diseases",\n        "rust",\n        "scab"\n      ],\n      "learning_rate": 0.0001,\n      "loss": "cross_entropy_loss",\n      "mixed_precision": true,\n      "num_classes": 4,\n      "offline_smoke": false,\n      "optimizer": "adamw",\n      "params_M": 86,\n      "pretrain_dataset": "LVD-1689M (self-supervised)",\n      "pretrained_hf_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",\n      "pretrained_name": "DINOv3-B/16 / LVD-1689M",\n      "recipe": {\n        "augmentation": {\n          "invariance": {\n            "color": true,\n            "crop_scale_min": 0.4,\n            "hflip": true,\n            "rot90": false,\n            "vflip": false\n          },\n          "schedule": "taper_last_20pct",\n          "tier": "heavy"\n        },\n        "epochs": 40,\n        "image_size": 224,\n        "learning_rate": 0.0001\n      },\n      "recipe_provenance": {\n        "augmentation": "tier[data_size=small\\u2192heavy]; invariance[domain_signal_missing]; schedule[small\\u2192taper_last_20pct]",\n        "epochs": "epochs_table[small,finetune]",\n        "image_size": "family_default=224",\n        "learning_rate": "lr_base[cnn,finetune]"\n      },\n      "recommended_epochs": 40,\n      "scratch_viable": false,\n      "strategy_ablation_group": "rank1_dinov3_strategy",\n      "strategy_ablation_variant": "partial_last4",\n      "train_csv": "/content/kaggle_data/plant-pathology-2020-fgvc7/train__jiaozi_labels.csv",\n      "train_norm_layers": true,\n      "tta": {\n        "enabled": true,\n        "num_augments": 2,\n        "transforms": [\n          "hflip"\n        ]\n      },\n      "unfreeze_last_n_blocks": 4,\n      "use_class_weights": true\n    },\n    "num_classes": 4,\n    "offline_smoke": false,\n    "optimizer": "adamw",\n    "params_M": 86.0,\n    "pretrained_hf_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",\n    "pretrained_name": "DINOv3-B/16 / LVD-1689M",\n    "rank": 2,\n    "score": 1.0,\n    "scratch_viable": false,\n    "strategy_ablation_group": "rank1_dinov3_strategy",\n    "strategy_ablation_variant": "partial_last4",\n    "task_type": "classification",\n    "tasks": [\n      "Load DINOv3-B/16 / LVD-1689M from facebook/dinov3-vitb16-pretrain-lvd1689m (86M params, pretrained on LVD-1689M (self-supervised))",\n      "Partial finetune: update head, norm layers, and the last 2 backbone blocks",\n      "Use Classification Head as the output head",\n      "Use CrossEntropyLoss as the training loss",\n      "Use AdamW as the optimizer"\n    ],\n    "train_norm_layers": true,\n    "tta": false,\n    "unfreeze_last_n_blocks": 4,\n    "use_pretrained": true\n  },\n  {\n    "alternatives": [\n      "dinov3"\n    ],\n    "augmentation": "basic",\n    "backbone": "dinov2",\n    "checkpoint": "facebook/dinov2-base",\n    "class_imbalance": true,\n    "data_size": "medium",\n    "embedding_dim": 32,\n    "finetune_strategy": "head_only",\n    "freeze_backbone": true,\n    "head": "classification_head",\n    "image_size": 224,\n    "learning_rate": 0.001,\n    "loss": "cross_entropy_loss",\n    "model_config": {\n      "backbone": "dinov2",\n      "batch_size": 8,\n      "class_imbalance": true,\n      "class_weight_power": 0.5,\n      "early_stopping_patience": 4,\n      "eval_batch_size": 16,\n      "evaluation_metric": "roc_auc",\n      "export_preds_path": "outputs/val_predictions_candidate_2.csv",\n      "finetune_strategy": "head_only",\n      "fold_file": "/content/kaggle_data/plant-pathology-2020-fgvc7/folds.json",\n      "fold_index": 0,\n      "freeze_backbone": true,\n      "head": "classification_head",\n      "image_column": "image_id",\n      "image_dir": "/content/kaggle_data/plant-pathology-2020-fgvc7/images",\n      "image_extension": ".jpg",\n      "image_path_template": "{image}",\n      "image_size": 224,\n      "label_column": "__jiaozi_label",\n      "label_columns": [\n        "healthy",\n        "multiple_diseases",\n        "rust",\n        "scab"\n      ],\n      "learning_rate": 0.001,\n      "loss": "cross_entropy_loss",\n      "mixed_precision": true,\n      "num_classes": 4,\n      "offline_smoke": false,\n      "optimizer": "adamw",\n      "params_M": 86,\n      "pretrain_dataset": "LVD-142M (self-supervised)",\n      "pretrained_hf_id": "facebook/dinov2-base",\n      "pretrained_name": "DINOv2-Base",\n      "recipe": {\n        "augmentation": {\n          "invariance": {\n            "color": true,\n            "crop_scale_min": 0.6,\n            "hflip": true,\n            "rot90": false,\n            "vflip": false\n          },\n          "schedule": "taper_last_20pct",\n          "tier": "medium"\n        },\n        "epochs": 25,\n        "image_size": 224,\n        "learning_rate": 0.001\n      },\n      "recipe_provenance": {\n        "augmentation": "tier[data_size=small\\u2192heavy | head_only\\u2192medium]; invariance[domain_signal_missing]; schedule[small\\u2192taper_last_20pct]",\n        "epochs": "epochs_table[small,head_only]",\n        "image_size": "family_default=224 | ok/14",\n        "learning_rate": "lr_base[transformer,head_only]"\n      },\n      "recommended_epochs": 25,\n      "scratch_viable": false,\n      "train_csv": "/content/kaggle_data/plant-pathology-2020-fgvc7/train__jiaozi_labels.csv",\n      "train_norm_layers": false,\n      "tta": {\n        "enabled": true,\n        "num_augments": 2,\n        "transforms": [\n          "hflip"\n        ]\n      },\n      "unfreeze_last_n_blocks": 0,\n      "use_class_weights": true\n    },\n    "num_classes": 4,\n    "offline_smoke": false,\n    "optimizer": "adamw",\n    "params_M": 86.0,\n    "pretrained_hf_id": "facebook/dinov2-base",\n    "pretrained_name": "DINOv2-Base",\n    "rank": 3,\n    "score": 0.673,\n    "scratch_viable": false,\n    "strategy_ablation_group": "",\n    "strategy_ablation_variant": "",\n    "task_type": "classification",\n    "tasks": [\n      "Load DINOv2-Base from facebook/dinov2-base (86M params, pretrained on LVD-142M (self-supervised))",\n      "Head-only finetune: freeze backbone, train head only",\n      "Use Classification Head as the output head",\n      "Use CrossEntropyLoss as the training loss",\n      "Use AdamW as the optimizer"\n    ],\n    "train_norm_layers": false,\n    "tta": false,\n    "unfreeze_last_n_blocks": 0,\n    "use_pretrained": true\n  },\n  {\n    "alternatives": [\n      "convnext",\n      "vit"\n    ],\n    "augmentation": "basic",\n    "backbone": "swin_transformer",\n    "checkpoint": "microsoft/swin-base-patch4-window7-224",\n    "class_imbalance": true,\n    "data_size": "medium",\n    "embedding_dim": 32,\n    "finetune_strategy": "full",\n    "freeze_backbone": false,\n    "head": "classification_head",\n    "image_size": 224,\n    "learning_rate": 3e-05,\n    "loss": "focal_loss",\n    "model_config": {\n      "backbone": "swin_transformer",\n      "batch_size": 8,\n      "class_imbalance": true,\n      "class_weight_power": 0.5,\n      "early_stopping_patience": 4,\n      "eval_batch_size": 16,\n      "evaluation_metric": "roc_auc",\n      "export_preds_path": "outputs/val_predictions_candidate_3.csv",\n      "finetune_strategy": "full",\n      "fold_file": "/content/kaggle_data/plant-pathology-2020-fgvc7/folds.json",\n      "fold_index": 0,\n      "freeze_backbone": false,\n      "head": "classification_head",\n      "image_column": "image_id",\n      "image_dir": "/content/kaggle_data/plant-pathology-2020-fgvc7/images",\n      "image_extension": ".jpg",\n      "image_path_template": "{image}",\n      "image_size": 224,\n      "label_column": "__jiaozi_label",\n      "label_columns": [\n        "healthy",\n        "multiple_diseases",\n        "rust",\n        "scab"\n      ],\n      "learning_rate": 3e-05,\n      "loss": "focal_loss",\n      "mixed_precision": true,\n      "num_classes": 4,\n      "offline_smoke": false,\n      "optimizer": "adamw",\n      "params_M": 88,\n      "pretrain_dataset": "ImageNet-22k",\n      "pretrained_hf_id": "microsoft/swin-base-patch4-window7-224",\n      "pretrained_name": "Swin-Base / ImageNet-22k",\n      "recipe": {\n        "augmentation": {\n          "invariance": {\n            "color": true,\n            "crop_scale_min": 0.4,\n            "hflip": true,\n            "rot90": false,\n            "vflip": false\n          },\n          "schedule": "taper_last_20pct",\n          "tier": "heavy"\n        },\n        "epochs": 40,\n        "image_size": 224,\n        "learning_rate": 3e-05\n      },\n      "recipe_provenance": {\n        "augmentation": "tier[data_size=small\\u2192heavy]; invariance[domain_signal_missing]; schedule[small\\u2192taper_last_20pct]",\n        "epochs": "epochs_table[small,finetune]",\n        "image_size": "family_default=224 | ok/32",\n        "learning_rate": "lr_base[transformer,finetune]"\n      },\n      "recommended_epochs": 40,\n      "scratch_viable": false,\n      "train_csv": "/content/kaggle_data/plant-pathology-2020-fgvc7/train__jiaozi_labels.csv",\n      "train_norm_layers": false,\n      "tta": {\n        "enabled": true,\n        "num_augments": 2,\n        "transforms": [\n          "hflip"\n        ]\n      },\n      "unfreeze_last_n_blocks": 0,\n      "use_class_weights": true\n    },\n    "num_classes": 4,\n    "offline_smoke": false,\n    "optimizer": "adamw",\n    "params_M": 88.0,\n    "pretrained_hf_id": "microsoft/swin-base-patch4-window7-224",\n    "pretrained_name": "Swin-Base / ImageNet-22k",\n    "rank": 4,\n    "score": 0.649,\n    "scratch_viable": false,\n    "strategy_ablation_group": "",\n    "strategy_ablation_variant": "",\n    "task_type": "classification",\n    "tasks": [\n      "Load Swin-Base / ImageNet-22k from microsoft/swin-base-patch4-window7-224 (88M params, pretrained on ImageNet-22k)",\n      "Full finetune: update all backbone and head weights",\n      "Use Classification Head as the output head",\n      "Use FocalLoss as the training loss",\n      "Use AdamW as the optimizer"\n    ],\n    "train_norm_layers": false,\n    "tta": false,\n    "unfreeze_last_n_blocks": 0,\n    "use_pretrained": true\n  }\n]')


def _metric_direction(metric_name: str | None) -> str:
    name = str(metric_name or "").lower()
    if any(token in name for token in ("loss", "error", "rmse", "mae")):
        return "min"
    return "max"


def _select_best_index(rows: list[dict[str, Any]]) -> int | None:
    best_index = None
    best_value = None
    best_direction = "max"
    for index, row in enumerate(rows):
        if row.get("status") != "success":
            continue
        value = row.get("metric_value")
        if value is None:
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        direction = _metric_direction(str(row.get("metric_name", "")))
        if best_index is None:
            best_index = index
            best_value = value
            best_direction = direction
            continue
        if direction != best_direction:
            direction = best_direction
        improved = value < best_value if direction == "min" else value > best_value
        if improved:
            best_index = index
            best_value = value
    return best_index


def run_all(configs: list[dict[str, Any]], seed: int = 123, epochs: int | None = None) -> list[dict[str, Any]]:
    rows = []
    normalized_configs = []
    for index, config in enumerate(configs, start=1):
        set_seed(seed)
        offline_smoke = as_bool(get_value(config, "offline_smoke", True), True)
        default_ep = 1 if offline_smoke else as_int(get_value(config, "recommended_epochs", 10), 10)
        ep = epochs if epochs is not None else default_ep
        ms = 1 if offline_smoke else 0
        normalized_configs.append(config)
        model, train_result = train_model(config, epochs=ep, max_steps=ms)
        eval_result = evaluate(model, config)
        row = compact_config_summary(config, rank_default=index)
        row.update(
            {
                "metric_name": eval_result.get("metric_name"),
                "metric_value": eval_result.get("metric_value"),
                "status": "success" if train_result.get("status") == "success" and eval_result.get("status") == "success" else "failed",
            }
        )
        rows.append(row)
    best_index = _select_best_index(rows)
    if best_index is not None:
        for index, row in enumerate(rows):
            row["selected_best"] = index == best_index
        Path("best_config.json").write_text(
            json.dumps(normalized_configs[best_index], indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep all Module 3 candidate configs.")
    parser.add_argument("--input", default="configs.json", help="JSON file with one or more configs.")
    parser.add_argument("--output", default=None, help="Optional path for the sweep result JSON.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Training epochs per candidate (default: 1 smoke / 10 real).")
    args = parser.parse_args()
    rows = run_all(load_configs(args.input, DEFAULT_CONFIGS), seed=args.seed, epochs=args.epochs)
    result_json = json.dumps(rows, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result_json + "\n", encoding="utf-8")
        print(f"Wrote sweep results to {output_path}")
    print(result_json)


if __name__ == "__main__":
    main()

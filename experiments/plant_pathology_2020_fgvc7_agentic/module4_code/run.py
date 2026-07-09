"""Single-configuration runner.

Smoke mode (default):  offline_smoke=true  → synthetic data, 1 epoch, 1 step.
Real training mode:    offline_smoke=false  → HuggingFace dataset, multi-epoch,
                       checkpoint saving.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evaluate import evaluate
from infer import predict
from train import train_model
from utils import as_bool, as_int, compact_config_summary, get_value, load_config, set_seed


DEFAULT_CONFIG = json.loads('{\n  "alternatives": [],\n  "augmentation": "basic",\n  "backbone": "dinov3",\n  "checkpoint": "facebook/dinov3-vitb16-pretrain-lvd1689m",\n  "class_imbalance": true,\n  "data_size": "medium",\n  "embedding_dim": 32,\n  "finetune_strategy": "partial",\n  "freeze_backbone": false,\n  "head": "classification_head",\n  "image_size": 224,\n  "learning_rate": 0.0001,\n  "loss": "cross_entropy_loss",\n  "model_config": {\n    "backbone": "dinov3",\n    "batch_size": 8,\n    "class_imbalance": true,\n    "class_weight_power": 0.5,\n    "early_stopping_patience": 4,\n    "eval_batch_size": 16,\n    "evaluation_metric": "roc_auc",\n    "export_preds_path": "outputs/val_predictions_candidate_1.csv",\n    "finetune_strategy": "partial",\n    "fold_file": "/content/kaggle_data/plant-pathology-2020-fgvc7/folds.json",\n    "fold_index": 0,\n    "freeze_backbone": false,\n    "head": "classification_head",\n    "image_column": "image_id",\n    "image_dir": "/content/kaggle_data/plant-pathology-2020-fgvc7/images",\n    "image_extension": ".jpg",\n    "image_path_template": "{image}",\n    "image_size": 224,\n    "label_column": "__jiaozi_label",\n    "label_columns": [\n      "healthy",\n      "multiple_diseases",\n      "rust",\n      "scab"\n    ],\n    "learning_rate": 0.0001,\n    "loss": "cross_entropy_loss",\n    "mixed_precision": true,\n    "num_classes": 4,\n    "offline_smoke": false,\n    "optimizer": "adamw",\n    "params_M": 86,\n    "pretrain_dataset": "LVD-1689M (self-supervised)",\n    "pretrained_hf_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",\n    "pretrained_name": "DINOv3-B/16 / LVD-1689M",\n    "recipe": {\n      "augmentation": {\n        "invariance": {\n          "color": true,\n          "crop_scale_min": 0.4,\n          "hflip": true,\n          "rot90": false,\n          "vflip": false\n        },\n        "schedule": "taper_last_20pct",\n        "tier": "heavy"\n      },\n      "epochs": 40,\n      "image_size": 224,\n      "learning_rate": 0.0001\n    },\n    "recipe_provenance": {\n      "augmentation": "tier[data_size=small\\u2192heavy]; invariance[domain_signal_missing]; schedule[small\\u2192taper_last_20pct]",\n      "epochs": "epochs_table[small,finetune]",\n      "image_size": "family_default=224",\n      "learning_rate": "lr_base[cnn,finetune]"\n    },\n    "recommended_epochs": 40,\n    "scratch_viable": false,\n    "strategy_ablation_group": "rank1_dinov3_strategy",\n    "strategy_ablation_variant": "partial_last2",\n    "train_csv": "/content/kaggle_data/plant-pathology-2020-fgvc7/train__jiaozi_labels.csv",\n    "train_norm_layers": true,\n    "tta": {\n      "enabled": true,\n      "num_augments": 2,\n      "transforms": [\n        "hflip"\n      ]\n    },\n    "unfreeze_last_n_blocks": 2,\n    "use_class_weights": true\n  },\n  "num_classes": 4,\n  "offline_smoke": false,\n  "optimizer": "adamw",\n  "params_M": 86.0,\n  "pretrained_hf_id": "facebook/dinov3-vitb16-pretrain-lvd1689m",\n  "pretrained_name": "DINOv3-B/16 / LVD-1689M",\n  "rank": 1,\n  "score": 1.0,\n  "scratch_viable": false,\n  "strategy_ablation_group": "rank1_dinov3_strategy",\n  "strategy_ablation_variant": "partial_last2",\n  "task_type": "classification",\n  "tasks": [\n    "Load DINOv3-B/16 / LVD-1689M from facebook/dinov3-vitb16-pretrain-lvd1689m (86M params, pretrained on LVD-1689M (self-supervised))",\n    "Partial finetune: update head, norm layers, and the last 2 backbone blocks",\n    "Use Classification Head as the output head",\n    "Use CrossEntropyLoss as the training loss",\n    "Use AdamW as the optimizer"\n  ],\n  "train_norm_layers": true,\n  "tta": false,\n  "unfreeze_last_n_blocks": 2,\n  "use_pretrained": true\n}')


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one experiment (smoke or real training).")
    parser.add_argument("--config", default="configs.json", help="JSON config path.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Training epochs (default: 1 for smoke, 10 for real).")
    parser.add_argument("--dataset", default=None,
                        help="Override dataset_id in config for real training.")
    args = parser.parse_args()

    set_seed(args.seed)
    config = load_config(args.config, DEFAULT_CONFIG)

    if args.dataset:
        config["dataset_id"] = args.dataset

    offline_smoke = as_bool(get_value(config, "offline_smoke", True), True)
    default_epochs = 1 if offline_smoke else as_int(get_value(config, "recommended_epochs", 10), 10)
    epochs = args.epochs if args.epochs is not None else default_epochs
    max_steps = 1 if offline_smoke else 0

    model, train_result = train_model(config, epochs=epochs, max_steps=max_steps)
    eval_result = evaluate(model, config)
    infer_result = predict(config=config, model=model)
    summary = {
        "status": "success",
        "config": compact_config_summary(config),
        "train": train_result,
        "evaluate": eval_result,
        "infer": infer_result,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

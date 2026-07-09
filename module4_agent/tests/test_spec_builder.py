from module4_agent.spec_builder import build_training_specs
import pytest


def test_build_training_specs_preserves_model_config_fields():
    candidates = [
        {
            "rank": 7,
            "score": "0.5",
            "model_config": {
                "task_type": "classification",
                "backbone": "efficientnet_b0",
                "pretrained_hf_id": "google/efficientnet-b0",
                "head": "classification_head",
                "loss": "cross_entropy_loss",
                "optimizer": "adamw",
                "finetune_strategy": "head_only",
                "freeze_backbone": False,
                "params_M": "5.3",
                "learning_rate": "0.0003",
                "augmentation": "stronger",
                "checkpoint": "google/efficientnet-b0",
            },
            "constraints": {"data_size": "small", "class_imbalance": True},
            "tasks": ["Freeze the backbone."],
        }
    ]

    spec = build_training_specs(candidates)[0]

    assert spec.rank == 7
    assert spec.score == 0.5
    assert spec.task_type == "classification"
    assert spec.backbone == "efficientnet_b0"
    assert spec.pretrained_hf_id == "google/efficientnet-b0"
    assert spec.loss == "cross_entropy_loss"
    assert spec.optimizer == "adamw"
    assert spec.finetune_strategy == "head_only"
    assert spec.freeze_backbone is True
    assert spec.params_M == 5.3
    assert spec.learning_rate == 3.0e-4
    assert spec.augmentation == "stronger"
    assert spec.data_size == "small"
    assert spec.class_imbalance is True
    assert spec.checkpoint == "google/efficientnet-b0"


def test_build_training_specs_preserves_model_config_tta():
    spec = build_training_specs(
        [
            {
                "model_config": {
                    "task_type": "classification",
                    "tta": True,
                }
            }
        ]
    )[0]

    assert spec.tta is True
    assert spec.to_config()["tta"] is True


def test_build_training_specs_reads_top_level_and_constraint_tta():
    top_level_spec = build_training_specs(
        [
            {
                "model_config": {"task_type": "classification"},
                "tta": "true",
            }
        ]
    )[0]
    constraint_spec = build_training_specs(
        [
            {
                "model_config": {"task_type": "classification"},
                "constraints": {"tta": 1},
            }
        ]
    )[0]

    assert top_level_spec.tta is True
    assert constraint_spec.tta is True


def test_training_spec_preserves_tta_recipe_and_fold_controls_together():
    spec = build_training_specs(
        [
            {
                "model_config": {
                    "task_type": "classification",
                    "tta": True,
                    "fold_file": "folds.json",
                    "fold_index": 2,
                    "export_preds_path": "val_preds.json",
                    "recipe": {
                        "augmentation": {
                            "tier": "medium",
                            "invariance": {"hflip": True},
                            "schedule": "taper_last_20pct",
                        }
                    },
                }
            }
        ]
    )[0]

    config = spec.to_config()

    assert config["tta"] is True
    assert config["model_config"]["tta"] is True
    assert config["model_config"]["fold_file"] == "folds.json"
    assert config["model_config"]["fold_index"] == 2
    assert config["model_config"]["export_preds_path"] == "val_preds.json"
    assert config["model_config"]["recipe"]["augmentation"]["schedule"] == "taper_last_20pct"


def test_build_training_specs_handles_older_candidate_shape():
    candidates = [
        {
            "rank": 1,
            "model_id": "hf-detr-small",
            "task_family": "detection",
            "pretrained_weights": "coco",
            "default_input_size": 128,
        }
    ]

    spec = build_training_specs(candidates)[0]

    assert spec.task_type == "object_detection"
    assert spec.backbone == "hf-detr-small"
    assert spec.head == "detection_head_anchor_free"
    assert spec.image_size == 128


def test_build_training_specs_supports_partial_finetune():
    specs = build_training_specs(
        [
            {
                "model_config": {
                    "task_type": "classification",
                    "backbone": "dinov3",
                    "finetune_strategy": "partial",
                    "freeze_backbone": True,
                    "unfreeze_last_n_blocks": "4",
                }
            }
        ]
    )
    spec = specs[0]

    assert spec.finetune_strategy == "partial"
    assert spec.freeze_backbone is False
    assert spec.unfreeze_last_n_blocks == 4
    assert spec.train_norm_layers is True
    assert {item.strategy_ablation_variant for item in specs} == {"partial_last4", "partial_last2"}
    assert {item.unfreeze_last_n_blocks for item in specs} == {2, 4}


def test_build_training_specs_extracts_structured_tasks():
    candidates = [
        {
            "format": "structured",
            "rank": 1,
            "tasks": [
                {
                    "action": "load_pretrained",
                    "hf_id": "ultralytics/assets",
                    "model_name": "YOLOv8-Nano",
                    "params_M": 3.2,
                    "finetune_base": "yolov8",
                },
                {
                    "action": "set_finetune_strategy",
                    "strategy": "full",
                    "freeze_backbone": False,
                    "scratch_viable": True,
                },
                {"action": "configure_head", "type": "detection_head_anchor_free"},
                {"action": "configure_loss", "type": "focal_loss"},
                {"action": "configure_optimizer", "type": "sgd_momentum"},
            ],
        }
    ]

    spec = build_training_specs(candidates)[0]

    assert spec.task_type == "object_detection"
    assert spec.backbone == "yolov8"
    assert spec.pretrained_hf_id == "ultralytics/assets"
    assert spec.loss == "focal_loss"
    assert spec.optimizer == "sgd_momentum"
    assert spec.finetune_strategy == "full"
    assert spec.freeze_backbone is False


def test_model_config_wins_when_tasks_conflict():
    candidates = [
        {
            "model_config": {
                "task_type": "classification",
                "backbone": "efficientnet_b0",
                "optimizer": "adamw",
            },
            "tasks": [
                {"action": "configure_optimizer", "type": "sgd_momentum"},
                "Use YOLO for detection.",
            ],
        }
    ]

    spec = build_training_specs(candidates)[0]

    assert spec.task_type == "classification"
    assert spec.backbone == "efficientnet_b0"
    assert spec.optimizer == "adamw"


def test_unknown_task_type_falls_back_safely():
    spec = build_training_specs([{"model_config": {"task_type": "not_real", "backbone": "custom"}}])[0]

    assert spec.task_type == "classification"
    assert spec.backbone == "custom"


def test_missing_model_config_and_non_dict_candidate_are_safe():
    specs = build_training_specs([{"rank": 1}, "bad-item"])

    assert len(specs) == 2
    assert specs[0].task_type == "classification"
    assert specs[1].task_type == "classification"


def test_empty_candidate_list_raises():
    with pytest.raises(ValueError):
        build_training_specs([])

from recommender.experiment_planner import generate_proposals


def test_planner_limits_experiments_and_changed_fields():
    baseline = {"augmentation": "basic", "learning_rate": 0.001, "recommended_epochs": 12}
    cards = [
        {
            "id": f"card_{index}",
            "strategy_name": f"Strategy {index}",
            "component": "augmentation",
            "priority": 1 - index / 10,
            "summary": "test",
            "experiment_template": {"augmentation": f"mode_{index}", "label_smoothing": 0.1},
        }
        for index in range(5)
    ]
    proposals = generate_proposals(baseline, cards, max_experiments=3, max_changed_variables=2)
    assert len(proposals) == 3
    assert all(len(item.changed_fields) <= 2 for item in proposals)
    assert all(item.config["recommended_epochs"] == 12 for item in proposals)


def test_planner_rejects_duplicate_unsupported_and_mixup_cutmix():
    baseline = {"augmentation": "basic"}
    cards = [
        {
            "id": "same",
            "strategy_name": "No change",
            "component": "augmentation",
            "experiment_template": {"augmentation": "basic"},
        },
        {
            "id": "unsafe",
            "strategy_name": "Unsafe",
            "component": "augmentation",
            "experiment_template": {"command": "rm -rf /"},
        },
        {
            "id": "both",
            "strategy_name": "Both",
            "component": "augmentation",
            "experiment_template": {"mixup_alpha": 0.2, "cutmix_alpha": 0.5},
        },
    ]
    assert generate_proposals(baseline, cards) == []


def test_planner_supports_medal_recipe_fields_and_prioritizes_resolution():
    baseline = {
        "image_size": 224,
        "finetune_strategy": "head_only",
        "unfreeze_last_n_blocks": 0,
        "backbone_learning_rate": 1.0e-4,
        "head_learning_rate": 1.0e-4,
    }
    cards = [
        {
            "id": "augmentation",
            "strategy_name": "RandAugment",
            "component": "augmentation",
            "priority": 1.0,
            "experiment_template": {"augmentation": "randaugment"},
        },
        {
            "id": "resolution",
            "strategy_name": "336 Resolution",
            "component": "resolution",
            "priority": 0.9,
            "experiment_template": {"image_size": 336},
        },
        {
            "id": "partial",
            "strategy_name": "Partial DINOv2",
            "component": "finetune",
            "priority": 0.95,
            "experiment_template": {
                "finetune_strategy": "partial",
                "unfreeze_last_n_blocks": 2,
            },
        },
        {
            "id": "rates",
            "strategy_name": "Discriminative LR",
            "component": "optimizer",
            "priority": 0.85,
            "experiment_template": {
                "backbone_learning_rate": 1.0e-5,
                "head_learning_rate": 3.0e-4,
            },
        },
    ]

    proposals = generate_proposals(baseline, cards, max_experiments=3)

    assert [proposal.strategy_card_ids[0] for proposal in proposals] == [
        "partial",
        "resolution",
        "rates",
    ]
    assert all(len(proposal.changed_fields) <= 2 for proposal in proposals)

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

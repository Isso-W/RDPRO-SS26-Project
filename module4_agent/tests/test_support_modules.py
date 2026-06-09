from module4_agent.prompts import CODER_PROMPT, REVIEWER_PROMPT
from module4_agent.smoke_harness import smoke_config, synthetic_batch


def test_smoke_harness_shapes_for_supported_tasks():
    x, y = synthetic_batch("classification", image_size=32, num_classes=4)
    assert tuple(x.shape) == (2, 3, 32, 32)
    assert tuple(y.shape) == (2,)

    x, mask = synthetic_batch("image_segmentation", image_size=16, num_classes=3)
    assert tuple(x.shape) == (2, 3, 16, 16)
    assert tuple(mask.shape) == (2, 16, 16)

    x, targets = synthetic_batch("object_detection", image_size=16, num_classes=3)
    assert tuple(x.shape) == (2, 3, 16, 16)
    assert "boxes" in targets[0]

    x, labels = synthetic_batch("feature_extraction", image_size=16)
    assert tuple(x.shape) == (2, 3, 16, 16)
    assert tuple(labels.shape) == (2,)


def test_smoke_config_and_prompt_placeholders_are_available():
    config = smoke_config("segmentation", image_size=64)

    assert config["task_type"] == "image_segmentation"
    assert config["image_size"] == 64
    assert "model_config" in CODER_PROMPT
    assert "smoke-test" in REVIEWER_PROMPT

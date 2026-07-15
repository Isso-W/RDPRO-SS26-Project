from __future__ import annotations

import importlib
import sys

import pytest

from module4_agent.code_generator import generate_files
from module4_agent.tests.test_code_generator import _specs


@pytest.fixture
def generated_train(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    generated = generate_files(_specs(), llm_provider="none")
    for name, content in generated.files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    module_names = ("train", "model", "model_utils", "smoke_data", "utils")
    for name in module_names:
        sys.modules.pop(name, None)
    train = importlib.import_module("train")
    yield train
    for name in module_names:
        sys.modules.pop(name, None)


def _op_names(transform):
    return [type(op).__name__ for op in transform.transforms]


def _recipe_config(
    tier: str = "medium",
    *,
    schedule: str = "constant",
    invariance: dict | None = None,
) -> dict:
    return {
        "task_type": "classification",
        "image_size": 32,
        "recipe": {
            "augmentation": {
                "tier": tier,
                "schedule": schedule,
                "invariance": invariance
                or {
                    "hflip": True,
                    "vflip": False,
                    "rot90": False,
                    "color": False,
                    "crop_scale_min": 0.75,
                    "randaugment": False,
                    "random_erasing": False,
                },
            }
        },
    }


def test_structured_tier_none_is_resize_only(generated_train):
    transform = generated_train._build_image_transform(
        _recipe_config(
            "none",
            invariance={
                "hflip": True,
                "vflip": True,
                "rot90": True,
                "color": True,
                "crop_scale_min": 0.5,
                "randaugment": True,
                "random_erasing": True,
            },
        ),
        "train",
    )

    assert _op_names(transform) == ["Resize", "ToTensor", "Normalize"]


def test_structured_augmentation_never_uses_randaugment(generated_train):
    transform = generated_train._build_image_transform(
        _recipe_config(
            "heavy",
            invariance={
                "hflip": False,
                "vflip": False,
                "rot90": False,
                "color": False,
                "crop_scale_min": 0.5,
                "randaugment": True,
                "random_erasing": True,
            },
        ),
        "train",
    )
    names = _op_names(transform)

    assert "RandAugment" not in names
    assert "RandomHorizontalFlip" not in names
    assert "RandomVerticalFlip" not in names
    assert "RandomRotation" not in names
    assert "ColorJitter" not in names


@pytest.mark.parametrize("alias", ["stronger", "stronger_v2"])
def test_legacy_stronger_aliases_use_strong_pipeline(generated_train, alias):
    strong = generated_train._build_image_transform(
        {"image_size": 32, "augmentation": "strong"},
        "train",
    )
    alias_transform = generated_train._build_image_transform(
        {"image_size": 32, "augmentation": alias},
        "train",
    )

    assert _op_names(alias_transform) == _op_names(strong)
    assert "RandomErasing" in _op_names(alias_transform)


def test_augmentation_helpers_prefer_recipe_and_read_schedule(generated_train):
    config = _recipe_config("medium", schedule="taper_last_20pct")
    config["augmentation"] = "strong"

    assert generated_train._augmentation_recipe(config) == config["recipe"]["augmentation"]
    assert generated_train._augmentation_schedule(config) == "taper_last_20pct"
    assert generated_train._augmentation_schedule({"augmentation": "strong"}) == ""


def _training_config(tmp_path, *, strategy: str) -> dict:
    config = _recipe_config("medium", schedule="taper_last_20pct")
    config.update(
        {
            "offline_smoke": False,
            "num_classes": 2,
            "batch_size": 2,
            "eval_batch_size": 2,
            "checkpoint_dir": str(tmp_path / f"checkpoints-{strategy}"),
            "finetune_strategy": strategy,
            "learning_rate": 1.0e-3,
            "scheduler": "none",
            "mixed_precision": False,
        }
    )
    return config


def _loader_spy(torch, calls):
    dataset = torch.utils.data.TensorDataset(
        torch.randn(4, 3, 8, 8),
        torch.tensor([0, 1, 0, 1]),
    )

    def build_loader(config, split="train", batch_size=32, deterministic=False):
        calls.append((split, deterministic))
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return build_loader


def test_taper_switches_full_finetune_to_deterministic_loader(
    generated_train,
    tmp_path,
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    calls: list[tuple[str, bool]] = []

    class TrainableModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Conv2d(3, 4, kernel_size=1)
            self.head = torch.nn.Linear(4, 2)

        def forward(self, x):
            features = self.backbone(x).mean(dim=(2, 3))
            return self.head(features)

    monkeypatch.setattr(generated_train, "build_model", lambda _config: TrainableModel())
    monkeypatch.setattr(generated_train, "_build_dataloader", _loader_spy(torch, calls))

    generated_train.train_model(
        _training_config(tmp_path, strategy="full"),
        epochs=5,
        max_steps=0,
    )

    assert [deterministic for split, deterministic in calls if split == "train"] == [False, True]


def test_taper_does_not_add_a_second_frozen_cache_loader(
    generated_train,
    tmp_path,
    monkeypatch,
):
    torch = pytest.importorskip("torch")
    calls: list[tuple[str, bool]] = []

    class FrozenModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Conv2d(3, 4, kernel_size=1)
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            self.head = torch.nn.Linear(4, 2)

        def forward(self, x):
            features = self.backbone(x).mean(dim=(2, 3))
            return self.head(features)

    monkeypatch.setattr(generated_train, "build_model", lambda _config: FrozenModel())
    monkeypatch.setattr(generated_train, "_build_dataloader", _loader_spy(torch, calls))

    generated_train.train_model(
        _training_config(tmp_path, strategy="head_only"),
        epochs=5,
        max_steps=0,
    )

    assert [deterministic for split, deterministic in calls if split == "train"] == [False, True]

import json

from module4_agent.code_generator import REQUIRED_GENERATED_FILES, generate_files
from module4_agent.spec_builder import build_training_specs


def _specs():
    return build_training_specs(
        [
            {
                "rank": 1,
                "model_config": {
                    "task_type": "classification",
                    "backbone": "efficientnet_b0",
                    "loss": "cross_entropy_loss",
                    "optimizer": "adamw",
                },
            },
            {
                "rank": 2,
                "model_config": {
                    "task_type": "feature_extraction",
                    "backbone": "dinov2_vits14",
                    "loss": "feature_mse_loss",
                    "optimizer": "adamw",
                },
            },
        ]
    )


def test_generate_files_contains_required_files_and_compiles():
    generated = generate_files(_specs(), llm_provider="none")

    assert set(REQUIRED_GENERATED_FILES).issubset(generated.files)
    assert "configs.json" in generated.files
    assert "generation_info.json" in generated.files
    assert "utils.py" in generated.files
    assert "imagenet_prior.py" in generated.files
    for filename, content in generated.files.items():
        if filename.endswith(".py"):
            compile(content, filename, "exec")


def test_generation_info_defaults_to_template(monkeypatch):
    monkeypatch.setenv("M4_LLM_PROVIDER", "none")

    generated = generate_files(_specs(), llm_provider="none")
    info = json.loads(generated.files["generation_info.json"])

    assert info["llm_provider"] == "none"
    assert info["llm_model"] == ""
    assert info["llm_attempted"] is False
    assert info["model_py_source"] == "template"
    assert info["llm_used"] is False
    assert info["template_fallback"] is True
    assert info["fallback_reason"] == ""


def test_invalid_llm_output_falls_back_to_compiling_template(monkeypatch):
    monkeypatch.setattr(
        "module4_agent.code_generator.generate_model_py",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "module4_agent.code_generator.get_last_generation_error",
        lambda: "provider returned an HTML page",
    )

    generated = generate_files(_specs(), llm_provider="openai")
    info = json.loads(generated.files["generation_info.json"])

    compile(generated.files["model.py"], "model.py", "exec")
    assert info["llm_attempted"] is True
    assert info["llm_used"] is False
    assert info["template_fallback"] is True
    assert info["fallback_reason"] == "provider returned an HTML page"


def test_run_experiments_embeds_and_sweeps_all_candidates():
    generated = generate_files(_specs(), llm_provider="none")
    content = generated.files["run_experiments.py"]

    assert "DEFAULT_CONFIGS" in content
    assert '"rank": 1' in content
    assert '"rank": 2' in content
    assert "from typing import Any" in content
    assert "for index, config in enumerate(configs" in content
    assert "model, train_result = train_model" in content


def test_run_uses_trained_model_for_evaluation():
    generated = generate_files(_specs(), llm_provider="none")

    assert "hidden[:, 0]" in generated.files["model_utils.py"]
    assert "patch_tokens.mean(dim=1)" in generated.files["model_utils.py"]
    assert "model, train_result = train_model" in generated.files["run.py"]
    assert "eval_result = evaluate(model, config)" in generated.files["run.py"]
    assert "def _build_dataloader" in generated.files["train.py"]
    assert "def _build_local_dataloader" in generated.files["train.py"]
    assert "train_csv" in generated.files["train.py"]
    assert "ImageFolder" in generated.files["train.py"]
    assert "torch.save" in generated.files["train.py"]
    assert "cohen_kappa_score" in generated.files["evaluate.py"]
    assert "roc_auc_score" in generated.files["evaluate.py"]
    assert "log_loss" in generated.files["evaluate.py"]
    assert "RandomResizedCrop" in generated.files["train.py"]
    assert "use_class_weights" in generated.files["train.py"]
    assert "label_smoothing" in generated.files["train.py"]
    assert "GradScaler" in generated.files["train.py"]
    assert "CosineAnnealingLR" in generated.files["train.py"]
    assert "SequentialLR" in generated.files["train.py"]
    assert "RandomVerticalFlip" not in generated.files["train.py"]
    assert "transforms.Resize(resize_size)" in generated.files["train.py"]
    assert "last_checkpoint.pt" in generated.files["train.py"]
    assert "early_stopping_patience" in generated.files["train.py"]
    assert "transforms.RandAugment" in generated.files["train.py"]
    assert "_apply_batch_regularization" in generated.files["train.py"]
    assert "MixUp and CutMix cannot be enabled" in generated.files["train.py"]
    assert "tta_horizontal_flip" in generated.files["evaluate.py"]
    assert "validation_probabilities.npz" in generated.files["evaluate.py"]
    assert 'augmentation in {"none", "off", "deterministic"}' in generated.files["train.py"]


def test_frozen_backbone_uses_cached_feature_path(tmp_path, monkeypatch):
    """A frozen backbone should trigger extract-once + train-head-on-cache."""
    torch = __import__("pytest").importorskip("torch")
    import importlib
    import os
    import sys

    import torch.nn as nn

    # Importing torch's deps in-process can mutate os.environ (e.g. KMP_DUPLICATE_LIB_OK);
    # snapshot and restore so we don't leak into later tests.
    env_snapshot = dict(os.environ)

    generated = generate_files(_specs(), llm_provider="none")
    for name, content in generated.files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    for mod in ("train", "model", "smoke_data", "utils"):
        sys.modules.pop(mod, None)
    train = importlib.import_module("train")

    class _FrozenBackbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            for parameter in self.parameters():
                parameter.requires_grad = False

        def forward(self, x):
            return self.conv(x)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _FrozenBackbone()
            self.head = nn.Linear(8, 3)

        def forward(self, x):
            feats = torch.nn.functional.adaptive_avg_pool2d(self.backbone(x), 1).flatten(1)
            return self.head(feats)

    def _fake_dataloader(config, split="train", batch_size=32, deterministic=False):
        count = 40 if split == "train" else 12
        dataset = torch.utils.data.TensorDataset(
            torch.randn(count, 3, 16, 16), torch.randint(0, 3, (count,))
        )
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    monkeypatch.setattr(train, "build_model", lambda config: _Model())
    monkeypatch.setattr(train, "_build_dataloader", _fake_dataloader)

    ckpt = tmp_path / "ckpt"
    config = {
        "task_type": "classification", "offline_smoke": False, "num_classes": 3,
        "backbone": "resnet18", "finetune_strategy": "head_only",
        "augmentation": "none",
        "learning_rate": 0.01, "batch_size": 8, "image_size": 16,
        "checkpoint_dir": str(ckpt),
    }
    try:
        _model, summary = train.train_model(config, epochs=3, max_steps=0, save_dir=str(ckpt))

        assert (ckpt / "best_model.pt").exists()
        assert list((ckpt / "feature_cache").glob("feat_train_*.pt"))
        assert list((ckpt / "feature_cache").glob("feat_val_*.pt"))
        blob = torch.load(ckpt / "best_model.pt", map_location="cpu", weights_only=False)
        assert blob.get("feature_cached") is True
        assert len(summary["validation_history"]) >= 1
    finally:
        for mod in ("train", "model", "smoke_data", "utils"):
            sys.modules.pop(mod, None)
        os.environ.clear()
        os.environ.update(env_snapshot)


def test_feedback_is_embedded_into_generated_readme():
    generated = generate_files(_specs(), feedback="Smoke test failed.", llm_provider="none")

    assert "Previous Review Notes" in generated.files["README_generated.md"]
    assert "Smoke test failed." in generated.files["README_generated.md"]


def test_generated_readme_documents_runtime_files():
    generated = generate_files(_specs(), llm_provider="none")
    readme = generated.files["README_generated.md"]

    assert "configs.json" in readme
    assert "generation_info.json" in readme
    assert "utils.py" in readme
    assert "model_utils.py" in readme
    assert "smoke_data.py" in readme
    assert "M4_LLM_PROVIDER=qwen" in readme
    assert "Smoke vs Real Training" in readme
    assert "Current Limitations" in readme
    assert "checkpoint" in readme.lower()


def test_generated_partial_finetune_unfreezes_last_blocks_and_uses_two_lrs(
    tmp_path, monkeypatch
):
    torch = __import__("pytest").importorskip("torch")
    import importlib
    import sys

    import torch.nn as nn

    generated = generate_files(_specs(), llm_provider="none")
    for name, content in generated.files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    for mod in ("model_utils", "train", "model", "smoke_data", "utils"):
        sys.modules.pop(mod, None)


def test_generated_evaluate_saves_validation_probability_artifact(tmp_path, monkeypatch):
    torch = __import__("pytest").importorskip("torch")
    import importlib
    import sys

    import torch.nn as nn

    generated = generate_files(_specs(), llm_provider="none")
    for name, content in generated.files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    for mod in ("evaluate", "smoke_data", "utils"):
        sys.modules.pop(mod, None)
    evaluate = importlib.import_module("evaluate")

    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 4 * 4, 3))
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.randn(8, 3, 4, 4),
            torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]),
        ),
        batch_size=4,
    )
    result = evaluate._eval_on_dataloader(
        model,
        loader,
        {
            "task_type": "classification",
            "num_classes": 3,
            "evaluation_metric": "log_loss",
            "checkpoint_dir": str(tmp_path / "checkpoints"),
        },
    )

    artifact = tmp_path / "checkpoints" / "validation_probabilities.npz"
    assert artifact.exists()
    assert result["validation_artifact"] == str(artifact)

    for mod in ("evaluate", "smoke_data", "utils"):
        sys.modules.pop(mod, None)
    model_utils = importlib.import_module("model_utils")
    train = importlib.import_module("train")

    class _Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])

    class _Core(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = _Encoder()
            self.layernorm = nn.LayerNorm(4)

    class _Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Core()

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _Backbone()
            self.head = nn.Linear(4, 3)

    model = _Model()
    config = {
        "finetune_strategy": "partial",
        "unfreeze_last_n_blocks": 2,
        "learning_rate": 1.0e-4,
        "backbone_learning_rate": 1.0e-5,
        "head_learning_rate": 3.0e-4,
        "optimizer": "adamw",
    }
    model_utils.apply_freeze(model, config)

    blocks = list(model.backbone.model.encoder.layer)
    assert all(not parameter.requires_grad for block in blocks[:2] for parameter in block.parameters())
    assert all(parameter.requires_grad for block in blocks[-2:] for parameter in block.parameters())
    assert all(parameter.requires_grad for parameter in model.backbone.model.layernorm.parameters())
    assert all(parameter.requires_grad for parameter in model.head.parameters())

    optimizer = train._build_optimizer(model, config)
    assert sorted(group["lr"] for group in optimizer.param_groups) == [1.0e-5, 3.0e-4]

    for mod in ("model_utils", "train", "model", "smoke_data", "utils"):
        sys.modules.pop(mod, None)


def test_generated_stratified_folds_are_disjoint_and_cover_all_samples(
    tmp_path, monkeypatch
):
    import importlib
    import sys

    generated = generate_files(_specs(), llm_provider="none")
    for name, content in generated.files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    for mod in ("train", "model", "smoke_data", "utils"):
        sys.modules.pop(mod, None)


def test_generated_imagenet_prior_maps_breeds_and_calibrates_blend(
    tmp_path, monkeypatch
):
    import importlib
    import sys

    import numpy as np
    import pytest

    generated = generate_files(_specs(), llm_provider="none")
    for name, content in generated.files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("imagenet_prior", None)
    prior = importlib.import_module("imagenet_prior")

    labels = [
        "black-and-tan_coonhound",
        "brabancon_griffon",
        "cardigan",
        "german_short-haired_pointer",
        "staffordshire_bullterrier",
        "walker_hound",
    ]
    categories = [
        "other",
        "black-and-tan coonhound",
        "Brabancon griffon",
        "Cardigan",
        "German short-haired pointer",
        "Staffordshire bullterrier",
        "Walker hound",
        "cardigan",
    ]
    assert prior.build_label_projection(labels, categories) == [1, 2, 3, 4, 5, 6]

    learned = np.asarray([[0.55, 0.45], [0.45, 0.55]], dtype=np.float32)
    imagenet = np.asarray([[0.95, 0.05], [0.05, 0.95]], dtype=np.float32)
    alpha, combined, loss = prior.calibrate_probability_blend(
        learned,
        imagenet,
        np.asarray([0, 1]),
        step=0.05,
    )
    assert alpha == pytest.approx(1.0)
    assert loss < prior.multiclass_log_loss([0, 1], learned)
    assert combined.sum(axis=1).tolist() == pytest.approx([1.0, 1.0])

    sys.modules.pop("imagenet_prior", None)
    train = importlib.import_module("train")

    labels = [label for label in range(3) for _ in range(9)]
    validation_sets = []
    for fold_index in range(3):
        train_indices, validation_indices = train._split_indices(
            labels,
            validation_fraction=0.2,
            seed=42,
            fold_count=3,
            fold_index=fold_index,
        )
        assert set(train_indices).isdisjoint(validation_indices)
        assert set(train_indices) | set(validation_indices) == set(range(len(labels)))
        validation_sets.append(set(validation_indices))

    assert validation_sets[0].isdisjoint(validation_sets[1])
    assert validation_sets[0].isdisjoint(validation_sets[2])
    assert validation_sets[1].isdisjoint(validation_sets[2])
    assert set().union(*validation_sets) == set(range(len(labels)))

    for mod in ("train", "model", "smoke_data", "utils"):
        sys.modules.pop(mod, None)

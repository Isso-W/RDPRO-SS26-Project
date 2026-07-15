import json

import pytest

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


def _recipe_specs():
    return build_training_specs(
        [
            {
                "rank": 1,
                "model_config": {
                    "task_type": "classification",
                    "backbone": "efficientnet_b0",
                    "loss": "cross_entropy_loss",
                    "optimizer": "adamw",
                    "recipe": {
                        "image_size": 384,
                        "learning_rate": 1.0e-4,
                        "epochs": 20,
                        "early_stopping_patience": 5,
                        "augmentation": {
                            "tier": "medium",
                            "invariance": {
                                "hflip": True,
                                "vflip": False,
                                "rot90": False,
                                "color": False,
                                "crop_scale_min": 0.75,
                                "randaugment": False,
                                "random_erasing": False,
                            },
                            "schedule": "taper_last_20pct",
                        },
                    },
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
    assert "_select_best_index" in content
    assert "best_config.json" in content


def test_run_uses_trained_model_for_evaluation():
    generated = generate_files(_specs(), llm_provider="none")

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
    assert "last_checkpoint.pt" in generated.files["train.py"]
    assert "early_stopping_patience" in generated.files["train.py"]
    assert "def _log_model_load_info" in generated.files["train.py"]
    assert "actual_model=" in generated.files["train.py"]
    assert "backbone_trainable=" in generated.files["train.py"]
    assert '"model_load_info": model_load_info' in generated.files["train.py"]


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


def test_transformer_backbone_uses_grouped_lr(tmp_path, monkeypatch):
    """Full-finetuning a transformer backbone splits LR: low backbone, full head."""
    pytest = __import__("pytest")
    pytest.importorskip("torch")
    import importlib
    import os
    import sys

    import torch.nn as nn

    env_snapshot = dict(os.environ)
    generated = generate_files(_specs(), llm_provider="none")
    for name, content in generated.files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    for mod in ("train", "model", "smoke_data", "utils"):
        sys.modules.pop(mod, None)
    train = importlib.import_module("train")

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Linear(8, 8)
            self.head = nn.Linear(8, 3)

    try:
        # transformer + full finetune -> two groups (backbone 1e-5, head 1e-3)
        opt = train._build_optimizer(_M(), {"backbone": "dinov2_base", "learning_rate": 1e-3})
        lrs = sorted(g["lr"] for g in opt.param_groups)
        assert len(opt.param_groups) == 2
        assert abs(lrs[0] - 1e-5) < 1e-9 and abs(lrs[1] - 1e-3) < 1e-9

        # CNN -> single group at full lr (no regression)
        opt_cnn = train._build_optimizer(_M(), {"backbone": "resnet50", "learning_rate": 1e-3})
        assert len(opt_cnn.param_groups) == 1

        # frozen transformer backbone -> single group (head only)
        m = _M()
        for p in m.backbone.parameters():
            p.requires_grad = False
        opt_frozen = train._build_optimizer(m, {"backbone": "dinov2_base", "learning_rate": 1e-3})
        assert len(opt_frozen.param_groups) == 1
    finally:
        for mod in ("train", "model", "smoke_data", "utils"):
            sys.modules.pop(mod, None)
        os.environ.clear()
        os.environ.update(env_snapshot)


def test_partial_finetune_unfreezes_last_blocks(tmp_path, monkeypatch):
    """partial finetune freezes early backbone blocks and trains the tail blocks."""
    pytest = __import__("pytest")
    torch = pytest.importorskip("torch")
    import importlib
    import os
    import sys

    import torch.nn as nn

    env_snapshot = dict(os.environ)
    generated = generate_files(_specs(), llm_provider="none")
    for name, content in generated.files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    for mod in ("model_utils", "train", "model", "smoke_data", "utils"):
        sys.modules.pop(mod, None)
    model_utils = importlib.import_module("model_utils")
    model_module = importlib.import_module("model")
    train = importlib.import_module("train")

    class _Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(3)])
            self.norm = nn.LayerNorm(4)

        def forward(self, x):
            for block in self.blocks:
                x = block(x)
            return self.norm(x)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = _Backbone()
            self.head = nn.Linear(4, 2)

    try:
        model = _Model()
        model_utils.apply_freeze(
            model,
            {
                "finetune_strategy": "partial",
                "unfreeze_last_n_blocks": 1,
                "train_norm_layers": False,
            },
        )

        assert all(not p.requires_grad for p in model.backbone.blocks[0].parameters())
        assert all(not p.requires_grad for p in model.backbone.blocks[1].parameters())
        assert all(p.requires_grad for p in model.backbone.blocks[2].parameters())
        assert all(not p.requires_grad for p in model.backbone.norm.parameters())
        assert all(p.requires_grad for p in model.head.parameters())
        assert train._backbone_is_frozen(model) is False

        model_with_norm = _Model()
        model_module._apply_finetune_strategy(
            model_with_norm,
            {
                "finetune_strategy": "partial",
                "unfreeze_last_n_blocks": 0,
                "train_norm_layers": True,
            },
        )
        assert all(not p.requires_grad for block in model_with_norm.backbone.blocks for p in block.parameters())
        assert all(p.requires_grad for p in model_with_norm.backbone.norm.parameters())
        assert all(p.requires_grad for p in model_with_norm.head.parameters())

        opt = train._build_optimizer(model, {"backbone": "dinov3", "learning_rate": 1e-3})
        assert len(opt.param_groups) == 2
        lrs = sorted(group["lr"] for group in opt.param_groups)
        assert abs(lrs[0] - 1e-5) < 1e-9 and abs(lrs[1] - 1e-3) < 1e-9
        assert getattr(model, "_unfreeze_last_n_blocks") == 1
        assert getattr(model, "_partial_unfrozen_params") > 0
    finally:
        for mod in ("model_utils", "train", "model", "smoke_data", "utils"):
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


def test_generated_utils_falls_back_to_recipe_values(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import importlib
    import sys

    generated = generate_files(_recipe_specs(), llm_provider="none")
    (tmp_path / "utils.py").write_text(generated.files["utils.py"], encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("utils", None)
    try:
        utils = importlib.import_module("utils")

        config = {
            "recipe": {
                "image_size": 384,
                "learning_rate": 1.0e-4,
                "epochs": 20,
                "early_stopping_patience": 5,
            }
        }
        assert utils.get_value(config, "image_size", 224) == 384
        assert utils.get_value(config, "learning_rate", 1.0e-3) == 1.0e-4
        assert utils.get_recipe_value(config, "epochs", 10) == 20
        assert utils.get_value(config, "early_stopping_patience", 0) == 5
        assert utils.get_value({**config, "image_size": 128}, "image_size", 224) == 128
    finally:
        sys.modules.pop("utils", None)


def test_recipe_dict_augmentation_builds_train_transform(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from PIL import Image
    import importlib
    import os
    import sys

    env_snapshot = dict(os.environ)
    generated = generate_files(_recipe_specs(), llm_provider="none")
    for name, content in generated.files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    for mod in ("train", "model", "smoke_data", "utils"):
        sys.modules.pop(mod, None)
    try:
        train = importlib.import_module("train")
        config = json.loads(generated.files["configs.json"])[0]
        transform = train._build_image_transform(config, "train")
        tensor = transform(Image.new("RGB", (512, 512), "white"))

        assert tuple(tensor.shape[-2:]) == (384, 384)
    finally:
        for mod in ("train", "model", "smoke_data", "utils"):
            sys.modules.pop(mod, None)
        os.environ.clear()
        os.environ.update(env_snapshot)


def test_legacy_string_augmentation_still_builds_train_transform(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from PIL import Image
    import importlib
    import os
    import sys

    env_snapshot = dict(os.environ)
    generated = generate_files(_specs(), llm_provider="none")
    for name, content in generated.files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    for mod in ("train", "model", "smoke_data", "utils"):
        sys.modules.pop(mod, None)
    try:
        train = importlib.import_module("train")
        transform = train._build_image_transform(
            {"image_size": 128, "augmentation": "strong"},
            "train",
        )
        tensor = transform(Image.new("RGB", (256, 256), "white"))

        assert tuple(tensor.shape[-2:]) == (128, 128)
    finally:
        for mod in ("train", "model", "smoke_data", "utils"):
            sys.modules.pop(mod, None)
        os.environ.clear()
        os.environ.update(env_snapshot)


def test_backbone_fallback_defaults_to_enabled_and_records_actual_source(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import importlib
    import sys

    generated = generate_files(_specs(), llm_provider="none")
    for name in ("utils.py", "model_utils.py"):
        (tmp_path / name).write_text(generated.files[name], encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    for mod in ("model_utils", "utils"):
        sys.modules.pop(mod, None)
    try:
        model_utils = importlib.import_module("model_utils")
        monkeypatch.setattr(model_utils, "_try_huggingface", lambda *_args: None)
        monkeypatch.setattr(model_utils, "_try_torchvision", lambda *_args, **_kwargs: None)

        backbone, channels = model_utils.load_backbone(
            {
                "backbone": "dinov3",
                "use_pretrained": True,
                "pretrained_hf_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
            }
        )
        class _Model(torch.nn.Module):
            def __init__(self, loaded_backbone):
                super().__init__()
                self.backbone = loaded_backbone
                self.head = torch.nn.Linear(channels, 2)

        model = _Model(backbone)
        info = model_utils.backbone_load_info(model, {"backbone": "dinov3"})

        assert channels == backbone.out_channels
        assert info["source"] == "tiny_fallback"
        assert info["actual_model"] == "TinyBackbone"
        assert info["fallback_reason"]
        assert info["total_params"] > 0
        assert info["trainable_params"] == info["total_params"]
    finally:
        for mod in ("model_utils", "utils"):
            sys.modules.pop(mod, None)


def test_explicitly_disabling_backbone_fallback_fails_loudly(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import importlib
    import sys

    generated = generate_files(_specs(), llm_provider="none")
    for name in ("utils.py", "model_utils.py"):
        (tmp_path / name).write_text(generated.files[name], encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    for mod in ("model_utils", "utils"):
        sys.modules.pop(mod, None)
    try:
        model_utils = importlib.import_module("model_utils")
        monkeypatch.setattr(model_utils, "_try_huggingface", lambda *_args: None)
        monkeypatch.setattr(model_utils, "_try_torchvision", lambda *_args, **_kwargs: None)

        with pytest.raises(RuntimeError, match="allow_backbone_fallback=true"):
            model_utils.load_backbone(
                {
                    "backbone": "dinov3",
                    "use_pretrained": True,
                    "pretrained_hf_id": "facebook/dinov3-vits16-pretrain-lvd1689m",
                    "allow_backbone_fallback": False,
                }
            )
    finally:
        for mod in ("model_utils", "utils"):
            sys.modules.pop(mod, None)


@pytest.mark.parametrize(
    "config",
    [
        {
            "backbone": "unknown_scratch_model",
            "use_pretrained": False,
            "allow_backbone_fallback": False,
        },
        {
            "backbone": "unknown_smoke_model",
            "use_pretrained": True,
            "pretrained_hf_id": "example/unavailable-model",
            "offline_smoke": True,
            "allow_backbone_fallback": False,
        },
    ],
)
def test_strict_fallback_does_not_break_scratch_or_offline_smoke(
    tmp_path,
    monkeypatch,
    config,
):
    pytest.importorskip("torch")
    import importlib
    import sys

    generated = generate_files(_specs(), llm_provider="none")
    for name in ("utils.py", "model_utils.py"):
        (tmp_path / name).write_text(generated.files[name], encoding="utf-8")

    monkeypatch.syspath_prepend(str(tmp_path))
    for mod in ("model_utils", "utils"):
        sys.modules.pop(mod, None)
    try:
        model_utils = importlib.import_module("model_utils")
        monkeypatch.setattr(model_utils, "_try_huggingface", lambda *_args: None)
        monkeypatch.setattr(model_utils, "_try_torchvision", lambda *_args, **_kwargs: None)

        backbone, channels = model_utils.load_backbone(config)

        assert type(backbone).__name__ == "TinyBackbone"
        assert channels == backbone.out_channels
    finally:
        for mod in ("model_utils", "utils"):
            sys.modules.pop(mod, None)

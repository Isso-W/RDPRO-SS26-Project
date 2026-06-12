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

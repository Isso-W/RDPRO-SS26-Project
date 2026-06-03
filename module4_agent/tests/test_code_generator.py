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
    generated = generate_files(_specs())

    assert set(REQUIRED_GENERATED_FILES).issubset(generated.files)
    assert "configs.json" in generated.files
    for filename, content in generated.files.items():
        if filename.endswith(".py"):
            compile(content, filename, "exec")


def test_run_experiments_embeds_and_sweeps_all_candidates():
    generated = generate_files(_specs())
    content = generated.files["run_experiments.py"]

    assert "DEFAULT_CONFIGS" in content
    assert '"rank": 1' in content
    assert '"rank": 2' in content
    assert "for index, config in enumerate(configs" in content


def test_generated_readme_documents_runtime_files():
    generated = generate_files(_specs())
    readme = generated.files["README_generated.md"]

    assert "configs.json" in readme
    assert "smoke_data.py" in readme
    assert "module4_summary.json" in readme
    assert "best_config.json" in readme
    assert "Smoke vs Real Training" in readme
    assert "Current Limitations" in readme

import json

from module4_agent.ablation import diff_controlled_fields, generate_ablation_variants
from module4_agent.code_generator import generate_files
from module4_agent.executor import write_generated_files
from module4_agent.experiment_loop import run_refinement_loop
from module4_agent.refinement import proxy_evaluate
from module4_agent.reviewer import review_generated
from module4_agent.schemas import SmokeResult
from module4_agent.spec_builder import build_training_specs


def _classification_specs():
    return build_training_specs(
        [
            {
                "rank": 1,
                "score": 0.9,
                "constraints": {"data_size": "small", "class_imbalance": True},
                "model_config": {
                    "task_type": "classification",
                    "backbone": "efficientnet_b0",
                    "pretrained_hf_id": "google/efficientnet-b0",
                    "loss": "cross_entropy_loss",
                    "optimizer": "adamw",
                    "finetune_strategy": "head_only",
                    "freeze_backbone": True,
                    "learning_rate": 1.0e-3,
                    "image_size": 64,
                },
                "alternatives": [{"backbone": "mobilenet_v3_small"}],
            }
        ]
    )


def test_proxy_baseline_result_generation_is_task_specific():
    spec = _classification_specs()[0]

    result = proxy_evaluate(spec, experiment_id="baseline", stage="baseline")

    assert result.stage == "baseline"
    assert result.metric_name == "accuracy"
    assert result.status == "success"
    assert result.config_summary["task_type"] == "classification"


def test_ablation_variants_modify_one_component_only():
    spec = _classification_specs()[0]

    variants = generate_ablation_variants(spec)

    assert variants
    components = {variant.modified_component for variant in variants}
    assert {"optimizer", "learning_rate", "augmentation", "finetune_strategy", "loss", "backbone"}.issubset(
        components
    )
    augmentation_values = {
        variant.modified_value for variant in variants if variant.modified_component == "augmentation"
    }
    assert {"stronger", "none"}.issubset(augmentation_values)
    for variant in variants:
        changes = diff_controlled_fields(spec, variant.training_spec)
        assert len(changes) == 1
        assert variant.modified_component in changes


def test_proxy_evaluation_is_stable_for_same_spec_and_seed():
    spec = _classification_specs()[0]

    first = proxy_evaluate(spec, experiment_id="a", stage="baseline", seed=99)
    second = proxy_evaluate(spec, experiment_id="b", stage="baseline", seed=99)

    assert first.metric_value == second.metric_value


def test_refinement_loop_stops_by_max_iterations(tmp_path):
    summary = run_refinement_loop(
        _classification_specs(),
        tmp_path,
        max_iters=1,
        improvement_threshold=1.0,
    )

    assert summary.stopped_reason == "max_iterations_reached"
    assert summary.iterations == 1
    assert summary.ablation_results


def test_refinement_loop_stops_by_improvement_threshold(tmp_path):
    summary = run_refinement_loop(
        _classification_specs(),
        tmp_path,
        max_iters=3,
        improvement_threshold=0.001,
    )

    assert summary.stopped_reason == "improvement_threshold_reached"
    assert summary.improvement >= 0.001


def test_experiment_logging_files_and_leaderboard_sorting(tmp_path):
    summary = run_refinement_loop(_classification_specs(), tmp_path, max_iters=1, improvement_threshold=1.0)

    experiments = tmp_path / "experiments.jsonl"
    leaderboard = tmp_path / "leaderboard.json"
    refinement_summary = tmp_path / "refinement_summary.json"
    best_config = tmp_path / "best_config.json"
    assert experiments.exists()
    assert leaderboard.exists()
    assert refinement_summary.exists()
    assert best_config.exists()
    assert json.loads(refinement_summary.read_text(encoding="utf-8"))["best_result"]["experiment_id"] == (
        summary.best_result.experiment_id
    )
    best_config_data = json.loads(best_config.read_text(encoding="utf-8"))
    assert best_config_data["_module4_refinement"]["experiment_id"] == summary.best_result.experiment_id

    leaderboard_data = json.loads(leaderboard.read_text(encoding="utf-8"))
    scores = [row["metric_value"] for row in leaderboard_data["rows"]]
    assert scores == sorted(scores, reverse=True)


def test_reviewer_accepts_refinement_artifacts(tmp_path):
    specs = _classification_specs()
    generated = generate_files(specs)
    write_generated_files(generated, tmp_path)
    run_refinement_loop(specs, tmp_path, max_iters=1, improvement_threshold=1.0)

    review = review_generated(
        generated,
        specs,
        smoke_result=SmokeResult(success=True, skipped=True),
        output_dir=tmp_path,
        refinement_enabled=True,
    )

    assert review.is_approved, review.feedback


def test_reviewer_rejects_missing_refinement_artifacts(tmp_path):
    specs = _classification_specs()
    generated = generate_files(specs)
    write_generated_files(generated, tmp_path)

    review = review_generated(
        generated,
        specs,
        smoke_result=SmokeResult(success=True, skipped=True),
        output_dir=tmp_path,
        refinement_enabled=True,
    )

    assert not review.is_approved
    assert any("Missing refinement artifacts" in error for error in review.errors)

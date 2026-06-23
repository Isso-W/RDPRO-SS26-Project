from pathlib import Path
import json
import sys
import time

from module4_agent.code_generator import generate_files
from module4_agent.executor import (
    SUBPROCESS_ENV_DEFAULTS,
    _run_python_script_inprocess,
    run_command,
    subprocess_env,
    write_generated_files,
)
from module4_agent.reviewer import review_generated
from module4_agent.schemas import CommandResult, GeneratedFiles, SmokeResult
from module4_agent.spec_builder import build_training_specs
from module4_agent.workflow import run_workflow


def test_workflow_runs_end_to_end_from_sample(tmp_path):
    sample = Path(__file__).resolve().parents[1] / "examples" / "sample_m3_output.json"
    output = tmp_path / "generated"

    result = run_workflow(sample, output, max_iter=1, timeout=60)

    assert result.is_approved, result.review_result.feedback
    assert (output / "model.py").exists()
    assert (output / "smoke_data.py").exists()
    assert (output / "run_experiments.py").exists()
    assert (output / "configs.json").exists()
    assert (output / "module4_summary.json").exists()
    assert len(result.specs) == 3
    assert result.smoke_result.success
    summary = json.loads((output / "module4_summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "approved"
    assert summary["iteration_history"][0]["is_approved"] is True


def test_workflow_supports_no_smoke(tmp_path):
    sample = Path(__file__).resolve().parents[1] / "examples" / "sample_m3_output.json"
    output = tmp_path / "generated"

    result = run_workflow(sample, output, max_iter=1, timeout=60, skip_smoke=True)

    assert result.is_approved
    assert result.smoke_result.skipped is True
    assert "static-only" in result.review_result.warnings[0]


def test_workflow_runs_refinement_after_review_passes(tmp_path):
    sample = Path(__file__).resolve().parents[1] / "examples" / "sample_m3_output.json"
    output = tmp_path / "generated"

    result = run_workflow(
        sample,
        output,
        max_iter=1,
        timeout=60,
        skip_smoke=True,
        run_refinement=True,
        max_refinement_iters=1,
        improvement_threshold=1.0,
    )

    assert result.is_approved, result.review_result.feedback
    assert result.refinement_summary is not None
    assert (output / "experiments.jsonl").exists()
    assert (output / "leaderboard.json").exists()
    assert (output / "refinement_summary.json").exists()
    assert (output / "best_config.json").exists()
    summary = json.loads((output / "module4_summary.json").read_text(encoding="utf-8"))
    assert summary["refinement_summary"]["stopped_reason"] == "max_iterations_reached"
    assert "best_config.json" in summary["generated_files"]


def test_reviewer_rejects_missing_files():
    specs = build_training_specs([{"model_config": {"task_type": "classification"}}])
    generated = GeneratedFiles(files={"model.py": "print('only one file')\n"})

    review = review_generated(generated, specs, smoke_result=SmokeResult(success=True))

    assert not review.is_approved
    assert any("Missing required files" in error for error in review.errors)


def test_reviewer_rejects_failed_smoke():
    specs = build_training_specs([{"model_config": {"task_type": "classification"}}])
    generated = generate_files(specs)

    review = review_generated(generated, specs, smoke_result=SmokeResult(success=False))

    assert not review.is_approved
    assert any("Smoke test failed" in error for error in review.errors)


def test_reviewer_rejects_missing_experiment_row_fields():
    specs = build_training_specs([{"model_config": {"task_type": "classification"}}])
    generated = generate_files(specs)
    smoke = SmokeResult(
        success=True,
        command_results=[
            CommandResult(
                command=[sys.executable, "run_experiments.py"],
                return_code=0,
                stdout='[{"metric_name": "accuracy", "status": "success"}]',
                stderr="",
                runtime_sec=0.01,
            )
        ],
    )

    review = review_generated(generated, specs, smoke_result=smoke)

    assert not review.is_approved
    assert any("missing fields" in error for error in review.errors)


def test_reviewer_dynamically_rejects_broken_head_only_freeze(tmp_path):
    specs = build_training_specs(
        [
            {
                "model_config": {
                    "task_type": "classification",
                    "finetune_strategy": "head_only",
                    "freeze_backbone": True,
                }
            }
        ]
    )
    generated = generate_files(specs)
    generated.files["model.py"] = generated.files["model.py"].replace(
        "parameter.requires_grad = False",
        "parameter.requires_grad = True",
    ).replace(
        "apply_freeze",
        "_no_freeze",
    )
    write_generated_files(generated, tmp_path)
    smoke = SmokeResult(
        success=True,
        command_results=[
            CommandResult(
                command=[sys.executable, "run_experiments.py"],
                return_code=0,
                stdout='[{"metric_name": "accuracy", "status": "success"}]',
                stderr="",
                runtime_sec=0.01,
            )
        ],
    )

    review = review_generated(generated, specs, smoke_result=smoke, output_dir=tmp_path)

    assert not review.is_approved
    assert any("did not freeze backbone-like parameters" in error for error in review.errors)


def test_executor_subprocess_environment_is_shared():
    env = subprocess_env()

    for key, value in SUBPROCESS_ENV_DEFAULTS.items():
        assert env[key] == value


def test_inprocess_fallback_captures_argparse_system_exit(tmp_path):
    script = tmp_path / "exit_script.py"
    script.write_text(
        "import argparse\n"
        "parser = argparse.ArgumentParser()\n"
        "parser.add_argument('--ok', action='store_true')\n"
        "parser.parse_args()\n",
        encoding="utf-8",
    )

    result = _run_python_script_inprocess(
        [sys.executable, "exit_script.py", "--bad-arg"],
        cwd=tmp_path,
        start=time.time(),
    )

    assert result.return_code == 2
    assert "unrecognized arguments" in result.stderr


def test_generated_scripts_accept_raw_module3_input(tmp_path):
    sample = Path(__file__).resolve().parents[1] / "examples" / "sample_m3_output_all_tasks.json"
    output = tmp_path / "generated"
    result = run_workflow(sample, output, max_iter=1, timeout=60, skip_smoke=True)

    assert result.is_approved
    command = run_command(
        [sys.executable, "run_experiments.py", "--input", str(sample)],
        cwd=output,
        timeout=60,
    )
    rows = json.loads(command.stdout)

    assert command.success
    assert (output / "best_config.json").exists()
    assert len(rows) == 4
    assert {row["task_type"] for row in rows} == {
        "classification",
        "object_detection",
        "image_segmentation",
        "feature_extraction",
    }

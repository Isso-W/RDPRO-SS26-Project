import json

from kaggle_orchestrator import (
    log_kaggle_outcome_if_scored,
    write_run_manifest,
    write_submission_receipt,
)


def test_write_run_manifest_records_kaggle_context(tmp_path):
    path = write_run_manifest(
        tmp_path,
        benchmark_key="plant-pathology-2020-fgv",
        info={
            "competition": "plant-pathology-2020-fgvc7",
            "metric": "categorical_accuracy",
            "train_csv": "/data/train.csv",
            "image_dir": "/data/images",
            "image_column": "image_id",
            "label_column": "__jiaozi_label",
            "label_columns": ["healthy", "rust"],
            "sample_submission": "/data/sample_submission.csv",
            "test_dir": "/data/test_images",
        },
        module3_input={"task_type": "classification", "data_size": "medium"},
        recommendations=[{"backbone": "efficientnet", "score": 0.9}],
        module4={"output_dir": str(tmp_path / "module4_code"), "summary": {"status": "approved"}},
    )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert manifest["benchmark_key"] == "plant-pathology-2020-fgv"
    assert manifest["competition"] == "plant-pathology-2020-fgvc7"
    assert manifest["data"]["label_columns"] == ["healthy", "rust"]
    assert manifest["recommendations"][0]["backbone"] == "efficientnet"


def test_receipt_without_score_does_not_log_memory(tmp_path):
    receipt = write_submission_receipt(
        tmp_path / "submission_receipt.json",
        benchmark_key="cassava",
        competition="cassava-leaf-disease-classification",
        submission_csv=tmp_path / "submission.csv",
        submitted=False,
    )
    memory = tmp_path / "outcomes.jsonl"

    result = log_kaggle_outcome_if_scored(receipt, memory_path=memory)

    assert result == {"logged": False, "reason": "missing_public_score"}
    assert not memory.exists()


def test_scored_receipt_logs_outcome_memory(tmp_path):
    manifest = write_run_manifest(
        tmp_path,
        benchmark_key="plant_pathology_2020",
        info={"competition": "plant-pathology-2020-fgvc7", "metric": "categorical_accuracy"},
        module3_input={
            "task_type": "classification",
            "data_size": "medium",
            "num_classes": 4,
            "constraints": {"class_imbalance": False},
            "data_stats": {"resolution_tier": "high", "color_mode": "rgb"},
        },
        recommendations=[{"backbone": "efficientnet", "pretrained": "efficientnet_b0"}],
        module4={"output_dir": str(tmp_path / "module4_code")},
    )
    project = tmp_path / "module4_code"
    project.mkdir()
    (project / "configs.json").write_text(
        json.dumps([
            {
                "backbone": "efficientnet",
                "model_config": {"pretrained_hf_id": "timm/efficientnet_b0"},
            }
        ]),
        encoding="utf-8",
    )
    receipt = write_submission_receipt(
        project / "submission_receipt.json",
        benchmark_key="plant_pathology_2020",
        competition="plant-pathology-2020-fgvc7",
        submission_csv=project / "submission.csv",
        submitted=True,
        status="complete",
        public_score="0.9123",
    )
    memory = tmp_path / "outcomes.jsonl"

    result = log_kaggle_outcome_if_scored(
        receipt,
        run_manifest_path=manifest,
        project_dir=project,
        memory_path=memory,
    )

    assert result["logged"] is True
    assert result["metric_value"] == 0.9123
    record = json.loads(memory.read_text(encoding="utf-8").strip())
    assert record["dataset_id"] == "plant-pathology-2020-fgvc7"
    assert record["fingerprint"]["num_classes"] == 4
    assert record["config"]["pretrained_hf_id"] == "timm/efficientnet_b0"
    assert record["result"]["metric_value"] == 0.9123

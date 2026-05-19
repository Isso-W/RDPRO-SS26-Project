from __future__ import annotations

import json
import unittest
from pathlib import Path

from cv_autodl_agent.io_utils import read_json
from cv_autodl_agent.schemas import DatasetManifest, RetrievedModelCandidate
from cv_autodl_agent.workflow import CVAutoDLWorkflow

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


class ExampleInputTests(unittest.TestCase):
    def test_all_example_inputs_validate(self) -> None:
        for example_name in ("classification", "segmentation", "detection", "cifar10", "food101"):
            manifest = DatasetManifest.from_dict(read_json(EXAMPLES / f"{example_name}_manifest.json"))
            candidates = [
                RetrievedModelCandidate.from_dict(item)
                for item in read_json(EXAMPLES / f"{example_name}_candidates.json")
            ]
            manifest.validate()
            for candidate in candidates:
                candidate.validate()

    def test_colab_demo_notebook_is_valid_json(self) -> None:
        notebook = json.loads((EXAMPLES / "colab_demo.ipynb").read_text(encoding="utf-8"))
        notebook_text = json.dumps(notebook)
        self.assertEqual(notebook["nbformat"], 4)
        self.assertGreaterEqual(len(notebook["cells"]), 5)
        self.assertIn("git', 'clone'", notebook_text)
        self.assertIn("os.chdir(repo_root)", notebook_text)

    def test_classification_example_matches_expected_summary_shape(self) -> None:
        manifest = DatasetManifest.from_dict(read_json(EXAMPLES / "classification_manifest.json"))
        candidates = [
            RetrievedModelCandidate.from_dict(item)
            for item in read_json(EXAMPLES / "classification_candidates.json")
        ]
        expected = read_json(EXAMPLES / "expected_classification_summary.json")

        result = CVAutoDLWorkflow().run(manifest, candidates, ROOT / ".tmp_example_test")
        self.assertEqual(result.selected_candidate.model_id, expected["selected_model"])
        self.assertEqual(result.review_report.status, expected["review_status"])
        self.assertEqual(result.ablation_summary.best_component_to_change, expected["ablation_best_component"])
        self.assertGreater(result.final_result.primary_metric_value or 0, result.baseline_result.primary_metric_value or 0)

    def test_cifar10_exports_real_colab_notebook(self) -> None:
        manifest = DatasetManifest.from_dict(read_json(EXAMPLES / "cifar10_manifest.json"))
        candidates = [
            RetrievedModelCandidate.from_dict(item)
            for item in read_json(EXAMPLES / "cifar10_candidates.json")
        ]

        result = CVAutoDLWorkflow().run(
            manifest,
            candidates,
            ROOT / ".tmp_cifar10_test",
            execution_mode="simulate",
            notebook_execution_mode="real",
        )
        notebook_text = Path(result.notebook_path).read_text(encoding="utf-8")
        train_text = (Path(result.project_dir) / "train.py").read_text(encoding="utf-8")
        requirements_text = (Path(result.project_dir) / "requirements.txt").read_text(encoding="utf-8")
        self.assertIn("EXECUTION_MODE = 'real'", notebook_text)
        self.assertIn("load_dataset(dataset_id", train_text)
        self.assertIn("from torchvision import models as tv_models", train_text)
        self.assertIn("torch.save", train_text)
        self.assertIn("if args.execution_mode == \"real\":", train_text)
        self.assertIn("payload = run_real_training(manifest, spec, output_dir)", train_text)
        self.assertIn("checkpoints", train_text)
        self.assertIn("best.pt", train_text)
        self.assertNotIn("import timm", train_text)
        self.assertNotIn("timm.create_model", train_text)
        self.assertNotIn("timm", requirements_text)


if __name__ == "__main__":
    unittest.main()

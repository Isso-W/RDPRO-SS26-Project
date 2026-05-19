from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cv_autodl_agent.schemas import DatasetManifest, RetrievedModelCandidate
from cv_autodl_agent.workflow import CVAutoDLWorkflow


class WorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.workflow = CVAutoDLWorkflow()

    def test_classification_workflow_generates_artifacts(self) -> None:
        manifest = DatasetManifest(
            dataset_name="cls-demo",
            task_family="classification",
            train_path="data/train",
            val_path="data/val",
            test_path="data/test",
            annotation_format="ImageFolder",
            recommended_metric="accuracy",
            num_classes=3,
            class_names=["cat", "dog", "bird"],
            label_source="folder_name",
            image_size_hint=224,
        )
        candidates = [
            RetrievedModelCandidate(
                rank=1,
                model_id="timm-resnet18",
                source="timm",
                task_family="classification",
                library="PyTorch",
                processor_or_transform="timm_default_transform",
                default_input_size=224,
                pretrained_weights="imagenet",
                license="Apache-2.0",
                training_notes="small and efficient",
                install_deps=["torch", "torchvision", "timm"],
            )
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = self.workflow.run(manifest, candidates, tmp_dir)
            project_dir = Path(result.project_dir)
            self.assertTrue((project_dir / "selected_candidate.json").exists())
            self.assertTrue((project_dir / "ablation_summary.json").exists())
            self.assertTrue((project_dir / "notebook.ipynb").exists())
            self.assertEqual(result.review_report.status, "pass")
            self.assertGreater(result.final_result.primary_metric_value or 0.0, 0.0)

    def test_detection_workflow_produces_single_edit_region(self) -> None:
        manifest = DatasetManifest(
            dataset_name="det-demo",
            task_family="detection",
            train_path="data/train",
            val_path="data/val",
            test_path="data/test",
            annotation_format="COCO",
            recommended_metric="map50",
            categories=["car", "person"],
            bbox_format="xywh",
            coco_json_path="annotations/train.json",
            image_size_hint=640,
        )
        candidates = [
            RetrievedModelCandidate(
                rank=1,
                model_id="hf-detr-small",
                source="huggingface",
                task_family="detection",
                library="PyTorch",
                processor_or_transform="AutoImageProcessor",
                default_input_size=640,
                pretrained_weights="coco",
                license="Apache-2.0",
                training_notes="small and efficient",
                install_deps=["torch", "torchvision", "transformers", "pycocotools"],
            )
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = self.workflow.run(manifest, candidates, tmp_dir)
            self.assertTrue(result.ablation_summary.recommended_edit_region.startswith("training_spec"))
            self.assertEqual(result.review_report.status, "pass")

    def test_segmentation_workflow_exports_notebook_cells(self) -> None:
        manifest = DatasetManifest(
            dataset_name="seg-demo",
            task_family="segmentation",
            train_path="data/images",
            val_path="data/val_images",
            test_path="data/test_images",
            annotation_format="paired_dirs",
            recommended_metric="miou",
            num_classes=2,
            class_names=["bg", "object"],
            mask_format="png",
            ignore_index=255,
            image_size_hint=512,
        )
        candidates = [
            RetrievedModelCandidate(
                rank=1,
                model_id="hf-segformer-b0",
                source="huggingface",
                task_family="segmentation",
                library="PyTorch",
                processor_or_transform="AutoImageProcessor",
                default_input_size=512,
                pretrained_weights="ade20k",
                license="Apache-2.0",
                training_notes="compact segmentation baseline",
                install_deps=["torch", "torchvision", "transformers"],
            )
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            result = self.workflow.run(manifest, candidates, tmp_dir)
            notebook = json.loads(Path(result.notebook_path).read_text())
            notebook_text = json.dumps(notebook)
            self.assertGreaterEqual(len(notebook["cells"]), 6)
            self.assertIn("--execution-mode {EXECUTION_MODE}", notebook_text)
            self.assertIn("checkpoints/best.pt", notebook_text)
            self.assertEqual(result.review_report.status, "pass")


if __name__ == "__main__":
    unittest.main()

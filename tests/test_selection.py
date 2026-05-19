from __future__ import annotations

import unittest

from cv_autodl_agent.schemas import DatasetManifest, RetrievedModelCandidate
from cv_autodl_agent.selectors import select_candidate


class SelectionTests(unittest.TestCase):
    def test_prefers_matching_classification_candidate(self) -> None:
        manifest = DatasetManifest(
            dataset_name="cls",
            task_family="classification",
            train_path="train",
            val_path="val",
            test_path="test",
            annotation_format="ImageFolder",
            recommended_metric="accuracy",
            num_classes=2,
            class_names=["a", "b"],
            label_source="folder_name",
            image_size_hint=224,
        )
        candidates = [
            RetrievedModelCandidate(
                rank=1,
                model_id="bad-detector",
                source="huggingface",
                task_family="detection",
                library="PyTorch",
                processor_or_transform="processor",
                default_input_size=640,
                pretrained_weights="coco",
                license="Apache-2.0",
                training_notes="fast",
                install_deps=["torch"],
            ),
            RetrievedModelCandidate(
                rank=2,
                model_id="good-classifier",
                source="timm",
                task_family="classification",
                library="PyTorch",
                processor_or_transform="transform",
                default_input_size=224,
                pretrained_weights="imagenet",
                license="Apache-2.0",
                training_notes="small",
                install_deps=["torch", "timm"],
            ),
        ]
        selected = select_candidate(manifest, candidates)
        self.assertEqual(selected.model_id, "good-classifier")


if __name__ == "__main__":
    unittest.main()

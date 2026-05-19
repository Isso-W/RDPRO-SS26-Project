from __future__ import annotations

import unittest

from cv_autodl_agent.exceptions import InputValidationError
from cv_autodl_agent.schemas import DatasetManifest


class ValidationTests(unittest.TestCase):
    def test_detection_requires_bbox_information(self) -> None:
        manifest = DatasetManifest(
            dataset_name="demo",
            task_family="detection",
            train_path="train",
            val_path="val",
            test_path="test",
            annotation_format="coco",
            recommended_metric="map50",
        )
        with self.assertRaises(InputValidationError):
            manifest.validate()


if __name__ == "__main__":
    unittest.main()

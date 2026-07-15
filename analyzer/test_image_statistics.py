"""
Tests for ImageStatisticsAnalyzer annotation-format detection.

Coverage: classification, object detection (dict-of-lists / list-of-dicts /
flat columns), segmentation masks, and unlabeled fallback.

Run with: python -m pytest analyzer/test_image_statistics.py -v
"""

import unittest
from collections import namedtuple
from unittest.mock import MagicMock

from PIL import Image

from analyzer.image_statistics import ImageStatisticsAnalyzer


def _make_split(rows, columns):
    """Build a mock object that behaves like a Hugging Face Dataset split.

    It supports the two indexing modes used by real HF datasets:
      - integer idx returns a row dict
      - string col returns a column list without decoding other columns
    """
    def _getitem(_self, key):
        if isinstance(key, str):
            return [row.get(key) for row in rows]
        return rows[key]

    split = MagicMock()
    split.column_names = columns
    split.__len__ = lambda self: len(rows)
    split.__getitem__ = _getitem
    split.__iter__ = lambda self: iter(rows)
    return split


def _make_dataset(splits_dict):
    """Build a mock object that behaves like a Hugging Face DatasetDict."""
    ds = MagicMock()
    ds.keys.return_value = list(splits_dict.keys())
    ds.__getitem__ = lambda self, key: splits_dict[key]
    ds.__contains__ = lambda self, key: key in splits_dict
    return ds


def _make_image(width=32, height=32, mode="RGB"):
    return Image.new(mode, (width, height))


class TestClassificationFormat(unittest.TestCase):

    def test_basic_classification(self):
        rows = [
            {"image": _make_image(), "label": 0},
            {"image": _make_image(), "label": 1},
            {"image": _make_image(), "label": 0},
            {"image": _make_image(), "label": 2},
        ]
        split = _make_split(rows, ["image", "label"])
        ds = _make_dataset({"train": split})

        analyzer = ImageStatisticsAnalyzer()
        report = analyzer.analyze(ds)

        self.assertEqual(report["annotation_format"], "classification")
        self.assertEqual(report["num_classes"], 3)
        self.assertEqual(report["total_images"], 4)
        self.assertEqual(report["class_distribution"], {0: 2, 1: 1, 2: 1})


class TestDetectionFormat(unittest.TestCase):

    def test_dict_of_lists_category(self):
        rows = [
            {"image": _make_image(), "objects": {"bbox": [[0, 0, 1, 1]], "category": [0, 1]}},
            {"image": _make_image(), "objects": {"bbox": [[0, 0, 1, 1]], "category": [2]}},
        ]
        split = _make_split(rows, ["image", "objects"])
        ds = _make_dataset({"train": split})

        report = ImageStatisticsAnalyzer().analyze(ds)

        self.assertEqual(report["annotation_format"], "detection")
        self.assertEqual(report["num_classes"], 3)
        self.assertEqual(report["class_distribution"], {0: 1, 1: 1, 2: 1})

    def test_list_of_dicts(self):
        rows = [
            {"image": _make_image(), "objects": [
                {"bbox": [0, 0, 1, 1], "category": 0},
                {"bbox": [0, 0, 1, 1], "category": 1},
            ]},
            {"image": _make_image(), "objects": [
                {"bbox": [0, 0, 1, 1], "category": 1},
            ]},
        ]
        split = _make_split(rows, ["image", "objects"])
        ds = _make_dataset({"train": split})

        report = ImageStatisticsAnalyzer().analyze(ds)

        self.assertEqual(report["annotation_format"], "detection")
        self.assertEqual(report["num_classes"], 2)
        self.assertEqual(report["class_distribution"], {0: 1, 1: 2})

    def test_flat_labels_column(self):
        rows = [
            {"image": _make_image(), "labels": [0, 1, 1]},
            {"image": _make_image(), "labels": [2]},
        ]
        split = _make_split(rows, ["image", "labels"])
        ds = _make_dataset({"train": split})

        report = ImageStatisticsAnalyzer().analyze(ds)

        self.assertEqual(report["annotation_format"], "detection_flat")
        self.assertEqual(report["num_classes"], 3)

    def test_unknown_objects_format(self):
        rows = [
            {"image": _make_image(), "objects": {"bbox": [[0, 0, 1, 1]]}},
        ]
        split = _make_split(rows, ["image", "objects"])
        ds = _make_dataset({"train": split})

        report = ImageStatisticsAnalyzer().analyze(ds)

        self.assertEqual(report["annotation_format"], "detection_unknown")
        self.assertIsNone(report["num_classes"])


class TestSegmentationFormat(unittest.TestCase):

    def test_label_is_mask_image(self):
        mask1 = Image.new("L", (4, 4), 0)
        mask1.putpixel((0, 0), 1)
        mask1.putpixel((1, 1), 2)

        mask2 = Image.new("L", (4, 4), 0)
        mask2.putpixel((2, 2), 3)

        rows = [
            {"image": _make_image(), "label": mask1},
            {"image": _make_image(), "label": mask2},
        ]
        split = _make_split(rows, ["image", "label"])
        ds = _make_dataset({"train": split})

        report = ImageStatisticsAnalyzer().analyze(ds)

        self.assertEqual(report["annotation_format"], "segmentation_mask")
        self.assertEqual(report["num_classes"], 3)
        self.assertEqual(report["class_distribution"], {})

    def test_annotation_column_mask(self):
        mask = Image.new("L", (4, 4), 0)
        mask.putpixel((0, 0), 5)

        rows = [
            {"image": _make_image(), "annotation": mask},
        ]
        split = _make_split(rows, ["image", "annotation"])
        ds = _make_dataset({"train": split})

        report = ImageStatisticsAnalyzer().analyze(ds)

        self.assertEqual(report["annotation_format"], "segmentation_mask")
        self.assertEqual(report["num_classes"], 1)


class TestNoLabels(unittest.TestCase):

    def test_no_label_column(self):
        rows = [
            {"image": _make_image()},
            {"image": _make_image()},
        ]
        split = _make_split(rows, ["image"])
        ds = _make_dataset({"train": split})

        report = ImageStatisticsAnalyzer().analyze(ds)

        self.assertEqual(report["annotation_format"], "none")
        self.assertIsNone(report["num_classes"])
        self.assertEqual(report["class_distribution"], {})

    def test_empty_split(self):
        split = _make_split([], ["image", "label"])
        ds = _make_dataset({"train": split})

        report = ImageStatisticsAnalyzer().analyze(ds)

        self.assertEqual(report["annotation_format"], "none")
        self.assertIsNone(report["num_classes"])


class TestSplitSelection(unittest.TestCase):

    def test_prefers_train_split(self):
        train_rows = [
            {"image": _make_image(), "label": 0},
            {"image": _make_image(), "label": 1},
        ]
        test_rows = [
            {"image": _make_image(), "label": 0},
        ]
        train_split = _make_split(train_rows, ["image", "label"])
        test_split = _make_split(test_rows, ["image", "label"])
        ds = _make_dataset({"train": train_split, "test": test_split})

        report = ImageStatisticsAnalyzer().analyze(ds)

        self.assertEqual(report["num_classes"], 2)
        self.assertEqual(report["total_images"], 3)

    def test_falls_back_to_first_split(self):
        rows = [
            {"image": _make_image(), "label": 0},
            {"image": _make_image(), "label": 1},
            {"image": _make_image(), "label": 2},
        ]
        split = _make_split(rows, ["image", "label"])
        ds = _make_dataset({"validation": split})

        report = ImageStatisticsAnalyzer().analyze(ds)

        self.assertEqual(report["num_classes"], 3)


class TestMergeWithNewFormats(unittest.TestCase):
    """Check that pipeline.merge_modules handles the newer report fields."""

    def test_segmentation_num_classes_preserved(self):
        from pipeline import merge_modules

        m1 = {
            "task_type": "image_segmentation",
            "data_size": "medium",
            "priority": "accuracy",
            "constraints": {"class_imbalance": False},
            "description": "segment buildings",
        }
        m2 = {
            "total_images": 5000,
            "num_classes": 21,
            "class_distribution": {},
            "annotation_format": "segmentation_mask",
        }
        merged = merge_modules(m1, m2)
        self.assertEqual(merged["num_classes"], 21)

    def test_no_labels_degrades_gracefully(self):
        from pipeline import merge_modules

        m1 = {
            "task_type": "feature_extraction",
            "data_size": "medium",
            "priority": "balanced",
            "constraints": {"class_imbalance": False},
            "description": "extract features",
        }
        m2 = {
            "total_images": 8000,
            "num_classes": None,
            "class_distribution": {},
            "annotation_format": "none",
        }
        merged = merge_modules(m1, m2)
        self.assertNotIn("num_classes", merged)
        self.assertFalse(merged["constraints"]["class_imbalance"])
        self.assertEqual(merged["data_size"], "medium")


if __name__ == "__main__":
    unittest.main(verbosity=2)

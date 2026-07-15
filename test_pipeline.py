"""
Pipeline integration-layer tests.

Coverage: data_size inference, class_imbalance inference, Module 1 + Module 2
merge behavior, and Module 1 parser fallback.

Run with:
    python -m pytest test_pipeline.py -v
    python test_pipeline.py
"""

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pipeline import (
    derive_class_imbalance,
    derive_color_mode,
    derive_data_size,
    derive_resolution_tier,
    merge_modules,
    parse_dataset_id,
    run_module4_generation,
)
from features_extraction_api import parse_module1_output
from env_loader import load_env_file


class TestEnvLoader(unittest.TestCase):

    def test_load_env_file_sets_missing_values_only(self):
        previous = {
            "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY"),
            "M4_LLM_PROVIDER": os.environ.get("M4_LLM_PROVIDER"),
        }
        os.environ["OPENAI_API_KEY"] = "existing-value"
        os.environ.pop("M4_LLM_PROVIDER", None)

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                env_path = Path(tmpdir) / ".env"
                env_path.write_text(
                    "OPENAI_API_KEY=from-file\n"
                    "M4_LLM_PROVIDER=openai\n",
                    encoding="utf-8",
                )

                self.assertTrue(load_env_file(env_path))
                self.assertEqual(os.environ["OPENAI_API_KEY"], "existing-value")
                self.assertEqual(os.environ["M4_LLM_PROVIDER"], "openai")
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


class TestDeriveDataSize(unittest.TestCase):

    def test_small(self):
        self.assertEqual(derive_data_size(500), "small")
        self.assertEqual(derive_data_size(3000), "small")

    def test_medium(self):
        self.assertEqual(derive_data_size(3001), "medium")
        self.assertEqual(derive_data_size(20000), "medium")

    def test_large(self):
        self.assertEqual(derive_data_size(20001), "large")
        self.assertEqual(derive_data_size(1_000_000), "large")

    def test_zero(self):
        self.assertEqual(derive_data_size(0), "small")

    # Two-signal logic: samples per class can downgrade the size tier.

    def test_per_class_downgrades_large_to_medium(self):
        """25k images over 200 classes gives 125 samples/class, so medium."""
        self.assertEqual(derive_data_size(25_000, num_classes=200), "medium")

    def test_per_class_downgrades_medium_to_small(self):
        """5k images over 100 classes gives 50 samples/class, so small."""
        self.assertEqual(derive_data_size(5_000, num_classes=100), "small")

    # Two-signal logic: total count still caps the tier on the cost side.

    def test_total_caps_high_per_class(self):
        """3k images over 2 classes has many samples/class, but is still small."""
        self.assertEqual(derive_data_size(3_000, num_classes=2), "small")

    def test_both_signals_large(self):
        """Both total count and samples/class must be large."""
        self.assertEqual(derive_data_size(100_000, num_classes=10), "large")

    # Task-specific thresholds: detection/segmentation have higher label cost.

    def test_detection_thresholds_halved(self):
        self.assertEqual(derive_data_size(1_500, task_type="object_detection"), "small")
        self.assertEqual(derive_data_size(2_000, task_type="object_detection"), "medium")
        self.assertEqual(derive_data_size(12_000, task_type="object_detection"), "large")

    def test_segmentation_thresholds_halved(self):
        self.assertEqual(derive_data_size(10_000, task_type="image_segmentation"), "medium")
        self.assertEqual(derive_data_size(10_001, task_type="image_segmentation"), "large")

    def test_per_class_ignored_for_detection(self):
        """The samples-per-class signal is only used for classification."""
        self.assertEqual(derive_data_size(12_000, num_classes=200, task_type="object_detection"), "large")

    def test_no_num_classes_falls_back_to_total(self):
        self.assertEqual(derive_data_size(25_000), "large")
        self.assertEqual(derive_data_size(25_000, num_classes=None), "large")


class TestDeriveClassImbalance(unittest.TestCase):

    def test_balanced(self):
        dist = {"cat": 500, "dog": 480, "bird": 520}
        self.assertFalse(derive_class_imbalance(dist))

    def test_imbalanced(self):
        dist = {"cat": 5000, "dog": 50, "bird": 4000}
        self.assertTrue(derive_class_imbalance(dist))

    def test_exactly_at_threshold(self):
        # max/min = 10, so it is still within the threshold.
        dist = {"a": 100, "b": 10}
        self.assertFalse(derive_class_imbalance(dist))

    def test_just_over_threshold(self):
        dist = {"a": 101, "b": 10}
        self.assertTrue(derive_class_imbalance(dist))

    def test_zero_count_class(self):
        dist = {"a": 500, "b": 0}
        self.assertTrue(derive_class_imbalance(dist))

    def test_empty(self):
        self.assertFalse(derive_class_imbalance({}))


class TestDeriveDataStats(unittest.TestCase):

    def test_resolution_tiers(self):
        self.assertEqual(derive_resolution_tier({"avg_width": 128, "avg_height": 200}), "low")
        self.assertEqual(derive_resolution_tier({"avg_width": 512, "avg_height": 384}), "medium")
        self.assertEqual(derive_resolution_tier({"avg_width": 1200, "avg_height": 768}), "high")

    def test_resolution_missing_or_invalid_defaults_medium(self):
        self.assertEqual(derive_resolution_tier({}), "medium")
        self.assertEqual(derive_resolution_tier({"avg_width": "bad", "avg_height": 768}), "medium")

    def test_color_mode_uses_dominant_mode(self):
        self.assertEqual(derive_color_mode({"mode_distribution": {"L": 90, "RGB": 10}}), "grayscale")
        self.assertEqual(derive_color_mode({"mode_distribution": {"RGB": 90, "L": 10}}), "rgb")
        self.assertEqual(derive_color_mode({}), "rgb")


class TestMergeModules(unittest.TestCase):

    def _make_m1(self, **overrides):
        base = {
            "task_type": "classification",
            "data_size": "medium",
            "priority": "balanced",
            "constraints": {
                "real_time": False, "edge_deployment": False,
                "class_imbalance": False, "cross_modal": False,
                "medical": False, "zero_shot": False, "few_shot": False,
            },
            "description": "test query",
        }
        base.update(overrides)
        return base

    def _make_m2(self, total_images=10000, class_dist=None):
        if class_dist is None:
            class_dist = {"a": 500, "b": 500}
        return {
            "total_images": total_images,
            "num_classes": len(class_dist),
            "class_distribution": class_dist,
            "split_sizes": {"train": total_images},
        }

    def test_data_size_from_module2(self):
        """data_size comes from Module 2, not the Module 1 placeholder."""
        m1 = self._make_m1()
        m2 = self._make_m2(total_images=500)
        merged = merge_modules(m1, m2)
        self.assertEqual(merged["data_size"], "small")

    def test_module1_fields_preserved(self):
        """task_type, priority, and description remain from Module 1."""
        m1 = self._make_m1(task_type="object_detection", priority="speed")
        m2 = self._make_m2()
        merged = merge_modules(m1, m2)
        self.assertEqual(merged["task_type"], "object_detection")
        self.assertEqual(merged["priority"], "speed")
        self.assertEqual(merged["description"], "test query")

    def test_imbalance_from_module2(self):
        """Module 2 can enable class_imbalance even when Module 1 did not."""
        m1 = self._make_m1()
        m2 = self._make_m2(class_dist={"a": 5000, "b": 10})
        merged = merge_modules(m1, m2)
        self.assertTrue(merged["constraints"]["class_imbalance"])

    def test_imbalance_from_module1(self):
        """A user-stated imbalance remains true even if the dataset looks balanced."""
        m1 = self._make_m1()
        m1["constraints"]["class_imbalance"] = True
        m2 = self._make_m2(class_dist={"a": 500, "b": 500})
        merged = merge_modules(m1, m2)
        self.assertTrue(merged["constraints"]["class_imbalance"])

    def test_other_constraints_untouched(self):
        """Module 2 does not overwrite unrelated constraints."""
        m1 = self._make_m1()
        m1["constraints"]["medical"] = True
        m2 = self._make_m2()
        merged = merge_modules(m1, m2)
        self.assertTrue(merged["constraints"]["medical"])
        self.assertFalse(merged["constraints"]["cross_modal"])

    def test_num_classes_passed_through(self):
        """num_classes is passed through for Module 4 head sizing."""
        m1 = self._make_m1()
        m2 = self._make_m2(class_dist={"a": 100, "b": 100, "c": 100})
        merged = merge_modules(m1, m2)
        self.assertEqual(merged["num_classes"], 3)

    def test_num_classes_absent_when_no_distribution(self):
        m1 = self._make_m1()
        m2 = {"total_images": 5000, "class_distribution": {}}
        merged = merge_modules(m1, m2)
        self.assertNotIn("num_classes", merged)

    def test_data_stats_passed_through_for_recipe_layer(self):
        m1 = self._make_m1()
        m2 = self._make_m2()
        m2.update({
            "avg_width": 1200,
            "avg_height": 800,
            "mode_distribution": {"L": 100, "RGB": 1},
            "format_distribution": {"PNG": 101},
        })

        merged = merge_modules(m1, m2)

        self.assertEqual(merged["data_stats"]["resolution_tier"], "high")
        self.assertEqual(merged["data_stats"]["color_mode"], "grayscale")
        self.assertEqual(merged["data_stats"]["avg_width"], 1200)
        self.assertEqual(merged["data_stats"]["mode_distribution"], {"L": 100, "RGB": 1})
        self.assertEqual(merged["data_stats"]["format_distribution"], {"PNG": 101})

    def test_data_size_uses_per_class_signal(self):
        """merge_modules uses class count: 25k images over 200 classes is medium."""
        m1 = self._make_m1()
        class_dist = {f"c{i}": 125 for i in range(200)}
        m2 = self._make_m2(total_images=25_000, class_dist=class_dist)
        merged = merge_modules(m1, m2)
        self.assertEqual(merged["data_size"], "medium")

    def test_merge_does_not_mutate_module1_output(self):
        """merge_modules should not mutate the Module 1 input."""
        m1 = self._make_m1()
        m2 = self._make_m2(class_dist={"a": 5000, "b": 10})
        merge_modules(m1, m2)
        self.assertFalse(m1["constraints"]["class_imbalance"])


class TestModule4Handoff(unittest.TestCase):

    def test_run_module4_generation_writes_input_and_passes_provider(self):
        task_lists = [
            {
                "format": "nl",
                "rank": 1,
                "score": 0.9,
                "model_config": {
                    "task_type": "classification",
                    "backbone": "efficientnet_b0",
                    "loss": "cross_entropy_loss",
                    "optimizer": "adamw",
                },
                "tasks": ["Use EfficientNet-B0."],
                "alternatives": [],
            }
        ]
        previous_provider = os.environ.get("M4_LLM_PROVIDER")

        class DummyResult:
            def to_summary(self):
                return {"status": "approved"}

        captured = {}

        def fake_run_workflow(input_path, output_dir, *, timeout, skip_smoke, run_refinement, llm_provider=None):
            captured["input_path"] = Path(input_path)
            captured["output_dir"] = Path(output_dir)
            captured["llm_provider"] = llm_provider
            captured["timeout"] = timeout
            captured["skip_smoke"] = skip_smoke
            captured["run_refinement"] = run_refinement
            return DummyResult()

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch("module4_agent.workflow.run_workflow", side_effect=fake_run_workflow):
                result = run_module4_generation(
                    task_lists,
                    tmpdir,
                    skip_smoke=True,
                    run_refinement=True,
                    timeout=7,
                    llm_provider="qwen",
                )

            input_path = Path(result["input_path"])
            self.assertTrue(input_path.exists())
            self.assertEqual(json.loads(input_path.read_text(encoding="utf-8")), task_lists)
            self.assertEqual(result["summary"], {"status": "approved"})
            self.assertEqual(captured["llm_provider"], "qwen")
            self.assertEqual(captured["timeout"], 7)
            self.assertTrue(captured["skip_smoke"])
            self.assertTrue(captured["run_refinement"])

        # Provider selection is passed as an argument and should not touch env vars.
        self.assertEqual(os.environ.get("M4_LLM_PROVIDER"), previous_provider)


class TestParseModule1Output(unittest.TestCase):

    def test_valid_json(self):
        raw = '{"task_type": "object_detection", "priority": "speed", "constraints": {"real_time": true}}'
        result = parse_module1_output(raw, "detect objects fast")
        self.assertEqual(result["task_type"], "object_detection")
        self.assertEqual(result["priority"], "speed")
        self.assertTrue(result["constraints"]["real_time"])
        self.assertEqual(result["description"], "detect objects fast")

    def test_markdown_code_block(self):
        raw = '```json\n{"task_type": "classification", "priority": "accuracy", "constraints": {}}\n```'
        result = parse_module1_output(raw, "classify")
        self.assertEqual(result["task_type"], "classification")
        self.assertEqual(result["priority"], "accuracy")

    def test_alias_mapping(self):
        raw = '{"task_type": "detection", "priority": "balanced", "constraints": {}}'
        result = parse_module1_output(raw, "detect")
        self.assertEqual(result["task_type"], "object_detection")

    def test_invalid_json_fallback(self):
        raw = "sorry I can't help with that"
        result = parse_module1_output(raw, "some query")
        self.assertEqual(result["task_type"], "classification")
        self.assertEqual(result["priority"], "balanced")
        self.assertFalse(result["constraints"]["real_time"])

    def test_invalid_enum_fallback(self):
        raw = '{"task_type": "regression", "priority": "fast", "constraints": {}}'
        result = parse_module1_output(raw, "q")
        self.assertEqual(result["task_type"], "classification")
        self.assertEqual(result["priority"], "balanced")

    def test_missing_constraints_filled(self):
        raw = '{"task_type": "image_segmentation", "priority": "accuracy", "constraints": {"medical": true}}'
        result = parse_module1_output(raw, "segment medical images")
        self.assertTrue(result["constraints"]["medical"])
        self.assertFalse(result["constraints"]["real_time"])
        self.assertFalse(result["constraints"]["edge_deployment"])

    def test_evaluation_metric_extracted(self):
        raw = '{"task_type": "classification", "priority": "accuracy", "evaluation_metric": "qwk"}'
        self.assertEqual(parse_module1_output(raw, "grade severity")["evaluation_metric"], "qwk")

    def test_evaluation_metric_alias_and_default(self):
        # alias AUC -> roc_auc
        raw = '{"task_type": "classification", "evaluation_metric": "AUC"}'
        self.assertEqual(parse_module1_output(raw, "q")["evaluation_metric"], "roc_auc")
        # unmentioned / invalid -> accuracy
        self.assertEqual(parse_module1_output('{"task_type": "classification"}', "q")["evaluation_metric"], "accuracy")
        self.assertEqual(
            parse_module1_output('{"task_type": "classification", "evaluation_metric": "top5"}', "q")["evaluation_metric"],
            "accuracy",
        )


class TestParseDatasetId(unittest.TestCase):

    def test_plain_id(self):
        self.assertEqual(parse_dataset_id("uoft-cs/cifar10"), ("uoft-cs/cifar10", None))

    def test_with_subset(self):
        self.assertEqual(
            parse_dataset_id("antofuller/mini-vtab:eurosat"),
            ("antofuller/mini-vtab", "eurosat"),
        )

    def test_local_path_no_colon(self):
        self.assertEqual(parse_dataset_id("./my_images"), ("./my_images", None))


if __name__ == "__main__":
    unittest.main(verbosity=2)

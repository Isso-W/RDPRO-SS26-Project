"""
Pipeline 整合层测试
覆盖: data_size 推断、class_imbalance 推断、Module 1+2 合并、Module 1 解析容错

运行: python -m pytest test_pipeline.py -v
      或 python test_pipeline.py
"""

import unittest

from pipeline import derive_data_size, derive_class_imbalance, merge_modules
from features_extraction_api import parse_module1_output


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


class TestDeriveClassImbalance(unittest.TestCase):

    def test_balanced(self):
        dist = {"cat": 500, "dog": 480, "bird": 520}
        self.assertFalse(derive_class_imbalance(dist))

    def test_imbalanced(self):
        dist = {"cat": 5000, "dog": 50, "bird": 4000}
        self.assertTrue(derive_class_imbalance(dist))

    def test_exactly_at_threshold(self):
        # max/min = 10，不超过阈值
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
        """data_size 应该由 Module 2 的 total_images 决定，而非 Module 1 的占位值"""
        m1 = self._make_m1()
        m2 = self._make_m2(total_images=500)
        merged = merge_modules(m1, m2)
        self.assertEqual(merged["data_size"], "small")

    def test_module1_fields_preserved(self):
        """task_type / priority / description 来自 Module 1，合并后不变"""
        m1 = self._make_m1(task_type="object_detection", priority="speed")
        m2 = self._make_m2()
        merged = merge_modules(m1, m2)
        self.assertEqual(merged["task_type"], "object_detection")
        self.assertEqual(merged["priority"], "speed")
        self.assertEqual(merged["description"], "test query")

    def test_imbalance_from_module2(self):
        """Module 1 没检测到不平衡，但 Module 2 数据显示不平衡 → True"""
        m1 = self._make_m1()
        m2 = self._make_m2(class_dist={"a": 5000, "b": 10})
        merged = merge_modules(m1, m2)
        self.assertTrue(merged["constraints"]["class_imbalance"])

    def test_imbalance_from_module1(self):
        """Module 1 用户说了不平衡，Module 2 数据均衡 → 仍然 True（OR 逻辑）"""
        m1 = self._make_m1()
        m1["constraints"]["class_imbalance"] = True
        m2 = self._make_m2(class_dist={"a": 500, "b": 500})
        merged = merge_modules(m1, m2)
        self.assertTrue(merged["constraints"]["class_imbalance"])

    def test_other_constraints_untouched(self):
        """Module 2 不影响 medical / cross_modal 等字段"""
        m1 = self._make_m1()
        m1["constraints"]["medical"] = True
        m2 = self._make_m2()
        merged = merge_modules(m1, m2)
        self.assertTrue(merged["constraints"]["medical"])
        self.assertFalse(merged["constraints"]["cross_modal"])


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


if __name__ == "__main__":
    unittest.main(verbosity=2)

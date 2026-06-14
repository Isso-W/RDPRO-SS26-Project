"""Tests for the accumulating/explainable recommender layer."""

import math
import tempfile
import unittest
from pathlib import Path

from recommender.fingerprint import dataset_fingerprint, fingerprint_distance
from recommender.outcome_memory import OutcomeMemory
from recommender.ranker import rank_candidates


class TestFingerprint(unittest.TestCase):
    def test_derives_signals_from_m2_report(self):
        m2 = {
            "num_classes": 5,
            "total_images": 9000,
            "avg_width": 600, "avg_height": 800,
            "mode_distribution": {"RGB": 9000},
        }
        m3 = {"task_type": "classification", "data_size": "medium",
              "constraints": {"class_imbalance": True}}
        fp = dataset_fingerprint(m2, m3)
        self.assertEqual(fp["task_type"], "classification")
        self.assertEqual(fp["num_classes"], 5)
        self.assertEqual(fp["resolution_tier"], "high")   # avg(600,800)=700
        self.assertEqual(fp["color_mode"], "rgb")
        self.assertTrue(fp["class_imbalance"])

    def test_distance_task_gate_and_ordering(self):
        a = {"task_type": "classification", "num_classes": 5, "data_size": "medium",
             "resolution_tier": "high", "class_imbalance": True, "color_mode": "rgb"}
        same = dict(a)
        diff_task = {**a, "task_type": "object_detection"}
        more_classes = {**a, "num_classes": 500}
        self.assertEqual(fingerprint_distance(a, same), 0.0)
        self.assertEqual(fingerprint_distance(a, diff_task), math.inf)
        self.assertGreater(fingerprint_distance(a, more_classes), 0.0)


class TestMemoryRanking(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "outcomes.jsonl"
        self.mem = OutcomeMemory(self.tmp)
        self.target = {"task_type": "classification", "num_classes": 5, "data_size": "medium",
                       "resolution_tier": "high", "class_imbalance": True, "color_mode": "rgb"}

        def fp(num_classes, data_size):
            return {"task_type": "classification", "num_classes": num_classes, "data_size": data_size,
                    "resolution_tier": "high", "class_imbalance": True, "color_mode": "rgb"}

        # efficientnet: strong on a very-similar dataset, weaker on a far one
        self.mem.log(fp(5, "medium"), {"backbone": "efficientnet"}, {"metric_value": 0.86}, "beans")
        self.mem.log(fp(200, "large"), {"backbone": "efficientnet"}, {"metric_value": 0.60}, "inat")
        # resnet: ok on a similar dataset
        self.mem.log(fp(5, "medium"), {"backbone": "resnet"}, {"metric_value": 0.70}, "cassava")
        # different task — must be ignored
        self.mem.log({"task_type": "object_detection", "num_classes": 5, "data_size": "medium",
                      "resolution_tier": "high", "class_imbalance": True, "color_mode": "rgb"},
                     {"backbone": "efficientnet"}, {"metric_value": 0.99}, "coco")

    def test_memory_overrides_heuristic_order(self):
        # Module 3 heuristic ranked resnet ABOVE efficientnet; dinov2 has no track record.
        candidates = [
            {"backbone": "resnet", "pretrained": "resnet50_imagenet", "score": 0.80},
            {"backbone": "efficientnet", "pretrained": "efficientnet_b0_imagenet", "score": 0.70},
            {"backbone": "dinov2", "pretrained": "dinov2_base", "score": 0.65},
        ]
        ranked = rank_candidates(candidates, self.target, self.mem, k=5)
        order = [c["backbone"] for c in ranked]
        # efficientnet's strong track record on similar data flips it above resnet
        self.assertEqual(order[0], "efficientnet")
        self.assertEqual(order[1], "resnet")
        # dinov2 has no memory -> cold start, ranked last
        self.assertEqual(order[2], "dinov2")
        self.assertEqual(ranked[2]["rank_basis"], "heuristic")
        self.assertGreater(ranked[0]["predicted_metric"], ranked[1]["predicted_metric"])

    def test_explanations_present(self):
        candidates = [{"backbone": "efficientnet", "pretrained": "efficientnet_b0_imagenet", "score": 0.7}]
        ranked = rank_candidates(candidates, self.target, self.mem, k=5)
        expl = ranked[0]["explanation"]
        self.assertIn("efficientnet", expl)
        self.assertIn("similar past run", expl)
        self.assertIn("beans", expl)   # closest dataset surfaced as evidence

    def test_cold_start_keeps_heuristic_order(self):
        empty = OutcomeMemory(Path(tempfile.mkdtemp()) / "empty.jsonl")
        candidates = [
            {"backbone": "resnet", "score": 0.8},
            {"backbone": "efficientnet", "score": 0.7},
        ]
        ranked = rank_candidates(candidates, self.target, empty, k=5)
        self.assertEqual([c["backbone"] for c in ranked], ["resnet", "efficientnet"])
        self.assertTrue(all(c["rank_basis"] == "heuristic" for c in ranked))


class TestPipelineGlue(unittest.TestCase):
    """Re-ranked candidates must stay compatible with downstream task-list building."""

    def test_preserves_retrieval_output_keys(self):
        # shape mimics retrieve_top3_hybrid output
        candidates = [
            {"backbone": "efficientnet", "head": "classification_head", "loss": "cross_entropy_loss",
             "optimizer": "adam", "pretrained": "efficientnet_b0_imagenet", "finetune_strategy": "either",
             "freeze_viable": True, "scratch_viable": True, "alt_backbones": [], "score": 0.7,
             "score_detail": {"structured": 0.7, "vector": 0.6}},
            {"backbone": "resnet", "head": "classification_head", "loss": "cross_entropy_loss",
             "optimizer": "adam", "pretrained": "resnet50_imagenet", "finetune_strategy": "either",
             "freeze_viable": True, "scratch_viable": True, "alt_backbones": [], "score": 0.8,
             "score_detail": {"structured": 0.8, "vector": 0.6}},
        ]
        fp = {"task_type": "classification", "num_classes": 5, "data_size": "medium",
              "resolution_tier": "high", "class_imbalance": False, "color_mode": "rgb"}
        empty = OutcomeMemory(Path(tempfile.mkdtemp()) / "m.jsonl")
        ranked = rank_candidates(candidates, fp, empty, k=5)
        required = {"backbone", "head", "loss", "optimizer", "pretrained",
                    "finetune_strategy", "freeze_viable", "scratch_viable", "alt_backbones", "score"}
        for r in ranked:
            self.assertTrue(required.issubset(r.keys()))
            self.assertIn("explanation", r)
            self.assertIn("rank_basis", r)


class TestOutcomeLogging(unittest.TestCase):
    def test_log_from_summary(self):
        from recommender.ranker import log_from_summary

        m2 = {"num_classes": 5, "total_images": 9000, "avg_width": 600, "avg_height": 800,
              "mode_distribution": {"RGB": 9000}}
        m3 = {"task_type": "classification", "data_size": "medium", "constraints": {}}
        summary = {
            "status": "success",
            "config": {"backbone": "efficientnet"},
            "evaluate": {"metric_name": "accuracy", "metric_value": 0.83, "macro_f1": 0.79, "status": "success"},
            "train": {"loss": 0.4},
        }
        mem = OutcomeMemory(Path(tempfile.mkdtemp()) / "m.jsonl")
        fp = log_from_summary(summary, m2, m3, config={"backbone": "efficientnet", "params_M": 5},
                              dataset_id="cassava", memory=mem)
        self.assertIsNotNone(fp)
        records = mem.all()
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["result"]["metric_value"], 0.83)
        self.assertEqual(records[0]["config"]["backbone"], "efficientnet")
        self.assertEqual(records[0]["dataset_id"], "cassava")

    def test_log_from_summary_skips_missing_metric(self):
        from recommender.ranker import log_from_summary

        mem = OutcomeMemory(Path(tempfile.mkdtemp()) / "m.jsonl")
        out = log_from_summary({"evaluate": {"metric_value": None}}, {}, {}, memory=mem)
        self.assertIsNone(out)
        self.assertEqual(mem.all(), [])


class TestSummaryParsing(unittest.TestCase):
    def test_extract_last_json(self):
        from run_and_log import extract_last_json

        text = (
            "[train] epoch 1/3 loss=0.5\n"
            "[train] Done.\n"
            '{\n  "status": "success",\n  "evaluate": {"metric_value": 0.81}\n}\n'
        )
        obj = extract_last_json(text)
        self.assertEqual(obj["status"], "success")
        self.assertEqual(obj["evaluate"]["metric_value"], 0.81)

    def test_extract_last_json_none_when_absent(self):
        from run_and_log import extract_last_json

        self.assertIsNone(extract_last_json("just logs, no json here"))


class TestCostMeter(unittest.TestCase):
    def test_accumulates_and_reports(self):
        import cost_meter
        cost_meter.reset()
        cost_meter.record_llm_call(120)
        cost_meter.record_llm_call(80)
        cost_meter.record_training(epochs=10, runs=1)
        r = cost_meter.report()
        self.assertEqual(r["llm_calls"], 2)
        self.assertEqual(r["llm_tokens"], 200)
        self.assertEqual(r["epochs"], 10)
        self.assertEqual(r["train_runs"], 1)
        self.assertIn("wall_clock_sec", r)

    def test_tokens_from_response_variants(self):
        import cost_meter

        class _Usage:
            total_tokens = 321

        class _Resp:
            usage = _Usage()

        self.assertEqual(cost_meter.tokens_from_response(_Resp()), 321)
        self.assertEqual(cost_meter.tokens_from_response({"usage": {"total_tokens": 7}}), 7)
        self.assertEqual(cost_meter.tokens_from_response(object()), 0)

    def test_cost_stored_in_memory_record(self):
        from recommender.ranker import log_from_summary

        summary = {"evaluate": {"metric_name": "accuracy", "metric_value": 0.7}}
        mem = OutcomeMemory(Path(tempfile.mkdtemp()) / "m.jsonl")
        log_from_summary(summary, {}, {}, config={"backbone": "resnet"}, memory=mem,
                         cost={"llm_calls": 1, "llm_tokens": 250, "epochs": 10, "wall_clock_sec": 42.0})
        self.assertEqual(mem.all()[0]["cost"]["llm_tokens"], 250)


if __name__ == "__main__":
    unittest.main()

"""Golden-case regression suite for Module 3 retrieval quality.

Each case pins one "query -> expected recommendation" baseline. Run this suite
after changing KB data or scoring logic to catch retrieval-quality regressions.

Assertions use two strengths:
  - top-1/component assertions for high-confidence cases
  - top-3 membership assertions where vector ranking can reasonably vary

Run with:
    python -m pytest retrieval/test_golden.py -q
or:
    cd retrieval && python -m pytest test_golden.py -q
"""

import unittest

if __package__:
    from .rag_retrieval import build_graph, build_vector_index, retrieve_top3_hybrid
else:
    from rag_retrieval import build_graph, build_vector_index, retrieve_top3_hybrid


# Shared retrieval resources, built once for all cases.

G = None
COL = None


def setUpModule():
    global G, COL
    G = build_graph()
    COL = build_vector_index()


def _query(task_type, data_size, priority, constraints=None, description=""):
    return {
        "task_type": task_type,
        "data_size": data_size,
        "priority": priority,
        "constraints": constraints or {},
        "description": description,
    }


def _backbones(results):
    return [r["backbone"] for r in results]


def _checkpoint_tiers(results):
    return [
        G.nodes[r["pretrained"]].get("size_tier")
        for r in results
        if r["pretrained"]
    ]


# Golden cases.

GOLDEN_INPUTS = {
    "edge_realtime_detection": _query(
        "object_detection", "small", "speed",
        {"real_time": True, "edge_deployment": True},
        "detect product defects on assembly line camera, Jetson Nano",
    ),
    "medical_seg_small": _query(
        "image_segmentation", "small", "accuracy",
        {"medical": True},
        "segment tumors in MRI scans, limited labeled data",
    ),
    "large_acc_classification": _query(
        "classification", "large", "accuracy",
        {},
        "fine-grained bird species classification, 200 classes",
    ),
    "zero_shot_classification": _query(
        "classification", "small", "balanced",
        {"zero_shot": True},
        "classify retail products without labeled training data",
    ),
    "zero_shot_cross_modal": _query(
        "classification", "small", "balanced",
        {"zero_shot": True, "cross_modal": True},
        "zero-shot open vocabulary classification",
    ),
    "few_shot_classification": _query(
        "classification", "small", "balanced",
        {"few_shot": True},
        "classify with only 10 labeled examples per class",
    ),
    "cross_modal_feature_extraction": _query(
        "feature_extraction", "medium", "balanced",
        {"cross_modal": True},
        "image-text retrieval for product search",
    ),
    "plain_small_classification": _query(
        "classification", "small", "balanced",
        {},
        "classify flower photos",
    ),
    "large_acc_detection": _query(
        "object_detection", "large", "accuracy",
        {},
        "detect vehicles in surveillance footage",
    ),
}


class TestGoldenCases(unittest.TestCase):
    """Scenario-level expectations for representative queries."""

    @classmethod
    def setUpClass(cls):
        cls.results = {
            name: retrieve_top3_hybrid(q, G, COL)
            for name, q in GOLDEN_INPUTS.items()
        }

    # 1. Edge real-time detection: YOLOv8 nano should be selected.
    def test_edge_realtime_detection(self):
        res = self.results["edge_realtime_detection"]
        self.assertEqual(_backbones(res)[0], "yolov8")
        self.assertEqual(res[0]["pretrained"], "yolov8n_coco")
        for tier in _checkpoint_tiers(res):
            self.assertIn(tier, {"nano", "small"})

    # 2. Small medical segmentation: UNet with dice loss.
    def test_medical_seg_small(self):
        res = self.results["medical_seg_small"]
        self.assertEqual(_backbones(res)[0], "unet")
        self.assertEqual(res[0]["loss"], "dice_loss")

    # 3. Large high-accuracy classification: self-supervised/base models appear,
    # and nano checkpoints are not recommended.
    def test_large_acc_classification(self):
        res = self.results["large_acc_classification"]
        backbones = _backbones(res)
        self.assertIn("dinov3", backbones)
        self.assertIn("dinov2", backbones)
        for tier in _checkpoint_tiers(res):
            self.assertNotIn(tier, {"nano"})

    # 4. Zero-shot classification: CLIP should appear, and every candidate must
    # advertise zero_shot capability.
    def test_zero_shot_classification_includes_clip(self):
        res = self.results["zero_shot_classification"]
        self.assertIn("clip_vit", _backbones(res))
        for r in res:
            self.assertIn(
                "zero_shot", G.nodes[r["backbone"]].get("capabilities", [])
            )

    # 5. Zero-shot + cross-modal: SigLIP2 is the preferred runtime choice.
    def test_zero_shot_cross_modal_top1_siglip2(self):
        res = self.results["zero_shot_cross_modal"]
        self.assertEqual(_backbones(res)[0], "siglip2")

    # 6. Few-shot classification: DINOv2 should be preferred with head-only tuning.
    def test_few_shot_classification_top1_dinov2(self):
        res = self.results["few_shot_classification"]
        self.assertEqual(_backbones(res)[0], "dinov2")
        self.assertEqual(res[0]["finetune_strategy"], "head_only")

    # 7. Cross-modal feature extraction: SigLIP2 with contrastive loss.
    def test_cross_modal_feature_extraction(self):
        res = self.results["cross_modal_feature_extraction"]
        self.assertEqual(_backbones(res)[0], "siglip2")
        self.assertEqual(res[0]["loss"], "infonce_loss")

    # 8. Plain small-data classification: use a pretrained checkpoint and do not
    # exceed the base tier.
    def test_plain_small_classification(self):
        res = self.results["plain_small_classification"]
        self.assertGreaterEqual(len(res), 2)
        for r in res:
            self.assertIsNotNone(
                r["pretrained"],
                f"{r['backbone']} was recommended from scratch for a small-data case",
            )
        for tier in _checkpoint_tiers(res):
            self.assertIn(tier, {"nano", "small", "base"})

    # 9. Large high-accuracy detection: dedicated YOLO detectors should appear.
    # Vector scores can affect order, so this checks membership only.
    def test_large_acc_detection_has_dedicated_detectors(self):
        res = self.results["large_acc_detection"]
        backbones = _backbones(res)
        self.assertIn("yolo26", backbones)
        self.assertIn("yolov8", backbones)


class TestGoldenInvariants(unittest.TestCase):
    """Structural invariants that must hold for every golden query."""

    @classmethod
    def setUpClass(cls):
        cls.results = {
            name: retrieve_top3_hybrid(q, G, COL)
            for name, q in GOLDEN_INPUTS.items()
        }

    def test_at_most_three_results(self):
        for name, res in self.results.items():
            with self.subTest(scenario=name):
                self.assertLessEqual(len(res), 3)
                self.assertGreaterEqual(len(res), 1)

    def test_scores_descending(self):
        for name, res in self.results.items():
            with self.subTest(scenario=name):
                scores = [r["score"] for r in res]
                self.assertEqual(scores, sorted(scores, reverse=True))

    def test_no_duplicate_backbones(self):
        for name, res in self.results.items():
            with self.subTest(scenario=name):
                backbones = _backbones(res)
                self.assertEqual(len(backbones), len(set(backbones)))

    def test_backbone_supports_task(self):
        for name, res in self.results.items():
            task = GOLDEN_INPUTS[name]["task_type"]
            with self.subTest(scenario=name):
                for r in res:
                    self.assertIn(task, G.nodes[r["backbone"]]["task_type"])

    def test_pretrained_or_scratch_viable(self):
        """Every candidate either has a checkpoint or is marked scratch-viable."""
        for name, res in self.results.items():
            with self.subTest(scenario=name):
                for r in res:
                    self.assertTrue(
                        r["pretrained"] is not None or r["scratch_viable"],
                        f"{r['backbone']} has no checkpoint and is not scratch-viable",
                    )


if __name__ == "__main__":
    unittest.main()

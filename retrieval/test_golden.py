"""Golden-case regression suite for Module 3 retrieval quality.

每个用例固化一个"查询 → 期望推荐"的基线。改动 KB 数据或打分逻辑后跑这套
测试，能立刻看出推荐质量是变好还是变坏。

断言强度分两档：
  - top1 / 组件断言：对答案有高置信度的场景才用
  - 成员资格断言（in top3）：排序受向量分影响、合理性存疑的场景只锁成员

运行方式：
    python -m pytest retrieval/test_golden.py -q
或：
    cd retrieval && python -m pytest test_golden.py -q
"""

import unittest

if __package__:
    from .rag_retrieval import build_graph, build_vector_index, retrieve_top3_hybrid
else:
    from rag_retrieval import build_graph, build_vector_index, retrieve_top3_hybrid


# ── 共享检索资源（构建一次，所有用例复用）──────────────────────────────────

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


# ── 黄金用例 ────────────────────────────────────────────────────────────────

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
    """场景级期望：固化每个典型查询的关键推荐结论。"""

    @classmethod
    def setUpClass(cls):
        cls.results = {
            name: retrieve_top3_hybrid(q, G, COL)
            for name, q in GOLDEN_INPUTS.items()
        }

    # 1. 边缘实时检测：yolov8 nano 是唯一正确答案，且禁止越带 checkpoint
    def test_edge_realtime_detection(self):
        res = self.results["edge_realtime_detection"]
        self.assertEqual(_backbones(res)[0], "yolov8")
        self.assertEqual(res[0]["pretrained"], "yolov8n_coco")
        for tier in _checkpoint_tiers(res):
            self.assertIn(tier, {"nano", "small"})

    # 2. 医学小数据分割：UNet + dice loss
    def test_medical_seg_small(self):
        res = self.results["medical_seg_small"]
        self.assertEqual(_backbones(res)[0], "unet")
        self.assertEqual(res[0]["loss"], "dice_loss")

    # 3. 大数据高精度分类：自监督/大模型方案在列，且不推 nano
    def test_large_acc_classification(self):
        res = self.results["large_acc_classification"]
        backbones = _backbones(res)
        self.assertIn("dinov3", backbones)
        self.assertIn("dinov2", backbones)
        for tier in _checkpoint_tiers(res):
            self.assertNotIn(tier, {"nano"})

    # 4. 零样本分类：CLIP 必须在列（它是零样本分类的教科书答案），
    #    且所有候选都具备 zero_shot capability
    def test_zero_shot_classification_includes_clip(self):
        res = self.results["zero_shot_classification"]
        self.assertIn("clip_vit", _backbones(res))
        for r in res:
            self.assertIn(
                "zero_shot", G.nodes[r["backbone"]].get("capabilities", [])
            )

    # 5. 零样本 + 跨模态：SigLIP2 是当前运行时的首选跨模态模型
    def test_zero_shot_cross_modal_top1_siglip2(self):
        res = self.results["zero_shot_cross_modal"]
        self.assertEqual(_backbones(res)[0], "siglip2")

    # 6. 少样本分类：DINOv2 应为首选（few_shot capability + 冻结骨干策略）
    def test_few_shot_classification_top1_dinov2(self):
        res = self.results["few_shot_classification"]
        self.assertEqual(_backbones(res)[0], "dinov2")
        self.assertEqual(res[0]["finetune_strategy"], "head_only")

    # 7. 跨模态特征提取：SigLIP2 首选 + 对比学习损失
    def test_cross_modal_feature_extraction(self):
        res = self.results["cross_modal_feature_extraction"]
        self.assertEqual(_backbones(res)[0], "siglip2")
        self.assertEqual(res[0]["loss"], "infonce_loss")

    # 8. 普通小数据分类：必须有预训练 checkpoint（小数据不该从头训练），
    #    且 checkpoint 不越过 base 档
    def test_plain_small_classification(self):
        res = self.results["plain_small_classification"]
        self.assertGreaterEqual(len(res), 2)
        for r in res:
            self.assertIsNotNone(
                r["pretrained"],
                f"{r['backbone']} 在小数据场景被推荐从头训练",
            )
        for tier in _checkpoint_tiers(res):
            self.assertIn(tier, {"nano", "small", "base"})

    # 9. 大数据高精度检测：YOLO26/YOLOv8 专用检测器必须在列。
    #    排序受向量分影响存疑（resnet 曾靠 v=1.0 登顶），只锁成员资格
    def test_large_acc_detection_has_dedicated_detectors(self):
        res = self.results["large_acc_detection"]
        backbones = _backbones(res)
        self.assertIn("yolo26", backbones)
        self.assertIn("yolov8", backbones)


class TestGoldenInvariants(unittest.TestCase):
    """结构不变量：对所有黄金输入都必须成立的性质。"""

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
        """每个候选要么有 checkpoint，要么明确标记可从头训练。"""
        for name, res in self.results.items():
            with self.subTest(scenario=name):
                for r in res:
                    self.assertTrue(
                        r["pretrained"] is not None or r["scratch_viable"],
                        f"{r['backbone']} 既无 checkpoint 也不可从头训练",
                    )


if __name__ == "__main__":
    unittest.main()

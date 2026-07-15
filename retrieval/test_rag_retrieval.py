"""
RAG retrieval test suite.

Run with: python test_rag_retrieval.py
"""

import unittest

if __package__:
    from .rag_retrieval import (
        build_graph,
        build_vector_index,
        retrieve_top3_hybrid,
        build_task_list,
        build_all_task_lists,
        MODULE1_EXAMPLES,
        _SIZE_TIER_ORDER,
        estimate_cost,
    )
else:
    from rag_retrieval import (
        build_graph,
        build_vector_index,
        retrieve_top3_hybrid,
        build_task_list,
        build_all_task_lists,
        MODULE1_EXAMPLES,
        _SIZE_TIER_ORDER,
        estimate_cost,
    )

_REQUIRED_FIELDS = {
    "backbone", "head", "loss", "optimizer",
    "pretrained", "scratch_viable", "finetune_strategy", "freeze_viable",
    "alt_backbones", "score", "score_detail",
}


def _make_input(
    task_type="classification",
    data_size="medium",
    priority="balanced",
    description="",
    **constraints,
):
    base = {"real_time": False, "edge_deployment": False, "class_imbalance": False,
            "cross_modal": False, "medical": False}
    base.update(constraints)
    return {
        "task_type":   task_type,
        "data_size":   data_size,
        "priority":    priority,
        "constraints": base,
        "description": description,
    }


def _checkpoint_tier(result, G):
    pid = result.get("pretrained")
    if pid is None:
        return None
    return G.nodes[pid].get("size_tier")


class TestSmoke(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.G   = build_graph()
        cls.col = build_vector_index()

    def _run(self, input_json):
        return retrieve_top3_hybrid(input_json, self.G, self.col)

    def test_module1_examples_no_crash(self):
        for name, inp in MODULE1_EXAMPLES.items():
            with self.subTest(example=name):
                results = self._run(inp)
                self.assertIsInstance(results, list, f"{name} did not return a list")

    def test_module1_examples_have_results(self):
        for name, inp in MODULE1_EXAMPLES.items():
            with self.subTest(example=name):
                results = self._run(inp)
                self.assertGreater(len(results), 0, f"{name} returned no results")

    def test_result_fields_complete(self):
        for name, inp in MODULE1_EXAMPLES.items():
            results = self._run(inp)
            for i, r in enumerate(results):
                with self.subTest(example=name, rank=i + 1):
                    missing = _REQUIRED_FIELDS - r.keys()
                    self.assertFalse(missing, f"Missing fields: {missing}")

    def test_scores_in_range(self):
        for name, inp in MODULE1_EXAMPLES.items():
            results = self._run(inp)
            for r in results:
                with self.subTest(example=name, backbone=r["backbone"]):
                    self.assertGreaterEqual(r["score"], 0.0)
                    self.assertLessEqual(r["score"], 1.0)

    def test_new_models_added_without_removing_existing_models(self):
        expected = {
            "dinov2", "dinov3",
            "yolov8", "yolo26",
            "clip_vit", "siglip2",
            "sam3",
        }
        self.assertTrue(expected <= set(self.G.nodes))


class TestBehavior(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.G   = build_graph()
        cls.col = build_vector_index()

    def _run(self, input_json):
        return retrieve_top3_hybrid(input_json, self.G, self.col)

    def _backbones(self, results):
        return [r["backbone"] for r in results]

    # ── Test 1: edge_deployment → only nano/small checkpoints ─────────────────

    def test_edge_deployment_checkpoint_size(self):
        inp = _make_input(
            task_type="object_detection",
            data_size="medium",
            priority="speed",
            edge_deployment=True,
        )
        results = self._run(inp)
        self.assertGreater(len(results), 0)
        allowed = {"nano", "small", None}
        for r in results:
            tier = _checkpoint_tier(r, self.G)
            self.assertIn(tier, allowed,
                          f"{r['backbone']} got checkpoint tier '{tier}' under edge_deployment")

    # ── Test 2: real_time → only nano/small checkpoints ───────────────────────

    def test_real_time_checkpoint_size(self):
        inp = _make_input(
            task_type="object_detection",
            data_size="medium",
            priority="speed",
            real_time=True,
        )
        results = self._run(inp)
        self.assertGreater(len(results), 0)
        allowed = {"nano", "small", None}
        for r in results:
            tier = _checkpoint_tier(r, self.G)
            self.assertIn(tier, allowed,
                          f"{r['backbone']} got checkpoint tier '{tier}' under real_time")

    # ── Test 3: speed priority → no accuracy_upgrade backbones ───────────────

    def test_speed_priority_excludes_accuracy_upgrade(self):
        inp = _make_input(
            task_type="object_detection",
            data_size="medium",
            priority="speed",
        )
        results = self._run(inp)
        accuracy_upgrade = {"detr", "rt_detr"}
        found = accuracy_upgrade & set(self._backbones(results))
        self.assertFalse(found,
                         f"accuracy_upgrade backbones appeared with priority=speed: {found}")

    # ── Test 4: accuracy + large → Mask2Former appears ────────────────────────

    def test_accuracy_large_includes_mask2former(self):
        inp = _make_input(
            task_type="image_segmentation",
            data_size="large",
            priority="accuracy",
        )
        results = self._run(inp)
        self.assertIn("mask2former", self._backbones(results),
                      "Mask2Former should appear for accuracy + large segmentation")

    # ── Test 5: medical → UNet appears ────────────────────────────────────────

    def test_medical_flag_includes_unet(self):
        inp = _make_input(
            task_type="image_segmentation",
            data_size="small",
            priority="balanced",
            medical=True,
        )
        results = self._run(inp)
        self.assertIn("unet", self._backbones(results),
                      "UNet should appear when medical=True")

    # ── Test 6: cross_modal → SigLIP2 ranks first ─────────────────────────────

    def test_cross_modal_siglip2_top1(self):
        inp = _make_input(
            task_type="feature_extraction",
            data_size="medium",
            priority="accuracy",
            cross_modal=True,
        )
        results = self._run(inp)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["backbone"], "siglip2",
                         f"Expected SigLIP2 at top, got {results[0]['backbone']}")

    # ── Test 7: no cross_modal → CLIP excluded ────────────────────────────────

    def test_no_cross_modal_excludes_clip(self):
        inp = _make_input(
            task_type="feature_extraction",
            data_size="medium",
            priority="balanced",
            cross_modal=False,
        )
        results = self._run(inp)
        self.assertNotIn("clip_vit", self._backbones(results),
                         "CLIP ViT should not appear without cross_modal=True")

    # ── Test 8: DETR/RT-DETR → head must be detection_head_transformer ────────

    def test_detr_requires_transformer_head(self):
        inp = _make_input(
            task_type="object_detection",
            data_size="large",
            priority="accuracy",
        )
        results = self._run(inp)
        for r in results:
            if r["backbone"] in ("detr", "rt_detr"):
                self.assertEqual(
                    r["head"], "detection_head_transformer",
                    f"{r['backbone']} should have detection_head_transformer, got {r['head']}"
                )

    # ── Test 9: DINOv2 checkpoint → finetune_strategy = head_only ─────────────

    def test_dinov2_finetune_strategy(self):
        # feature_extraction → frozen (head_only): frozen features are the point
        inp = _make_input(
            task_type="feature_extraction",
            data_size="medium",
            priority="balanced",
        )
        results = self._run(inp)
        for r in results:
            if r["backbone"] == "dinov2" and r["pretrained"] is not None:
                self.assertEqual(
                    r["finetune_strategy"], "head_only",
                    f"DINOv2 (feature_extraction) should be head_only, got {r['finetune_strategy']}"
                )

    def test_dinov2_full_finetune_on_medium_classification(self):
        # classification + enough data → full finetune (resolved from "either"),
        # so it competes fairly instead of being a frozen linear probe
        inp = _make_input(task_type="classification", data_size="medium", priority="accuracy",
                          description="fine-grained classification")
        for r in self._run(inp):
            if r["backbone"] == "dinov2" and r["pretrained"] is not None:
                self.assertEqual(r["finetune_strategy"], "full")

    def test_dinov2_frozen_when_few_shot(self):
        # few-shot → frozen, avoid overfitting a big ViT on tiny data
        inp = _make_input(task_type="classification", data_size="medium", priority="accuracy",
                          few_shot=True, description="few shot")
        for r in self._run(inp):
            if r["backbone"] == "dinov2" and r["pretrained"] is not None:
                self.assertEqual(r["finetune_strategy"], "head_only")

    def test_dinov3_classification_uses_partial_finetune(self):
        inp = _make_input(
            task_type="classification",
            data_size="medium",
            priority="accuracy",
            description="fine-grained visual classification where feature quality matters",
        )
        for r in self._run(inp):
            if r["backbone"] == "dinov3" and r["pretrained"] is not None:
                self.assertEqual(r["finetune_strategy"], "partial")
                self.assertIn(r["unfreeze_last_n_blocks"], {2, 4})
                self.assertTrue(r["train_norm_layers"])
                return
        self.fail("DINOv3 recommendation not found")

    # ── Test 10: class_imbalance → focal_loss selected ────────────────────────

    def test_class_imbalance_uses_focal_loss(self):
        inp = _make_input(
            task_type="classification",
            data_size="medium",
            priority="balanced",
            class_imbalance=True,
        )
        results = self._run(inp)
        self.assertGreater(len(results), 0)
        top = results[0]
        self.assertEqual(top["loss"], "focal_loss",
                         f"Expected focal_loss with class_imbalance, got {top['loss']}")

    # ── Test 11: invalid task_type → empty list ────────────────────────────────

    def test_invalid_task_type_returns_empty(self):
        inp = _make_input(task_type="nonexistent_task")
        results = self._run(inp)
        self.assertEqual(results, [], "Invalid task_type should return empty list")

    # ── Test 12: zero_shot → only capable backbones ────────────────────────────

    def test_zero_shot_filters_to_capable_backbones(self):
        inp = _make_input(
            task_type="feature_extraction",
            data_size="small",
            priority="balanced",
            zero_shot=True,
        )
        results = self._run(inp)
        self.assertGreater(len(results), 0, "zero_shot should return at least one result")
        capable = {"dinov2", "dinov3", "clip_vit", "siglip2", "sam3"}
        for r in results:
            self.assertIn(r["backbone"], capable,
                          f"{r['backbone']} lacks zero_shot capability but appeared in results")

    # ── Test 13: few_shot → capable backbones ranked higher ───────────────────

    def test_few_shot_boosts_capable_backbones(self):
        inp = _make_input(
            task_type="feature_extraction",
            data_size="small",
            priority="balanced",
            few_shot=True,
        )
        results = self._run(inp)
        self.assertGreater(len(results), 0)
        capable = {"dinov2", "dinov3", "clip_vit", "siglip2", "sam3"}
        # At least one few-shot-capable backbone should appear.
        found = capable & {r["backbone"] for r in results}
        self.assertTrue(found, f"No few_shot-capable backbone in results: {self._backbones(results)}")

    # ── Test 14: real-time detection → YOLO26 is available ───────────────────

    def test_real_time_detection_includes_yolo26(self):
        inp = _make_input(
            task_type="object_detection",
            data_size="medium",
            priority="speed",
            real_time=True,
        )
        results = self._run(inp)
        self.assertEqual(results[0]["backbone"], "yolo26")
        self.assertIn("yolo26", self._backbones(results))
        yolo26 = next(r for r in results if r["backbone"] == "yolo26")
        self.assertIn(yolo26["pretrained"], {"yolo26n_coco", "yolo26s_coco"})

    # ── Test 15: DINOv3 appears for high-quality feature extraction ───────────

    def test_feature_quality_includes_dinov3(self):
        inp = _make_input(
            task_type="feature_extraction",
            data_size="large",
            priority="accuracy",
            few_shot=True,
            description="extract dense visual features for retrieval and few-shot classification",
        )
        results = self._run(inp)
        self.assertEqual(results[0]["backbone"], "dinov3")
        self.assertIn("dinov3", self._backbones(results))
        self.assertEqual(results[0]["finetune_strategy"], "head_only")

    # ── Test 16: cross-modal retrieval includes SigLIP2 ──────────────────────

    def test_cross_modal_includes_siglip2(self):
        inp = _make_input(
            task_type="feature_extraction",
            data_size="medium",
            priority="accuracy",
            cross_modal=True,
            description="image text retrieval with multilingual labels",
        )
        results = self._run(inp)
        self.assertIn("siglip2", self._backbones(results))

    # ── Test 17: promptable/open-vocabulary segmentation includes SAM3 ───────

    def test_promptable_segmentation_includes_sam3(self):
        inp = _make_input(
            task_type="image_segmentation",
            data_size="large",
            priority="accuracy",
            open_vocabulary=True,
            text_prompt=True,
            description="segment every object matching a text prompt in images and videos",
        )
        results = self._run(inp)
        self.assertIn("sam3", self._backbones(results))


class TestBudget(unittest.TestCase):
    """Constraint-aware selection for max_params_m / max_flops_g budgets."""

    @classmethod
    def setUpClass(cls):
        cls.G   = build_graph()
        cls.col = build_vector_index()

    def _run(self, input_json):
        return retrieve_top3_hybrid(input_json, self.G, self.col)

    def test_param_budget_excludes_over_budget_backbones(self):
        inp = _make_input(priority="balanced", max_params_m=12)
        results = self._run(inp)
        self.assertGreater(len(results), 0)
        for r in results:
            cost = estimate_cost(r.get("pretrained"), inp, self.G)
            if cost["params_m"] is not None:
                self.assertLessEqual(cost["params_m"], 12,
                                     f"{r['backbone']}/{r.get('pretrained')} over param budget")

    def test_param_budget_downgrades_checkpoint(self):
        # ResNet normally picks resnet50; a 12M budget should downgrade to resnet18.
        inp = _make_input(priority="balanced", max_params_m=12)
        picks = {r["backbone"]: r.get("pretrained") for r in self._run(inp)}
        if "resnet" in picks:
            self.assertEqual(picks["resnet"], "resnet18_imagenet")

    def test_flops_budget_excludes_over_budget(self):
        inp = _make_input(priority="balanced", max_flops_g=2.0)
        for r in self._run(inp):
            cost = estimate_cost(r.get("pretrained"), inp, self.G)
            if cost["flops_g"] is not None:
                self.assertLessEqual(cost["flops_g"], 2.0)

    def test_no_budget_keeps_heavy_models(self):
        # Without a budget, heavier backbones such as DINOv2 may appear.
        without = {r["backbone"] for r in self._run(_make_input(priority="balanced"))}
        within = {r["backbone"] for r in self._run(_make_input(priority="balanced", max_params_m=12))}
        # A budget narrows the candidate set.
        self.assertGreaterEqual(len(without), len(within))

    def test_flops_scales_with_image_size(self):
        small = estimate_cost("resnet18_imagenet", _make_input(image_size=224), self.G)
        large = estimate_cost("resnet18_imagenet", _make_input(image_size=448), self.G)
        # 448 has twice the side length of 224, so area scales by 4.
        self.assertAlmostEqual(large["flops_g"], small["flops_g"] * 4, delta=0.1)


class TestTaskList(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.G   = build_graph()
        cls.col = build_vector_index()
        # Use a cross-modal case so CLIP appears with a concrete checkpoint.
        cls.results = retrieve_top3_hybrid(
            _make_input(
                task_type="feature_extraction",
                data_size="medium",
                priority="accuracy",
                cross_modal=True,
            ),
            cls.G, cls.col,
        )
        cls.top = cls.results[0]

    # ── structured format ─────────────────────────────────────────────────────

    def test_structured_top_level_keys(self):
        tl = build_task_list(self.top, self.G, fmt="structured")
        for key in ("format", "backbone", "backbone_name", "tasks", "alternatives"):
            self.assertIn(key, tl, f"Missing key '{key}' in structured output")
        self.assertEqual(tl["format"], "structured")

    def test_structured_tasks_is_list(self):
        tl = build_task_list(self.top, self.G, fmt="structured")
        self.assertIsInstance(tl["tasks"], list)
        self.assertGreater(len(tl["tasks"]), 0)

    def test_structured_task_ids_unique(self):
        tl = build_task_list(self.top, self.G, fmt="structured")
        ids = [t["id"] for t in tl["tasks"]]
        self.assertEqual(len(ids), len(set(ids)), "Duplicate task IDs")

    def test_structured_load_model_task_has_hf_id(self):
        # top result has a checkpoint (CLIP)
        tl = build_task_list(self.top, self.G, fmt="structured")
        load = next(t for t in tl["tasks"] if t["id"] == "load_model")
        self.assertIn("hf_id", load)
        self.assertTrue(load["hf_id"])

    def test_structured_scratch_path_no_hf_id(self):
        # Build a scratch result with no pretrained checkpoint.
        fake = dict(self.top)
        fake["pretrained"] = None
        tl = build_task_list(fake, self.G, fmt="structured")
        load = next(t for t in tl["tasks"] if t["id"] == "load_model")
        self.assertEqual(load["action"], "train_from_scratch")
        self.assertNotIn("hf_id", load)

    # ── nl format ─────────────────────────────────────────────────────────────

    def test_nl_top_level_keys(self):
        tl = build_task_list(self.top, self.G, fmt="nl")
        for key in ("format", "model_config", "tasks", "alternatives"):
            self.assertIn(key, tl, f"Missing key '{key}' in nl output")
        self.assertEqual(tl["format"], "nl")

    def test_nl_tasks_are_strings(self):
        tl = build_task_list(self.top, self.G, fmt="nl")
        for t in tl["tasks"]:
            self.assertIsInstance(t, str, f"NL task is not a string: {t!r}")

    def test_nl_model_config_has_backbone(self):
        tl = build_task_list(self.top, self.G, fmt="nl")
        self.assertIn("backbone", tl["model_config"])

    def test_nl_model_config_attaches_classification_recipe(self):
        input_json = _make_input(
            task_type="classification",
            data_size="small",
            priority="accuracy",
            fine_grained=True,
        )
        input_json["data_stats"] = {"resolution_tier": "high", "color_mode": "rgb"}
        fake = {
            "backbone": "efficientnet",
            "pretrained": "efficientnet_b0_imagenet",
            "head": "classification_head",
            "loss": "cross_entropy_loss",
            "optimizer": "adamw",
            "scratch_viable": False,
            "finetune_strategy": "full",
            "freeze_viable": False,
            "alt_backbones": [],
            "score": 1.0,
        }

        tl = build_task_list(
            fake,
            self.G,
            fmt="nl",
            input_json=input_json,
            data_stats=input_json["data_stats"],
        )

        recipe = tl["model_config"]["recipe"]
        provenance = tl["model_config"]["recipe_provenance"]
        self.assertEqual(recipe["image_size"], 384)
        self.assertEqual(recipe["learning_rate"], 1.0e-4)
        self.assertEqual(recipe["epochs"], 40)
        self.assertEqual(recipe["augmentation"]["tier"], "heavy")
        self.assertIn("fine_grained+high_res bump", provenance["image_size"])

    def test_invalid_fmt_raises(self):
        with self.assertRaises(ValueError):
            build_task_list(self.top, self.G, fmt="xml")

    # ── build_all_task_lists ──────────────────────────────────────────────────

    def test_all_task_lists_length(self):
        tls = build_all_task_lists(self.results, self.G, fmt="structured")
        self.assertEqual(len(tls), len(self.results))

    def test_all_task_lists_rank_field(self):
        tls = build_all_task_lists(self.results, self.G, fmt="structured")
        for i, tl in enumerate(tls, 1):
            self.assertEqual(tl["rank"], i, f"rank mismatch at position {i}")

    def test_all_task_lists_score_field(self):
        tls = build_all_task_lists(self.results, self.G, fmt="nl")
        for tl in tls:
            self.assertIn("score", tl)
            self.assertIsNotNone(tl["score"])


class TestPhaseBLossEdges(unittest.TestCase):
    """Phase B: _select_components consumes preferred_when loss edges."""

    def _loss_for(self, graph, constraints):
        if __package__:
            from .rag_retrieval import _select_components
        else:
            from rag_retrieval import _select_components
        q = _make_input(task_type="classification", **constraints)
        return _select_components("resnet", "classification", q, graph)["loss"]

    def test_edge_path_fires_distinctly_from_hardcoded(self):
        # Prove the edge path is active, not only the fallback rule. Medical
        # normally falls back to cross_entropy; after adding a synthetic
        # focal->CE edge, focal can only come from graph-edge selection.
        base = self._loss_for(build_graph(), {"medical": True})
        self.assertEqual(base, "cross_entropy_loss")  # default without the edge

        g = build_graph()
        g.add_edge("focal_loss", "cross_entropy_loss",
                   relation="preferred_when", condition={"any": ["medical=True"]})
        self.assertEqual(self._loss_for(g, {"medical": True}), "focal_loss")

    def test_class_imbalance_picks_focal_via_edge(self):
        # The real KB edge focal->CE(class_imbalance) is consumed by Phase B.
        self.assertEqual(self._loss_for(build_graph(), {"class_imbalance": True}),
                         "focal_loss")

    def test_no_imbalance_falls_back_to_cross_entropy(self):
        self.assertEqual(self._loss_for(build_graph(), {}), "cross_entropy_loss")


if __name__ == "__main__":
    unittest.main(verbosity=2)

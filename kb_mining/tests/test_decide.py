"""test_decide.py — 用真实 build_graph()，注入假 retrieve_fn（无需 chroma）。"""

from __future__ import annotations

import pytest

from kb_mining import decide

build_graph = pytest.importorskip("retrieval.rag_retrieval").build_graph


def comp(data_size):
    return {"traits": {"fine_grained": True, "class_imbalance": True,
                       "medical": True, "data_size": data_size},
            "start": "2022-01", "end": "2022-06"}


TCOMP = {
    "c_med_med": comp("medium"),
    "c_ci_small": comp("small"),
    "c_med_small": comp("small"),
    "c_fg_med": comp("medium"),
}


def make_row(trait, ctype, kb_id, comp_slug):
    return {"trait": trait, "component_type": ctype, "kb_id": kb_id,
            "passed": True, "support": 0.9, "breadth": 3,
            "evidence": [{"competition": comp_slug, "rank": 1, "raw": kb_id,
                          "citation": "x"}]}


def fake_retrieve(top1_by_trait):
    def fn(query, graph):
        trait = next(iter(query["constraints"]))
        top1 = top1_by_trait[trait]
        return [{"backbone": top1, "loss": "cross_entropy_loss"},
                {"backbone": "resnet", "loss": "focal_loss"},
                {"backbone": "vit", "loss": "dice_loss"}]
    return fn


def test_five_tiers_each_hit_once():
    g = build_graph()
    fn = fake_retrieve({"medical": "unet", "class_imbalance": "vit",
                        "fine_grained": "convnext"})
    rows = [
        make_row("medical", "backbone", "unet", "c_med_med"),           # 0 confirmed
        make_row("class_imbalance", "backbone", "resnet", "c_ci_small"),  # 1 field-fix
        make_row("class_imbalance", "backbone", "mobilenet_v3", "c_ci_small"),  # 2 edge-tune
        make_row("medical", "backbone", "efficientnet", "c_med_small"),  # 3 new-edge
        make_row("fine_grained", "backbone", "swin_transformer", "c_fg_med"),  # 4 schema-ext
    ]
    tiers = [decide.classify_row(r, g, fn, TCOMP)["tier"] for r in rows]
    assert tiers == [0, 1, 2, 3, 4]


def test_conflict_marked_on_reverse_edge():
    g = build_graph()
    # 让反向边 mobilenet_v3 -> efficientnet 的条件与将建议的新边条件相交
    g["mobilenet_v3"]["efficientnet"]["condition"] = {"any": ["medical=True"]}
    fn = fake_retrieve({"medical": "mobilenet_v3"})
    # kb_id=efficientnet, medical → 档 3 新边 (efficientnet, mobilenet_v3, {any:[medical=True]})
    row = make_row("medical", "backbone", "efficientnet", "c_med_small")
    prop = decide.classify_row(row, g, fn, TCOMP)
    assert prop["tier"] == 3
    assert prop["conflict"] is True
    assert "CONFLICT" in prop["note"]


def test_cross_role_backbone_produces_no_edge():
    # 检测里 engine 编码器共识(efficientnet) vs RAG top-1 frame 检测器(yolov8)
    # → 跨角色，归档 5，不提边。
    g = build_graph()
    fn = fake_retrieve({"medical": "yolov8"})
    row = {"trait": "medical", "component_type": "backbone", "kb_id": "efficientnet",
           "role": "engine", "task_type": "object_detection", "passed": True,
           "evidence": [{"competition": "c_med_med", "rank": 1, "raw": "efficientnet",
                         "citation": "x"}]}
    prop = decide.classify_row(row, g, fn, TCOMP)
    assert prop["tier"] == 5
    assert "跨角色" in prop["action"]


def test_detection_loss_demoted_to_finding():
    g = build_graph()
    fn = fake_retrieve({"class_imbalance": "focal_loss"})
    row = {"trait": "class_imbalance", "component_type": "loss", "kb_id": "cross_entropy_loss",
           "role": None, "task_type": "object_detection", "passed": True, "support": 0.71,
           "evidence": [{"competition": "c_ci_small", "rank": 1, "raw": "BCE+lovasz",
                         "citation": "x"}]}
    prop = decide.classify_row(row, g, fn, TCOMP)
    assert prop["tier"] == 6 and prop["kind"] == "finding"
    assert "组合损失" in prop["action"]


def test_seg_frame_in_detection_is_finding():
    g = build_graph()
    fn = fake_retrieve({"class_imbalance": "yolov8"})
    row = {"trait": "class_imbalance", "component_type": "backbone", "kb_id": "unet",
           "role": "frame", "task_type": "object_detection", "passed": True,
           "support": 0.6, "breadth": 2,
           "evidence": [{"competition": "c_ci_small", "rank": 1, "raw": "nnU-Net",
                         "citation": "x"}]}
    prop = decide.classify_row(row, g, fn, TCOMP)
    assert prop["tier"] == 6 and prop["kind"] == "finding"
    assert "分割网解检测" in prop["action"]


def test_cross_condition_reverse_edge_conflict():
    g = build_graph()
    # 反向边 mobilenet_v3->efficientnet 的条件用 class_imbalance（与将建议的 medical 不相交）
    g["mobilenet_v3"]["efficientnet"]["condition"] = {"any": ["class_imbalance=True"]}
    fn = fake_retrieve({"medical": "mobilenet_v3"})
    row = {"trait": "medical", "component_type": "backbone", "kb_id": "efficientnet",
           "role": "engine", "task_type": "classification", "passed": True,
           "evidence": [{"competition": "c_med_small", "rank": 1, "raw": "efficientnet",
                         "citation": "x"}]}
    prop = decide.classify_row(row, g, fn, TCOMP)
    assert prop["tier"] == 3
    assert prop["conflict"] is True
    assert "跨条件" in prop["note"]      # 跨条件反向对也判 CONFLICT


def test_stacking_warning():
    proposals = [
        {"tier": 3, "component_type": "backbone", "kb_id": "resnet", "trait": "medical"},
        {"tier": 3, "component_type": "backbone", "kb_id": "resnet", "trait": "class_imbalance"},
        {"tier": 4, "component_type": "backbone", "kb_id": "vit", "trait": "fine_grained"},
    ]
    warns = decide.stacking_warnings(proposals)
    assert len(warns) == 1
    assert "resnet" in warns[0]


def test_confirmed_needs_no_apply():
    g = build_graph()
    fn = fake_retrieve({"medical": "unet"})
    prop = decide.classify_row(make_row("medical", "backbone", "unet", "c_med_med"),
                               g, fn, TCOMP)
    assert prop["tier"] == 0
    assert prop["conflict"] is False
    assert "archetype_top3_after" not in prop      # 档 0 不试应用

"""test_aggregate.py — 全离线，手写 facts fixture 精确断言共识数值与侧表。"""

from __future__ import annotations

import pytest

from kb_mining import aggregate


def make_fact(comp, kind, families, best=None, pseudo=False, loss_kb=None,
              loss_raw=None, metric=False, members_raw=None, fis=None, rank=1):
    return {
        "competition": comp, "kind": kind, "families": families,
        "best_single_family": best, "used_pseudo_labeling": pseudo,
        "loss_kb": loss_kb, "loss_raw": loss_raw, "loss_is_metric_learning": metric,
        "members_raw": members_raw or [{"raw_model": f} for f in families],
        "family_image_size": fis or {}, "citations": [f"cite {comp}"], "rank": rank,
    }


# 测试用竞赛：cf1/cf2 fine_grained；old 为 swin 发布前(2020-06)的 class_imbalance 赛
def _c(slug, start, end, **traits):
    base = {"fine_grained": False, "class_imbalance": False, "medical": False}
    base.update(traits)
    return {"slug": slug, "start": start, "end": end,
            "task_type": "classification", "traits": base}


TC = {
    "cf1": _c("cf1", "2022-01", "2022-06", fine_grained=True),
    "cf2": _c("cf2", "2022-01", "2022-06", fine_grained=True),
    "old": _c("old", "2020-06", "2021-06", class_imbalance=True),
}


# ── 票重 ────────────────────────────────────────────────────────────────────
def test_vote_weights_single():
    w = aggregate.family_vote_weights(make_fact("cf1", "single", ["efficientnet"]))
    assert w == {"efficientnet": 1.0}


def test_vote_weights_ensemble_best_vs_rest():
    w = aggregate.family_vote_weights(
        make_fact("cf1", "ensemble", ["resnet", "vit"], best="resnet"))
    assert w == {"resnet": 1.0, "vit": 0.5}


def test_vote_weights_unclear():
    w = aggregate.family_vote_weights(make_fact("cf1", "unclear", ["resnet"]))
    assert w == {"resnet": 0.5}


def test_vote_weights_pseudo_discount():
    w = aggregate.family_vote_weights(
        make_fact("cf1", "single", ["efficientnet"], pseudo=True))
    assert w == {"efficientnet": pytest.approx(0.8)}


def test_loss_vote_takes_max_weight():
    # ensemble best=resnet(1.0), vit(0.5) → loss 票重取 1.0
    loss_kb, w = aggregate.loss_vote(
        make_fact("cf1", "ensemble", ["resnet", "vit"], best="resnet", loss_kb="focal_loss"))
    assert loss_kb == "focal_loss" and w == 1.0


# ── 共存性 ──────────────────────────────────────────────────────────────────
def test_coexists():
    assert aggregate.coexists("resnet", "2020-06") is True     # 2015-12 < 2020-06
    assert aggregate.coexists("swin_transformer", "2020-06") is False  # 2021-03 不早于
    assert aggregate.coexists("swin_transformer", "2021-06") is True
    assert aggregate.coexists("unknown", "2020-06") is True    # 未知一律放行


# ── 共识数值 ────────────────────────────────────────────────────────────────
def test_consensus_support_and_breadth():
    facts = [
        make_fact("cf1", "single", ["efficientnet"]),
        make_fact("cf2", "single", ["efficientnet"]),
        make_fact("cf1", "ensemble", ["resnet", "vit"], best="resnet"),
    ]
    rows = aggregate.compute_consensus(facts, TC)
    bb = {r["kb_id"]: r for r in rows
          if r["trait"] == "fine_grained" and r["component_type"] == "backbone"}
    # total = 1.0(eff)+1.0(eff)+1.0(resnet)+0.5(vit) = 3.5
    assert bb["efficientnet"]["votes"] == 2.0
    assert bb["efficientnet"]["total_votes"] == 3.5
    assert bb["efficientnet"]["support"] == pytest.approx(2 / 3.5, abs=1e-3)
    assert bb["efficientnet"]["breadth"] == 2
    assert bb["efficientnet"]["passed"] is True
    assert bb["resnet"]["breadth"] == 1
    assert bb["resnet"]["passed"] is False       # breadth < 2


def test_consensus_coexistence_excludes_from_numerator_and_denominator():
    # old(start 2020-06)：swin 不共存 → 既不计票也不计分母
    facts = [make_fact("old", "ensemble", ["swin_transformer", "resnet"],
                       best="swin_transformer")]
    rows = aggregate.compute_consensus(facts, TC)
    ci = {r["kb_id"]: r for r in rows
          if r["trait"] == "class_imbalance" and r["component_type"] == "backbone"}
    assert "swin_transformer" not in ci          # 共存过滤掉
    # 只剩 resnet 的 0.5 计入分母 → support 1.0
    assert ci["resnet"]["votes"] == 0.5
    assert ci["resnet"]["total_votes"] == 0.5
    assert ci["resnet"]["support"] == 1.0


def test_consensus_threshold_boundary():
    # 构造 support 恰 0.5、breadth 恰 2 → passed（>= 边界）
    facts = [
        make_fact("cf1", "single", ["efficientnet"]),
        make_fact("cf2", "single", ["efficientnet"]),
        make_fact("cf1", "single", ["resnet"]),
        make_fact("cf2", "single", ["resnet"]),
    ]
    rows = aggregate.compute_consensus(facts, TC)
    bb = {r["kb_id"]: r for r in rows
          if r["trait"] == "fine_grained" and r["component_type"] == "backbone"}
    assert bb["efficientnet"]["support"] == 0.5
    assert bb["efficientnet"]["breadth"] == 2
    assert bb["efficientnet"]["passed"] is True   # 0.5>=0.5 且 2>=2


def test_backbone_role_groups_have_separate_denominators():
    # 一个分割方案：U-Net(车架) + EfficientNet/ResNet(发动机 encoder)。
    # 车架组与发动机组各自分母，unet 不被 encoder 稀释。
    tc = {"seg": {**_c("seg", "2022-01", "2022-06", medical=True),
                  "task_type": "image_segmentation"}}
    facts = [make_fact("seg", "ensemble", ["unet", "efficientnet", "resnet"], best="unet")]
    rows = aggregate.compute_consensus(facts, tc)
    frame = {r["kb_id"]: r for r in rows
             if r.get("role") == "frame" and r["trait"] == "medical"}
    engine = {r["kb_id"]: r for r in rows
              if r.get("role") == "engine" and r["trait"] == "medical"}
    assert frame["unet"]["support"] == 1.0          # 车架组独占，不被 encoder 稀释
    assert set(engine) == {"efficientnet", "resnet"}
    # ensemble best=unet(1.0)；efficientnet/resnet 各 0.5 → engine 分母 1.0
    assert engine["efficientnet"]["support"] == 0.5
    assert frame["unet"]["total_votes"] == 1.0      # 分母独立
    assert engine["efficientnet"]["total_votes"] == 1.0


def test_dominance_passes_below_half_support():
    # efficientnet 0.375 < 0.5，但 ratio 1.5 / margin 0.125 / breadth 3 达标 → dominance 通过
    tc = {f"c{i}": _c(f"c{i}", "2022-03", "2022-06", fine_grained=True) for i in range(4)}
    facts = [
        make_fact("c0", "single", ["efficientnet"]),
        make_fact("c1", "single", ["efficientnet"]),
        make_fact("c2", "single", ["efficientnet"]),   # eff: 3.0, breadth 3
        make_fact("c0", "single", ["resnet"]),
        make_fact("c3", "single", ["resnet"]),          # resnet: 2.0, breadth 2
        make_fact("c1", "single", ["vit"]),
        make_fact("c3", "single", ["vit"]),             # vit: 2.0
        make_fact("c2", "single", ["convnext"]),        # convnext: 1.0  → total 8
    ]
    rows = aggregate.compute_consensus(facts, tc)
    eng = {r["kb_id"]: r for r in rows if r["trait"] == "fine_grained"
           and r.get("role") == "engine"}
    assert eng["efficientnet"]["support"] == 0.375     # < 0.5
    assert eng["efficientnet"]["dominance"] is True
    assert eng["efficientnet"]["passed"] is True       # 靠 dominance 过
    assert eng["efficientnet"]["ratio"] == 1.5
    assert eng["resnet"]["passed"] is False            # 非头名，不 dominance


def test_task_type_pools_are_separate():
    # 分类赛的 efficientnet 与检测赛的 yolov8 不在同一 (trait) 池里竞争
    tc = {
        "cls": _c("cls", "2023-06", "2023-12", medical=True),
        "det": {**_c("det", "2023-06", "2023-12", medical=True),
                "task_type": "object_detection"},
    }
    facts = [
        make_fact("cls", "single", ["efficientnet"]),
        make_fact("det", "single", ["yolov8"]),
    ]
    rows = aggregate.compute_consensus(facts, tc)
    cls_bb = {(r["kb_id"]) for r in rows
              if r["task_type"] == "classification" and r["component_type"] == "backbone"}
    det_bb = {(r["kb_id"]) for r in rows
              if r["task_type"] == "object_detection" and r["component_type"] == "backbone"}
    assert cls_bb == {"efficientnet"}      # 各池独立
    assert det_bb == {"yolov8"}
    # 每池 support 都是 1.0（各自池内独占）
    for r in rows:
        if r["component_type"] == "backbone" and r["trait"] == "medical":
            assert r["support"] == 1.0


def test_loss_voting_excluded_for_metric_learning_competition():
    tc = dict(TC)
    tc["ml"] = {**_c("ml", "2022-01", "2022-06", fine_grained=True),
                "loss_voting": False}
    facts = [make_fact("ml", "single", ["efficientnet"], loss_kb="focal_loss",
                       loss_raw="focal")]
    rows = aggregate.compute_consensus(facts, tc)
    loss_rows = [r for r in rows if r["component_type"] == "loss"]
    assert loss_rows == []                         # loss 投票整场排除


# ── 侧表 ────────────────────────────────────────────────────────────────────
def test_side_tables():
    facts = [
        make_fact("cf1", "ensemble", ["efficientnet", "resnet", "unknown"],
                  best="efficientnet",
                  members_raw=[{"raw_model": "tf_efficientnet_b4"},
                               {"raw_model": "resnet50"},
                               {"raw_model": "cropnet"}],
                  fis={"efficientnet": 512, "resnet": 384}),
        make_fact("cf2", "single", ["unknown"], loss_kb="unknown",
                  loss_raw="arcface", metric=True,
                  members_raw=[{"raw_model": "cropnet"}]),
    ]
    unknown, recipes, cooccur = aggregate.build_side_tables(facts, TC)
    assert unknown["models"]["cropnet"] == 2                 # 两篇都出现
    assert unknown["losses"]["arcface"]["metric_learning"] is True
    # 共现：efficientnet-resnet（unknown 不计入共现，因非 KB 家族）
    assert cooccur["efficientnet"]["resnet"] == 1
    assert "unknown" not in cooccur
    # recipes：efficientnet|fine_grained 有 512
    assert recipes["efficientnet|fine_grained"]["mode"] == 512

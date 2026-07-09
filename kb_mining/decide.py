"""decide.py — consensus.json × 现有 KB → data/proposals.md（五档决策）。

**只写建议，不改任何 KB 数据。** 对每条 passed 共识行，用原型查询打到现有
检索管道，按五档归类（0 confirmed / 1 field-fix / 2 edge-tune / 3 new-edge /
4 schema-ext），并附冲突检查与堆叠纪律告警。

retrieve_fn(query, graph)->list[config] 可注入：生产包 retrieve_top3_hybrid，
测试注入假检索（无需 chroma）。

CLI:  python -m kb_mining.decide
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable

from kb_mining import catalog

DEFAULT_CONSENSUS = Path("kb_mining/data/consensus.json")
OUT_MD = Path("kb_mining/data/proposals.md")

# 必须与 rag_retrieval._matches_condition 的 checks 键集合一致
VALID_CONDITION_KEYS = {
    "real_time=True", "edge_deployment=True", "class_imbalance=True",
    "cross_modal=True", "no_text_modality=True", "medical=True",
    "zero_shot=True", "few_shot=True", "data_size=small", "large_data=True",
    "high_accuracy_priority=True", "feature_quality_priority=True",
}

TIER_NAMES = {
    0: "confirmed", 1: "field-fix", 2: "edge-tune", 3: "new-edge", 4: "schema-ext",
    5: "cross-role（无对应 RAG 槽位）", 6: "finding（不提边）",
}

# 分割元架构：出现在检测任务里即"用分割网解检测"，记 finding
_SEG_FRAMES = {"unet", "segformer", "mask2former"}


def trait_key(trait: str) -> str:
    return f"{trait}=True"


# ── 原型查询 ────────────────────────────────────────────────────────────────
def archetype_query(trait: str, data_size: str = "medium",
                    task_type: str = "classification") -> dict:
    """某 (task_type, trait) 的原型查询。fine_grained 非合法 constraint 键，写入也被忽略。"""
    return {
        "task_type": task_type,
        "data_size": data_size,
        "priority": "balanced",
        "constraints": {trait: True},
        "description": f"{trait.replace('_', ' ')} {task_type.replace('_', ' ')}",
    }


def evidence_data_size_mode(row: dict, competitions: dict) -> str:
    sizes = [competitions[e["competition"]]["traits"]["data_size"]
             for e in row.get("evidence", []) if e.get("competition") in competitions]
    return Counter(sizes).most_common(1)[0][0] if sizes else "medium"


def _top1(configs: list[dict], ctype: str) -> str | None:
    if not configs:
        return None
    return configs[0].get("backbone") if ctype == "backbone" else configs[0].get("loss")


def _condition_keys(cond: dict) -> set[str]:
    return set(cond.get("all", [])) | set(cond.get("any", []))


def _existing_preferred_edges(graph, src: str) -> list[tuple]:
    out = []
    if src not in graph:
        return out
    for succ in graph.successors(src):
        e = graph[src][succ]
        if e.get("relation") == "preferred_when":
            out.append((src, succ, e.get("condition", {})))
    return out


# ── 五档归类 ────────────────────────────────────────────────────────────────
def classify_row(
    row: dict,
    graph,
    retrieve_fn: Callable[[dict, object], list[dict]],
    competitions: dict,
) -> dict:
    """对一条 passed 共识行归类，返回 proposal dict。"""
    trait = row["trait"]
    ctype = row["component_type"]
    A = row["kb_id"]
    key = trait_key(trait)
    ds = evidence_data_size_mode(row, competitions)
    task_type = row.get("task_type", "classification")
    query = archetype_query(trait, ds, task_type)

    configs = retrieve_fn(query, graph)
    top1 = _top1(configs, ctype)
    top3_before = [c.get("backbone") for c in configs]

    prop = {"task_type": task_type, "trait": trait, "component_type": ctype, "kb_id": A,
            "role": row.get("role"), "data_size": ds, "current_top1": top1,
            "archetype_top3_before": top3_before, "kind": "proposal",
            "by_dominance": bool(row.get("dominance")),
            "conflict": False, "note": ""}

    # finding-A：检测 loss 是组合损失（分类头 + 分割/回归加权求和），压平成单票是
    # 伪象——只记 finding，不提边（看 raw：BCE{4-class} + [0.75*lovasz+0.25*BCE] 之类）。
    if ctype == "loss" and task_type == "object_detection":
        prop.update(tier=6, kind="finding", action=(
            f"检测任务的 loss 共识（'{A}' support={row.get('support')}）来自组合损失，"
            f"压平成单票是伪象，不代表 CE/focal 之间的选择——记为 finding，不提边。"))
        return prop

    # finding-B：分割架构（U-Net/SegFormer/Mask2Former）出现在检测任务共识里——
    # 这是"用分割网解检测赛"的路径，KB 目前无法表达；记 finding，不改分割任务的边。
    if ctype == "backbone" and task_type == "object_detection" and A in _SEG_FRAMES:
        prop.update(tier=6, kind="finding", action=(
            f"分割架构 '{A}' 出现在检测任务共识（support={row.get('support')}，"
            f"breadth={row.get('breadth')}）——'用分割网解检测赛'的路径，KB 无法表达此"
            f"跨任务用法；记为 finding，不动分割边。"))
        return prop

    # 档 5：跨角色——backbone 共识的角色与 RAG top-1 的角色不同（如检测里 engine
    # 编码器共识 vs RAG 选的 frame 检测器）。RAG 未建模该角色槽位，无对应决策，不提边。
    if ctype == "backbone":
        row_role = row.get("role")
        top1_role = catalog.family_role(top1) if top1 else None
        if row_role and top1_role and row_role != top1_role:
            prop.update(tier=5, action=(
                f"跨角色：'{A}'（{row_role}）与当前 top-1 '{top1}'（{top1_role}）角色不同。"
                f"RAG 对 {task_type} 只选 {top1_role}，未建模 {row_role} 槽位——"
                f"本条共识暂无对应 RAG 决策，不提边。"))
            return prop

    # 档 0：已确认
    if top1 == A:
        prop.update(tier=0, action=f"'{A}' 已是 {trait} 原型查询的 top-1，无需改动。")
        return prop

    # 档 1：field-fix（仅 backbone，检查 data_size 列表）
    if ctype == "backbone" and A in graph:
        node_sizes = graph.nodes[A].get("data_size", [])
        if node_sizes and ds not in node_sizes:
            prop.update(tier=1,
                        action=f"'{A}' 节点 data_size={node_sizes} 不含证据竞赛众数档 "
                               f"'{ds}'；建议补入 '{ds}'。")
            _annotate_apply(prop, graph, retrieve_fn, query, ctype,
                            edge=None, field=(A, "data_size", ds))
            return prop

    # 档 2：edge-tune（A 已有 preferred_when 边，但条件不含 T）
    existing = _existing_preferred_edges(graph, A)
    if existing and all(key not in _condition_keys(cond) for _, _, cond in existing):
        tgt, cond = existing[0][1], existing[0][2]
        prop.update(tier=2,
                    action=f"'{A}' 已有 preferred_when 边 → '{tgt}'（条件 {cond}）；"
                           f"建议把 '{key}' 并入该边条件（all→any 或加键）。")
        _annotate_apply(prop, graph, retrieve_fn, query, ctype,
                        edge=(A, tgt, {"any": sorted(_condition_keys(cond) | {key})}))
        return prop

    # 档 3：new-edge（T 是合法 condition 键）
    if key in VALID_CONDITION_KEYS:
        target = top1 or "<current top-1>"
        prop.update(tier=3,
                    action=f"新增边 ('{A}', '{target}', preferred_when)，条件 "
                           f"{{'any': ['{key}']}}；目标=当前 top-1（打分不消费目标，纯语义）。")
        _annotate_apply(prop, graph, retrieve_fn, query, ctype,
                        edge=(A, target, {"any": [key]}))
        return prop

    # 档 4：schema-ext（T 非合法 condition 键，如 fine_grained）
    target = top1 or "<current top-1>"
    prop.update(tier=4,
                action=f"'{key}' 非合法 constraint 键。建议：①constraints 加键 '{trait}' "
                       f"②Module 1 prompt 同步产出该键 ③再加档 3 的边 "
                       f"('{A}', '{target}', preferred_when, {{'any': ['{key}']}})。"
                       f"影响面：Module 1 + 输入 schema + 检索。")
    return prop


def _annotate_apply(prop, graph, retrieve_fn, query, ctype, edge=None, field=None):
    """试应用到 graph 副本、重跑原型查询，记录 top-3 变化 + 反向边冲突。"""
    g2 = copy.deepcopy(graph)
    if field is not None:
        node_id, fld, val = field
        vals = list(g2.nodes[node_id].get(fld, []))
        if val not in vals:
            vals.append(val)
        g2.nodes[node_id][fld] = vals
    if edge is not None:
        src, tgt, cond = edge
        if src in g2 and tgt in g2:
            g2.add_edge(src, tgt, relation="preferred_when", condition=cond)
            # 反向边冲突：只要存在反向的 (tgt, src, preferred_when) 就是冲突——
            # 条件相交（同 trait）固然冲突；条件不相交（跨条件，如 medical vs
            # class_imbalance）也冲突：在同时满足两条件的查询下两边同时命中、方向
            # 相反，Phase B 按候选顺序取首个命中 → 行为顺序依赖。
            if g2.has_edge(tgt, src) and g2[tgt][src].get("relation") == "preferred_when":
                rev_keys = _condition_keys(g2[tgt][src].get("condition", {}))
                new_keys = _condition_keys(cond)
                prop["conflict"] = True
                if rev_keys & new_keys:
                    prop["note"] = (f"CONFLICT：已存在反向边 '{tgt}'→'{src}' 且条件相交"
                                    f"（{sorted(rev_keys & new_keys)}），需短训 A/B 仲裁，暂不应用。")
                else:
                    prop["note"] = (f"CONFLICT：存在跨条件反向边 '{tgt}'→'{src}'（条件 "
                                    f"{sorted(rev_keys)}），与本边（{sorted(new_keys)}）在"
                                    f"同时满足两条件的查询下方向相反，Phase B 顺序依赖；"
                                    f"需短训 A/B 仲裁，暂不应用。")
    try:
        after = [c.get("backbone") for c in retrieve_fn(query, g2)]
    except Exception:
        after = None
    prop["archetype_top3_after"] = after
    if after is not None and after != prop["archetype_top3_before"]:
        prop["note"] = (prop["note"] + " " if prop["note"] else "") + \
            f"若应用，原型 top-3：{prop['archetype_top3_before']} → {after}。"


# ── 堆叠纪律 ────────────────────────────────────────────────────────────────
def stacking_warnings(proposals: list[dict]) -> list[str]:
    """档 3/4 建议里，同一源 backbone 获 >1 条新挖掘边 → 告警（应合并为一条 any 边）。"""
    by_src: dict[str, list[str]] = defaultdict(list)
    for p in proposals:
        if p["tier"] in (3, 4) and p["component_type"] == "backbone":
            by_src[p["kb_id"]].append(p["trait"])
    return [f"backbone '{src}' 将获 {len(traits)} 条挖掘边（traits: {', '.join(traits)}）"
            f"——应合并为一条 `any` 边，避免 bonus 堆叠。"
            for src, traits in by_src.items() if len(traits) > 1]


# ── Markdown ────────────────────────────────────────────────────────────────
def render_md(proposals: list[dict], warnings: list[str]) -> str:
    lines = ["# proposals.md — KB 改动建议（五档决策）", "",
             "> 本文件仅为建议；KB 数据由人看完后另行提交。", ""]
    if warnings:
        lines.append("## ⚠ 堆叠纪律告警")
        lines += [f"- {w}" for w in warnings] + [""]

    def emit(p):
        flag = " **[CONFLICT]**" if p["conflict"] else ""
        dom = " **[DOMINANCE]**" if p.get("by_dominance") else ""
        lines.append(f"- **[{p.get('task_type','classification')} · {p['component_type']}] "
                     f"{p['kb_id']}** ×「{p['trait']}」{flag}{dom}")
        lines.append(f"  - {p['action']}")
        if p.get("note"):
            lines.append(f"  - {p['note']}")

    for tier in (0, 1, 2, 3, 4, 5):    # 提案（含跨角色隔离）
        sub = [p for p in proposals if p["tier"] == tier]
        if not sub:
            continue
        lines.append(f"## 档 {tier} — {TIER_NAMES[tier]}（{len(sub)} 条）")
        lines.append("")
        for p in sub:
            emit(p)
        lines.append("")

    findings = [p for p in proposals if p["tier"] == 6]   # findings：不提边的观察
    if findings:
        lines.append(f"## Findings（{len(findings)} 条，仅记录，不改 KB）")
        lines.append("")
        for p in findings:
            emit(p)
        lines.append("")
    return "\n".join(lines)


# ── 生产检索包装 ────────────────────────────────────────────────────────────
def make_real_retrieve_fn():
    from retrieval.rag_retrieval import build_vector_index, retrieve_top3_hybrid
    repo_root = Path(__file__).resolve().parent.parent
    col = build_vector_index(persist_path=str(repo_root / "retrieval" / "chroma_db_kb"))

    def retrieve_fn(query: dict, graph) -> list[dict]:
        return retrieve_top3_hybrid(query, graph, col)

    return retrieve_fn


# ── 编排 ────────────────────────────────────────────────────────────────────
def run_decide(
    consensus_path: Path = DEFAULT_CONSENSUS,
    graph=None,
    retrieve_fn=None,
    competitions: dict | None = None,
) -> list[dict]:
    competitions = competitions or catalog.COMPETITIONS
    if graph is None:
        from retrieval.rag_retrieval import build_graph
        graph = build_graph()
    retrieve_fn = retrieve_fn or make_real_retrieve_fn()

    rows = json.loads(consensus_path.read_text(encoding="utf-8"))
    passed = [r for r in rows if r.get("passed")]
    proposals = [classify_row(r, graph, retrieve_fn, competitions) for r in passed]
    warnings = stacking_warnings(proposals)

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text(render_md(proposals, warnings), encoding="utf-8")

    by_tier = Counter(p["tier"] for p in proposals)
    print(f"[decide] proposals={len(proposals)} by_tier="
          + " ".join(f"{TIER_NAMES[t]}={by_tier[t]}" for t in sorted(by_tier)))
    if warnings:
        print(f"[decide] [WARN] 堆叠告警 {len(warnings)} 条（见 proposals.md 顶部）")
    return proposals


def _cli() -> None:
    ap = argparse.ArgumentParser(description="consensus.json → proposals.md")
    ap.add_argument("--consensus", type=Path, default=DEFAULT_CONSENSUS)
    args = ap.parse_args()
    run_decide(args.consensus)


if __name__ == "__main__":
    _cli()

"""Turn mined consensus rows into human-reviewed KB proposals.

This module writes suggestions only; it does not modify KB data. For each
passed consensus row, it runs an archetype query against the current retrieval
pipeline and classifies the result into proposal tiers:
0 confirmed, 1 field-fix, 2 edge-tune, 3 new-edge, 4 schema-ext.
It also records conflict checks and edge-stacking warnings.

retrieve_fn(query, graph)->list[config] is injectable. Production uses
retrieve_top3_hybrid; tests use fake retrieval without Chroma.

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

# Must stay aligned with rag_retrieval._matches_condition check keys.
VALID_CONDITION_KEYS = {
    "real_time=True", "edge_deployment=True", "class_imbalance=True",
    "cross_modal=True", "no_text_modality=True", "medical=True",
    "zero_shot=True", "few_shot=True", "data_size=small", "large_data=True",
    "high_accuracy_priority=True", "feature_quality_priority=True",
}

TIER_NAMES = {
    0: "confirmed", 1: "field-fix", 2: "edge-tune", 3: "new-edge", 4: "schema-ext",
    5: "cross-role/no matching RAG slot", 6: "finding/no edge proposal",
}

# Segmentation frames appearing in detection consensus are recorded as findings.
_SEG_FRAMES = {"unet", "segformer", "mask2former"}


def trait_key(trait: str) -> str:
    return f"{trait}=True"


# Archetype queries.
def archetype_query(trait: str, data_size: str = "medium",
                    task_type: str = "classification") -> dict:
    """Build an archetype query for one (task_type, trait) pair."""
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


# Tiered classification.
def classify_row(
    row: dict,
    graph,
    retrieve_fn: Callable[[dict, object], list[dict]],
    competitions: dict,
) -> dict:
    """Classify one passed consensus row into a proposal dict."""
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

    # Finding A: detection losses are often composite. Flattening them into a
    # single CE/focal vote would be misleading, so record a finding only.
    if ctype == "loss" and task_type == "object_detection":
        prop.update(tier=6, kind="finding", action=(
            f"The detection loss consensus ('{A}', support={row.get('support')}) "
            f"comes from composite losses. Treat it as a finding, not a CE/focal edge."))
        return prop

    # Finding B: segmentation frames used for detection are a cross-task pattern
    # that the current KB schema cannot express cleanly.
    if ctype == "backbone" and task_type == "object_detection" and A in _SEG_FRAMES:
        prop.update(tier=6, kind="finding", action=(
            f"Segmentation frame '{A}' appears in detection consensus "
            f"(support={row.get('support')}, breadth={row.get('breadth')}). "
            f"Record the cross-task pattern as a finding and leave segmentation edges unchanged."))
        return prop

    # Tier 5: cross-role consensus. For example, a detection consensus may name
    # an encoder engine while RAG selects a detector frame.
    if ctype == "backbone":
        row_role = row.get("role")
        top1_role = catalog.family_role(top1) if top1 else None
        if row_role and top1_role and row_role != top1_role:
            prop.update(tier=5, action=(
                f"Cross-role consensus: '{A}' ({row_role}) differs from current top-1 "
                f"'{top1}' ({top1_role}). RAG currently selects the {top1_role} slot for "
                f"{task_type}, so this row has no matching RAG decision slot."))
            return prop

    # Tier 0: already confirmed.
    if top1 == A:
        prop.update(tier=0, action=f"'{A}' is already top-1 for the {trait} archetype query.")
        return prop

    # Tier 1: field fix for backbone data_size.
    if ctype == "backbone" and A in graph:
        node_sizes = graph.nodes[A].get("data_size", [])
        if node_sizes and ds not in node_sizes:
            prop.update(tier=1,
                        action=f"'{A}' has data_size={node_sizes}, but evidence points to "
                               f"'{ds}'. Add '{ds}' to the node if the evidence checks out.")
            _annotate_apply(prop, graph, retrieve_fn, query, ctype,
                            edge=None, field=(A, "data_size", ds))
            return prop

    # Tier 2: tune an existing preferred_when edge.
    existing = _existing_preferred_edges(graph, A)
    if existing and all(key not in _condition_keys(cond) for _, _, cond in existing):
        tgt, cond = existing[0][1], existing[0][2]
        prop.update(tier=2,
                    action=f"'{A}' already has a preferred_when edge to '{tgt}' "
                           f"with condition {cond}. Add '{key}' to that condition.")
        _annotate_apply(prop, graph, retrieve_fn, query, ctype,
                        edge=(A, tgt, {"any": sorted(_condition_keys(cond) | {key})}))
        return prop

    # Tier 3: add a new edge when the trait is a valid condition key.
    if key in VALID_CONDITION_KEYS:
        target = top1 or "<current top-1>"
        prop.update(tier=3,
                    action=f"Add edge ('{A}', '{target}', preferred_when) with "
                           f"{{'any': ['{key}']}}. The target is the current top-1; "
                           f"the scoring path uses the source and condition.")
        _annotate_apply(prop, graph, retrieve_fn, query, ctype,
                        edge=(A, target, {"any": [key]}))
        return prop

    # Tier 4: schema extension for traits that are not valid condition keys.
    target = top1 or "<current top-1>"
    prop.update(tier=4,
                action=f"'{key}' is not a valid constraint key. Add '{trait}' to the "
                       f"input schema, teach Module 1 to emit it, then add the edge "
                       f"('{A}', '{target}', preferred_when, {{'any': ['{key}']}}). "
                       f"Impact area: Module 1, input schema, and retrieval.")
    return prop


def _annotate_apply(prop, graph, retrieve_fn, query, ctype, edge=None, field=None):
    """Apply a proposal on a graph copy and record top-3 changes/conflicts."""
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
            # A reverse preferred_when edge is a conflict. Even disjoint
            # conditions can conflict when a query satisfies both conditions,
            # making Phase B order-dependent.
            if g2.has_edge(tgt, src) and g2[tgt][src].get("relation") == "preferred_when":
                rev_keys = _condition_keys(g2[tgt][src].get("condition", {}))
                new_keys = _condition_keys(cond)
                prop["conflict"] = True
                if rev_keys & new_keys:
                    prop["note"] = (f"CONFLICT: reverse edge '{tgt}' -> '{src}' already exists "
                                    f"with overlapping conditions {sorted(rev_keys & new_keys)}. "
                                    f"Run a short A/B check before applying.")
                else:
                    prop["note"] = (f"CONFLICT: reverse edge '{tgt}' -> '{src}' exists with "
                                    f"conditions {sorted(rev_keys)}. Together with this edge "
                                    f"({sorted(new_keys)}), queries satisfying both conditions "
                                    f"would become order-dependent. Run a short A/B check first.")
    try:
        after = [c.get("backbone") for c in retrieve_fn(query, g2)]
    except Exception:
        after = None
    prop["archetype_top3_after"] = after
    if after is not None and after != prop["archetype_top3_before"]:
        prop["note"] = (prop["note"] + " " if prop["note"] else "") + \
            f"If applied, archetype top-3 changes from {prop['archetype_top3_before']} to {after}."


# Edge-stacking checks.
def stacking_warnings(proposals: list[dict]) -> list[str]:
    """Warn when one source backbone receives multiple mined edges."""
    by_src: dict[str, list[str]] = defaultdict(list)
    for p in proposals:
        if p["tier"] in (3, 4) and p["component_type"] == "backbone":
            by_src[p["kb_id"]].append(p["trait"])
    return [f"backbone '{src}' would receive {len(traits)} mined edges "
            f"(traits: {', '.join(traits)}); merge them into one `any` edge to avoid bonus stacking."
            for src, traits in by_src.items() if len(traits) > 1]


# Markdown rendering.
def render_md(proposals: list[dict], warnings: list[str]) -> str:
    lines = ["# KB Update Proposals", "",
             "> This file contains suggestions only. KB data should be edited separately after review.", ""]
    if warnings:
        lines.append("## Edge-Stacking Warnings")
        lines += [f"- {w}" for w in warnings] + [""]

    def emit(p):
        flag = " **[CONFLICT]**" if p["conflict"] else ""
        dom = " **[DOMINANCE]**" if p.get("by_dominance") else ""
        lines.append(f"- **[{p.get('task_type','classification')} / {p['component_type']}] "
                     f"{p['kb_id']}** x `{p['trait']}`{flag}{dom}")
        lines.append(f"  - {p['action']}")
        if p.get("note"):
            lines.append(f"  - {p['note']}")

    for tier in (0, 1, 2, 3, 4, 5):
        sub = [p for p in proposals if p["tier"] == tier]
        if not sub:
            continue
        lines.append(f"## Tier {tier} - {TIER_NAMES[tier]} ({len(sub)})")
        lines.append("")
        for p in sub:
            emit(p)
        lines.append("")

    findings = [p for p in proposals if p["tier"] == 6]
    if findings:
        lines.append(f"## Findings ({len(findings)}, record only)")
        lines.append("")
        for p in findings:
            emit(p)
        lines.append("")
    return "\n".join(lines)


# Production retrieval wrapper.
def make_real_retrieve_fn():
    from retrieval.rag_retrieval import build_vector_index, retrieve_top3_hybrid
    repo_root = Path(__file__).resolve().parent.parent
    col = build_vector_index(persist_path=str(repo_root / "retrieval" / "chroma_db_kb"))

    def retrieve_fn(query: dict, graph) -> list[dict]:
        return retrieve_top3_hybrid(query, graph, col)

    return retrieve_fn


# Orchestration.
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
        print(f"[decide] [WARN] {len(warnings)} edge-stacking warnings; see proposals.md")
    return proposals


def _cli() -> None:
    ap = argparse.ArgumentParser(description="consensus.json → proposals.md")
    ap.add_argument("--consensus", type=Path, default=DEFAULT_CONSENSUS)
    args = ap.parse_args()
    run_decide(args.consensus)


if __name__ == "__main__":
    _cli()

"""aggregate.py — facts.jsonl → consensus.{json,md} + 三张侧表（纯函数，无 LLM）。

共识 = 在"具有特征 T 的合格竞赛"内，family A 的加权得票占比（support）与
覆盖竞赛数（breadth）。投票规则、资格过滤见 kb_mining_plan §4。

判断留白的落定：
  - 分母含 unknown（诚实分母 = 全部获胜模型选择；unknown 占比高本身是 KB 缺口
    信号），但不为 unknown 发 consensus 行——它进 unknown_components 侧表。
  - metric-learning 竞赛（catalog `loss_voting=False`）整场排除 loss 投票。

CLI:  python -m kb_mining.aggregate [--support-min 0.5] [--breadth-min 2]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from kb_mining import catalog

DEFAULT_IN = Path("kb_mining/data/facts.jsonl")
OUT_JSON = Path("kb_mining/data/consensus.json")
OUT_MD = Path("kb_mining/data/consensus.md")
OUT_UNKNOWN = Path("kb_mining/data/unknown_components.json")
OUT_RECIPES = Path("kb_mining/data/recipes.json")
OUT_COOCC = Path("kb_mining/data/ensemble_cooccurrence.json")

SUPPORT_MIN = 0.50
BREADTH_MIN = 2

# DOMINANCE 判据：组内头名相对第二名"够强 + 够广 + 够拉开"，即便 support < 0.5
# 也认定为共识（碎片化任务里唯一能出可用 backbone 提案的判据）。
DOMINANCE_RATIO_MIN = 1.5    # 头名 support / 第二名 support
DOMINANCE_BREADTH_MIN = 3    # 头名跨竞赛数
DOMINANCE_MARGIN_MIN = 0.10  # 头名 support − 第二名 support
BOOLEAN_TRAITS = ("fine_grained", "class_imbalance", "medical", "multi_label")
MIN_END = "2021-01"


# ── 投票权重（纯函数）───────────────────────────────────────────────────────
def family_vote_weights(fact: dict) -> dict[str, float]:
    """一篇 fact 内每个 family 的票重（含伪标签折扣）。families 已去重。"""
    kind = fact.get("kind")
    best = fact.get("best_single_family")
    disc = 0.8 if fact.get("used_pseudo_labeling") else 1.0
    out: dict[str, float] = {}
    for fam in fact.get("families", []):
        if kind == "single":
            w = 1.0
        elif kind == "ensemble":
            w = 1.0 if fam == best else 0.5
        else:  # unclear
            w = 0.5
        out[fam] = w * disc
    return out


def loss_vote(fact: dict) -> tuple[str | None, float]:
    """一篇 fact 的 loss 票（篇级：loss_kb + 该篇最高家族票重）。无 loss → (None,0)。"""
    loss_kb = fact.get("loss_kb")
    if not loss_kb:
        return None, 0.0
    weights = family_vote_weights(fact)
    w = max(weights.values()) if weights else 0.0
    return loss_kb, w


def coexists(family: str, comp_start: str) -> bool:
    """family 是否在竞赛开赛前已发布（共存性）。未知 family 一律放行。"""
    rel = catalog.FAMILY_RELEASE.get(family)
    if rel is None:
        return True   # unknown：无法判定发布时间，放行（仅进分母/侧表，不发行）
    return rel < comp_start


# ── 共识计算 ────────────────────────────────────────────────────────────────
def compute_consensus(
    facts: list[dict],
    competitions: dict[str, dict],
    support_min: float = SUPPORT_MIN,
    breadth_min: int = BREADTH_MIN,
) -> list[dict]:
    """对每个 (task_type, trait, component_type, kb_id) 产出一行共识（含未过阈值的）。

    按 task_type 分池：检测 backbone 不与分类 backbone 在同一 trait 下竞争。
    """
    task_types = sorted({competitions[f["competition"]]["task_type"]
                         for f in facts if f["competition"] in competitions})
    rows: list[dict] = []
    for tt in task_types:
        tt_facts = [f for f in facts
                    if competitions.get(f["competition"], {}).get("task_type") == tt]
        for trait in BOOLEAN_TRAITS:
            rows.extend(_backbone_consensus(tt_facts, competitions, trait,
                                            support_min, breadth_min, tt))
            rows.extend(_loss_consensus(tt_facts, competitions, trait,
                                        support_min, breadth_min, tt))
    rows.sort(key=lambda r: (r["task_type"], r["trait"], r["component_type"],
                             r.get("role") or "", -r["support"]))
    return rows


def _apply_dominance(rows, support_min, breadth_min) -> list[dict]:
    """组内计算头名的 DOMINANCE，并据 (多数决 OR dominance) 设 passed。"""
    ranked = sorted(rows, key=lambda r: r["support"], reverse=True)
    for r in rows:
        r["dominance"] = False
        r["ratio"] = None
        r["margin"] = None
        r["runner_up"] = None
    if ranked:
        top = ranked[0]
        runner_sup = ranked[1]["support"] if len(ranked) > 1 else 0.0
        margin = top["support"] - runner_sup
        ratio = top["support"] / runner_sup if runner_sup > 0 else float("inf")
        top["dominance"] = bool(top["breadth"] >= DOMINANCE_BREADTH_MIN
                                and ratio >= DOMINANCE_RATIO_MIN
                                and margin >= DOMINANCE_MARGIN_MIN)
        top["margin"] = round(margin, 3)
        top["ratio"] = None if ratio == float("inf") else round(ratio, 2)
        top["runner_up"] = ranked[1]["kb_id"] if len(ranked) > 1 else None
    for r in rows:
        majority = r["support"] >= support_min and r["breadth"] >= breadth_min
        r["passed"] = bool(majority or r["dominance"])
    return rows


def _backbone_consensus(facts, competitions, trait, support_min, breadth_min,
                        task_type) -> list[dict]:
    """backbone 共识：车架 / 发动机各成一组，各自分母、各算 support（各比各的）。"""
    votes = {"frame": defaultdict(float), "engine": defaultdict(float)}
    comps = {"frame": defaultdict(set), "engine": defaultdict(set)}
    evidence = {"frame": defaultdict(list), "engine": defaultdict(list)}
    role_total = {"frame": 0.0, "engine": 0.0}
    unknown_votes = 0.0
    contributing_comps: set = set()

    for fact in facts:
        comp = competitions.get(fact["competition"])
        if comp is None or not comp["traits"].get(trait):
            continue
        assert comp["end"] >= MIN_END, f"{comp['slug']} end<{MIN_END}"
        for kb_id, w, emittable, ev in _backbone_contribs(fact, comp):
            contributing_comps.add(fact["competition"])
            role = catalog.family_role(kb_id) if emittable else None
            if role is None:          # unknown（或无角色）：只计 kb_coverage 分母
                unknown_votes += w
                continue
            role_total[role] += w
            votes[role][kb_id] += w
            comps[role][kb_id].add(fact["competition"])
            if ev:
                evidence[role][kb_id].append(ev)

    all_known = role_total["frame"] + role_total["engine"]
    denom_cov = all_known + unknown_votes
    rows = []
    for role in ("frame", "engine"):
        total = role_total[role]
        group = []
        for kb_id, v in votes[role].items():
            group.append({
                "task_type": task_type, "trait": trait,
                "component_type": "backbone", "role": role, "kb_id": kb_id,
                "support": round(v / total if total else 0.0, 4),
                "breadth": len(comps[role][kb_id]),
                "votes": round(v, 3), "total_votes": round(total, 3),
                "unknown_votes": round(unknown_votes, 3),
                "kb_coverage": round(all_known / denom_cov if denom_cov else 0.0, 3),
                "role_share": round(total / denom_cov if denom_cov else 0.0, 3),
                "n_competitions": len(contributing_comps),
                "evidence": evidence[role][kb_id],
            })
        rows.extend(_apply_dominance(group, support_min, breadth_min))  # 组内各比各的
    return rows


def _loss_consensus(facts, competitions, trait, support_min, breadth_min,
                    task_type) -> list[dict]:
    votes: dict[str, float] = defaultdict(float)
    comps: dict[str, set] = defaultdict(set)
    evidence: dict[str, list] = defaultdict(list)
    total = 0.0
    unknown_votes = 0.0
    contributing_comps: set = set()

    for fact in facts:
        comp = competitions.get(fact["competition"])
        if comp is None or not comp["traits"].get(trait):
            continue
        assert comp["end"] >= MIN_END, f"{comp['slug']} end<{MIN_END}"
        for kb_id, w, is_emittable, ev in _loss_contribs(fact, comp):
            contributing_comps.add(fact["competition"])
            if not is_emittable:
                unknown_votes += w
                continue
            total += w
            votes[kb_id] += w
            comps[kb_id].add(fact["competition"])
            if ev:
                evidence[kb_id].append(ev)

    coverage = total / (total + unknown_votes) if (total + unknown_votes) else 0.0
    rows = []
    for kb_id, v in votes.items():
        rows.append({
            "task_type": task_type, "trait": trait,
            "component_type": "loss", "role": None, "kb_id": kb_id,
            "support": round(v / total if total else 0.0, 4),
            "breadth": len(comps[kb_id]),
            "votes": round(v, 3), "total_votes": round(total, 3),
            "unknown_votes": round(unknown_votes, 3),
            "kb_coverage": round(coverage, 3),
            "n_competitions": len(contributing_comps),
            "evidence": evidence[kb_id],
        })
    return _apply_dominance(rows, support_min, breadth_min)


def _backbone_contribs(fact, comp):
    """→ list[(kb_id, weight, is_emittable, evidence|None)]，含共存过滤。"""
    out = []
    weights = family_vote_weights(fact)
    raw_by_fam = _raw_models_by_family(fact)
    cite = (fact.get("citations") or [None])[0]
    for fam, w in weights.items():
        if not coexists(fam, comp["start"]):
            continue   # 共存不过：不计票不计分母
        emittable = fam in catalog.FAMILY_RELEASE   # 已知家族才发行
        ev = {"competition": fact["competition"], "rank": fact.get("rank"),
              "raw": ", ".join(raw_by_fam.get(fam, [])), "citation": cite} if emittable else None
        out.append((fam, w, emittable, ev))
    return out


def _loss_contribs(fact, comp):
    if not comp.get("loss_voting", True):
        return []   # metric-learning 竞赛整场排除 loss 投票
    loss_kb, w = loss_vote(fact)
    if not loss_kb or w == 0:
        return []
    emittable = loss_kb != "unknown"
    cite = (fact.get("citations") or [None])[0]
    ev = {"competition": fact["competition"], "rank": fact.get("rank"),
          "raw": fact.get("loss_raw"), "citation": cite} if emittable else None
    return [(loss_kb, w, emittable, ev)]


def _raw_models_by_family(fact) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for m in fact.get("members_raw", []):
        raw = m.get("raw_model")
        if raw:
            out[catalog.map_model(raw)].append(raw)
    return out


# ── 三张侧表 ────────────────────────────────────────────────────────────────
def build_side_tables(
    facts: list[dict],
    competitions: dict[str, dict] | None = None,
) -> tuple[dict, dict, dict]:
    competitions = competitions or catalog.COMPETITIONS
    unknown_models: Counter = Counter()
    unknown_losses: dict[str, dict] = {}
    recipes: dict[str, dict] = defaultdict(lambda: defaultdict(list))
    cooccur: dict[str, Counter] = defaultdict(Counter)

    for fact in facts:
        comp_traits = None  # recipes 需要 trait；下面按竞赛特征展开
        # unknown 模型
        for fam, raws in _raw_models_by_family(fact).items():
            if fam == "unknown":
                for r in raws:
                    unknown_models[r] += 1
        # unknown loss
        loss_kb = fact.get("loss_kb")
        if loss_kb == "unknown" and fact.get("loss_raw"):
            r = fact["loss_raw"]
            rec = unknown_losses.setdefault(
                r, {"count": 0, "metric_learning": False, "hybrid": False})
            rec["count"] += 1
            rec["metric_learning"] = rec["metric_learning"] or bool(fact.get("loss_is_metric_learning"))
            rec["hybrid"] = rec["hybrid"] or catalog.is_hybrid_loss(r)
        # ensemble 共现
        fams = [f for f in fact.get("families", []) if f in catalog.FAMILY_RELEASE]
        if fact.get("kind") == "ensemble" and len(fams) >= 2:
            for a, b in combinations(sorted(set(fams)), 2):
                cooccur[a][b] += 1
                cooccur[b][a] += 1

    # recipes: (family, trait) → image_size 分布（用竞赛特征展开 trait）
    for fact in facts:
        comp = competitions.get(fact["competition"])
        if not comp:
            continue
        fis = fact.get("family_image_size", {})
        for fam, size in fis.items():
            if fam not in catalog.FAMILY_RELEASE:
                continue
            for trait in BOOLEAN_TRAITS:
                if comp["traits"].get(trait):
                    recipes[f"{fam}|{trait}"]["image_sizes"].append(size)

    # recipes 收敛为 {key: {mode, distribution}}
    recipes_out = {}
    for key, d in recipes.items():
        sizes = d["image_sizes"]
        if sizes:
            recipes_out[key] = {
                "mode": Counter(sizes).most_common(1)[0][0],
                "distribution": dict(Counter(sizes)),
                "n": len(sizes),
            }

    unknown_out = {
        "models": dict(unknown_models.most_common()),
        "losses": dict(sorted(unknown_losses.items(), key=lambda kv: -kv[1]["count"])),
    }
    cooccur_out = {a: dict(c.most_common()) for a, c in cooccur.items()}
    return unknown_out, recipes_out, cooccur_out


# ── Markdown 渲染 ───────────────────────────────────────────────────────────
def render_md(rows: list[dict], competitions: dict[str, dict]) -> str:
    any_unverified = any(not c.get("traits_verified") for c in competitions.values())
    lines = ["# consensus.md — 数据集特征 → 组件共识", ""]
    if any_unverified:
        lines += ["> ⚠ 存在 `traits_verified=False` 的竞赛——特征卡尚未人工核对，"
                  "以下共识为初判。逐竞赛核对后重跑本表。", ""]
    lines += [f"> 阈值：support ≥ {SUPPORT_MIN}，breadth ≥ {BREADTH_MIN}。"
              "`passed` 列标记是否达标。", ""]

    # 每个显示组：(标题, 过滤器)。backbone 分车架/发动机两张子表。
    groups = [
        ("backbone·车架 frame", lambda r: r["component_type"] == "backbone"
         and r.get("role") == "frame"),
        ("backbone·发动机 engine", lambda r: r["component_type"] == "backbone"
         and r.get("role") == "engine"),
        ("loss", lambda r: r["component_type"] == "loss"),
    ]
    task_types = sorted({r["task_type"] for r in rows})
    for tt in task_types:
        lines.append(f"# task_type = {tt}")
        lines.append("")
        for trait in BOOLEAN_TRAITS:
            for gname, gfilter in groups:
                sub = [r for r in rows if r["task_type"] == tt
                       and r["trait"] == trait and gfilter(r)]
                if not sub:
                    continue
                cov = sub[0].get("kb_coverage", 0.0)
                share = sub[0].get("role_share")
                extra = f"，本组占全部 backbone 票 {share:.0%}" if share is not None else ""
                cov_note = f"（KB 覆盖率 {cov:.0%}{extra}）" if cov < 0.95 or share else ""
                lines.append(f"## {trait} — {gname} {cov_note}")
                lines.append("")
                lines.append("| kb_id | support | breadth | votes/total | passed | raw（归并痕迹） |")
                lines.append("|---|---|---|---|---|---|")
                for r in sub:
                    raws = sorted({e["raw"] for e in r["evidence"] if e.get("raw")
                                   and e["raw"] != r["kb_id"]})
                    raw_note = ", ".join(raws)[:60] if raws else ""
                    if r["passed"] and r.get("dominance"):
                        mark = f"✔dom(×{r['ratio']},+{r['margin']})" if r.get("ratio") \
                            else "✔dom"
                    elif r["passed"]:
                        mark = "✔"
                    else:
                        mark = ""
                    lines.append(
                        f"| {r['kb_id']} | {r['support']:.2f} | {r['breadth']} | "
                        f"{r['votes']:.1f}/{r['total_votes']:.1f} | {mark} | {raw_note} |")
                lines.append("")
    return "\n".join(lines)


# ── 编排 ────────────────────────────────────────────────────────────────────
def run_aggregate(
    in_path: Path = DEFAULT_IN,
    support_min: float = SUPPORT_MIN,
    breadth_min: int = BREADTH_MIN,
    competitions: dict[str, dict] | None = None,
) -> list[dict]:
    competitions = competitions or catalog.COMPETITIONS
    facts = [json.loads(l) for l in in_path.open(encoding="utf-8") if l.strip()]

    rows = compute_consensus(facts, competitions, support_min, breadth_min)
    unknown, recipes, cooccur = build_side_tables(facts, competitions)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_MD.write_text(render_md(rows, competitions), encoding="utf-8")
    OUT_UNKNOWN.write_text(json.dumps(unknown, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_RECIPES.write_text(json.dumps(recipes, ensure_ascii=False, indent=2), encoding="utf-8")
    OUT_COOCC.write_text(json.dumps(cooccur, ensure_ascii=False, indent=2), encoding="utf-8")

    n_pass = sum(1 for r in rows if r["passed"])
    print(f"[aggregate] rows={len(rows)} passed={n_pass} "
          f"unknown_models={len(unknown['models'])} recipes={len(recipes)}")
    return rows


def _cli() -> None:
    ap = argparse.ArgumentParser(description="facts.jsonl → consensus + 侧表")
    ap.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    ap.add_argument("--support-min", type=float, default=SUPPORT_MIN)
    ap.add_argument("--breadth-min", type=int, default=BREADTH_MIN)
    args = ap.parse_args()
    run_aggregate(args.in_path, args.support_min, args.breadth_min)


if __name__ == "__main__":
    _cli()

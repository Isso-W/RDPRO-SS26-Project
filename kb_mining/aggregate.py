"""Aggregate facts.jsonl into consensus files and side tables.

Consensus is measured by support and breadth for family A within eligible
competitions that have trait T. Voting rules and eligibility filters are
documented in docs/kb_mining_protocol.md.

Boundary choices:
  - unknown stays in the denominator, because a high unknown share is itself a
    KB coverage signal. It is not emitted as a consensus row; it goes to the
    unknown_components side table.
  - metric-learning competitions with catalog `loss_voting=False` are excluded
    from loss voting.

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

# Dominance accepts a clear group leader even when support is below 0.5. This
# keeps fragmented tasks from losing useful backbone proposals.
DOMINANCE_RATIO_MIN = 1.5    # top support / runner-up support
DOMINANCE_BREADTH_MIN = 3    # number of competitions behind the top family
DOMINANCE_MARGIN_MIN = 0.10  # top support minus runner-up support
BOOLEAN_TRAITS = ("fine_grained", "class_imbalance", "medical", "multi_label")
MIN_END = "2021-01"


# Vote weights.
def family_vote_weights(fact: dict) -> dict[str, float]:
    """Return per-family vote weights for one fact."""
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
    """Return the loss vote for one fact, or (None, 0.0)."""
    loss_kb = fact.get("loss_kb")
    if not loss_kb:
        return None, 0.0
    weights = family_vote_weights(fact)
    w = max(weights.values()) if weights else 0.0
    return loss_kb, w


def coexists(family: str, comp_start: str) -> bool:
    """Return whether a family existed before the competition start date."""
    rel = catalog.FAMILY_RELEASE.get(family)
    if rel is None:
        return True   # unknown stays in denominator/side table only
    return rel < comp_start


# Consensus computation.
def compute_consensus(
    facts: list[dict],
    competitions: dict[str, dict],
    support_min: float = SUPPORT_MIN,
    breadth_min: int = BREADTH_MIN,
) -> list[dict]:
    """Emit one row for each (task_type, trait, component_type, kb_id).

    Pools are separated by task_type so, for example, detection backbones do not
    compete with classification backbones under the same trait.
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
    """Mark rows as passed by majority or within-group dominance."""
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
    """Compute backbone consensus in separate frame and engine groups."""
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
            if role is None:          # unknown/no-role votes count for coverage only
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
        rows.extend(_apply_dominance(group, support_min, breadth_min))
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
    """Return backbone contributions after coexistence filtering."""
    out = []
    weights = family_vote_weights(fact)
    raw_by_fam = _raw_models_by_family(fact)
    cite = (fact.get("citations") or [None])[0]
    for fam, w in weights.items():
        if not coexists(fam, comp["start"]):
            continue
        emittable = fam in catalog.FAMILY_RELEASE
        ev = {"competition": fact["competition"], "rank": fact.get("rank"),
              "raw": ", ".join(raw_by_fam.get(fam, [])), "citation": cite} if emittable else None
        out.append((fam, w, emittable, ev))
    return out


def _loss_contribs(fact, comp):
    if not comp.get("loss_voting", True):
        return []   # exclude metric-learning competitions from loss voting
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


# Side tables.
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
        comp_traits = None  # recipes expand traits below
        # Unknown models.
        for fam, raws in _raw_models_by_family(fact).items():
            if fam == "unknown":
                for r in raws:
                    unknown_models[r] += 1
        # Unknown loss.
        loss_kb = fact.get("loss_kb")
        if loss_kb == "unknown" and fact.get("loss_raw"):
            r = fact["loss_raw"]
            rec = unknown_losses.setdefault(
                r, {"count": 0, "metric_learning": False, "hybrid": False})
            rec["count"] += 1
            rec["metric_learning"] = rec["metric_learning"] or bool(fact.get("loss_is_metric_learning"))
            rec["hybrid"] = rec["hybrid"] or catalog.is_hybrid_loss(r)
        # Ensemble co-occurrence.
        fams = [f for f in fact.get("families", []) if f in catalog.FAMILY_RELEASE]
        if fact.get("kind") == "ensemble" and len(fams) >= 2:
            for a, b in combinations(sorted(set(fams)), 2):
                cooccur[a][b] += 1
                cooccur[b][a] += 1

    # Recipes: (family, trait) -> image_size distribution.
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

    # Collapse recipes to {key: {mode, distribution}}.
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


# Markdown rendering.
def render_md(rows: list[dict], competitions: dict[str, dict]) -> str:
    any_unverified = any(not c.get("traits_verified") for c in competitions.values())
    lines = ["# Consensus: Dataset Traits to Components", ""]
    if any_unverified:
        lines += ["> Some competitions still have `traits_verified=False`. Treat these "
                  "rows as preliminary until the feature cards are checked.", ""]
    lines += [f"> Thresholds: support >= {SUPPORT_MIN}, breadth >= {BREADTH_MIN}. "
              "The `passed` column marks rows that meet the rule.", ""]

    # Display groups. Backbones are split into frame and engine subtables.
    groups = [
        ("backbone frame", lambda r: r["component_type"] == "backbone"
         and r.get("role") == "frame"),
        ("backbone engine", lambda r: r["component_type"] == "backbone"
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
                extra = f", group share of backbone votes {share:.0%}" if share is not None else ""
                cov_note = f"(KB coverage {cov:.0%}{extra})" if cov < 0.95 or share else ""
                lines.append(f"## {trait} - {gname} {cov_note}")
                lines.append("")
                lines.append("| kb_id | support | breadth | votes/total | passed | raw aliases |")
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


# Orchestration.
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
    ap = argparse.ArgumentParser(description="facts.jsonl to consensus and side tables")
    ap.add_argument("--in", dest="in_path", type=Path, default=DEFAULT_IN)
    ap.add_argument("--support-min", type=float, default=SUPPORT_MIN)
    ap.add_argument("--breadth-min", type=int, default=BREADTH_MIN)
    args = ap.parse_args()
    run_aggregate(args.in_path, args.support_min, args.breadth_min)


if __name__ == "__main__":
    _cli()

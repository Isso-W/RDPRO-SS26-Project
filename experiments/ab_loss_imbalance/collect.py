"""collect.py - outcomes.jsonl to per-testbed and overall verdicts.

Decision rule, preregistered in the protocol: for each fold i, compute the
paired difference `d_i = metric(CE, fold_i) - metric(focal, fold_i)`. Let
dbar=mean(d), SE=std/sqrt(n), and tie_band=max(MARGIN_FLOOR, 2*SE):
  dbar >= tie_band   -> CE_WINS
  dbar <= -tie_band  -> FOCAL_WINS
  otherwise          -> TIE
Two-testbed merge: a side wins only with at least one supporting testbed and no
opposing testbed; otherwise the result is TIE.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from experiments.ab_loss_imbalance.configs import (
    ARMS, MARGIN_FLOOR, TESTBEDS,
)

CE = "cross_entropy_loss"
FOCAL = "focal_loss"
DEFAULT_OUTCOMES = Path("experiments/ab_loss_imbalance/results/outcomes.jsonl")


# Per-testbed verdict.
def testbed_verdict(deltas: list[float], margin_floor: float = MARGIN_FLOOR) -> dict:
    """Return {verdict, dbar, se, band, n} for paired differences."""
    n = len(deltas)
    dbar = statistics.fmean(deltas) if deltas else 0.0
    se = (statistics.stdev(deltas) / (n ** 0.5)) if n >= 2 else float("inf")
    band = max(margin_floor, 2 * se) if se != float("inf") else float("inf")
    if band == float("inf"):
        verdict = "TIE"                      # not enough folds; stay conservative
    elif dbar >= band:
        verdict = "CE_WINS"
    elif dbar <= -band:
        verdict = "FOCAL_WINS"
    else:
        verdict = "TIE"
    return {"verdict": verdict, "dbar": dbar, "se": se, "band": band, "n": n}


def merge_verdicts(testbed_verdicts: list[str]) -> str:
    """Merge testbed verdicts with support required and no opposing result."""
    if any(v == "FOCAL_WINS" for v in testbed_verdicts):
        if any(v == "CE_WINS" for v in testbed_verdicts):
            return "TIE"                      # conflicting testbeds
        return "FOCAL_WINS" if all(v != "CE_WINS" for v in testbed_verdicts) else "TIE"
    if any(v == "CE_WINS" for v in testbed_verdicts):
        return "CE_WINS"
    return "TIE"


# Paired-difference extraction.
def paired_deltas(records: list[dict], testbed: str, metric: str) -> list[float]:
    """Extract per-fold CE-focal differences for one testbed and metric."""
    by_arm_fold: dict[tuple, float] = {}
    for r in records:
        if r.get("benchmark") != testbed:
            continue
        val = (r.get("val_metric") or {}).get(metric)
        if val is None:
            continue
        by_arm_fold[(r["arm"], r["fold"])] = float(val)
    deltas = []
    folds = sorted({f for (_, f) in by_arm_fold})
    for f in folds:
        if (CE, f) in by_arm_fold and (FOCAL, f) in by_arm_fold:
            deltas.append(by_arm_fold[(CE, f)] - by_arm_fold[(FOCAL, f)])
    return deltas


# Summary.
def summarize(records: list[dict]) -> dict:
    per_testbed = {}
    for testbed, tb in TESTBEDS.items():
        deltas = paired_deltas(records, testbed, tb["metric"])
        res = testbed_verdict(deltas)
        res["metric"] = tb["metric"]
        res["deltas"] = deltas
        per_testbed[testbed] = res
    overall = merge_verdicts([r["verdict"] for r in per_testbed.values()])
    return {"per_testbed": per_testbed, "overall": overall}


def render(summary: dict) -> str:
    lines = ["# A/B Arbitration Result - CE vs focal under class_imbalance", ""]
    for testbed, r in summary["per_testbed"].items():
        lines.append(f"## {testbed} (primary metric {r['metric']}, n={r['n']} folds)")
        if r["deltas"]:
            ds = ", ".join(f"{d:+.4f}" for d in r["deltas"])
            se = "inf" if r["se"] == float("inf") else f"{r['se']:.4f}"
            band = "inf" if r["band"] == float("inf") else f"{r['band']:.4f}"
            lines.append(f"  d(CE-focal) by fold: [{ds}]")
            lines.append(f"  dbar={r['dbar']:+.4f}  SE={se}  tie band=+/-{band}  -> **{r['verdict']}**")
        else:
            lines.append(f"  No paired fold data -> **{r['verdict']}**")
        lines.append("")
    lines.append(f"## Overall verdict: **{summary['overall']}**")
    return "\n".join(lines)


def load_outcomes(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def _cli() -> None:
    ap = argparse.ArgumentParser(description="Summarize A/B fold-level results into a verdict")
    ap.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    args = ap.parse_args()
    summary = summarize(load_outcomes(args.outcomes))
    print(render(summary))


if __name__ == "__main__":
    _cli()

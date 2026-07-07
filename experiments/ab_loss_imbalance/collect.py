"""collect.py — outcomes.jsonl → 台级 + 总 verdict（纯函数，无网络）。

裁决规则（预注册，见 plan §0）：对每折 i 的配对差
`Δ_i = metric(CE, fold_i) − metric(focal, fold_i)`，取 Δ̄=mean，SE=std/√n，
平局带 = max(MARGIN_FLOOR, 2·SE)：
  Δ̄ ≥ 带 → CE_WINS；Δ̄ ≤ −带 → FOCAL_WINS；其间 → TIE（现状赢含糊局面）。
双台合并：无一台 focal 胜且≥1 台 CE 胜 → CE_WINS（focal 对称）；其余 → TIE。
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


# ── 台级裁决（纯函数）───────────────────────────────────────────────────────
def testbed_verdict(deltas: list[float], margin_floor: float = MARGIN_FLOOR) -> dict:
    """配对差列表 → {verdict, dbar, se, band, n}。verdict ∈ CE_WINS/FOCAL_WINS/TIE。"""
    n = len(deltas)
    dbar = statistics.fmean(deltas) if deltas else 0.0
    se = (statistics.stdev(deltas) / (n ** 0.5)) if n >= 2 else float("inf")
    band = max(margin_floor, 2 * se) if se != float("inf") else float("inf")
    if band == float("inf"):
        verdict = "TIE"                      # 折不足，无法裁决 → 保守 TIE
    elif dbar >= band:
        verdict = "CE_WINS"
    elif dbar <= -band:
        verdict = "FOCAL_WINS"
    else:
        verdict = "TIE"
    return {"verdict": verdict, "dbar": dbar, "se": se, "band": band, "n": n}


def merge_verdicts(testbed_verdicts: list[str]) -> str:
    """双台合并：翻案需无反对 + 至少一台支持；否则 TIE。"""
    if any(v == "FOCAL_WINS" for v in testbed_verdicts):
        if any(v == "CE_WINS" for v in testbed_verdicts):
            return "TIE"                      # 两台互相矛盾 → 现状赢
        return "FOCAL_WINS" if all(v != "CE_WINS" for v in testbed_verdicts) else "TIE"
    if any(v == "CE_WINS" for v in testbed_verdicts):
        return "CE_WINS"
    return "TIE"


# ── 配对差提取 ──────────────────────────────────────────────────────────────
def paired_deltas(records: list[dict], testbed: str, metric: str) -> list[float]:
    """从 outcomes 记录里取某台 CE−focal 的逐折配对差（缺折则跳过该折）。"""
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


# ── 汇总 ────────────────────────────────────────────────────────────────────
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
    lines = ["# A/B 仲裁结果 — CE vs focal（class_imbalance）", ""]
    for testbed, r in summary["per_testbed"].items():
        lines.append(f"## {testbed}（主指标 {r['metric']}，n={r['n']} 折）")
        if r["deltas"]:
            ds = ", ".join(f"{d:+.4f}" for d in r["deltas"])
            se = "inf" if r["se"] == float("inf") else f"{r['se']:.4f}"
            band = "inf" if r["band"] == float("inf") else f"{r['band']:.4f}"
            lines.append(f"  Δ(CE−focal) 逐折: [{ds}]")
            lines.append(f"  Δ̄={r['dbar']:+.4f}  SE={se}  平局带=±{band}  → **{r['verdict']}**")
        else:
            lines.append(f"  （无配对折数据）→ **{r['verdict']}**")
        lines.append("")
    lines.append(f"## 总 verdict：**{summary['overall']}**")
    return "\n".join(lines)


def load_outcomes(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def _cli() -> None:
    ap = argparse.ArgumentParser(description="汇总 A/B 折级结果 → verdict")
    ap.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES)
    args = ap.parse_args()
    summary = summarize(load_outcomes(args.outcomes))
    print(render(summary))


if __name__ == "__main__":
    _cli()

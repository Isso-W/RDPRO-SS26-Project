"""Summarise the outcome memory into a quality-vs-cost table (and optional plot).

This is the raw material for the efficiency selling point: for every logged run, show
the achieved quality next to its cost (LLM tokens/calls, epochs, wall-clock). Later, the
same view compares Jiaozi against an agentic pipeline (AIDE / MLE-STAR) — Jiaozi should
sit in the cheap corner.

  python -m recommender.report --memory recommender/outcomes.jsonl
  python -m recommender.report --plot quality_vs_cost.png        # needs matplotlib
"""

from __future__ import annotations

import argparse

from .outcome_memory import OutcomeMemory


def summarize(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        result = record.get("result", {}) or {}
        cost = record.get("cost", {}) or {}
        config = record.get("config", {}) or {}
        rows.append({
            "dataset": record.get("dataset_id"),
            "backbone": config.get("backbone"),
            "metric_name": result.get("metric_name"),
            "metric": result.get("metric_value"),
            "tokens": cost.get("llm_tokens"),
            "llm_calls": cost.get("llm_calls"),
            "epochs": cost.get("epochs"),
            "wall_clock_sec": cost.get("wall_clock_sec"),
        })
    return rows


def format_table(rows: list[dict]) -> str:
    if not rows:
        return "(no outcomes logged yet)"
    cols = ["dataset", "backbone", "metric_name", "metric", "tokens", "llm_calls", "epochs", "wall_clock_sec"]
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    lines = [header, sep]
    for r in rows:
        lines.append("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    return "\n".join(lines)


def plot_quality_vs_cost(rows: list[dict], out_png: str, cost_key: str = "tokens") -> str | None:
    """Scatter of quality vs a cost axis. Returns the path, or None if matplotlib is absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[report] matplotlib not available; skipping plot.")
        return None

    pts = [(r.get(cost_key), r.get("metric"), r.get("backbone")) for r in rows
           if r.get(cost_key) is not None and r.get("metric") is not None]
    if not pts:
        print("[report] no (cost, metric) points to plot.")
        return None

    fig, ax = plt.subplots(figsize=(6, 4))
    for x, y, label in pts:
        ax.scatter(x, y)
        ax.annotate(str(label), (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(cost_key)
    ax.set_ylabel("quality (metric)")
    ax.set_title("Quality vs cost")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    print(f"[report] Saved plot -> {out_png}")
    return out_png


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarise the outcome memory.")
    parser.add_argument("--memory", default=None, help="Outcome-memory JSONL (default: recommender/outcomes.jsonl).")
    parser.add_argument("--plot", default=None, help="Save a quality-vs-cost scatter to this PNG.")
    parser.add_argument("--cost-axis", default="tokens", choices=["tokens", "llm_calls", "epochs", "wall_clock_sec"])
    args = parser.parse_args()

    memory = OutcomeMemory(args.memory) if args.memory else OutcomeMemory()
    rows = summarize(memory.all())
    print(format_table(rows))
    print(f"\n{len(rows)} outcome(s) in {memory.path}")
    if args.plot:
        plot_quality_vs_cost(rows, args.plot, cost_key=args.cost_axis)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

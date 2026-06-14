"""Memory-aware ranking of Module 3 candidates, with an explanation per pick.

For each candidate, predict its metric on the new dataset from the outcome memory
(similarity-weighted average over past runs of the *same backbone* on *similar
datasets*). Candidates with a track record are ranked by that predicted metric;
candidates with none keep their KB-heuristic order and are listed after (cold
start). Every candidate carries a human-readable rationale.
"""

from __future__ import annotations

from .outcome_memory import OutcomeMemory


def _predict_from_memory(candidate: dict, fingerprint: dict, memory: OutcomeMemory, k: int):
    """Similarity-weighted metric prediction for a candidate. Returns (pred|None, hits)."""
    backbone = candidate.get("backbone")
    hits = memory.query_similar(fingerprint, k=k, backbone=backbone)
    num = den = 0.0
    used = []
    for hit in hits:
        metric = hit.get("result", {}).get("metric_value")
        if metric is None:
            continue
        weight = 1.0 / (1.0 + hit["distance"])
        num += weight * float(metric)
        den += weight
        used.append(hit)
    if den == 0:
        return None, []
    return num / den, used


def _explain(candidate: dict, fingerprint: dict, hits: list[dict]) -> str:
    backbone = candidate.get("backbone")
    checkpoint = candidate.get("pretrained")
    parts = [f"{backbone}" + (f" / {checkpoint}" if checkpoint else " (train from scratch)")]

    if candidate.get("rank_basis") == "memory" and hits:
        closest = min(hits, key=lambda h: h["distance"])
        ds = closest.get("dataset_id") or "a similar dataset"
        cm = closest.get("result", {}).get("metric_value")
        parts.append(
            f"predicted {candidate['predicted_metric']} from {len(hits)} similar past run(s); "
            f"closest: {ds} scored {cm}"
        )
    else:
        parts.append("no prior outcomes for similar tasks — ranked by KB heuristic + vector match (cold start)")

    sig = f"task={fingerprint.get('task_type')}, classes={fingerprint.get('num_classes')}, data={fingerprint.get('data_size')}"
    if fingerprint.get("class_imbalance"):
        sig += ", imbalanced"
    parts.append(sig)
    return "; ".join(parts)


def rank_candidates(
    candidates: list[dict],
    fingerprint: dict,
    memory: OutcomeMemory,
    k: int = 5,
    minimize: bool = False,
) -> list[dict]:
    """Re-rank Module 3 candidates using the outcome memory; annotate each with a rationale.

    `minimize=True` for metrics where lower is better (e.g. log_loss).
    Returns a new list of candidate dicts (originals untouched), each with added
    `predicted_metric`, `memory_support`, `rank_basis`, `explanation`.
    """
    with_mem: list[dict] = []
    without_mem: list[dict] = []

    for candidate in candidates:
        pred, hits = _predict_from_memory(candidate, fingerprint, memory, k)
        enriched = dict(candidate)
        enriched["predicted_metric"] = round(pred, 4) if pred is not None else None
        enriched["memory_support"] = len(hits)
        enriched["rank_basis"] = "memory" if pred is not None else "heuristic"
        enriched["explanation"] = _explain(enriched, fingerprint, hits)
        (with_mem if pred is not None else without_mem).append(enriched)

    # Candidates with a track record first, ordered by predicted metric.
    with_mem.sort(key=lambda c: c["predicted_metric"], reverse=not minimize)
    # Cold-start candidates keep their incoming (heuristic) order.
    return with_mem + without_mem


def recommend(
    candidates: list[dict],
    m2_report: dict,
    m3_input: dict,
    memory=None,
    k: int = 5,
) -> list[dict]:
    """High-level entry point: fingerprint the dataset, then memory-rank + explain.

    `candidates` is Module 3's retrieval output; `m2_report` the Module 2 analysis;
    `m3_input` the merged Module 1+2 input. Returns ranked, explained candidates.
    """
    from .fingerprint import dataset_fingerprint
    from .outcome_memory import OutcomeMemory

    fingerprint = dataset_fingerprint(m2_report, m3_input)
    mem = memory if memory is not None else OutcomeMemory()
    metric = str((m3_input or {}).get("evaluation_metric", "accuracy")).lower()
    minimize = metric in {"log_loss", "multiclass_log_loss", "rmse"}
    return rank_candidates(candidates, fingerprint, mem, k=k, minimize=minimize)


def log_run(
    m2_report: dict,
    m3_input: dict,
    config: dict,
    result: dict,
    dataset_id: str | None = None,
    memory=None,
    cost: dict | None = None,
) -> dict:
    """Close the loop: fingerprint the dataset and log one run's outcome to memory.

    Call this after a real training run so the next recommendation is better informed.
    Returns the fingerprint that was logged.
    """
    from .fingerprint import dataset_fingerprint
    from .outcome_memory import OutcomeMemory

    fingerprint = dataset_fingerprint(m2_report, m3_input)
    mem = memory if memory is not None else OutcomeMemory()
    mem.log(fingerprint, config, result, dataset_id=dataset_id, cost=cost)
    return fingerprint


def log_from_summary(
    summary: dict,
    m2_report: dict,
    m3_input: dict,
    config: dict | None = None,
    dataset_id: str | None = None,
    memory=None,
    cost: dict | None = None,
) -> dict | None:
    """Log an outcome from a generated run.py summary (the {train, evaluate, infer} dict).

    `config` (the project's flattened configs.json) is preferred for the logged config —
    it carries backbone / checkpoint / params; otherwise the summary's compact config is
    used. Returns the logged fingerprint, or None if there's no usable metric.
    """
    evaluate = summary.get("evaluate", {}) or {}
    if evaluate.get("metric_value") is None:
        return None

    logged_config = dict(config) if config else dict(summary.get("config", {}))
    result = {
        "metric_name": evaluate.get("metric_name", "accuracy"),
        "metric_value": evaluate.get("metric_value"),
        "macro_f1": evaluate.get("macro_f1"),
        "status": evaluate.get("status", summary.get("status")),
    }
    return log_run(m2_report, m3_input, logged_config, result,
                   dataset_id=dataset_id, memory=memory, cost=cost)

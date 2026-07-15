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

    basis = candidate.get("rank_basis")
    if basis == "memory" and hits:
        closest = min(hits, key=lambda h: h["distance"])
        ds = closest.get("dataset_id") or "a similar dataset"
        cm = closest.get("result", {}).get("metric_value")
        parts.append(
            f"predicted {candidate['predicted_metric']} from {len(hits)} similar past run(s); "
            f"closest: {ds} scored {cm}"
        )
    elif basis == "logme":
        parts.append(
            f"no track record — LogME transferability {candidate.get('logme_score')} measured on this "
            "dataset's frozen features (cold start)"
        )
    else:
        parts.append("no track record or LogME — ranked by KB heuristic + vector match (cold start)")

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
    logme_scores: dict | None = None,
) -> list[dict]:
    """Re-rank Module 3 candidates using the outcome memory; annotate each with a rationale.

    `minimize=True` for metrics where lower is better (e.g. log_loss).
    `logme_scores` maps backbone -> LogME transferability; used as the cold-start signal
    (better than the KB heuristic) when memory has no track record for a candidate.

    Tiers, best first: memory-backed (by predicted metric) > LogME-scored (by LogME) >
    heuristic-only (incoming order). Each candidate gains `predicted_metric`,
    `memory_support`, `logme_score`, `rank_basis`, `explanation`.
    """
    logme_scores = logme_scores or {}
    with_mem: list[dict] = []
    with_logme: list[dict] = []
    heuristic: list[dict] = []

    for candidate in candidates:
        pred, hits = _predict_from_memory(candidate, fingerprint, memory, k)
        enriched = dict(candidate)
        enriched["predicted_metric"] = round(pred, 4) if pred is not None else None
        enriched["memory_support"] = len(hits)
        logme = logme_scores.get(candidate.get("backbone"))
        enriched["logme_score"] = round(logme, 4) if logme is not None else None

        if pred is not None:
            enriched["rank_basis"] = "memory"
            with_mem.append(enriched)
        elif logme is not None:
            enriched["rank_basis"] = "logme"
            with_logme.append(enriched)
        else:
            enriched["rank_basis"] = "heuristic"
            heuristic.append(enriched)
        enriched["explanation"] = _explain(enriched, fingerprint, hits)

    with_mem.sort(key=lambda c: c["predicted_metric"], reverse=not minimize)
    with_logme.sort(key=lambda c: c["logme_score"], reverse=True)  # higher LogME = better
    # heuristic-only candidates keep their incoming (KB) order
    return with_mem + with_logme + heuristic


def recommend(
    candidates: list[dict],
    m2_report: dict,
    m3_input: dict,
    memory=None,
    k: int = 5,
    logme_scores: dict | None = None,
) -> list[dict]:
    """High-level entry point: fingerprint the dataset, then memory-rank + explain.

    `candidates` is Module 3's retrieval output; `m2_report` the Module 2 analysis;
    `m3_input` the merged Module 1+2 input. `logme_scores` (backbone -> LogME) is an
    optional cold-start signal. Returns ranked, explained candidates.
    """
    from .fingerprint import dataset_fingerprint
    from .outcome_memory import OutcomeMemory

    fingerprint = dataset_fingerprint(m2_report, m3_input)
    mem = memory if memory is not None else OutcomeMemory()
    metric = str((m3_input or {}).get("evaluation_metric", "accuracy")).lower()
    minimize = metric in {"log_loss", "multiclass_log_loss", "rmse"}
    return rank_candidates(candidates, fingerprint, mem, k=k, minimize=minimize,
                           logme_scores=logme_scores)


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

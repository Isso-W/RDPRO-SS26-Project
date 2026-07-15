# Accumulating, explainable recommender (`recommender/`)

The positional moat vs an autonomous black-box agent (e.g. MLE-STAR): a recommender that
**accumulates** across tasks, **explains** every pick, and is **cheap** — none of which a
per-task, from-scratch, opaque search loop can do. Sits on top of Module 3's
constraint-aware candidate shortlist.

## What it does

```
Module 3 (KB + rules + budget filter)  → in-budget candidate shortlist
        │
        ▼
recommender.recommend(candidates, m2_report, m3_input, memory)
        │  1. dataset_fingerprint  — semantic signals from the Module 2 report
        │  2. memory lookup        — similar past runs of the same backbone
        │  3. rank + explain       — by predicted metric; cold start → heuristic
        ▼
ranked candidates, each with predicted_metric / memory_support / rank_basis / explanation
```

After a run, `OutcomeMemory.log(fingerprint, config, result)` appends the outcome, so the
next recommendation is better informed — the system improves as it is used.

## Modules

- **`fingerprint.py`** — `dataset_fingerprint(m2_report, m3_input)` → semantic signals
  (task, num_classes, data_size, resolution_tier, color_mode, class_imbalance), reusing the
  image statistics Module 2 already computes. `fingerprint_distance` compares two
  (task_type is a hard gate).
- **`outcome_memory.py`** — `OutcomeMemory`: a JSONL log of `(fingerprint, config, result)`;
  `query_similar(fingerprint, k, backbone)` returns nearest past runs. Inspectable, portable,
  and doubles as training data for a learned predictor later.
- **`ranker.py`** — `rank_candidates` / `recommend`: similarity-weighted metric prediction
  per candidate (kNN over same-backbone records), rank by it, cold-start candidates fall back
  to the KB heuristic. Every candidate gets a rationale.

## Why this beats hardcoded rules (and MLE-STAR can't follow)

- **Accumulation**: ranking is driven by *measured outcomes on similar datasets*, not fixed
  heuristics — and it gets better with every logged run. MLE-STAR forgets between tasks.
- **Explainability**: each pick cites its evidence (the closest past dataset + its score, or
  "cold start, KB heuristic"). MLE-STAR is a black box.
- **Cheap**: instance-based (kNN), no training, no agentic loop.

## Ranking tiers (best → worst signal)

1. **memory** — similarity-weighted metric from past runs of the same backbone on similar
   datasets (accumulated, free).
2. **logme** — `logme.py` LogME transferability measured on *this* dataset's frozen features;
   the cold-start signal when memory has no track record. Higher = better. (Implemented;
   feature extraction is the caller's job and feeds `logme_scores` into the ranker.)
3. **heuristic** — KB structured + vector score; last resort when neither above is available.

## Roadmap

1. **Now** — kNN memory + LogME cold-start + explanation (done).
2. **Cold-start seeding** — seed the memory from public transfer benchmarks (timm / VTAB) so
   it is useful before we have our own runs.
4. **Learned predictor** — once the log is large, fit a lightweight regressor
   (meta-features × config → metric) to replace the kNN.
5. **Calibration** — the eval harness logs real outcomes that both grow the memory and check
   the predictions.

## Companion pillar: automated KB updating (future)

The recommender ranks *within* Module 3's candidate pool — so it's capped by what the KB
covers, and the KB is hand-curated and narrow. The missing companion is an **auto-updating
KB**: ingest new backbones / checkpoints from HuggingFace / papers-with-code into the
*persistent structured graph* (not MLE-STAR's ephemeral per-task web search). This combines
MLE-STAR's currency with our structure + explainability, and keeps the candidate pool fresh
so the recommender isn't bottlenecked. Tracked as a separate workstream.

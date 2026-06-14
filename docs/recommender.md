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

## Roadmap

1. **Now** — kNN over outcome memory + explanation (this module), cold-start = KB heuristic.
2. **Signal upgrade** — replace/augment the memory prediction with a **LogME / linear-probe**
   transferability score (dataset-specific, near-zero cost; see `module3_improvements.md` §7).
3. **Cold-start seeding** — seed the memory from public transfer benchmarks (timm / VTAB) so
   it is useful before we have our own runs.
4. **Learned predictor** — once the log is large, fit a lightweight regressor
   (meta-features × config → metric) to replace the kNN.
5. **Calibration** — the eval harness logs real outcomes that both grow the memory and check
   the predictions.

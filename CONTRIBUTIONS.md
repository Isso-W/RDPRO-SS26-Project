# Contributions

This file reports contributions visible in the Git history used to assemble this release. It does not infer work from names in documents, and it does not assign unrecorded work. Author aliases that clearly share the same public identity are grouped below; email addresses are intentionally omitted.

The release was assembled from the fixed source snapshots `9d3f257`, `61e42e1`, and `0122829`. Path-level history and the commits named below are the evidence for the module attribution.

## Xuanyan Wang

- Prepared the course-project snapshot that established the integrated repository layout (`664a281`).
- Added early-stopping recipe metadata (`61e42e1`).
- Files affected by these recorded changes include `pipeline.py`, `recipe/`, `retrieval/`, and the repository-wide project snapshot.

## Isso-W

- Built and integrated substantial parts of the Module 1-to-4 pipeline, including dataset-analysis support, retrieval behavior, the outcome-memory recommender, the deterministic recipe layer, and experiment-cost logging.
- Added the knowledge-base mining workflow and the paired loss-imbalance experiment (`d36901e`, `bff11cd`).
- Added the full-chain notebook and recipe-driven augmentation flow (`f64c883`, `b5f75b6`), plus Module 4 backbone-loading validation and provenance reporting (`e6f0f8a`, `3f9994f`).
- Principal paths: `pipeline.py`, `features_extraction_api.py`, `retrieval/`, `recommender/`, `recipe/`, `module4_agent/`, `kb_mining/`, and `experiments/ab_loss_imbalance/`.

## Zeyu Wang (`muzhi777`)

- Integrated the modern GraphRAG vision-model additions and preserved the runtime baseline behavior during conflict resolution (`e2a36e5`, `9d3f257`).
- Added and corrected DINOv3 pooling, partial fine-tuning, ablation selection, and training-backbone logging (`81793d5`, `9c5b8e2`, `9af7196`, `6977a01`).
- Repaired full-chain retrieval tests and supplied the minimal knowledge-mining CSV fixtures (`8f3015b`).
- Implemented the standalone benchmark reproduction through snapshot `0122829`, including its adapters, executor, metrics, tests, and GPU handling.
- Prepared the public executed-Notebook evidence, per-cell logs, normalized leaderboard records, and experiment reports (`fc99cdb`, `b214883`).
- Translated the public technical documentation and reviewer-facing Notebook text into English while preserving execution counts and numeric outputs (`6889be4`).
- Principal paths: [`retrieval/`](retrieval/), [`recommender/`](recommender/), [`module4_agent/`](module4_agent/), [`kb_mining/tests/fixtures/`](kb_mining/tests/fixtures/), [`experiments/mlestar_kaggle_benchmarks/`](experiments/mlestar_kaggle_benchmarks/), [`experiments/notebook_runs/`](experiments/notebook_runs/), and [`docs/`](docs/).

## Haoyue Chen (`haoyue-chen`)

- Authored and expanded the early dataset analyzer (`73eaaf7`, `ce7ae8f`) that appears in the source lineage used by the fixed runtime and full-chain snapshots.
- Principal historical path: `analyzer.py`, whose integrated successor is `dataset_analyzer.py`.

## Earlier integration contributors

- `codetraveller66` contributed the initial requirement-extraction API from which `features_extraction_api.py` evolved (`7e1a3fd`).
- `wang` contributed pipeline and Module 4 integration cleanup (`a181856`, `d28376f`).
- `deideifan` contributed early README maintenance (`84d8410`, `f8709a6`). Those historical edits are acknowledged here even though the course-release README is based on the separately supplied draft.

## How to verify

Run these commands from a clone that contains the source refs:

```bash
git show --stat 9d3f257
git show --stat 61e42e1
git show --stat 0122829
git log --format='%h %an %s' -- pipeline.py retrieval recipe module4_agent kb_mining
```

This document describes code provenance, not grading weight or percentage credit.

# CLAUDE.md — Jiaozi Project

## Project Overview

Jiaozi is a CV model recommendation system. Given a task description (task type, dataset size, constraints), it recommends a full model configuration (backbone + head + loss + optimizer + pretrained checkpoint).

Key files:
- `pipeline.py` — full Module 1→2→3→4 pipeline entry point (`python pipeline.py --query ... --dataset ...`); also holds the Module 2→3 field mapping (`derive_data_size`, `derive_class_imbalance`, `merge_modules`)
- `features_extraction_api.py` — Module 1: LLM extraction of `task_type` / `priority` / `constraints` from natural language (`module1_pipeline`)
- `ingestion/`, `analyzer/`, `processors/`, `features/` — Module 2: dataset loading and analysis (pipeline uses `ImageLoader` + `ImageStatisticsAnalyzer`)
- `retrieval/rag_retrieval.py` — Module 3: the active KB and hybrid retrieval implementation
- `retrieval/test_rag_retrieval.py` — Module 3 test suite (smoke, behavior, task list, zero_shot)
- `module4_agent/` — Module 4: code-generation agent (spec_builder → code_generator / llm_codegen → reviewer → smoke harness → refinement)
- `docs/MODULE3_API.md` — API reference for Module 4, includes Module 2→3 interface alignment section
- `docs/report_module3.md` — mid-term report material for Module 3

## Module 3 Architecture

### Knowledge Base Structure

Two layers working together:

**NetworkX DiGraph** — component relationships:
- Node types: `backbone`, `pretrained_model`, `head`, `loss`, `optimizer`
- Edge types: `compatible_with`, `has_pretrained`, `alternative_to`, `preferred_when`, `requires`

`requires` edges (added 2026-05-28): fixed connections for integrated architectures (DETR, RT-DETR). Head and loss reached via `requires` cannot be swapped.

**ChromaDB vector index** — semantic search over backbone descriptions only (pretrained models are reached via graph traversal, not direct embedding).

### Retrieval Pipeline (Hybrid — Scheme C, updated 2026-05-28)

1. **Scale-band filter** — derive acceptable `size_tier` range from hard constraints (`edge_deployment` / `real_time` → `{nano, small}`; `data_size=small` → `{nano, small, base}`; etc.). Backbones with no in-band checkpoint and insufficient data for scratch training are excluded.
2. **Tier filter** — `accuracy_upgrade` only when `priority=accuracy`; `special_case` requires its activating constraint (e.g. `medical`, `zero_shot`, `edge_deployment`). `zero_shot=True` is a hard filter: only backbones with `"zero_shot"` in `capabilities` pass.
3. **Structured scoring** — data_size match (0–2) + priority vs complexity (0–2) + `preferred_when` bonuses (+1.5 each) + `few_shot` capability bonus (+1.5). Normalised to [0, 1].
4. **Vector scoring** — input converted to natural language, cosine similarity against backbone descriptions via `all-MiniLM-L6-v2`. Normalised to [0, 1].
5. **Weighted merge** — structured 60% + vector 40% → Top 3.
6. **Graph traversal** — for each Top-3 backbone, use pre-computed checkpoint from scale-band step; resolve head/loss/optimizer via `requires` then `compatible_with` edges; emit training strategy.

### `preferred_when` Edge Semantics

A `preferred_when` edge from A → B means "prefer A over B when condition holds." Only edges where the **source is a backbone** are consumed by the scoring loop — edges between losses or pretrained models are currently dead data (known issue, not yet addressed).

### Condition Format (updated 2026-05-25)

`EDGE_CONDITIONS` stores conditions as structured dicts, not plain strings. The format is:

```python
{"condition": {"all": ["key1", "key2"]}}   # AND — all keys must match
{"condition": {"any": ["key1", "key2"]}}   # OR  — any key must match
```

`_matches_condition(condition: dict, input_json: dict) -> bool` evaluates these dicts against the input. Valid condition keys include: `"real_time=True"`, `"edge_deployment=True"`, `"class_imbalance=True"`, `"cross_modal=True"`, `"medical=True"`, `"zero_shot=True"`, `"few_shot=True"`, `"large_data=True"`, `"data_size=small"`, `"high_accuracy_priority=True"`, `"feature_quality_priority=True"`.

## Knowledge Base — Current State (2026-05-28)

14 backbone architectures, 22 HuggingFace pretrained checkpoints, 7 heads, 6 losses, 3 optimizers.

### Backbone node fields

- `tier`: per-task role — `"default"` / `"accuracy_upgrade"` / `"special_case"`
- `scratch_viable_from`: minimum `data_size` for from-scratch training (`"small"` / `"medium"` / `"large"` / `None`)
- `domain_transfer`: `"strong"` / `"moderate"` / `"weak"` — collected but **not yet used in scoring** (known gap)
- `capabilities`: list — e.g. `["zero_shot", "few_shot", "open_vocabulary"]`. Currently set on DINOv2 and CLIP ViT only.

### Pretrained model node fields

- `size_tier`: `"nano"` / `"small"` / `"base"` / `"large"` / `"xlarge"`
- `finetune_strategy`: `"full"` / `"head_only"` / `"either"`
- `freeze_viable`: bool
- `recommended_when`: dict (defined but not yet consumed by retrieval)

### Head node fields

- `params_scale`: `"none"` / `"minimal"` / `"moderate"` / `"heavy"`

## Module 4 Interface

`build_task_list(result, graph, fmt)` and `build_all_task_lists(results, graph, fmt)` convert retrieval output to Module 4-consumable task lists.

- `fmt="structured"` — fixed `action` types (`load_pretrained`, `train_from_scratch`, `set_finetune_strategy`, `configure_head`, `configure_loss`, `configure_optimizer`)
- `fmt="nl"` — natural language task list + `model_config` metadata dict

See `docs/MODULE3_API.md` for full field reference and Module 2→3 alignment questions.

## Input Schema (updated 2026-05-28)

```python
{
    "task_type":   str,   # classification | object_detection | image_segmentation | feature_extraction
    "data_size":   str,   # small | medium | large
    "priority":    str,   # speed | accuracy | balanced
    "constraints": dict,  # boolean flags:
                          #   real_time, edge_deployment, class_imbalance,
                          #   cross_modal, medical,
                          #   zero_shot, few_shot   ← added 2026-05-28
    "description": str,   # free text, used for vector search
}
```

## Known Issues / Open Work

- `preferred_when` edges on loss and pretrained nodes are defined but never scored
- `_select_components` head/optimizer selection still uses `candidates[0]` (order-dependent, fragile)
- `domain_transfer` field on backbones is collected but not used in scoring
- `recommended_when` on pretrained_model nodes is defined but not consumed
- Missing component type: LR Scheduler (relevant for transformer finetuning)
- Vector index only covers backbones; heads and losses are not semantically searchable

## Future: domain-specific pretrained models

Biology and medical imaging have the most HuggingFace coverage (e.g., UNI for pathology, BioViL for chest X-ray, SAM-Med2D for segmentation). Plan: add a `domain_specialty` field to `pretrained_model` nodes and populate with well-maintained checkpoints per domain as needed. Requires first refining `constraints.medical` into sub-modalities (xray / mri / pathology / etc.). Not in current scope — add when domain-specific use cases arise.

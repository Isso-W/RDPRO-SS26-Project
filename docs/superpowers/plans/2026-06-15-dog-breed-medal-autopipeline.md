# Dog Breed Medal AutoPipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the Jiaozi Module 1 -> 2 -> 3 -> 4 and MCP knowledge loop so Dog Breed training uses competition-capable configurations and can iteratively approach the MLE-STAR medal-equivalent log-loss threshold.

**Architecture:** Keep `integration_update_colab.ipynb` parameter-free. Module 3 emits model-aware recipes, Module 4 executes those recipes correctly, AutoPipeline validates its retrieved candidates before selecting the baseline, and the MCP agent searches only strategy-card-controlled changes. Prediction artifacts become reusable so TTA, folds, calibration, and probability ensembles remain inside the generated project.

**Tech Stack:** Python, PyTorch, torchvision, Hugging Face Transformers, scikit-learn, FastMCP, SQLite FTS, Kaggle API, pytest, Google Colab.

---

### Task 1: Correct DINOv2 Classification Features

**Files:**
- Modify: `module4_agent/code_generator.py`
- Test: `module4_agent/tests/test_code_generator.py`

- [ ] **Step 1: Write failing generator-contract tests**

Add assertions that generated `model_utils.py` selects the DINOv2 class token instead of averaging the full token sequence, and that generated `train.py` only enables deterministic feature caching when augmentation is explicitly disabled:

```python
assert "hidden[:, 0]" in generated.files["model_utils.py"]
assert 'augmentation in {"none", "off", "deterministic"}' in generated.files["train.py"]
```

- [ ] **Step 2: Run the focused test**

Run:

```bash
pytest module4_agent/tests/test_code_generator.py -q
```

Expected: the two new assertions fail.

- [ ] **Step 3: Implement model-aware pooling and honest augmentation**

Make `_HFBackbone` retain `model.config.model_type`. For DINOv2/ViT-like classifiers return `last_hidden_state[:, 0]`; keep mean pooling only as the generic fallback. Change feature-cache eligibility so `basic`, `strong`, and `randaugment` always execute their stochastic transforms during training.

- [ ] **Step 4: Verify focused tests**

Run:

```bash
pytest module4_agent/tests/test_code_generator.py -q
```

Expected: PASS.

### Task 2: Add Partial Fine-Tuning and Discriminative Learning Rates

**Files:**
- Modify: `module4_agent/schemas.py`
- Modify: `module4_agent/spec_builder.py`
- Modify: `module4_agent/code_generator.py`
- Modify: `retrieval/rag_retrieval.py`
- Test: `module4_agent/tests/test_spec_builder.py`
- Test: `module4_agent/tests/test_code_generator.py`
- Test: `retrieval/test_rag_retrieval.py`

- [ ] **Step 1: Write failing partial-finetune tests**

Cover a recipe with:

```python
{
    "finetune_strategy": "partial",
    "unfreeze_last_n_blocks": 2,
    "backbone_learning_rate": 1e-5,
    "head_learning_rate": 3e-4,
}
```

Assert the spec preserves those fields, the generated model unfreezes only the final encoder blocks plus the head, and the optimizer builds separate backbone/head parameter groups.

- [ ] **Step 2: Run focused tests**

Run:

```bash
pytest module4_agent/tests/test_spec_builder.py module4_agent/tests/test_code_generator.py retrieval/test_rag_retrieval.py -q
```

Expected: FAIL because `partial` is currently normalized to `head_only`.

- [ ] **Step 3: Implement the partial strategy**

Extend the schema and spec builder with `unfreeze_last_n_blocks`, `backbone_learning_rate`, and `head_learning_rate`. In generated code, freeze the complete backbone, locate Hugging Face `encoder.layer` or `encoder.layers`, unfreeze the last N blocks, and keep the classifier head trainable. Build AdamW parameter groups using the strategy-card values.

- [ ] **Step 4: Verify focused tests**

Run the command from Step 2 and expect PASS.

### Task 3: Make Module 3 Emit Fine-Grained Accuracy Recipes

**Files:**
- Modify: `features_extraction_api.py`
- Modify: `pipeline.py`
- Modify: `retrieval/rag_retrieval.py`
- Test: `test_pipeline.py`
- Test: `retrieval/test_golden.py`

- [ ] **Step 1: Write failing task-signal tests**

For the Dog Breed query, assert Module 1/merge output includes:

```python
{
    "domain": "fine_grained_classification",
    "priority": "accuracy",
    "evaluation_metric": "log_loss",
}
```

Assert Module 3 does not treat DINOv2 `head_only` as the only viable training recipe for a fine-grained accuracy task.

- [ ] **Step 2: Run focused tests**

Run:

```bash
pytest test_pipeline.py retrieval/test_golden.py -q
```

Expected: FAIL on missing domain/priority-aware recipe behavior.

- [ ] **Step 3: Implement signal propagation**

Preserve fine-grained and calibrated-probability intent from Module 1 through `merge_modules`. Add domain-aware scoring and emit a partial-finetune recipe for DINOv2 when accuracy is the priority and the target has many visually similar classes.

- [ ] **Step 4: Verify focused tests**

Run the command from Step 2 and expect PASS.

### Task 4: Validate AutoPipeline Candidates Before Baseline Selection

**Files:**
- Create: `autopipeline/candidate_selector.py`
- Modify: `dog_breed_workflow.py`
- Modify: `pipeline.py`
- Test: `tests/test_candidate_selector.py`
- Test: `tests/test_dog_breed_workflow.py`

- [ ] **Step 1: Write failing successive-halving tests**

Provide three generated configs and mocked metrics. Assert the selector:

```python
result = select_candidate(
    project_dir=project,
    configs=configs,
    target_metric="log_loss",
    probe_epochs=2,
)
assert result["selected_index"] == 1
assert len(result["trials"]) == 3
```

It must minimize log loss, preserve data split/seed, and write each probe through `python -u run.py --config ...`.

- [ ] **Step 2: Run focused tests**

Run:

```bash
pytest tests/test_candidate_selector.py tests/test_dog_breed_workflow.py -q
```

Expected: FAIL because the workflow always trains `configs[0]`.

- [ ] **Step 3: Implement candidate calibration**

Run a short controlled probe for each Module 3 candidate, select by validation log loss, then train the selected candidate for its full `recommended_epochs`. Store all probe outcomes in `OutcomeMemory` so later runs can rank from measured evidence.

- [ ] **Step 4: Verify focused tests**

Run the command from Step 2 and expect PASS.

### Task 5: Extend MCP/RAG With High-Value Dog Breed Strategies

**Files:**
- Modify: `recommender/experiment_planner.py`
- Create: `knowledge_base/strategy_cards/strategy_finetune_dinov2_partial_001.json`
- Create: `knowledge_base/strategy_cards/strategy_resolution_336_001.json`
- Create: `knowledge_base/strategy_cards/strategy_calibration_temperature_001.json`
- Modify: `knowledge/schemas.py`
- Test: `tests/test_experiment_planner.py`
- Test: `tests/test_knowledge_store.py`

- [ ] **Step 1: Write failing planner tests**

Assert `image_size`, partial-finetune fields, and temperature calibration are accepted only when present in strategy-card templates. Assert at most two changed variables and no duplicate/unsupported proposal.

- [ ] **Step 2: Run focused tests**

Run:

```bash
pytest tests/test_experiment_planner.py tests/test_knowledge_store.py -q
```

Expected: FAIL because these fields are not currently supported.

- [ ] **Step 3: Add strategy cards and planner support**

Add executable templates for partial DINOv2, 336 resolution, and temperature calibration. Keep JSON authoritative and let the existing FTS index rebuild from the new cards. Rank finetuning and resolution ahead of weak batch regularizers after a frozen-head baseline underfits.

- [ ] **Step 4: Verify focused tests**

Run the command from Step 2 and expect PASS.

### Task 6: Add Reusable Probability Artifacts, TTA, and Ensembles

**Files:**
- Create: `ensemble.py`
- Modify: `kaggle_submit.py`
- Modify: `dog_breed_workflow.py`
- Modify: `module4_agent/code_generator.py`
- Test: `tests/test_ensemble.py`
- Test: `tests/test_kaggle_submit.py`

- [ ] **Step 1: Write failing probability-combination tests**

Create two probability matrices with matching IDs/classes. Assert weighted averaging preserves column order, clips safely, renormalizes every row, and rejects mismatched IDs or labels.

- [ ] **Step 2: Run focused tests**

Run:

```bash
pytest tests/test_ensemble.py tests/test_kaggle_submit.py -q
```

Expected: FAIL because no reusable ensemble API exists.

- [ ] **Step 3: Implement probability artifacts**

Save validation and test probabilities for every successful baseline/MCP experiment. Support horizontal-flip TTA, temperature scaling fitted only on validation logits, and weighted probability averaging. Choose ensemble members from validation log loss without reading Kaggle scores.

- [ ] **Step 4: Verify focused tests**

Run the command from Step 2 and expect PASS.

### Task 7: Add Stratified Cross-Validation as an Agent-Controlled Strategy

**Files:**
- Modify: `module4_agent/code_generator.py`
- Modify: `dog_breed_workflow.py`
- Create: `knowledge_base/strategy_cards/strategy_validation_stratified_5fold_001.json`
- Test: `module4_agent/tests/test_code_generator.py`
- Test: `tests/test_dog_breed_workflow.py`

- [ ] **Step 1: Write failing fold tests**

Assert every example appears in validation exactly once, folds are deterministic for the configured seed, class coverage is preserved, and fold checkpoints/probabilities have stable names.

- [ ] **Step 2: Run focused tests**

Run:

```bash
pytest module4_agent/tests/test_code_generator.py tests/test_dog_breed_workflow.py -q
```

Expected: FAIL because only one 80/20 split exists.

- [ ] **Step 3: Implement fold execution**

Add `fold_index` and `num_folds` config support using stratified deterministic folds. The workflow trains folds serially, aggregates OOF metrics, averages test probabilities, and writes fold cost/outcomes back through MCP.

- [ ] **Step 4: Verify focused tests**

Run the command from Step 2 and expect PASS.

### Task 8: Update Notebook Contract and Run the Full Regression Suite

**Files:**
- Modify: `build_integration_update_notebook.py`
- Regenerate: `integration_update_colab.ipynb`
- Modify: `tests/test_notebook_contract.py`
- Modify: `README.md`
- Modify: `docs/MCP_KNOWLEDGE_AGENT.md`

- [ ] **Step 1: Extend notebook contract tests**

Assert the notebook clones `mcp_knowledge`, contains no hand-written model hyperparameters, displays candidate probes/folds/ensemble weights, and reports validation plus official Kaggle scores separately.

- [ ] **Step 2: Regenerate the notebook**

Run:

```bash
python build_integration_update_notebook.py
```

Expected: `integration_update_colab.ipynb` is valid JSON and contains the new reporting cells.

- [ ] **Step 3: Run all relevant tests**

Run:

```bash
pytest tests module4_agent/tests retrieval/test_golden.py retrieval/test_rag_retrieval.py recommender/test_recommender.py test_pipeline.py -q
```

Expected: PASS.

- [ ] **Step 4: Push and execute Colab**

Push `mcp_knowledge`, open the canonical notebook from that branch, run all on GPU, submit the validation-selected model/ensemble, and store the complete report.

- [ ] **Step 5: Continue score-guided optimization**

Use only validation/OOF metrics to choose the next MCP-controlled experiments. The completion gate is an official Kaggle log loss at or below the declared MLE-STAR medal-equivalent threshold; a lower-quality score keeps the goal active.

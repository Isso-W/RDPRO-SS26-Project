# Dog Breed Medal Prior And CV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the Jiaozi Module 1 -> 2 -> 3 -> Recommender -> 4 Dog Breed workflow with ImageNet breed-prior calibration and automatic fold ensembling, then verify the result with Kaggle multiclass log loss.

**Architecture:** Module 3 continues to choose the trainable backbone and Module 4 continues to generate the only executable training project. MCP can retrieve an ImageNet-prior strategy card; generated evaluation calibrates a blend between the learned 120-class probabilities and the pretrained ImageNet dog-class probabilities. After the baseline and three controlled experiments select a configuration, the workflow trains the same selected configuration on deterministic stratified folds and averages test probabilities, without counting folds as new experiment configurations.

**Tech Stack:** Python, PyTorch, torchvision, NumPy, pandas, scikit-learn, FastMCP, pytest, Google Colab, Kaggle API.

---

### Task 1: Add Deterministic Fold Splits To Module 4

**Files:**
- Modify: `module4_agent/code_generator.py`
- Test: `module4_agent/tests/test_code_generator.py`

- [ ] **Step 1: Write a generated-runtime test**

Generate a project with `fold_count=3` and assert that `_split_indices` produces disjoint validation indices for folds 0, 1, and 2 whose union covers every sample exactly once.

- [ ] **Step 2: Run the focused test and verify failure**

Run:

```bash
.venv/bin/python -m pytest module4_agent/tests/test_code_generator.py -q
```

Expected: the new fold assertions fail because generated `_split_indices` only supports one validation fraction.

- [ ] **Step 3: Implement stratified fold selection**

Change the generated helper signature to:

```python
def _split_indices(
    labels: list[int],
    validation_fraction: float,
    seed: int,
    fold_count: int = 1,
    fold_index: int = 0,
):
```

For `fold_count > 1`, shuffle each class deterministically and assign validation members using `position % fold_count == fold_index`. Preserve the existing fraction behavior for `fold_count == 1`. Read `fold_count` and `fold_index` from the controlled config in both CSV and ImageFolder loaders.

- [ ] **Step 4: Re-run the focused test**

Run:

```bash
.venv/bin/python -m pytest module4_agent/tests/test_code_generator.py -q
```

Expected: PASS.

### Task 2: Add ImageNet Breed-Prior Projection

**Files:**
- Modify: `module4_agent/code_generator.py`
- Modify: `kaggle_submit.py`
- Test: `module4_agent/tests/test_code_generator.py`
- Test: `tests/test_kaggle_submit.py`

- [ ] **Step 1: Add failing mapping and calibration tests**

Cover normalized Kaggle labels such as `german_shepherd`, `brabancon_griffon`, `pembroke`, and `walker_hound`, and verify that all 120 contiguous ImageNet dog categories can be projected into sample-submission order. Add a calibration test where a validation-selected convex blend improves log loss.

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest module4_agent/tests/test_code_generator.py tests/test_kaggle_submit.py -q
```

Expected: FAIL because prior projection and calibration do not exist.

- [ ] **Step 3: Generate the prior helper**

Generated evaluation code will load the torchvision `DEFAULT` classifier corresponding to the selected backbone when `imagenet_prior_blend` is enabled. Normalize label/category names, map the 120 training labels to ImageNet category indices, softmax the native 1000 logits, retain and renormalize mapped dog probabilities, and search a deterministic blend grid:

```python
combined = (1.0 - alpha) * learned_probabilities + alpha * prior_probabilities
```

Select `alpha` by validation log loss and save `prior_alpha` plus the mapping in `validation_probabilities.npz`.

- [ ] **Step 4: Apply the calibrated prior during submission**

`kaggle_submit.py` reads the validation artifact for each member, reconstructs the same native pretrained model and mapping, and applies the saved `prior_alpha` before TTA/ensemble output is written.

- [ ] **Step 5: Re-run the focused tests**

Run:

```bash
.venv/bin/python -m pytest module4_agent/tests/test_code_generator.py tests/test_kaggle_submit.py -q
```

Expected: PASS.

### Task 3: Teach MCP/RAG To Retrieve The Prior Strategy

**Files:**
- Create: `knowledge_base/strategy_cards/strategy_inference_imagenet_breed_prior_001.json`
- Modify: `recommender/experiment_planner.py`
- Modify: `knowledge/store.py`
- Modify: `agents/mle_experiment_agent.py`
- Test: `tests/test_knowledge_store.py`
- Test: `tests/test_experiment_planner.py`

- [ ] **Step 1: Add failing retrieval and planner tests**

Assert that a Dog Breed log-loss query retrieves the prior card in the low-token top five and that the planner accepts only the controlled field `imagenet_prior_blend`.

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_knowledge_store.py tests/test_experiment_planner.py -q
```

Expected: FAIL because the card and supported field are absent.

- [ ] **Step 3: Add the evidence-backed card and ranking**

The card template is:

```json
{"imagenet_prior_blend": "auto"}
```

Its evidence points to the Kaggle dataset description and the fixed public ImageNet-pretrained solution. Boost retrieval for `ImageNet`, `breed prior`, `native classifier`, and `log loss`; rank the inference card before generic TTA when the dataset domain is fine-grained Dog Breed.

- [ ] **Step 4: Re-run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_knowledge_store.py tests/test_experiment_planner.py -q
```

Expected: PASS.

### Task 4: Train Automatic Selected-Config Folds

**Files:**
- Create: `autopipeline/fold_ensemble.py`
- Modify: `autopipeline/__init__.py`
- Modify: `dog_breed_workflow.py`
- Modify: `ensemble.py`
- Modify: `kaggle_submit.py`
- Test: `tests/test_fold_ensemble.py`
- Test: `tests/test_dog_breed_workflow.py`
- Test: `tests/test_ensemble.py`

- [ ] **Step 1: Write failing orchestration tests**

Assert that the selected configuration is copied without changing model hyperparameters, fold 0 reuses the existing checkpoint when compatible, folds 1 and 2 execute only:

```text
python -u run.py --config <controlled-json>
```

and submission members receive validation-derived non-negative weights.

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bash
.venv/bin/python -m pytest tests/test_fold_ensemble.py tests/test_dog_breed_workflow.py tests/test_ensemble.py -q
```

Expected: FAIL because selected-config fold orchestration does not exist.

- [ ] **Step 3: Implement three-fold finalization**

Train the selected configuration with `fold_count=3` and `fold_index` 0..2. Keep the Module 4 selected backbone, optimizer, learning rates, augmentation, scheduler, image size, seed, and recommended epochs unchanged. Record every fold in `OutcomeMemory`, calculate inverse-log-loss weights with clipping, and pass the member list to submission.

- [ ] **Step 4: Re-run focused tests**

Run:

```bash
.venv/bin/python -m pytest tests/test_fold_ensemble.py tests/test_dog_breed_workflow.py tests/test_ensemble.py -q
```

Expected: PASS.

### Task 5: Notebook, Regression, Colab, And Official Score

**Files:**
- Modify: `integration_update_colab.ipynb`
- Modify: `README.md`
- Test: `tests/test_integration_notebook.py`

- [ ] **Step 1: Regenerate the notebook without hyperparameter inputs**

Display prior calibration alpha, fold metrics, fold weights, final ensemble members, training cost, submission status, and official score. The notebook must not expose model hyperparameters.

- [ ] **Step 2: Run all local tests**

Run:

```bash
.venv/bin/python -m pytest tests module4_agent/tests test_pipeline.py -q
PYTHONPATH=retrieval:. .venv/bin/python -m pytest retrieval/test_golden.py retrieval/test_rag_retrieval.py recommender/test_recommender.py -q
```

Expected: all tests pass.

- [ ] **Step 3: Commit and push `mcp_knowledge`**

Run:

```bash
git add autopipeline agents dog_breed_workflow.py ensemble.py integration_update_colab.ipynb kaggle_submit.py knowledge knowledge_base module4_agent recommender tests README.md docs/superpowers/plans/2026-06-15-dog-breed-medal-prior-cv.md
git commit -m "feat: add Dog Breed prior and fold ensemble"
git push origin mcp_knowledge
```

- [ ] **Step 4: Run the canonical Colab notebook**

Verify the notebook clones the committed SHA, completes Module 1 -> 2 -> 3 -> Recommender -> 4, runs the baseline, at most three MCP configurations, selected-config folds, and submission.

- [ ] **Step 5: Audit the official score**

Success requires an actual completed Kaggle submission with public/private log loss at or below the agreed medal-equivalent threshold. If the score misses the threshold, write the measured outcome to MCP memory and continue with the next evidence-backed Jiaozi strategy rather than claiming completion.

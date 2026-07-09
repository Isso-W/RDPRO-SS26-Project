# Cassava Auto-CV Codegen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use Jiaozi Auto-CV to select a model, generate Cassava competition training code, execute the generated code in Google Colab, and compare its trained checkpoint with published Kaggle scores.

**Architecture:** Extend `DatasetManifest` with CSV-label and competition benchmark metadata. Add a real `csv_labels` path to the existing generated classification template, while preserving Hugging Face support. A Colab driver notebook downloads Kaggle data, invokes `CVAutoDLWorkflow` in simulate mode to select/refine/generate the project, then executes the generated `train.py` in real mode and renders the generated metrics.

**Tech Stack:** Jiaozi `CVAutoDLWorkflow`, Python, PyTorch, torchvision, pandas, Kaggle CLI, Google Colab.

---

### Task 1: Cassava Input Contracts

**Files:**
- Modify: `cv_autodl_agent/schemas.py`
- Create: `examples/cassava_manifest.json`
- Create: `examples/cassava_candidates.json`
- Modify: `tests/test_validation.py`
- Modify: `tests/test_examples.py`

- [x] Add CSV label path, validation fraction, Kaggle competition name, and benchmark score fields.
- [x] Validate CSV classification manifests require label and column metadata.
- [x] Add Cassava manifest and candidates and verify Auto-CV selects EfficientNet-B3.

### Task 2: Generated CSV Training Code

**Files:**
- Modify: `cv_autodl_agent/templates.py`
- Modify: `cv_autodl_agent/codegen.py`
- Modify: `tests/test_examples.py`

- [x] Add deterministic stratified splitting and CSV image dataset loading to generated `train.py`.
- [x] Add EfficientNet-B3 and ConvNeXt-Tiny model mappings.
- [x] Add mixed precision, class-weighted loss, scheduler handling, history, predictions, and checkpoint output.
- [x] Write benchmark comparisons into generated `metrics.json`.
- [x] Add pandas to generated requirements for CSV manifests.

### Task 3: Auto-CV Colab Driver

**Files:**
- Replace: `examples/cassava_competition_colab.ipynb`
- Modify: `tests/test_examples.py`

- [x] Download official competition data after token upload.
- [x] Invoke `CVAutoDLWorkflow` with Cassava manifest and candidate inputs.
- [x] Display the selected candidate, refined training spec, review result, and generated source path.
- [x] Execute the generated `train.py` in real mode with the Kaggle data root.
- [x] Render generated history, confusion matrix, checkpoint path, and benchmark comparison.

### Task 4: Remove Parallel Handwritten Trainer

**Files:**
- Delete: `competition/__init__.py`
- Delete: `competition/cassava.py`
- Delete: `tests/test_cassava_competition.py`
- Modify: `pyproject.toml`

- [x] Remove the standalone competition trainer so Jiaozi-generated code is the only training path.
- [x] Restore package discovery to the Auto-CV package.

### Task 5: Documentation and Verification

**Files:**
- Modify: `README.md`
- Modify: `docs/CV_AUTODL_TECHNICAL_DOC.md`

- [x] Document the exact Auto-CV generation-to-training chain and score caveat.
- [x] Run all unit tests, notebook JSON/code-cell compilation, Python compilation, and `git diff --check`.

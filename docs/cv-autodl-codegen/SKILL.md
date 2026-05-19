---
name: cv-autodl-codegen
description: Use this skill when working on the Jiaozi CV Auto-DL code generation submodule, especially when consuming DatasetManifest and RetrievedModelCandidate inputs, generating PyTorch classification training code, running ablation/refinement, reviewing generated code, and exporting Colab notebooks for CIFAR-10 or Food101.
---

# CV Auto-DL Codegen

## Purpose

Use this skill to modify or operate the Jiaozi CV Auto-DL codegen submodule.

The submodule starts after upstream retrieval. It does not parse natural language and does not run RAG itself. It consumes:

- `DatasetManifest`
- `RetrievedModelCandidate[]`

It produces:

- generated training project files
- ablation results
- review report
- Colab notebook
- real classification checkpoint when run in Colab real mode

## Repository Locations

Core package:

```text
cv_autodl_agent/
```

Important files:

```text
cv_autodl_agent/schemas.py
cv_autodl_agent/workflow.py
cv_autodl_agent/templates.py
cv_autodl_agent/codegen.py
cv_autodl_agent/ablation.py
cv_autodl_agent/review.py
cv_autodl_agent/notebook.py
```

Examples:

```text
examples/cifar10_manifest.json
examples/cifar10_candidates.json
examples/food101_manifest.json
examples/food101_candidates.json
examples/colab_demo.ipynb
```

Tests:

```text
tests/
```

## Workflow

Follow this sequence:

```text
validate inputs
-> select candidate
-> build TrainingSpec
-> generate train.py / dataset.py / inference.py / requirements.txt
-> run baseline
-> run ablation
-> summarize ablation
-> apply targeted refinement
-> review generated project
-> export Colab notebook
```

Main entrypoint:

```python
from cv_autodl_agent import CVAutoDLWorkflow
```

CLI entrypoint:

```bash
python3 -m cv_autodl_agent \
  --manifest examples/cifar10_manifest.json \
  --candidates examples/cifar10_candidates.json \
  --output-dir demo_run \
  --execution-mode simulate \
  --notebook-execution-mode real
```

## Real Training Rules

Real training is currently supported for classification only.

The generated `train.py` must:

- use `datasets.load_dataset`
- support `uoft-cs/cifar10`
- support `ethz/food101`
- use `torchvision.models` and PyTorch
- create a full classification model
- replace the final classifier layer with `num_classes`
- save `checkpoints/best.pt`
- write `metrics.json`
- keep `execution-mode=simulate` fallback

Do not replace the real training path with a simulate-only implementation.

## Model Mapping

The generated `train.py` maps candidate model IDs into torchvision classifiers.

Supported names:

```text
resnet18
resnet34
resnet50
mobilenet_v3_small
efficientnet_b0
```

Known aliases:

```text
timm-resnet18 -> resnet18
torchvision-resnet18 -> resnet18
hf-* -> resnet18 fallback
```

For unsupported names, fallback to `resnet18` to preserve Colab demo reliability.

## Colab Rules

`examples/colab_demo.ipynb` should run from Colab even if the current notebook directory is not the repo root.

The bootstrap cell should:

- check for `cv_autodl_agent/`
- clone `https://github.com/Isso-W/Jiaozi.git` into `/content/Jiaozi` if needed
- change directory to the repository root
- add the repo root to `sys.path`

Remember: Colab sees the pushed GitHub repository, not local unpushed files.

## Testing Checklist

After code changes, run:

```bash
python3 -m unittest discover -s tests -v
```

Tests should confirm:

- example manifests validate
- CIFAR-10 and Food101 manifests validate
- generated `train.py` contains `load_dataset`
- generated `train.py` contains `torchvision`
- generated `train.py` contains `torch.save`
- generated `train.py` writes `checkpoints/best.pt`
- generated `train.py` keeps `run_simulated_training`
- generated `train.py` does not rely on `timm.create_model`
- Colab notebook defaults generated training to `real`

## Common Fixes

If Colab says it cannot find `cv_autodl_agent`, update the notebook bootstrap or push the latest repository changes.

If real training fails because a dataset column is missing, check:

```json
{
  "image_column": "img",
  "label_column": "label"
}
```

CIFAR-10 uses `img`; Food101 uses `image`.

If Colab is too slow, reduce:

```json
{
  "max_train_samples": 1024,
  "max_val_samples": 256,
  "max_epochs": 1
}
```

If training quality is too low, increase those same fields.

## Boundaries

Do not add natural language parsing or RAG retrieval into this submodule unless explicitly requested. Those belong to upstream modules.

Do not treat LangGraph as a Colab GPU executor. LangGraph can orchestrate the workflow, but real GPU execution requires Colab manual run, Colab Enterprise, Vertex AI, or another remote GPU backend.

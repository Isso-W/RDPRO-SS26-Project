# CV Auto-DL Codegen Agent Technical Document

## 1. Overview

This module implements the code generation, review, ablation, and Colab export part of a CV Auto-DL pipeline.

The upstream modules are responsible for:

- parsing the user request
- producing a `DatasetManifest`
- retrieving the top candidate models as `RetrievedModelCandidate[]`

This module starts after retrieval. It consumes structured inputs, selects a compatible candidate, generates training code, runs a baseline, performs ablation, applies targeted refinement, reviews the generated project, and exports a Colab notebook.

The current real training path supports image classification on Hugging Face datasets:

- `uoft-cs/cifar10`
- `ethz/food101`

Segmentation and detection templates are present as workflow targets, but their current execution path remains simulated until real task-specific training loops are added.

## 2. Scope

Supported in the current implementation:

- image classification workflow
- Hugging Face dataset loading for CIFAR-10 and Food101
- torchvision/PyTorch model training
- generated `train.py`, `dataset.py`, `inference.py`, `requirements.txt`, `notebook.ipynb`
- baseline run
- component-level ablation
- targeted refinement
- static review and structured review report
- Colab real training mode
- local simulate fallback

Not yet supported as real training:

- segmentation training loop
- detection training loop
- fully automated remote Colab execution from LangGraph
- production-scale hyperparameter search

## 3. Architecture

The workflow is implemented in `cv_autodl_agent.workflow.CVAutoDLWorkflow`.

Pipeline:

```text
DatasetManifest + RetrievedModelCandidate[]
  -> validate inputs
  -> select candidate
  -> build TrainingSpec
  -> generate project files
  -> run baseline
  -> run ablation variants
  -> summarize ablation
  -> apply targeted refinement
  -> review generated project
  -> export Colab notebook
```

Main modules:

- `schemas.py`: dataclass contracts for manifest, candidates, training specs, execution results, ablation summaries, review reports, and workflow output
- `selectors.py`: deterministic candidate scoring and fallback selection
- `planner.py`: heuristic `TrainingSpec` generation
- `codegen.py`: writes generated project files
- `templates.py`: source templates for `train.py`, `dataset.py`, and `inference.py`
- `executor.py`: runs generated training scripts and parses metrics
- `ablation.py`: builds and runs component-level ablation variants
- `review.py`: static and semantic checks for generated projects
- `notebook.py`: exports Colab notebooks
- `__main__.py`: CLI entrypoint

## 4. Input Contracts

### DatasetManifest

Required common fields:

```json
{
  "dataset_name": "cifar10-huggingface",
  "task_family": "classification",
  "train_path": "train",
  "val_path": "test",
  "test_path": "test",
  "annotation_format": "huggingface",
  "recommended_metric": "accuracy"
}
```

Classification fields:

```json
{
  "num_classes": 10,
  "class_names": ["airplane", "automobile"],
  "label_source": "label",
  "image_size_hint": 224
}
```

Hugging Face fields:

```json
{
  "hf_dataset_id": "uoft-cs/cifar10",
  "image_column": "img",
  "label_column": "label",
  "train_split": "train",
  "val_split": "test",
  "test_split": "test",
  "max_train_samples": 1024,
  "max_val_samples": 256,
  "max_epochs": 1
}
```

### RetrievedModelCandidate

Example:

```json
{
  "rank": 1,
  "model_id": "timm-resnet18",
  "source": "timm",
  "task_family": "classification",
  "library": "PyTorch",
  "processor_or_transform": "torchvision_transforms",
  "default_input_size": 224,
  "pretrained_weights": "imagenet",
  "license": "Apache-2.0",
  "training_notes": "small image classification baseline suitable for Colab",
  "install_deps": ["torch", "torchvision", "datasets"]
}
```

The current real training implementation maps supported candidate names to torchvision classifiers. For example, `timm-resnet18` is normalized to torchvision `resnet18`.

## 5. Generated Project

For each selected candidate, the workflow generates a project directory like:

```text
candidate_01_timm-resnet18/
  manifest.json
  training_spec.json
  refined_training_spec.json
  train.py
  dataset.py
  inference.py
  requirements.txt
  notebook.ipynb
  baseline_result.json
  ablation_plan.json
  ablation_trials.json
  ablation_summary.json
  review_report.json
  runs/
```

When `train.py` runs in real mode, it outputs:

```text
output_dir/
  metrics.json
  checkpoints/
    best.pt
```

`metrics.json` contains:

```json
{
  "status": "success",
  "primary_metric_name": "accuracy",
  "primary_metric_value": 0.812,
  "checkpoint_path": "output_dir/checkpoints/best.pt",
  "artifacts_path": "output_dir"
}
```

## 6. Real Training Path

Real training is enabled by:

```bash
python train.py \
  --manifest manifest.json \
  --spec refined_training_spec.json \
  --output-dir colab_demo_output/real_cifar10_train \
  --execution-mode real
```

The generated `train.py` does the following:

- loads the Hugging Face dataset via `datasets.load_dataset`
- uses the configured image and label columns
- selects small subsets for fast Colab demonstration
- builds torchvision transforms
- creates a torchvision classifier such as `resnet18`
- replaces the final classification layer with `num_classes`
- trains with `torch.optim.AdamW`
- evaluates validation accuracy
- saves the best checkpoint to `checkpoints/best.pt`
- writes `metrics.json`

Supported torchvision classifiers:

- `resnet18`
- `resnet34`
- `resnet50`
- `mobilenet_v3_small`
- `efficientnet_b0`

Unsupported names fallback to `resnet18` to keep the Colab demo robust.

## 7. Simulate Fallback

`execution-mode=simulate` is kept for fast local testing. It does not train a model, but it preserves the same output contract:

- `metrics.json`
- `checkpoints/best.pt`

This lets tests and workflow logic run without downloading datasets or requiring GPU.

## 8. Ablation and Refinement

The ablation engine tests a small set of component-level variants:

- transforms
- freeze strategy
- optimizer and scheduler
- batch size
- learning rate

For classification, a typical winner is:

```json
{
  "best_component_to_change": "transforms",
  "recommended_edit_region": "training_spec.transforms"
}
```

The workflow then applies only that edit region and reruns the generated project.

## 9. Colab Usage

Open:

```text
examples/colab_demo.ipynb
```

The first notebook cell bootstraps the repository:

- if the notebook already runs from the repository root, it uses the local files
- otherwise it clones `https://github.com/Isso-W/Jiaozi.git` into `/content/Jiaozi`
- then it changes directory into the repo root

Important: Colab clones the remote GitHub repository. Local changes must be committed and pushed before Colab can see them.

The notebook then:

- loads `examples/cifar10_manifest.json`
- loads `examples/cifar10_candidates.json`
- runs the agent workflow in simulate mode to generate and review code
- exports a real-mode generated notebook
- runs the generated `train.py` in real mode
- prints `metrics.json`
- checks whether `checkpoints/best.pt` exists

## 10. CLI Usage

Generate a CIFAR-10 project locally:

```bash
python3 -m cv_autodl_agent \
  --manifest examples/cifar10_manifest.json \
  --candidates examples/cifar10_candidates.json \
  --output-dir demo_run \
  --execution-mode simulate \
  --notebook-execution-mode real
```

Generate a Food101 project:

```bash
python3 -m cv_autodl_agent \
  --manifest examples/food101_manifest.json \
  --candidates examples/food101_candidates.json \
  --output-dir food101_run \
  --execution-mode simulate \
  --notebook-execution-mode real
```

## 11. Testing

Run:

```bash
python3 -m unittest discover -s tests -v
```

Current tests validate:

- all example inputs parse and validate
- generated notebook JSON is valid
- Colab bootstrap exists
- CIFAR-10 generated notebook defaults to real mode
- generated `train.py` includes `datasets.load_dataset`
- generated `train.py` includes torchvision model creation
- generated `train.py` saves `checkpoints/best.pt`
- generated `train.py` retains simulate fallback
- segmentation and detection workflows still produce valid simulated projects

## 12. Current Limitations and Next Steps

Limitations:

- real training currently supports classification only
- Food101 defaults to a small subset for fast Colab runs
- the model search result is consumed as structured metadata, not yet generated by a live RAG service inside this module
- LangGraph orchestration hook exists, but the current production path uses deterministic Python orchestration

Recommended next steps:

- add real segmentation training
- add real detection training
- add a richer model mapping table
- push checkpoints to Google Drive or Hugging Face Hub
- optionally deploy real remote execution through Colab Enterprise, Vertex AI, RunPod, Modal, or another GPU runner

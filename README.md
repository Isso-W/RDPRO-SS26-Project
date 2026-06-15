# Jiaozi

## Dog Breed MCP Experiment Agent

The `mcp_knowledge` workflow adds a two-stage, low-token experiment loop for Kaggle
Dog Breed Identification:

```text
fixed sources -> Knowledge Learner -> strategy cards -> Module 1-4 AutoPipeline
-> baseline -> MCP MLE Agent (max 3 experiments) -> selected-config 3-fold
-> Kaggle probability submission
```

Fine-grained runs probe all three Module 3 candidates before selecting the
baseline. DINOv2 recipes support official class-token plus pooled-patch
features, partial last-block fine-tuning, discriminative learning rates,
warmup, higher-resolution strategy cards, TTA, and validation-selected
probability ensembles. The Dog Breed path can also retrieve an evidence-backed
ImageNet breed-prior card, calibrate its probability blend on validation data,
and average deterministic stratified-fold checkpoints for final submission.

Open `integration_update_colab.ipynb` in Colab and run all cells. Training choices
come from Jiaozi and retrieved strategy cards; the notebook does not override model
or training hyperparameters. See `docs/MCP_KNOWLEDGE_AGENT.md` for credentials,
storage, MCP tools, and output details.

Jiaozi is a CV Auto-DL prototype. Given a natural-language task request and a
HuggingFace image dataset id, it recommends CV model configurations and can
generate runnable local training/evaluation/inference code.

## Colab

Open the Colab notebook for the `integration-update` branch:

[Open `integration_update_colab.ipynb` in Colab](https://colab.research.google.com/github/Isso-W/Jiaozi/blob/mcp_knowledge/integration_update_colab.ipynb)

For direct GPT-generated training on four Kaggle competitions and ten public
image-classification datasets:

[Open `vision_benchmarks_colab.ipynb` in Colab](https://colab.research.google.com/github/Isso-W/Jiaozi/blob/codex/integration-update-colab/vision_benchmarks_colab.ipynb)

The benchmark notebook defaults to formal Cassava training: full data,
EfficientNet-B3 at 300px, mixed precision, strong augmentation, class weights,
cosine scheduling, per-epoch validation, early stopping, and resumable
checkpoints in Google Drive.

## Pipeline

The active integrated entry point is `pipeline.py`:

```text
User request + HuggingFace dataset id
-> Module 1: parse task type, priority, and constraints with an LLM
-> Module 2: analyze dataset size, classes, and class imbalance
-> Module 3: retrieve up to three ranked CV model configurations
-> Module 4: generate training/evaluation/inference code
-> Local smoke test + deterministic review
```

The active Module 4 implementation is `module4_agent/`.

## Environment

Use one Python interpreter consistently for install and test commands. On the
current macOS setup, `/usr/bin/python3` was used for validation:

```bash
/usr/bin/python3 -m pip install -r requirements.txt
```

If you prefer a virtual environment:

```bash
/usr/bin/python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## LLM Configuration

Local secrets should go in `.env`, which is ignored by git. Start from the
template:

```bash
cp .env.example .env
```

For Qwen/DashScope:

```bash
JIAOZI_LLM_PROVIDER=qwen
M4_LLM_PROVIDER=qwen
JIAOZI_DASHSCOPE_API_KEY=...
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
M1_QWEN_MODEL=qwen-plus
M4_QWEN_MODEL=qwen-plus
```

For offline Module 4 template generation:

```bash
M4_LLM_PROVIDER=none
```

Module 1 still needs an LLM provider for natural-language parsing. Module 4 can
fall back to deterministic templates when its LLM provider is unavailable.

## Run The Integrated Pipeline

```bash
/usr/bin/python3 pipeline.py \
  --query "classify images on a small dataset" \
  --dataset uoft-cs/cifar10 \
  --fmt nl \
  --module4-output generated_pipeline \
  --module4-no-smoke
```

`pipeline.py` writes Module 3 candidates to
`generated_pipeline/module3_candidates.json`, then asks Module 4 to generate the
training/evaluation/inference project.

To run Module 4 smoke tests as part of the pipeline, omit `--module4-no-smoke`.

## Run Module 4 Directly

```bash
/usr/bin/python3 -m module4_agent \
  --input module4_agent/examples/sample_m3_output.json \
  --output generated/
```

For static generation/review only:

```bash
/usr/bin/python3 -m module4_agent \
  --input module4_agent/examples/sample_m3_output.json \
  --output generated/ \
  --no-smoke
```

Generated projects include:

```text
configs.json
generation_info.json
utils.py
model_utils.py
smoke_data.py
model.py
train.py
evaluate.py
infer.py
run.py
run_experiments.py
requirements.txt
README_generated.md
module4_summary.json
```

`generation_info.json` records whether `model.py` came from an LLM provider or
from the template fallback.

## Tests

Pipeline and Module 2->3 mapping:

```bash
/usr/bin/python3 -m unittest test_pipeline.py -v
```

Module 3 retrieval behavior:

```bash
cd retrieval
PYTHONPYCACHEPREFIX=/private/tmp/jiaozi-pycache /usr/bin/python3 -m unittest test_rag_retrieval.py -v
PYTHONPYCACHEPREFIX=/private/tmp/jiaozi-pycache /usr/bin/python3 -m unittest test_golden.py -v
```

Module 4 direct smoke:

```bash
M4_LLM_PROVIDER=none /usr/bin/python3 -m module4_agent \
  --input module4_agent/examples/sample_m3_output.json \
  --output /private/tmp/jiaozi-m4-smoke
```

Module 4 pytest suite:

```bash
/usr/bin/python3 -m pytest module4_agent/tests -q
```

## Current Scope

- Local smoke tests use synthetic data and do not perform long training.
- Module 4 keeps `offline_smoke=true` by default, so smoke runs do not download
  HuggingFace checkpoints.
- Real checkpoint loading is available in generated code when
  `offline_smoke=false` and `use_pretrained=true`.
- Object detection and segmentation metrics are smoke-compatible placeholders,
  not real benchmark scores.
- The project uses a lightweight workflow agent design rather than a heavy
  external agent framework such as LangGraph or AutoGen.

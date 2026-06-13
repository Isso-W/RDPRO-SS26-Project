# Jiaozi Usage Guide

Practical how-to for four common tasks:

1. [Visualise the pipeline with skrub (DataOps graph)](#1-visualise-the-pipeline-with-skrub)
2. [Run real training](#2-run-real-training)
3. [Persist data & checkpoints on Google Drive (Colab)](#3-persist-data--checkpoints-on-google-drive-colab)
4. [Pull a Kaggle competition test set](#4-pull-a-kaggle-competition-test-set)

**Prerequisites**

```bash
pip install -r requirements.txt
cp .env.example .env        # then fill in your LLM key — Module 1 needs one
```

Module 1 (requirement parsing) calls an LLM. Default provider is `qwen`
(`JIAOZI_DASHSCOPE_API_KEY`); to use OpenAI/an OpenAI-compatible proxy instead set
`JIAOZI_LLM_PROVIDER=openai`, `OPENAI_API_KEY`, and (for a proxy) `OPENAI_BASE_URL`.

---

## 1. Visualise the pipeline with skrub

The whole Module 1→2→3 pipeline is also expressed as a skrub **DataOps** lazy DAG in
`skrub_pipeline.py`, so you can show its computational graph.

```bash
# Text description of the graph (no extra deps)
python skrub_pipeline.py --graph

# Just the Module 2 sub-graph — no LLM key needed
python skrub_pipeline.py --module2-only --graph

# Render an SVG + PNG (needs graphviz, see below)
python skrub_pipeline.py --draw                 # -> pipeline_graph.svg / .png

# Actually execute the DAG end to end
python skrub_pipeline.py --query "classify cassava leaf disease" --dataset dpdl-benchmark/cassava --run
```

Drawing needs the graphviz **system binary** plus `pydot`:

```bash
pip install pydot
# macOS:  brew install graphviz
# Ubuntu: sudo apt-get install graphviz
# Windows: winget install Graphviz.Graphviz   (then add ...\Graphviz\bin to PATH)
```

From Python (e.g. a notebook, where `draw_graph()` renders inline):

```python
from skrub_pipeline import build_pipeline, build_module2_pipeline

dag = build_pipeline("classify cassava leaf disease", "dpdl-benchmark/cassava")
print(dag.skb.describe_steps())   # text
dag.skb.draw_graph()              # SVG (inline in Jupyter/Colab)
result = dag.skb.eval()           # run it
```

`pipeline.get_skrub_dag(query, dataset_id)` returns the same DAG from the main module.

---

## 2. Run real training

By default Module 4 generates **smoke** code (`offline_smoke=true`: a tiny backbone +
synthetic data, for a fast 1-step sanity check). Real training uses the recommended
backbone, the real dataset, multi-epoch training, and checkpointing.

### Generate real-training code

```bash
# Local test runner: generate a real-training project (smoke is auto-skipped)
python run_for_testing.py \
  --dataset dpdl-benchmark/cassava \
  --query "Classify cassava leaf disease images, balancing accuracy and speed" \
  --module4 --real-training
# -> test_runs/<timestamp>/module4_code/
```

Equivalently via the pipeline entry point:

```bash
python pipeline.py \
  --query "..." --dataset dpdl-benchmark/cassava \
  --module4-output ./out --module4-real-training
```

`--real-training` / `--module4-real-training` sets `offline_smoke=false` and skips the
local smoke harness. It **generates** the project; it does not train yet.

### Actually train

```bash
cd test_runs/<timestamp>/module4_code      # or ./out
python -u run.py --epochs 15                # -u = stream per-epoch logs live
```

`run.py` runs train → evaluate → infer and prints a JSON summary (look for the
`"evaluate"` block). `--epochs` defaults to the pipeline-recommended count
(`recommended_epochs`, derived from data size + finetune strategy) when omitted.

Notes:
- Detection / segmentation real dataloaders aren't implemented yet — they fall back to
  synthetic data.
- For a **frozen** backbone (e.g. a `head_only` recommendation like DINOv2), training
  auto-uses the feature cache: it extracts backbone features once, then trains the head
  on the cache — far faster, and a proper linear probe.

### On Colab

Open `integration_update_colab.ipynb`, set a GPU runtime, and run the cells. In the run
cell set `REAL_TRAINING = True` (and optionally `EPOCHS`); the training cell runs the
generated `run.py` on the GPU.

---

## 3. Persist data & checkpoints on Google Drive (Colab)

A fresh Colab runtime is wiped on disconnect, so by default the dataset re-downloads
(~10+ min) and trained checkpoints are lost. Point both at Drive.

### Dataset cache

Mount Drive and set `HF_HOME` **before** the pipeline runs (subprocesses inherit it):

```python
from google.colab import drive
drive.mount('/content/drive')

import os
os.environ['HF_HOME'] = '/content/drive/MyDrive/Jiaozi/hf_cache'
os.makedirs(os.environ['HF_HOME'], exist_ok=True)
```

The dataset downloads once to Drive; later runtimes reuse it. (Reading from Drive/FUSE is
a bit slower than local disk, but far faster than re-downloading.) The
`integration_update_colab.ipynb` notebook has this as a dedicated cell.

### Checkpoints + auto-resume

The notebook's training cell injects a Drive `checkpoint_dir` (keyed by backbone+dataset)
and `resume_checkpoint=auto` into `configs.json` before running `run.py`, so:
- checkpoints land under `/content/drive/MyDrive/Jiaozi/checkpoints/<backbone>_<dataset>/`,
- a re-run continues from the last checkpoint instead of starting over.

To do it manually for any generated project, add these top-level keys to `configs.json`
(they're read via `normalize_config` → `get_value`):

```json
{ "checkpoint_dir": "/content/drive/MyDrive/Jiaozi/checkpoints/myrun", "resume_checkpoint": "auto" }
```

To train fresh, delete that Drive folder or set `SAVE_CHECKPOINTS_TO_DRIVE = False`.

---

## 4. Pull a Kaggle competition test set

`ingestion/kaggle_loader.py` downloads a Kaggle competition (train + hidden test set) via
the Kaggle API, keyed off `vision_benchmark_catalog.py`.

### Credentials (one of)

- `~/.kaggle/kaggle.json` → `{"username": "...", "key": "..."}`, or
- env vars `KAGGLE_USERNAME` and `KAGGLE_KEY`.

You must also **accept the competition rules** on its Kaggle page once, or downloads 403.

```bash
pip install kaggle
```

### Download + locate

```bash
python -m ingestion.kaggle_loader cassava --data-root ./kaggle_data
```

or from Python (e.g. Colab, caching onto Drive):

```python
from ingestion.kaggle_loader import ingest_benchmark
info = ingest_benchmark("cassava", data_root="/content/drive/MyDrive/Jiaozi/kaggle")
```

`info` returns paths ready for Module 4's local CSV dataloader:

```python
{
  "train_csv": ".../train.csv",
  "image_dir": ".../train_images",
  "image_column": "image_id",
  "label_column": "label",
  "test_dir": ".../test_images",        # hidden labels — predict + submit to score
  "sample_submission": ".../sample_submission.csv",
  "num_classes": 5, "metric": "accuracy",
}
```

Available Kaggle keys in the catalog: `cassava`, `state_farm`, `siim_isic`,
`diabetic_retinopathy`. (HuggingFace-sourced keys like `cifar10` use the
`pipeline --dataset` path instead and raise here.)

### Train on it

Feed `train_csv` / `image_dir` / `image_column` / `label_column` into a generated
project's `configs.json` (top-level), then run `run.py` — Module 4's
`_build_local_dataloader` already reads these.

> The competition **test** set has hidden labels: you can't score it locally. Run
> inference over `test_dir`, write predictions in the `sample_submission.csv` format, and
> submit with `kaggle competitions submit -c <competition> -f submission.csv -m "..."`.
> (A prediction/submission helper is not built yet.)

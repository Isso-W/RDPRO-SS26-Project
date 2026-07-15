# Data Engineering for AI & ML [Project]

## GraphRAG-based AutoML Benchmark Agent

This course release uses runtime snapshot `9d3f257` as its compatibility baseline, incorporates the full-chain additions through `61e42e1`, and isolates the standalone benchmark experiment from snapshot `0122829`.

### Submission deliverables

- **Project implementation source code:** the root `pipeline.py` entry point and the `ingestion/`, `analyzer/`, `retrieval/`, `recommender/`, `recipe/`, and `module4_agent/` modules.
- **Reproducible scripted experimentation source code:** `experiments/ab_loss_imbalance/` and the isolated `experiments/mlestar_kaggle_benchmarks/` project, with commands and result boundaries documented in `EXPERIMENTS.md` and `EXPERIMENTAL_RESULTS.md`.

GraphRAG-based AutoML Benchmark Agent is a computer-vision AutoML prototype. It takes a task request and an image dataset, recommends model configurations, and generates runnable experiment code.

Its role is similar to an MLE-style benchmark agent, but the starting point is different. The system does not begin by freely editing code and trying many experiments. It first builds a small set of valid configurations for the supported image task types: `classification`, `object_detection`, `image_segmentation`, and `feature_extraction`. Those configurations are then used by the code generation and validation steps.

MLE-style benchmark protocols are used as design references for evaluation, fold control, and result reporting. They are not runtime dependencies of this project.

The report is organized in three parts. Sections 1-6 describe the active runtime pipeline. Sections 7-9 explain how the knowledge base can be checked and updated. Sections 10-14 cover environment setup, running commands, validation, and submission files.

## 1. Project Goal

Given:

- a natural-language user request, and
- an image dataset reference available to the selected runner,

the system analyzes the task and dataset, retrieves suitable model configurations from a knowledge base, and generates runnable training/evaluation/inference code.

In the current code, the dataset reference can be a HuggingFace image dataset id, a cataloged Kaggle benchmark entry, or a local image folder / CSV image layout. Real-data loading is implemented mainly for `classification` and `feature_extraction`; `object_detection` and `image_segmentation` are supported in recommendation and smoke validation unless a task-specific loader is added.

Local execution is kept small. The local run should check that the selected configuration is valid and that the generated code runs. Longer training is intended for Colab or GPU infrastructure.

Each retrieved candidate is treated as one experiment configuration. The generated code can compare several candidates under the same data assumptions, metric choice, and smoke-test settings. For that reason, this document also explains where the knowledge base comes from, how updates are reviewed, and how recommendations are validated.

The next issue is practical: before code generation or refinement, how does the system decide which image-task configurations are valid enough to try?

## 2. Why KB + RAG in an MLE-Style Agent?

We use MLE-STAR as a reference point because it connects search, experiment feedback, and targeted refinement. The difference is where this project starts. Before running experiments, it tries to make the candidate model space explicit.

For the supported image tasks, many failures happen before training begins: the task type may be mapped incorrectly, the head and loss may not match the backbone, or a recommended checkpoint may be unsuitable for the data and compute constraints. Running more experiments does not solve this if the candidate space itself is poorly structured. For that reason, the KB/RAG layer sits before code generation and experiment execution. It does not replace MLE-style evaluation; it narrows the set of configurations that reach the evaluation stage.

With this order, only a smaller set of candidates reaches Module 4:

- the knowledge base stores image-model components, compatibility rules, pretrained checkpoints, and task constraints;
- graph rules reject invalid backbone/head/loss/optimizer combinations before code is generated;
- semantic retrieval maps flexible user requests into structured candidate configurations;
- Module 4 treats each candidate as an experiment configuration and validates the generated code through smoke tests and reviewer checks.

Compared with a direct experimental-search implementation, this project first puts model choices into inspectable `model_config` objects. Later refinement can still compare candidates and update controlled fields, but those changes stay inside the structure provided by the knowledge base.

The generated project still contains the expected benchmark pieces: runnable experiments, comparable candidates, and validation logs. The difference is that the first image-model decisions are constrained by the KB. The handoff is simple: the KB defines valid choices, RAG retrieves and ranks candidates, and Module 4 turns the selected configuration into code.

## 3. From Knowledge Base to RAG to Code Generation

The knowledge base comes first because image-model selection is not only a language task. A valid recommendation must respect task type, data size, deployment constraints, compatible heads, losses, optimizers, and available pretrained checkpoints. These relationships are easier to maintain as a graph than as free-form prompt text.

A graph alone would be too rigid for real user requests. Users usually write "fast traffic camera detection" or "small medical image dataset" instead of explicit fields such as `task_type=object_detection`, `priority=speed`, or `constraints.medical=true`. Module 3 therefore adds RAG on top of the knowledge base: semantic retrieval makes the input side flexible, while graph rules keep the output side valid and explainable.

```text
Knowledge base
  image-model components and compatibility rules

        |
        v

Graph + vector retrieval
  graph constraints keep recommendations valid
  semantic retrieval matches flexible user language

        |
        v

Ranked Module 3 candidates
  model_config is the source of truth
  tasks provide explanation/prompt context

        |
        v

Module 4 code generation
  model_config -> TrainingSpec -> runnable Python files
```

The contract between Module 3 and Module 4 is small. Module 4 does not select a new model on its own; it consumes the structured `model_config` from Module 3, converts it into an internal training specification, and generates runnable code for model building, training, evaluation, inference, and experiment comparison.

With that contract, the system is easier to trace: each stage turns one structured object into the next.

## 4. End-to-End System

```text
User request + image dataset id
        |
        v
+-----------------------------+
| Module 1: Requirement Parse |
| Natural language -> JSON    |
+-----------------------------+
        |
        v
+-----------------------------+
| Module 2: Dataset Analysis  |
| size, classes, imbalance,   |
| resolution, color mode      |
+-----------------------------+
        |
        v
+-----------------------------+
| Merge Layer                 |
| Module 1 intent + Module 2  |
| dataset statistics          |
+-----------------------------+
        |
        v
+-----------------------------+
| Module 3: KB / RAG          |
| graph rules + semantic      |
| retrieval                   |
+-----------------------------+
        |
        v
+-----------------------------+
| Optional recommender/recipe |
| memory ranking + training   |
| hyperparameter suggestions  |
+-----------------------------+
        |
        v
+-----------------------------+
| Module 4: Code Generation   |
| model.py, train.py, eval,   |
| inference, experiment runs  |
+-----------------------------+
        |
        v
+-----------------------------+
| Local smoke tests / review  |
| then Colab or GPU training  |
+-----------------------------+
```

`pipeline.py` runs the full chain. Module 4 can also be run directly when a Module 3 candidate JSON file is already available.

The diagram shows the responsibility of each stage. The data contract is shown below.

## 5. Core Data Flow

The system passes structured payloads between stages. This keeps module interfaces testable and keeps model selection separate from code generation.

```text
Module 1 output
  task_type, priority, constraints, evaluation_metric

Module 2 output
  total_images, class_distribution, resolution, color_mode

        |
        v

Merged Module 3 input
  task_type
  data_size
  priority
  constraints.class_imbalance
  data_stats
  num_classes

        |
        v

Module 3 candidates
  rank, score, model_config, tasks, alternatives

        |
        v

Module 4 generated project
  model.py
  train.py
  evaluate.py
  infer.py
  run.py
  run_experiments.py
  README_generated.md
```

A small example makes the contract more concrete:

```text
Input
  query: "classify natural images"
  dataset reference: uoft-cs/cifar10

Module 3 candidate
  rank: 1
  model_config:
    task_type: classification
    backbone: efficientnet_b0
    head: classification_head
    loss: cross_entropy_loss
    optimizer: adamw
    finetune_strategy: head_only

Module 4 output
  generated/
    model.py
    train.py
    evaluate.py
    infer.py
    run.py
    run_experiments.py
    README_generated.md

Review summary
  status: approved
  smoke_success: true
  num_candidates: 3
```

The important contract is that `model_config` drives Module 4. Natural-language task descriptions can help with explanations, but generated code is based on the structured fields.

## 6. Module Overview

The modules below are listed in pipeline order. Each subsection starts with the files to inspect first.

### Module 1: Requirement Parsing

Main files:

- `features_extraction_api.py`
- `env_loader.py`
- `pipeline.py`

Module 1 converts a user request into fields such as `task_type`, `priority`, constraints, and the preferred evaluation metric. In the integrated pipeline, this is the only step that has to interpret open-ended user language. The LLM provider is configured through environment variables.

It does not choose the model; it produces the structured task intent used by the later modules.

Example intent mapping:

- "classify leaf disease" -> `classification`
- "detect vehicles in traffic camera images" -> `object_detection`
- "segment medical masks" -> `image_segmentation`
- "extract image embeddings" -> `feature_extraction`

### Module 2: Dataset Analysis

Main files:

- `ingestion/`
- `analyzer/`
- `dataset_analyzer.py`
- `pipeline.py`

Module 2 loads or samples an image dataset and computes lightweight statistics. These statistics are used to infer:

- dataset size tier: `small`, `medium`, or `large`;
- number of classes;
- class imbalance;
- image resolution tier;
- color mode.

The merge logic in `pipeline.py` combines the user's intent with the measured dataset properties before calling Module 3.

Module 2 does not choose an architecture; it supplies dataset facts that make the retrieval query more specific.

### Module 3: Knowledge Base and Retrieval

Main files:

- `retrieval/rag_retrieval.py`
- `retrieval/test_rag_retrieval.py`
- `retrieval/test_golden.py`
- `docs/MODULE3_API.md`

Module 3 selects the model configurations for the supported image tasks. It combines a structured image-model component graph with a vector index. In the current implementation, the vector index is backed by ChromaDB.

```text
Structured component graph
  nodes:
    backbone
    pretrained_model
    head
    loss
    optimizer

  edges:
    compatible_with
    has_pretrained
    alternative_to
    preferred_when

        +

Semantic retrieval
  component descriptions
  ChromaDB collection
  SentenceTransformer embeddings

        |
        v

Top-ranked model configurations
```

Module 3 returns up to three ranked candidates. Each candidate can include:

- `model_config`: structured model/training specification;
- `tasks`: natural-language task list for prompt or explanation context;
- `alternatives`: optional alternative models/backbones.

Ranking uses both graph constraints and semantic retrieval scores. The graph checks task compatibility, constraints, and preferred-condition edges. The vector index helps match the user's description to relevant image-model components.

### Recommender and Recipe Layer

Main files:

- `recommender/`
- `recipe/`
- `cost_meter.py`
- `run_and_log.py`
- `recommender/outcome_memory.py`

The recommender runs after Module 3. It can rerank candidates using outcomes from similar past runs and record the ranking basis. The recipe layer adds training suggestions such as learning rate, image size, augmentation strength, and early stopping settings.

By default, new outcome-memory records are written to `recommender/outcomes.jsonl`. This is a generated local run artifact and is intentionally ignored rather than checked in.

It can change ranking and training suggestions, but it does not replace the Module 3 candidate-generation step.

It is optional in the pipeline:

- `--use-recommender` enables memory-based reranking;
- `--use-recipe` injects recipe hyperparameters into Module 4 configs.

### Module 4: Downstream Code Generation

Main files:

- `module4_agent/spec_builder.py`
- `module4_agent/code_generator.py`
- `module4_agent/executor.py`
- `module4_agent/reviewer.py`
- `module4_agent/workflow.py`
- `module4_agent/refinement.py`
- `module4_agent/experiment_loop.py`
- `module4_agent/tests/`

Module 4 consumes Module 3 candidate outputs and writes a local Python training project. The generated files cover model construction, one-step smoke training, evaluation, inference, and multi-candidate experiment runs.

It turns selected configurations into runnable files and checks that the generated project is internally consistent.

```text
Module 3 candidate JSON
        |
        v
TrainingSpec objects
        |
        v
Generated Python project
        |
        v
Smoke execution
        |
        v
Deterministic review
        |
        v
Optional refinement loop
```

Module 4 keeps local work small:

- local runs use synthetic data by default;
- smoke tests do not download large checkpoints;
- full training is left for Colab/GPU;
- the generated code follows Module 3's `model_config` instead of inventing a new model recommendation.

Since Module 3 makes the model choices, the next question is where its knowledge comes from and how those rules are maintained.

## 7. Knowledge Base Sources

The runtime path described above ends with Module 4 code generation and local validation. The next three sections describe the maintenance path for the knowledge base used by Module 3.

The project uses knowledge in two places. At runtime, Module 3 reads the curated registry in `retrieval/rag_retrieval.py`. Separately, the mining code collects external evidence that can support or challenge that registry. Mined evidence is not used as a second runtime KB, and it is not applied automatically.

### 7.1 Curated Registry

The curated registry defines:

- image-model component nodes;
- compatibility edges;
- preferred-condition edges;
- HuggingFace checkpoint metadata;
- alternatives and fallback behavior.

### 7.2 Kaggle Write-Up Mining

Main files:

- `kb_mining/`
- `kb_mining/data/source_check.md`
- `kb_mining/data/consensus.md`
- `kb_mining/data/proposals.md`
- `docs/kb_mining_protocol.md`

The mining code uses the official Meta Kaggle dump as an evidence source. It extracts patterns from winning-solution write-ups and compares them with the current KB choices. The question is whether successful competition solutions support the current recommendation rules, suggest a small correction, or expose a conflict that needs further validation.

The output is not a new knowledge base. It is a set of evidence-backed proposals that can be reviewed before any runtime behavior is changed.

```text
Meta Kaggle official dump
  Competitions.csv
  ForumTopics.csv
  ForumMessages.csv

        |
        v

harvest.py
  collect winning-solution posts

        |
        v

extract.py
  extract model/loss/strategy facts

        |
        v

aggregate.py
  compute support, breadth, dominance

        |
        v

decide.py
  classify as confirmed, field fix,
  edge tuning, new edge, schema extension,
  conflict, or finding

        |
        v

Human review before KB update
```

`source_check.md` documents that the Meta Kaggle join path is valid:

```text
Competitions.ForumId == ForumTopics.ForumId
ForumTopics.FirstForumMessageId == ForumMessages.Id
```

`consensus.md` summarizes dataset-trait-to-component evidence. Some competition trait cards are marked `traits_verified=False`; those entries are treated as preliminary evidence until checked manually.

## 8. Evidence-Guided Knowledge Base Refinement

This section is about maintaining the KB, not about the normal runtime path. Module 3 starts from a curated runtime KB. That makes the recommendation logic inspectable, but the rules still need evidence beyond the initial design. The mining path adds that check: external evidence is mined, normalized, aggregated, and converted into reviewable refinement proposals. External text does not rewrite the retrieval graph automatically; it only produces proposed KB changes.

Review is needed because Kaggle write-ups are not controlled experiments. They can contain noisy descriptions, ensemble recipes, domain-specific tricks, or competition-specific constraints. The code summarizes evidence and classifies proposals, while the final decision remains human-reviewed.

The update path connects the sources in Section 7 to the validation work in Section 9:

```text
Curated runtime KB
        |
        v
External evidence
  Meta Kaggle write-ups
  benchmark results
        |
        v
Fact extraction and consensus
        |
        v
Refinement proposals
  confirmed
  field/edge update
  new relation/schema extension
  conflict or finding
        |
        v
Human review
        |
        v
Accepted KB refinement + tests
```

During review, we check source validity, support, breadth, conflicts, and the affected KB fields or edges. Accepted changes are applied to the runtime KB in `retrieval/rag_retrieval.py`. Retrieval artifacts are rebuilt when descriptions change, and golden or retrieval-behavior tests are updated before downstream smoke validation.

Mining output is evidence, not executable authority. A proposal that agrees with the existing KB can strengthen the current behavior. A small mismatch can be reviewed as a field or edge update. A proposal that conflicts with the current KB is treated as a hypothesis for a controlled validation experiment. That conflict case is the focus of the next section.

## 9. Validating Knowledge-Base Refinement Proposals

### 9.1 A/B Experiment for a KB Conflict

Main files:

- `experiments/ab_loss_imbalance/`
- `docs/ab_loss_imbalance_protocol.md`
- `docs/ab_loss_imbalance_summary.md`
- `experiments/ab_loss_imbalance/results/outcomes.jsonl`

The A/B run is not an end-to-end evaluation of the whole agent. It checks one KB refinement proposal. The mining pipeline found a specific conflict: for class-imbalanced datasets, should the default loss prefer focal loss or cross entropy?

We use this validation step when mined evidence points in a different direction from the current KB rule. Support counts alone are not enough because the effect may depend on dataset domain, metric choice, fold split, or training noise. The A/B experiment turns the proposed KB change into a controlled comparison.

The repository contains the paired-fold setup, metrics, verdict logic, and ten checked-in Cassava fold records (two arms by five folds). The Cassava collector reconstructs a `CE_WINS` testbed verdict, but the untested severe-imbalance/medical regime remains pending. This dataset-limited result is not an automatic KB edit and does not change the global focal-loss default.

### 9.2 Benchmark Protocol Reference

When a KB proposal needs a real experiment, the validation should follow the same discipline as an MLE-style benchmark rather than an informal training run. The protocol reference defines:

- fixed folds and seeds;
- matched budgets;
- OOF-only model selection;
- separate public Kaggle submission reporting;
- deterministic baseline vs search/refinement/ensemble comparison arms.

Detailed experiment cases, artifacts, and completed or pending scores belong in `EXPERIMENTAL_RESULTS.md`. The active runtime does not depend on those reference experiments.

## 10. Environment Configuration

Use a dedicated environment for the root project, which supports Python 3.9 or newer. The standalone benchmark under `experiments/mlestar_kaggle_benchmarks/` has an independent Python 3.11-or-newer environment and must be installed separately. Local secrets should be stored in `.env`, which is ignored by git; `.env.example` provides the template.

Common configuration cases:

- Module 1 needs an LLM provider when parsing free-form natural-language requests.
- Module 4 can run with `M4_LLM_PROVIDER=none`, which uses deterministic code templates for local smoke validation.
- Kaggle credentials are only needed for Kaggle data download, Meta Kaggle mining, or benchmark notebooks.
- GPU access is not required for local smoke tests, but is required for full training or benchmark-scale experiments.

The main environment variables are:

| Variable | Required when | Used by | Notes |
| --- | --- | --- | --- |
| `JIAOZI_LLM_PROVIDER` | parsing free-form requests | Module 1 | Usually `qwen` or `openai`. |
| `JIAOZI_DASHSCOPE_API_KEY` | using Qwen/DashScope | Module 1 and Module 4 | Kept in `.env`; never committed. |
| `DASHSCOPE_BASE_URL` | using DashScope-compatible API | Module 1 and Module 4 | Defaults to DashScope's compatible endpoint. |
| `OPENAI_API_KEY` | using OpenAI | Module 1 or Module 4 | Only needed if an OpenAI provider is selected. |
| `M4_LLM_PROVIDER` | optional for Module 4 | Module 4 | `none` uses templates; `qwen`, `openai`, or `vertex` request LLM generation. |
| `KAGGLE_USERNAME` / `KAGGLE_KEY` | Kaggle workflows | ingestion, mining, notebooks | Needed for download, submission, or Meta Kaggle data access. |
| `KAGGLE_API_TOKEN` | Kaggle workflows in Colab | notebooks | Store as a Colab Secret; never place the token in notebook source. |
| `HF_TOKEN` | gated HuggingFace checkpoints | generated training code | Only needed for restricted model weights. |

## 11. Running the Project

Install dependencies:

```bash
python -m pip install -e '.[dev]'
```

Open a notebook from the published `main` branch:

- [Full-chain Colab notebook](https://colab.research.google.com/github/Isso-W/Jiaozi/blob/main/jiaozi_fullchain.ipynb)
- [Integrated pipeline Colab notebook](https://colab.research.google.com/github/Isso-W/Jiaozi/blob/main/integration_update_colab.ipynb)
- [Vision benchmark Colab notebook](https://colab.research.google.com/github/Isso-W/Jiaozi/blob/main/vision_benchmarks_colab.ipynb)
- [Kaggle benchmark Colab notebook](https://colab.research.google.com/github/Isso-W/Jiaozi/blob/main/kaggle_benchmark_colab.ipynb)

The notebooks read credentials from Colab Secrets and do not contain API keys.

Run the integrated pipeline:

```bash
python pipeline.py \
  --query "classify natural images" \
  --dataset uoft-cs/cifar10 \
  --fmt nl \
  --module4-output generated_pipeline \
  --module4-no-smoke
```

Run Module 4 directly:

```bash
python -m module4_agent \
  --input module4_agent/examples/sample_m3_output.json \
  --output generated/
```

Useful optional flags:

```bash
--use-recommender
--use-recipe
--module4-run-refinement
--module4-real-training
--module4-llm-provider qwen
```

Rebuild the checked-in Cassava A/B summary:

```bash
python -m experiments.ab_loss_imbalance.collect
```

Run the isolated scripted experiment under Python 3.11:

```bash
cd experiments/mlestar_kaggle_benchmarks
python3.11 -m pip install -e '.[vision,dev]'
python3.11 scripts/run_smoke_experiment.py --output-dir /tmp/mlestar-smoke
```

The standalone smoke command uses synthetic data and disables submissions.

## 12. Validation Strategy

Validation starts with cheap checks before moving to expensive experiments.

Most unit and smoke tests run offline once dependencies are installed. Integrated dataset loading, Kaggle mining, LLM extraction, and real benchmark training may require network access, API credentials, or GPU resources depending on the selected workflow.

Current boundaries:

- Local smoke tests validate generated code and configuration consistency; they are not benchmark training runs.
- Full training, Kaggle submission, and public-score reporting require the appropriate data access, credentials, and usually GPU/Colab resources.
- General real-data loaders are implemented for classification-style image datasets and feature extraction. Detection and segmentation need task-specific loaders for bounding boxes or masks.
- Knowledge-base mining produces reviewable proposals. It does not automatically rewrite the runtime retrieval graph.

### Unit and Behavior Tests

```bash
python -m unittest test_pipeline.py -v
(cd retrieval && python -m unittest test_rag_retrieval.py -v)
(cd retrieval && python -m unittest test_golden.py -v)
python -m pytest module4_agent/tests -q
python -m pytest recipe/tests -q
python -m pytest kb_mining/tests -q
python -m pytest experiments/ab_loss_imbalance/tests -q
```

### Module 4 Smoke Validation

```bash
M4_LLM_PROVIDER=none python -m module4_agent \
  --input module4_agent/examples/sample_m3_output.json \
  --output /tmp/graphrag-automl-m4-smoke
```

The smoke path checks that generated code can compile, build lightweight models, run tiny synthetic-data training/evaluation/inference, and pass the candidate consistency review.

### Benchmark-Level Validation

Any benchmark claim should follow the stricter MLE-style benchmark protocol:

- fixed folds;
- identical seeds;
- matched time budgets;
- OOF-only selection;
- public Kaggle scores only when the API accepts a submission.

## 13. Repository Map

```text
.
|-- pipeline.py                       # integrated full-chain entry point
|-- features_extraction_api.py         # Module 1 requirement parsing
|-- ingestion/                         # dataset loading utilities
|-- analyzer/                          # image statistics and dataset analysis
|-- retrieval/                         # Module 3 KB/RAG recommender
|-- recommender/                       # outcome-memory reranking
|-- recipe/                            # training recipe suggestions
|-- module4_agent/                     # Module 4 code-generation workflow
|-- kb_mining/                         # evidence mining for KB updates
|-- experiments/ab_loss_imbalance/      # paired loss experiment + Cassava records
|-- experiments/mlestar_kaggle_benchmarks/ # isolated Python 3.11 experiment
|-- docs/                              # design notes and API docs
|-- CONTRIBUTIONS.md                  # history-backed module attribution
|-- EXPERIMENTS.md                    # reproducible experiment commands
|-- EXPERIMENTAL_RESULTS.md           # completed and pending results
|-- jiaozi_fullchain.ipynb            # complete M1-to-M4 Colab demo
|-- integration_update_colab.ipynb    # integration notebook
|-- vision_benchmarks_colab.ipynb     # vision benchmark notebook
`-- kaggle_benchmark_colab.ipynb      # Kaggle benchmark notebook
```

## 14. Submission-Oriented Documents

For final project submission, the README should be accompanied by:

- `README.md`: polished version of this project guide;
- `CONTRIBUTIONS.md`: which member wrote which parts, with links to files or sections;
- `EXPERIMENTS.md`: how to run scripted experiments and where logs are stored;
- `EXPERIMENTAL_RESULTS.md`: what experiments were run, what they show, and which results are still planned rather than completed.

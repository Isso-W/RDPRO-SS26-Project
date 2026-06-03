# CV Auto-DL

This repository currently keeps the active Module 3 retrieval layer and the new
Module 4 local code-generation agent.

## Module 4: Local Code-Generation Agent

Module 4 consumes ranked candidate configurations from Module 3 and generates a
small runnable PyTorch project for local CPU smoke testing. The generated code is
intended as a stable prototype that users can later adapt for Colab/GPU training.

Module 3 is the Knowledge Base / retrieval module. It recommends up to three CV
model configurations from structured user and task constraints. Module 4 treats
each Module 3 candidate as one experiment configuration and sweeps all of them in
`run_experiments.py`.

The structured `model_config` is the source of truth. Natural-language `tasks`
are only used as explanatory context. If they conflict, Module 4 follows
`model_config`.

### Expected Input

Module 4 accepts a JSON file containing either one candidate object, a list of
candidate objects, or an object with a `candidates` list.

```json
[
  {
    "format": "nl",
    "rank": 1,
    "score": 0.92,
    "model_config": {
      "task_type": "classification",
      "backbone": "efficientnet_b0",
      "pretrained_hf_id": "google/efficientnet-b0",
      "head": "classification_head",
      "loss": "cross_entropy_loss",
      "optimizer": "adamw",
      "finetune_strategy": "head_only",
      "freeze_backbone": true
    },
    "tasks": [
      "Load EfficientNet-B0 pretrained on ImageNet.",
      "Use head-only fine-tuning and freeze the backbone."
    ],
    "alternatives": []
  }
]
```

Supported task types:

- `classification`
- `object_detection`
- `image_segmentation`
- `feature_extraction`

### CLI Usage

```bash
python -m module4_agent --input module4_agent/examples/sample_m3_output.json --output generated/
```

All supported task families can be smoke-tested with:

```bash
python -m module4_agent --input module4_agent/examples/sample_m3_output_all_tasks.json --output generated_all_tasks/
```

To generate and run static review without executing smoke subprocesses:

```bash
python -m module4_agent --input module4_agent/examples/sample_m3_output.json --output generated/ --no-smoke
```

To also run the experiment-level refinement loop after code generation, smoke
testing, and review pass:

```bash
python -m module4_agent --input module4_agent/examples/sample_m3_output.json --output generated/ --run-refinement
python -m module4_agent --input module4_agent/examples/sample_m3_output.json --output generated/ --run-refinement --max-refinement-iters 2 --improvement-threshold 0.01
```

The CLI will:

1. load Module 3 candidate configs;
2. build internal training specs;
3. generate all required files into the output directory;
4. run CPU smoke tests with synthetic data;
5. run deterministic reviewer checks;
6. optionally run baseline → ablation → targeted refinement when
   `--run-refinement` is enabled;
7. write `module4_summary.json` and print the same JSON-like final summary.

### Generated Files

```text
generated/
  configs.json
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

When `--run-refinement` is enabled, Module 4 also writes:

```text
generated/
  experiments.jsonl
  leaderboard.json
  refinement_summary.json
  best_config.json
```

`configs.json` contains the normalized candidate configs that Module 4 actually
generated from Module 3 output. `run.py` runs the first configuration by default:

```bash
python run.py --config configs.json
```

`run_experiments.py` sweeps every Module 3 candidate using the same random seed
and synthetic data setup for fair smoke comparison:

```bash
python run_experiments.py --input configs.json
```

`module4_summary.json` records the final workflow status, generated files,
smoke command results, reviewer feedback, warnings/errors, iteration history,
per-candidate summaries, and optional refinement results.

### Smoke Test vs Proxy Evaluation

Module 4 now has two separate iteration levels:

- Code correctness loop: Coder → Executor → Reviewer. This generates runnable
  files, executes tiny CPU smoke tests on synthetic tensors, and checks that the
  generated project is complete and consistent with Module 3 configs.
- Experiment refinement loop: baseline → ablation → targeted refinement →
  proxy evaluation. This is enabled only with `--run-refinement`.

The refinement loop selects the top-ranked Module 3 candidate as the baseline,
generates controlled ablations that modify only one allowed component at a time,
evaluates them with a deterministic proxy metric, applies a small targeted
refinement to the best ablation, and repeats until the improvement threshold is
reached, max iterations are reached, or no variant improves.

Allowed refinement components are:

- `optimizer`
- `learning_rate`
- `augmentation`
- `finetune_strategy`
- `loss`
- `backbone` / `checkpoint`, only when Module 3 supplied alternatives

The loop does not modify task type, metric choice, logging format, data split
policy, experiment loop structure, or unrelated task fields.

`experiments.jsonl` records every baseline, ablation, and refinement result.
`leaderboard.json` sorts successful proxy results by metric value, with higher
being better for the current proxy metrics. `refinement_summary.json` reports
the baseline result, best result, selected component, improvement, and stop
reason. `best_config.json` exports the best proxy-selected config directly,
with `_module4_refinement` metadata for traceability.

These proxy metrics are workflow-level signals, not real benchmark scores.
They reward reasonable choices in a stable way, such as task-compatible
optimizers, focal loss for imbalanced classification, Dice-style segmentation
losses, pretrained configs for small data, and full fine-tuning for larger data.
When generated smoke code is available, the loop also folds in the tiny
synthetic smoke-training loss as a deterministic local signal. Real validation
metrics can be integrated later without changing the overall loop structure.

### Reviewer Checks

The deterministic reviewer rejects generated code when:

- required generated files are missing;
- generated Python files do not compile;
- smoke tests fail, unless `--no-smoke` was explicitly used;
- `run_experiments.py` does not visibly sweep all candidates;
- `configs.json` does not match the internal `TrainingSpec` list;
- important `model_config` fields are not represented;
- metric names do not match task types;
- experiment rows omit required summary fields;
- `head_only` does not freeze backbone-like parameters, or `full` unexpectedly
  freezes parameters.
- refinement artifacts are missing when `--run-refinement` is enabled;
- ablation/refinement rows modify multiple components at once;
- refinement changes forbidden fields such as `task_type` or metric choice.

### Current Limitations

- Real long training is not performed locally.
- Generated code uses lightweight dummy PyTorch models by default.
- Refinement uses deterministic proxy metrics, not real benchmark metrics.
- Real HuggingFace checkpoint loading can be added later.
- Object detection and segmentation are smoke-test compatible first, not full
  benchmark training implementations.
- LangGraph and an LLM reviewer are optional future extensions; the current
  implementation uses deterministic templates and deterministic review checks.

### Tests

```bash
pytest module4_agent/tests
pytest retrieval/test_rag_retrieval.py
pytest
```

Module 3 retrieval tests use a deterministic local embedding fallback by
default so they do not require HuggingFace network access. Set
`CV_AUTODL_EMBEDDINGS=hf` to use the real SentenceTransformer embedding path.

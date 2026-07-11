# MLE-STAR DataOps Agent

`mlestar` is a reproducible implementation of the MLE-STAR method, adapted so
that every generated project is a skrub DataOps DAG rather than an unstructured
training script. It is not a claim of identical provider/model results to the
Google implementation.

Each candidate project contains `pipeline.py` with five immutable component
boundaries: `data_loading`, `data_preparation`, `model`, `training`, and
`prediction`. The generated functions are named `load_data`, `prepare_data`,
`build_model`, `train_model`, and `predict_or_submit`. The executor runs them
through `skrub.var` and `skrub.deferred`, records the DAG description in
`dataops_report.json`, and keeps the run context and component trace owned by
the framework.

## Local setup

```bash
python -m pip install -r requirements.txt
```

Create a task file such as `task.json`:

```json
{
  "task_id": "my_binary_task",
  "modality": "tabular",
  "target_columns": ["target"],
  "id_column": "id",
  "metric": {"name": "roc_auc", "greater_is_better": true},
  "components": [
    {"name": "data_loading"},
    {"name": "data_preparation"},
    {"name": "model"},
    {"name": "training"},
    {"name": "prediction"}
  ]
}
```

Generate a safe offline project plan first:

```bash
python -m mlestar run \
  --task task.json \
  --data-root ./input \
  --run-dir ./runs/my_binary_task \
  --plan-only
```

To request LLM-generated source, configure `JIAOZI_DASHSCOPE_API_KEY` (Qwen) or
`OPENAI_API_KEY` (OpenAI) and use `--llm-provider qwen` or `openai`. The LLM
must return a JSON file envelope; unsafe/invalid output is rejected and replaced
with a deterministic plan-only fallback. A real run requires the generated
prediction component to write OOF predictions and the task metric itself; proxy
scores are never selected.

## Run artifacts

The run directory contains `task.json`, `inventory.json`, `search_evidence.json`,
`candidates.json`, `audit.jsonl`, `experiments.jsonl`, each generated project,
and `final_report.json`. The inventory fingerprint and code SHA-256 make every
result traceable. `execution_receipt.json` is written even if a project fails.

Generated code is audited before execution for test-statistics leakage, unused
inputs, dynamic execution, subprocess/network access, unsafe absolute writes,
and missing imports. The executor then uses a fresh process with LLM/Kaggle
credentials removed from its environment.

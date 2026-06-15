# MCP Knowledge-Guided Dog Breed Experiments

`integration_update_colab.ipynb` is the canonical training entrypoint. It clones
`mcp_knowledge`, reads LLM and Kaggle credentials from Colab Secrets, mounts Drive,
and persists all learning and experiment artifacts under:

```text
/content/drive/MyDrive/Jiaozi/
  knowledge_base/
  kaggle_data/
  workspace/dog_breed/module4_code/
  reports/
```

## Architecture

The Knowledge Learner uses the fixed source list in `knowledge/sources.py`. It calls
the FastMCP stdio server to ingest text, create compact summaries, extract strategy
cards, merge evidence, and rebuild the SQLite FTS index. At least three sources must
succeed.

The MLE Experiment Agent has no web or raw-source API. It calls these MCP tools in
order:

```text
search_strategy_cards
get_past_experiments
generate_experiment_configs
run_experiment
read_metrics
compare_results
write_experiment_result
```

Retrieval returns at most five compact cards and ten historical outcomes. The planner
creates at most three experiments and changes at most two strategy-controlled fields
per experiment. Baseline split, seed, and `recommended_epochs` are preserved.

For fine-grained Dog Breed runs, AutoPipeline first probes the Module 3 Top-3
candidates on the same stratified holdout. The selected baseline then enters the
MCP loop. Strategy cards can execute partial DINOv2 fine-tuning,
discriminative backbone/head learning rates, 336-pixel training, label
smoothing, and horizontal-flip TTA.

## Safety

`run_experiment` accepts structured JSON only. The project must be below
`JIAOZI_WORKSPACE_ROOT`, and the only executed process is:

```text
python -u run.py --config <generated-json>
```

Each experiment receives an isolated checkpoint directory. Shell commands, external
working directories, and caller-provided executable paths are rejected.

## Colab Secrets

Provide one LLM credential:

- `JIAOZI_DASHSCOPE_API_KEY`, or
- `OPENAI_API_KEY`

Provide Kaggle credentials as either:

- `KAGGLE_JSON`, containing the full credential JSON, or
- `KAGGLE_USERNAME` and `KAGGLE_KEY`

The Kaggle competition rules must already be accepted for
`dog-breed-identification`.

## Outputs

The report records the AutoPipeline selection, fixed-source strategy cards, controlled
configuration differences, validation Log Loss, accuracy, macro-F1, best epoch,
runtime and token cost, selected checkpoint, Git commit, submission status, and
official Kaggle score when available. A timeout or rejected submission keeps
`submission.csv` and records a null score.

Every successful run also writes validation probabilities. Jiaozi searches
non-negative ensemble weights on validation data and submits the ensemble only
when it improves validation Log Loss over the best individual model.

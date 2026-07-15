# `ab_loss_imbalance`: A/B arbitration of CE vs focal loss

This experiment resolves the conflict over whether `class_imbalance` should default to focal loss or cross-entropy. `docs/ab_loss_imbalance_protocol.md` is the authoritative specification and defines the pre-registered decision rule.

## Components

- `configs.py`: Frozen experiment matrix and decision constants.
- `collect.py`: Pure functions that compute paired fold differences, testbed verdicts, and the combined verdict.
- `run_ab.py`: Creates folds, generates the project, trains each `(arm, fold)` pair, and appends outcomes. Requires data and a GPU.
- `tests/`: Offline tests for the frozen configuration, verdict logic, and runner helpers.
- `results/outcomes.jsonl`: One append-only record per fold; completed runs can be resumed safely.

## Current status

- The offline foundation is complete: paired-fold injection, fold-integrity checks, validation-prediction export, frozen experiment matrices, verdict aggregation, and resume logic all have tests.
- Prediction export writes both `y_prob` and `y_score`; `run_ab.py` accepts either field.
- Cassava is complete: two arms × five folds, for 10 records total. The default `collect` command reconstructs the `CE_WINS` testbed result from `results/outcomes.jsonl`.
- SIIM-ISIC remains incomplete. The single Cassava result is not enough to change the global focal-loss default. See the repository-level `EXPERIMENTAL_RESULTS.md` for scope and follow-up work.

## Running the experiment

```bash
# Offline tests
python -m pytest experiments/ab_loss_imbalance/tests/ module4_agent/tests/test_fold_injection.py -q

# Smoke-test one arm/fold pair, then run both testbeds
python -m experiments.ab_loss_imbalance.run_ab --testbed cassava --only focal_loss:0
python -m experiments.ab_loss_imbalance.run_ab --testbed cassava
python -m experiments.ab_loss_imbalance.run_ab --testbed siim_isic

# Aggregate the verdict
python -m experiments.ab_loss_imbalance.collect
```

After applying any verdict-specific KB action from §4 of the protocol, `cd retrieval && pytest test_golden.py test_rag_retrieval.py -q` must pass.

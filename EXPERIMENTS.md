# Reproducible Experiments

The repository separates the active Jiaozi runtime from two reproducible experiment areas. Generated artifacts, raw competition data, credentials, and model weights are intentionally excluded from version control.

## 1. CE versus focal loss on class-imbalanced data

The paired-fold harness is in `experiments/ab_loss_imbalance/`. Its checked-in Cassava records are at `experiments/ab_loss_imbalance/results/outcomes.jsonl`.

Rebuild the summary from the ten checked-in fold records:

```bash
python -m experiments.ab_loss_imbalance.collect
```

Run the offline logic tests:

```bash
python -m pytest experiments/ab_loss_imbalance/tests \
  module4_agent/tests/test_fold_injection.py -q
```

Run new training only on a machine with accepted Kaggle competition terms, local credentials, the dataset, and a GPU:

```bash
python -m experiments.ab_loss_imbalance.run_ab \
  --testbed cassava \
  --data-root /path/to/kaggle_data \
  --output /path/to/ab_runs
```

The experiment uses seed 42 and a shared fold file for both loss arms. The collector reports paired `CE - focal` differences. A result from one dataset does not change the project-wide focal-loss default.

## 2. Standalone benchmark reproduction

The independent Python 3.11 project is isolated at `experiments/mlestar_kaggle_benchmarks/`. Install and run it from that directory so its dependencies do not alter the root environment:

```bash
cd experiments/mlestar_kaggle_benchmarks
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[vision,dev]'
python -m pytest -q
python scripts/run_smoke_experiment.py --output-dir /tmp/mlestar-smoke
```

The scripted command fixes seed 13, uses only the synthetic Leaf Classification data, writes a configuration/environment/data-hash manifest, and asserts that no submission file is produced. It is an offline reproducibility check, not a Kaggle score. Real benchmark work uses fixed folds, identical seeds, matched budgets, and OOF-only model selection as specified in `experiments/mlestar_kaggle_benchmarks/docs/EVALUATION_PROTOCOL.md`.

## 3. Root pipeline and Module 4 smoke run

Install the root project and execute the deterministic path:

```bash
python -m pip install -e '.[dev]'
M4_LLM_PROVIDER=none python -m module4_agent \
  --input module4_agent/examples/sample_m3_output.json \
  --output /tmp/jiaozi-module4-smoke
```

This validates generated code on synthetic inputs. It is not evidence of real-data model quality.

## Reproducibility records

For every new real experiment, retain these fields in the run directory:

- source revision and clean/dirty state;
- Python, operating-system, package, and accelerator information;
- seed, fold assignment, complete configuration, and command;
- SHA-256 hashes for configuration and input manifests;
- per-fold OOF metrics and elapsed time;
- submission status, which must remain disabled unless explicitly requested.

Do not commit `.env`, Kaggle tokens, raw Meta Kaggle forum text, raw competition datasets, checkpoints, generated projects, or submission files.

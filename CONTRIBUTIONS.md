# Contributions

This file summarizes the main work done by each project member and links the
work to the relevant code, experiment, or documentation files.

## Xuanyan Wang

Xuanyan worked on the main agent pipeline, the Module 4 code-generation agent,
and the final integration of the runtime path.

- Integrated the full pipeline flow from parsed user intent and dataset analysis
  to model recommendation, code generation, and local validation.
  - Related files: [`pipeline.py`](pipeline.py), [`run_for_testing.py`](run_for_testing.py), [`configs/pipeline.yaml`](configs/pipeline.yaml), [`test_pipeline.py`](test_pipeline.py)

- Connected the early pipeline stages to the later recommendation and code
  generation steps, including the data passed between Module 1, Module 2, Module
  3, and Module 4.
  - Related files: [`features_extraction_api.py`](features_extraction_api.py), [`dataset_analyzer.py`](dataset_analyzer.py), [`ingestion/`](ingestion/), [`analyzer/`](analyzer/)

- Developed Module 4, which turns Module 3 candidate configurations into a
  runnable local training project.
  - Related files: [`module4_agent/spec_builder.py`](module4_agent/spec_builder.py), [`module4_agent/code_generator.py`](module4_agent/code_generator.py), [`module4_agent/executor.py`](module4_agent/executor.py), [`module4_agent/reviewer.py`](module4_agent/reviewer.py), [`module4_agent/workflow.py`](module4_agent/workflow.py)

- Added the Module 4 smoke-test path, controlled refinement support, ablation
  handling, and experiment tracking.
  - Related files: [`module4_agent/smoke_harness.py`](module4_agent/smoke_harness.py), [`module4_agent/ablation.py`](module4_agent/ablation.py), [`module4_agent/refinement.py`](module4_agent/refinement.py), [`module4_agent/experiment_loop.py`](module4_agent/experiment_loop.py), [`module4_agent/experiment_tracker.py`](module4_agent/experiment_tracker.py), [`module4_agent/tests/`](module4_agent/tests/)

- Worked on recipe and runtime support used around the generated experiments.
  - Related files: [`recipe/`](recipe/), [`recommender/`](recommender/), [`cost_meter.py`](cost_meter.py), [`run_and_log.py`](run_and_log.py)

## Letian Wang

Letian worked on Module 3, including the knowledge base, RAG retrieval, and the
model recommendation interface used by the downstream agent.

- Built and maintained the Module 3 knowledge-base and retrieval logic for
  recommending computer-vision model configurations.
  - Related files: [`retrieval/rag_retrieval.py`](retrieval/rag_retrieval.py), [`retrieval/test_rag_retrieval.py`](retrieval/test_rag_retrieval.py), [`retrieval/test_golden.py`](retrieval/test_golden.py)

- Defined the Module 3 output contract used by Module 4, including ranked
  candidates and structured `model_config` fields.
  - Related files: [`docs/MODULE3_API.md`](docs/MODULE3_API.md), [`docs/report_module3.md`](docs/report_module3.md), [`docs/module3_4_technical_en.md`](docs/module3_4_technical_en.md)

- Worked on knowledge-base maintenance and the evidence-mining path used to
  review possible KB updates.
  - Related files: [`kb_mining/`](kb_mining/), [`kb_mining/data/`](kb_mining/data/), [`kb_mining/tests/`](kb_mining/tests/), [`docs/kb_mining_protocol.md`](docs/kb_mining_protocol.md)

- Helped connect the KB/RAG recommendation step with the integrated agent flow.
  - Related files: [`pipeline.py`](pipeline.py), [`module4_agent/examples/`](module4_agent/examples/), [`docs/module3_improvements.md`](docs/module3_improvements.md)

## Zeyu Wang

Zeyu worked on the MLE-style Kaggle benchmark reproduction and the standalone
benchmark framework used for comparison and protocol reference.

- Implemented the standalone MLE-style benchmark reproduction code.
  - Related files: [`experiments/mlestar_kaggle_benchmarks/`](experiments/mlestar_kaggle_benchmarks/)

- Added benchmark adapters, experiment execution, metric handling, search and
  refinement components, and tests for the standalone benchmark package.
  - Related files: [`experiments/mlestar_kaggle_benchmarks/mlestar/`](experiments/mlestar_kaggle_benchmarks/mlestar/), [`experiments/mlestar_kaggle_benchmarks/tests/`](experiments/mlestar_kaggle_benchmarks/tests/), [`experiments/mlestar_kaggle_benchmarks/scripts/run_smoke_experiment.py`](experiments/mlestar_kaggle_benchmarks/scripts/run_smoke_experiment.py)

- Connected Kaggle benchmark metadata and helper scripts to the root project.
  - Related files: [`run_kaggle_benchmark.py`](run_kaggle_benchmark.py), [`vision_benchmark_catalog.py`](vision_benchmark_catalog.py), [`kaggle_submit.py`](kaggle_submit.py)

- Wrote supporting documentation for the benchmark reproduction and evaluation
  protocol.
  - Related files: [`experiments/mlestar_kaggle_benchmarks/README.md`](experiments/mlestar_kaggle_benchmarks/README.md), [`experiments/mlestar_kaggle_benchmarks/docs/`](experiments/mlestar_kaggle_benchmarks/docs/)

## Mingyue Fan

Mingyue worked on benchmark execution and result collection for the selected
Kaggle image tasks.

- Ran and checked benchmark notebooks for the selected task set, including image
  classification, multi-label classification, segmentation, and detection cases.
  - Related files: [`experiments/notebook_runs/notebooks/`](experiments/notebook_runs/notebooks/), [`experiments/notebook_runs/logs/`](experiments/notebook_runs/logs/)

- Helped collect notebook outputs, leaderboard records, and normalized result
  files for reviewer inspection.
  - Related files: [`experiments/notebook_runs/manifest.json`](experiments/notebook_runs/manifest.json), [`experiments/notebook_runs/results/rdpro_experiment_v2_scores.csv`](experiments/notebook_runs/results/rdpro_experiment_v2_scores.csv), [`experiments/notebook_runs/results/source_manifest.json`](experiments/notebook_runs/results/source_manifest.json)

- Contributed to the experiment description and result discussion.
  - Related files: [`EXPERIMENTS.md`](EXPERIMENTS.md), [`EXPERIMENTAL_RESULTS.md`](EXPERIMENTAL_RESULTS.md)

## Mingda Zhang

Mingda worked on benchmark execution, evidence checking, and result reporting for
the Kaggle task set.

- Ran and reviewed benchmark notebooks for the selected MLE-bench Lite and
  custom extension tasks.
  - Related files: [`experiments/notebook_runs/notebooks/`](experiments/notebook_runs/notebooks/), [`experiments/notebook_runs/logs/`](experiments/notebook_runs/logs/)

- Checked stored notebook outputs and per-cell logs used as experiment evidence.
  - Related files: [`experiments/notebook_runs/export_evidence.py`](experiments/notebook_runs/export_evidence.py), [`experiments/notebook_runs/test_evidence.py`](experiments/notebook_runs/test_evidence.py), [`experiments/notebook_runs/manifest.json`](experiments/notebook_runs/manifest.json)

- Contributed to the final experiment records and result summary.
  - Related files: [`EXPERIMENTS.md`](EXPERIMENTS.md), [`EXPERIMENTAL_RESULTS.md`](EXPERIMENTAL_RESULTS.md)

## Shared Final Review

The final submission documents were reviewed and edited jointly so that the
repository could be read as a single project rather than as separate branches.

- Related files: [`README.md`](README.md), [`CONTRIBUTIONS.md`](CONTRIBUTIONS.md), [`EXPERIMENTS.md`](EXPERIMENTS.md), [`EXPERIMENTAL_RESULTS.md`](EXPERIMENTAL_RESULTS.md), [`docs/`](docs/)

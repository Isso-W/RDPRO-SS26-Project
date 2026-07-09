# MCP Knowledge-Guided MLE Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a two-stage, MCP-enabled system that learns compact ML strategy cards from external notes, retrieves only local knowledge at experiment time, runs controlled Module 4 experiments, and writes measured outcomes back into the existing accumulating recommender memory.

**Architecture:** Keep the existing Module 1-4 pipeline, `recommender.OutcomeMemory`, generated Module 4 projects, and `cost_meter` as the execution foundation. Add a JSON + SQLite FTS knowledge store for source summaries and strategy cards, expose pure Python services through an official FastMCP server, then add a Knowledge Learner Agent and a low-token MLE Experiment Agent that call the same service functions. Real training remains opt-in and can only execute a fixed `python -u run.py --config <generated-json> --epochs <n>` command inside an allowlisted workspace root.

**Tech Stack:** Python 3.10+, standard-library `sqlite3` FTS5, dataclasses, existing OpenAI-compatible provider integration, official `mcp[cli]` Python SDK, pytest, existing PyTorch/Module 4 generated projects.

---

## Scope And Reuse Decisions

- `recommender/outcome_memory.py` remains the canonical append-only experiment history. Do not create a second experiment database.
- `knowledge_base/source_summaries/` and `knowledge_base/strategy_cards/` are canonical JSON documents. SQLite stores a searchable copy and can be rebuilt.
- Phase 1 ingestion accepts caller-provided text or an existing local file. It does not crawl arbitrary URLs.
- `module4_agent/code_generator.py` remains the owner of generated training/evaluation behavior.
- MCP tool functions are thin wrappers around testable services. Agents may call those services directly in unit tests; MCP is the external protocol boundary.
- The MCP server defaults to stdio. It must log to stderr because stdout is reserved for JSON-RPC.
- The MLE agent defaults to plan-only mode. Real training requires `execute=True` or the CLI `--execute` flag.

## Target File Map

**Create**

- `knowledge/__init__.py` - public knowledge-store API.
- `knowledge/schemas.py` - source, summary, evidence, strategy-card, proposal, and comparison contracts.
- `knowledge/store.py` - atomic JSON persistence and SQLite FTS index.
- `knowledge/scoring.py` - deterministic strategy relevance and priority scoring.
- `knowledge/seeds.py` - initial card loader.
- `knowledge_base/raw_sources/.gitkeep` - runtime source location.
- `knowledge_base/source_summaries/.gitkeep` - runtime summary location.
- `knowledge_base/strategy_cards/randaugment.json` - seed card.
- `knowledge_base/strategy_cards/cutmix.json` - seed card.
- `knowledge_base/strategy_cards/mixup.json` - seed card.
- `knowledge_base/strategy_cards/label_smoothing.json` - seed card.
- `knowledge_base/strategy_cards/tta.json` - seed card.
- `knowledge_base/strategy_cards/efficientnet_b4.json` - seed card.
- `knowledge_base/experiments/.gitkeep` - default outcome-memory directory.
- `knowledge_base/index/.gitkeep` - generated SQLite index directory.
- `recommender/experiment_planner.py` - controlled proposal generation and duplicate rejection.
- `module4_agent/result_parser.py` - shared JSON result extraction.
- `mcp_server/__init__.py` - MCP package marker.
- `mcp_server/context.py` - configured store, memory, workspace, and runner dependencies.
- `mcp_server/server.py` - FastMCP registration and transport entry point.
- `mcp_server/tools/__init__.py` - tool package marker.
- `mcp_server/tools/knowledge_tools.py` - ingestion, summarization, extraction, upsert, and search services.
- `mcp_server/tools/experiment_tools.py` - history, planning, execution, metrics, comparison, and write-back services.
- `mcp_server/tools/report_tools.py` - report service.
- `agents/__init__.py` - agent package marker.
- `agents/prompts.py` - strict JSON prompts for source summarization and card extraction.
- `agents/knowledge_learner_agent.py` - stage 1 orchestration.
- `agents/mle_experiment_agent.py` - stage 2 orchestration.
- `tests/test_knowledge_store.py` - persistence/index tests.
- `tests/test_strategy_scoring.py` - ranking tests.
- `tests/test_experiment_planner.py` - mutation and duplicate tests.
- `tests/test_experiment_tools.py` - safe execution and comparison tests.
- `tests/test_knowledge_learner_agent.py` - mocked LLM learning tests.
- `tests/test_mcp_server.py` - registered tool and direct-call tests.
- `tests/test_mle_experiment_agent.py` - plan-only and mocked execute-loop tests.
- `tests/fixtures/cassava_solution_note.md` - deterministic ingestion fixture.
- `docs/MCP_KNOWLEDGE_AGENT.md` - setup, tool contracts, and demo.

**Modify**

- `pyproject.toml` - Python floor, MCP dependency, package discovery, CLI scripts.
- `requirements.txt` - MCP runtime dependency.
- `.env.example` - knowledge-agent, KB-root, workspace-root, and MCP transport settings.
- `.gitignore` - generated SQLite, outcomes, run artifacts, and raw source content.
- `module4_agent/schemas.py` - strategy-controlled config fields.
- `module4_agent/spec_builder.py` - parse strategy-controlled fields.
- `module4_agent/code_generator.py` - RandAugment, MixUp, CutMix, TTA, and compact config output.
- `module4_agent/tests/test_code_generator.py` - generated strategy behavior tests.
- `recommender/outcome_memory.py` - richer metadata, config hashes, and filtered history queries.
- `recommender/ranker.py` - pass experiment metadata through the existing write-back path.
- `recommender/__init__.py` - export planner APIs.
- `recommender/report.py` - strategy IDs, delta, and status columns.
- `recommender/test_recommender.py` - backward-compatible memory tests.
- `run_and_log.py` - use the shared parser and the actual Module 4 output path.
- `README.md` - two-stage architecture and quick start.

## Core Data Contracts

Use these fields consistently across storage, tools, and agents:

```python
@dataclass
class SourceRecord:
    id: str
    source_name: str
    source_type: str
    content_path: str
    url: str
    content_sha256: str
    created_at: str


@dataclass
class SourceSummary:
    id: str
    source_id: str
    models: list[str]
    augmentations: list[str]
    losses: list[str]
    optimizers: list[str]
    schedulers: list[str]
    inference: list[str]
    ensemble: list[str]
    created_at: str


@dataclass
class ExperimentProposal:
    experiment_name: str
    strategy_card_ids: list[str]
    config: dict
    changed_fields: list[str]
    config_hash: str
    rationale: str


@dataclass
class ExperimentComparison:
    best_experiment: str | None
    improved: bool
    target_metric: str
    baseline_value: float | None
    best_value: float | None
    metric_delta: float | None
    keep_strategy_card_ids: list[str]
    discard_strategy_card_ids: list[str]
```

`SourceRecord.id` is `source_<slug>_<sha256[:8]>`. `SourceSummary.id` is `summary_<source_id>`. Every API accepting `source_summary_id` uses `SourceSummary.id`, while `SourceSummary.source_id` links back to the raw source.

### Task 1: Package Configuration And Domain Contracts

**Files:**
- Create: `knowledge/__init__.py`
- Create: `knowledge/schemas.py`
- Modify: `pyproject.toml`
- Modify: `requirements.txt`
- Modify: `.env.example`
- Modify: `.gitignore`
- Test: `tests/test_knowledge_store.py`

- [ ] **Step 1: Write failing schema round-trip tests**

```python
from knowledge.schemas import Evidence, StrategyCard


def test_strategy_card_round_trip():
    card = StrategyCard(
        id="strategy_aug_randaugment_001",
        task_type="classification",
        domain="plant_disease",
        strategy_name="RandAugment",
        component="augmentation",
        summary="Strong image-classification augmentation.",
        use_when=["baseline overfits"],
        avoid_when=["model underfits"],
        compatible_with=["efficientnet_b3"],
        target_metrics=["macro_f1"],
        experiment_template={
            "augmentation": "randaugment",
            "randaugment_num_ops": 2,
            "randaugment_magnitude": 9,
        },
        evidence=[Evidence(source_id="cassava_solution_001", note="Used by a top solution.")],
        risk="Too much augmentation can hurt convergence.",
        risk_level="medium",
        priority=0.0,
    )
    assert StrategyCard.from_dict(card.to_dict()) == card
```

- [ ] **Step 2: Run the test and verify the package is missing**

Run: `uv run --no-project --with pytest python -m pytest tests/test_knowledge_store.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'knowledge'`.

- [ ] **Step 3: Add JSON-friendly dataclasses and validation**

Implement `Evidence`, `SourceRecord`, `SourceSummary`, `StrategyCard`, `ExperimentProposal`, and `ExperimentComparison` in `knowledge/schemas.py`. Use `to_dict()` based on `dataclasses.asdict`, explicit `from_dict()` constructors, UTC ISO timestamps, and these validation rules:

```python
VALID_TASK_TYPES = {
    "classification",
    "object_detection",
    "image_segmentation",
    "feature_extraction",
}
VALID_COMPONENTS = {
    "augmentation",
    "loss",
    "optimizer",
    "scheduler",
    "inference",
    "ensemble",
    "backbone",
    "finetune",
}
VALID_RISK_LEVELS = {"low", "medium", "high"}
```

`StrategyCard.validate()` must reject an empty ID/name/summary, unsupported enums, a priority outside `[0, 1]`, or a non-dict `experiment_template`.

- [ ] **Step 4: Add package and runtime configuration**

In `pyproject.toml`:

```toml
requires-python = ">=3.10"
dependencies = [
  "chromadb",
  "datasets",
  "networkx",
  "numpy",
  "openai>=2.0.0",
  "pandas",
  "pillow",
  "sentence-transformers",
  "torch",
  "torchvision",
  "transformers",
  "mcp[cli]>=1.2.0",
]

[project.scripts]
jiaozi-mcp = "mcp_server.server:main"
jiaozi-knowledge-learn = "agents.knowledge_learner_agent:main"
jiaozi-mle-agent = "agents.mle_experiment_agent:main"

[tool.setuptools.packages.find]
include = [
  "agents*",
  "analyzer*",
  "features*",
  "ingestion*",
  "knowledge*",
  "mcp_server*",
  "module4_agent*",
  "processors*",
  "recommender*",
  "retrieval*",
]
```

Add `mcp[cli]>=1.2.0` to `requirements.txt`. Add:

```dotenv
JIAOZI_KB_ROOT=knowledge_base
JIAOZI_WORKSPACE_ROOT=workspace
JIAOZI_MCP_TRANSPORT=stdio
KNOWLEDGE_LLM_PROVIDER=qwen
KNOWLEDGE_QWEN_MODEL=qwen-plus
KNOWLEDGE_OPENAI_MODEL=gpt-4.1-mini
```

Ignore generated files while preserving seed cards and `.gitkeep` files:

```gitignore
knowledge_base/index/*.sqlite3*
knowledge_base/experiments/*.jsonl
knowledge_base/raw_sources/*
!knowledge_base/raw_sources/.gitkeep
workspace/
```

- [ ] **Step 5: Run schema tests**

Run: `uv run --no-project --with pytest python -m pytest tests/test_knowledge_store.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml requirements.txt .env.example .gitignore knowledge tests/test_knowledge_store.py
git commit -m "feat: add knowledge domain contracts"
```

### Task 2: JSON And SQLite FTS Knowledge Store

**Files:**
- Create: `knowledge/store.py`
- Create: `knowledge_base/raw_sources/.gitkeep`
- Create: `knowledge_base/source_summaries/.gitkeep`
- Create: `knowledge_base/experiments/.gitkeep`
- Create: `knowledge_base/index/.gitkeep`
- Modify: `knowledge/__init__.py`
- Test: `tests/test_knowledge_store.py`

- [ ] **Step 1: Add failing persistence, upsert, and search tests**

```python
def test_upsert_merges_evidence_and_rebuilds_search_index(tmp_path, sample_card):
    store = KnowledgeStore(tmp_path)
    store.upsert_strategy_card(sample_card)
    newer = replace(
        sample_card,
        evidence=[Evidence(source_id="paper_002", note="Independent evidence.")],
    )
    merged = store.upsert_strategy_card(newer)

    assert {item.source_id for item in merged.evidence} == {
        "cassava_solution_001",
        "paper_002",
    }
    assert store.search_strategy_cards("randaugment classification", limit=5)[0].id == sample_card.id
```

Also test that two cards with the same normalized `(task_type, component, strategy_name)` merge instead of creating duplicates, and that `rebuild_index()` restores search after deleting the SQLite file.

- [ ] **Step 2: Verify tests fail**

Run: `uv run --no-project --with pytest python -m pytest tests/test_knowledge_store.py -q`

Expected: FAIL because `KnowledgeStore` does not exist.

- [ ] **Step 3: Implement paths, atomic JSON writes, and FTS schema**

`KnowledgeStore.__init__(root)` must create:

```text
raw_sources/
source_summaries/
strategy_cards/
experiments/
index/
```

Use a temporary sibling file plus `Path.replace()` for JSON writes. Initialize:

```sql
CREATE TABLE IF NOT EXISTS strategy_cards (
    id TEXT PRIMARY KEY,
    identity_key TEXT UNIQUE NOT NULL,
    task_type TEXT NOT NULL,
    domain TEXT NOT NULL,
    component TEXT NOT NULL,
    strategy_name TEXT NOT NULL,
    priority REAL NOT NULL,
    card_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS strategy_cards_fts USING fts5(
    id UNINDEXED,
    strategy_name,
    task_type,
    domain,
    component,
    summary,
    use_when,
    avoid_when,
    compatible_with,
    target_metrics
);
```

Expose these exact methods:

- `KnowledgeStore.ingest_source(source_name: str, source_type: str, content: str, url: str = "") -> SourceRecord`
- `KnowledgeStore.get_source(source_id: str) -> SourceRecord`
- `KnowledgeStore.save_source_summary(summary: SourceSummary) -> SourceSummary`
- `KnowledgeStore.get_source_summary(summary_id: str) -> SourceSummary`
- `KnowledgeStore.upsert_strategy_card(card: StrategyCard) -> StrategyCard`
- `KnowledgeStore.get_strategy_card(card_id: str) -> StrategyCard`
- `KnowledgeStore.list_strategy_cards() -> list[StrategyCard]`
- `KnowledgeStore.search_strategy_cards(query: str, limit: int = 20) -> list[StrategyCard]`
- `KnowledgeStore.rebuild_index() -> int`

Sanitize IDs to `[a-z0-9][a-z0-9_-]*`; never use caller input as a filesystem path.

- [ ] **Step 4: Run store tests**

Run: `uv run --no-project --with pytest python -m pytest tests/test_knowledge_store.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add knowledge knowledge_base tests/test_knowledge_store.py
git commit -m "feat: add local strategy knowledge store"
```

### Task 3: Strategy Scoring, Compact Retrieval, And Seed Cards

**Files:**
- Create: `knowledge/scoring.py`
- Create: `knowledge/seeds.py`
- Create: `knowledge_base/strategy_cards/randaugment.json`
- Create: `knowledge_base/strategy_cards/cutmix.json`
- Create: `knowledge_base/strategy_cards/mixup.json`
- Create: `knowledge_base/strategy_cards/label_smoothing.json`
- Create: `knowledge_base/strategy_cards/tta.json`
- Create: `knowledge_base/strategy_cards/efficientnet_b4.json`
- Test: `tests/test_strategy_scoring.py`

- [ ] **Step 1: Write failing scoring tests**

```python
def test_context_and_outcomes_raise_relevant_card_priority():
    score = score_strategy_card(
        card=randaugment_card,
        task="classification",
        current_model="efficientnet_b3",
        target_metric="macro_f1",
        observed_results=[
            {"strategy_card_ids": [randaugment_card.id], "metric_delta": 0.018},
        ],
    )
    assert score.priority > 0.55
    assert "compatible" in score.reason
    assert "positive past result" in score.reason


def test_search_returns_compact_payload_only(store):
    rows = search_compact_cards(
        store,
        task="classification",
        current_model="efficientnet_b3",
        target_metric="macro_f1",
        top_k=5,
        observed_results=[],
    )
    assert set(rows[0]) == {
        "id", "strategy_name", "component", "summary", "use_when",
        "avoid_when", "experiment_template", "priority", "reason",
    }
```

- [ ] **Step 2: Implement the deterministic score**

Use:

```python
priority = clamp(
    0.4 * evidence_score
    + 0.3 * compatibility_score
    + 0.2 * past_success_score
    - 0.1 * risk_score,
    0.0,
    1.0,
)
```

Where:

- `evidence_score = min(unique_source_count / 3, 1.0)`.
- `compatibility_score = 0.45 * task_match + 0.25 * model_match + 0.15 * metric_match + 0.15 * condition_match`.
- `past_success_score = 0.5` with no linked outcomes; otherwise the fraction with `metric_delta > 0`.
- `risk_score` maps `low=0.1`, `medium=0.5`, `high=0.9`.
- `condition_match` is `1.0` when any normalized `use_when` term appears in the query context, otherwise `0.5`.

Sort by `priority` descending, then strategy name. Return only the compact fields asserted above.

- [ ] **Step 3: Add six valid seed cards**

Each seed must contain a unique ID, evidence, risk level, target metrics, and an executable template. Use these templates:

```json
{"augmentation": "randaugment", "randaugment_num_ops": 2, "randaugment_magnitude": 9}
{"cutmix_alpha": 1.0}
{"mixup_alpha": 0.2}
{"label_smoothing": 0.1}
{"tta": true}
{"backbone": "efficientnet_b4", "image_size": 380}
```

`knowledge.seeds.load_seed_cards(store)` reads the tracked JSON files and upserts all six.

- [ ] **Step 4: Run scoring tests**

Run: `uv run --no-project --with pytest python -m pytest tests/test_strategy_scoring.py -q`

Expected: PASS with deterministic ordering.

- [ ] **Step 5: Commit**

```bash
git add knowledge knowledge_base/strategy_cards tests/test_strategy_scoring.py
git commit -m "feat: add scored strategy card retrieval"
```

### Task 4: Enrich Outcome Memory And Generate Controlled Proposals

**Files:**
- Create: `recommender/experiment_planner.py`
- Modify: `recommender/outcome_memory.py`
- Modify: `recommender/ranker.py`
- Modify: `recommender/__init__.py`
- Modify: `recommender/test_recommender.py`
- Test: `tests/test_experiment_planner.py`

- [ ] **Step 1: Add failing backward-compatibility and planner tests**

Test that legacy JSONL rows still load, while new rows contain:

```json
{
  "experiment_id": "round1_randaugment",
  "parent_experiment_id": "baseline",
  "strategy_card_ids": ["strategy_aug_randaugment_001"],
  "config_hash": "8ec6eac4c82f2d7f5f7132b13ca03eb9351fc5b9503f569c92cb4b28f4930f36",
  "metric_delta": 0.0152,
  "notes": "Generated by the MLE experiment agent."
}
```

Planner tests must assert:

- at most `max_experiments`,
- at most `max_changed_variables` relative to baseline,
- exact duplicate configs are rejected by canonical SHA-256 hash,
- previously failed config hashes are rejected,
- augmentation/loss/scheduler proposals precede backbone changes,
- baseline-only keys such as dataset paths and class counts are preserved.

- [ ] **Step 2: Extend `OutcomeMemory` without breaking old callers**

Keep existing positional arguments valid and add keyword-only metadata:

```python
def log(
    self,
    fingerprint: dict,
    config: dict,
    result: dict,
    dataset_id: str | None = None,
    cost: dict | None = None,
    *,
    experiment_id: str | None = None,
    parent_experiment_id: str | None = None,
    strategy_card_ids: list[str] | None = None,
    notes: str | None = None,
) -> dict:
```

Return the record that was written. Compute `config_hash` from recursively sorted JSON after removing runtime-only keys `checkpoint_dir`, `resume_checkpoint`, and `output_dir`. Add `OutcomeMemory.query_experiments(*, task: str | None = None, metric: str | None = None, top_k: int = 10, include_failed: bool = True) -> list[dict]`.

- [ ] **Step 3: Implement proposal generation**

Implement this exact function signature:

```text
MUTABLE_FIELDS = {
    "augmentation",
    "randaugment_num_ops",
    "randaugment_magnitude",
    "mixup_alpha",
    "cutmix_alpha",
    "label_smoothing",
    "loss",
    "optimizer",
    "scheduler",
    "learning_rate",
    "finetune_strategy",
    "freeze_backbone",
    "backbone",
    "pretrained_hf_id",
    "image_size",
    "tta",
}

def generate_experiment_proposals(
    baseline: dict,
    strategy_cards: list[dict],
    past_experiments: list[dict],
    *,
    max_experiments: int = 3,
    max_changed_variables: int = 2,
) -> list[ExperimentProposal]
```

For each card, overlay only `MUTABLE_FIELDS`, count actual changes, reject zero-change/over-limit/duplicate/known-failed configs, and name proposals `exp_<sequence>_<normalized_strategy_name>`.

- [ ] **Step 4: Run memory and planner tests**

Run: `uv run --no-project --with pytest python -m pytest recommender/test_recommender.py tests/test_experiment_planner.py -q`

Expected: PASS, including all existing recommender tests.

- [ ] **Step 5: Commit**

```bash
git add recommender tests/test_experiment_planner.py
git commit -m "feat: plan experiments from strategy memory"
```

### Task 5: Make Generated Module 4 Projects Execute Strategy Templates

**Files:**
- Modify: `module4_agent/schemas.py`
- Modify: `module4_agent/spec_builder.py`
- Modify: `module4_agent/code_generator.py`
- Modify: `module4_agent/tests/test_code_generator.py`
- Modify: `module4_agent/tests/test_spec_builder.py`

- [ ] **Step 1: Add failing generated-code assertions**

Add tests asserting generated `train.py` contains and uses:

```text
transforms.RandAugment
mixup_alpha
cutmix_alpha
label_smoothing
tta
torch.flip
```

Add a spec-builder test that a candidate containing the six seed-template fields preserves them in `TrainingSpec.to_config()`.

- [ ] **Step 2: Extend `TrainingSpec` and parser fields**

Add:

```python
scheduler: str = "cosine"
label_smoothing: float = 0.0
randaugment_num_ops: int = 2
randaugment_magnitude: int = 9
mixup_alpha: float = 0.0
cutmix_alpha: float = 0.0
tta: bool = False
```

Parse the same keys from `model_config` in `build_training_specs()`.

- [ ] **Step 3: Generate RandAugment**

In generated `_build_image_transform`, add a branch for `augmentation == "randaugment"`:

```python
return transforms.Compose([
    transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(
        num_ops=as_int(get_value(config, "randaugment_num_ops", 2), 2),
        magnitude=as_int(get_value(config, "randaugment_magnitude", 9), 9),
    ),
    transforms.ToTensor(),
    normalize,
])
```

- [ ] **Step 4: Generate mutually exclusive MixUp/CutMix**

Add `_mix_classification_batch(x, target, config, num_classes)` returning `(mixed_x, target_a, target_b, lam)`. Prefer CutMix when `cutmix_alpha > 0`; otherwise use MixUp when `mixup_alpha > 0`; otherwise return the original batch with `lam=1.0`. In the classification training path compute:

```python
loss = (
    lam * _loss_for_output(output, target_a, config, class_weights=class_weights)
    + (1.0 - lam) * _loss_for_output(output, target_b, config, class_weights=class_weights)
)
```

Use `torch.distributions.Beta(alpha, alpha).sample()` and clamp CutMix boxes to image bounds.

- [ ] **Step 5: Generate two-view classification TTA**

In generated classification validation/evaluation, when `tta` is true:

```python
logits = model(x)
flipped_logits = model(torch.flip(x, dims=[3]))
logits = (logits + flipped_logits) / 2.0
```

Do not enable TTA for detection, segmentation, or feature extraction.

- [ ] **Step 6: Run Module 4 unit tests**

Run: `uv run --no-project --with pytest python -m pytest module4_agent/tests/test_spec_builder.py module4_agent/tests/test_code_generator.py -q`

Expected: PASS. If the environment lacks PyTorch, run the static generation tests selected with `-k "not smoke"` and run the full suite in the project environment before merge.

- [ ] **Step 7: Commit**

```bash
git add module4_agent
git commit -m "feat: execute knowledge-guided training strategies"
```

### Task 6: Safe Experiment Execution, Metrics, Comparison, And Write-Back

**Files:**
- Create: `module4_agent/result_parser.py`
- Create: `mcp_server/__init__.py`
- Create: `mcp_server/context.py`
- Create: `mcp_server/tools/__init__.py`
- Create: `mcp_server/tools/experiment_tools.py`
- Modify: `run_and_log.py`
- Test: `tests/test_experiment_tools.py`

- [ ] **Step 1: Add failing command-safety tests**

Test:

```python
def test_runner_uses_fixed_python_command(tmp_path, fake_generated_project):
    runner = ExperimentRunner(workspace_root=tmp_path)
    result = runner.run(
        experiment_name="exp_001_randaugment",
        project_dir=fake_generated_project,
        config={"task_type": "classification", "augmentation": "randaugment"},
        epochs=2,
        timeout=30,
    )
    assert result.command[:3] == [sys.executable, "-u", "run.py"]
    assert "--config" in result.command
    assert result.shell is False


def test_runner_rejects_paths_outside_workspace(tmp_path):
    runner = ExperimentRunner(workspace_root=tmp_path / "allowed")
    with pytest.raises(ValueError, match="workspace root"):
        runner.run("exp_1", tmp_path / "outside", {}, epochs=1)
```

Also test invalid experiment names, missing `run.py`, epochs outside `1..500`, timeout outside `1..86400`, result JSON parsing, minimize-metric comparison, and write-back metadata.

- [ ] **Step 2: Move shared result parsing**

Move `extract_last_json()` from `run_and_log.py` into `module4_agent/result_parser.py`; import it from both call sites. Fix `run_and_log.py` to use:

```python
project = Path(result["module4"]["output_dir"]).resolve()
```

instead of assuming an extra `module4_code/` directory.

- [ ] **Step 3: Implement the allowlisted runner**

`ExperimentRunner.run()` must:

1. Resolve both workspace root and project path.
2. reject a project not contained by the workspace root,
3. require `run.py`,
4. write `experiments/<experiment_name>.json` atomically,
5. execute with `subprocess.run(command, cwd=project, shell=False, capture_output=True, text=True, timeout=timeout)`,
6. persist `experiments/<experiment_name>.run.json` with command, return code, stdout/stderr tails, timing, and parsed summary,
7. return a JSON-friendly run record containing `"shell": false`.

- [ ] **Step 4: Implement experiment service functions**

Expose these exact service signatures:

- `get_past_experiments(task: str, metric: str, top_k: int = 10) -> dict`
- `generate_experiment_configs(baseline: dict, strategy_cards: list[dict], past_experiments: list[dict], max_experiments: int = 3, max_changed_variables: int = 2) -> dict`
- `run_experiment(experiment_name: str, project_dir: str, config: dict, epochs: int, timeout: int = 3600) -> dict`
- `read_metrics(run_record_path: str) -> dict`
- `compare_results(baseline: dict, experiments: list[dict], target_metric: str) -> dict`
- `write_experiment_result(experiment_name: str, config: dict, metrics: dict, fingerprint: dict, dataset_id: str, strategy_card_ids: list[str], baseline_metric: float | None = None, notes: str = "", cost: dict | None = None) -> dict`

`compare_results` treats `log_loss`, `multiclass_log_loss`, and `rmse` as lower-is-better. It returns `ExperimentComparison`, including the best experiment, target-metric delta, and strategy-card IDs to keep or discard. The next round performs a fresh strategy search instead of carrying raw source context forward.

- [ ] **Step 5: Run experiment-tool tests**

Run: `uv run --no-project --with pytest python -m pytest tests/test_experiment_tools.py recommender/test_recommender.py -q`

Expected: PASS without launching real training; use a fixture `run.py` that prints a deterministic JSON summary.

- [ ] **Step 6: Commit**

```bash
git add module4_agent/result_parser.py mcp_server run_and_log.py tests/test_experiment_tools.py
git commit -m "feat: add safe experiment execution services"
```

### Task 7: Knowledge Learner Services And Agent

**Files:**
- Create: `agents/prompts.py`
- Create: `agents/knowledge_learner_agent.py`
- Create: `mcp_server/tools/knowledge_tools.py`
- Create: `tests/fixtures/cassava_solution_note.md`
- Create: `tests/test_knowledge_learner_agent.py`
- Modify: `module4_agent/llm_codegen.py`

- [ ] **Step 1: Add failing mocked-LLM pipeline test**

```python
def test_learning_pipeline_ingests_summarizes_extracts_and_upserts(tmp_path, monkeypatch):
    responses = iter([
        json.dumps({
            "models": ["EfficientNet-B4"],
            "augmentations": ["RandAugment"],
            "losses": ["Label Smoothing"],
            "optimizers": ["AdamW"],
            "schedulers": ["CosineAnnealingLR"],
            "inference": ["TTA"],
            "ensemble": [],
        }),
        json.dumps({"cards": [{
            "id": "strategy_aug_randaugment_001",
            "task_type": "classification",
            "domain": "plant_disease",
            "strategy_name": "RandAugment",
            "component": "augmentation",
            "summary": "Strong image classification augmentation.",
            "use_when": ["baseline overfits"],
            "avoid_when": ["model underfits"],
            "compatible_with": ["efficientnet_b4"],
            "target_metrics": ["macro_f1"],
            "experiment_template": {
                "augmentation": "randaugment",
                "randaugment_num_ops": 2,
                "randaugment_magnitude": 9
            },
            "risk": "Tune strength.",
            "risk_level": "medium"
        }]}),
    ])
    monkeypatch.setattr("agents.knowledge_learner_agent.call_llm", lambda *args, **kwargs: next(responses))

    result = KnowledgeLearnerAgent(KnowledgeStore(tmp_path)).learn(
        source_name="cassava_solution",
        source_type="note",
        content=FIXTURE_TEXT,
        url="https://example.invalid/cassava",
    )
    assert result["cards_upserted"] == ["strategy_aug_randaugment_001"]
```

- [ ] **Step 2: Promote the existing LLM call helper to a public API**

Rename `module4_agent.llm_codegen._call_llm` to `call_llm`, keep `_call_llm = call_llm` as a compatibility alias, and add optional model environment selection for `KNOWLEDGE_LLM_PROVIDER`, `KNOWLEDGE_QWEN_MODEL`, and `KNOWLEDGE_OPENAI_MODEL`.

- [ ] **Step 3: Add strict JSON prompts and parsing**

`SOURCE_SUMMARY_SYSTEM_PROMPT` requires exactly:

```json
{
  "models": [],
  "augmentations": [],
  "losses": [],
  "optimizers": [],
  "schedulers": [],
  "inference": [],
  "ensemble": []
}
```

`STRATEGY_CARD_SYSTEM_PROMPT` requires a top-level `cards` array with every `StrategyCard` field except computed `priority` and `observed_results`. Strip optional Markdown fences, reject invalid JSON, validate every card, and attach source evidence before upsert.

- [ ] **Step 4: Implement knowledge service functions and orchestration**

Expose these exact service signatures:

- `ingest_external_source(source_name: str, source_type: str, content: str, url: str = "") -> dict`
- `summarize_source(source_id: str) -> dict`
- `extract_strategy_cards(source_summary_id: str) -> dict`
- `upsert_strategy_card(strategy_card: dict) -> dict`
- `search_strategy_cards(task: str, current_model: str, target_metric: str, top_k: int = 5) -> dict`

`KnowledgeLearnerAgent.learn()` calls the first four in order and returns source ID, summary ID, and upserted card IDs. Add a CLI accepting `--file`, `--source-name`, `--source-type`, and optional `--url`.

- [ ] **Step 5: Run learner tests**

Run: `uv run --no-project --with pytest python -m pytest tests/test_knowledge_learner_agent.py tests/test_knowledge_store.py tests/test_strategy_scoring.py -q`

Expected: PASS with no network call.

- [ ] **Step 6: Commit**

```bash
git add agents mcp_server/tools/knowledge_tools.py module4_agent/llm_codegen.py tests
git commit -m "feat: add offline knowledge learner agent"
```

### Task 8: FastMCP Server Registration And Protocol Tests

**Files:**
- Create: `mcp_server/server.py`
- Create: `mcp_server/tools/report_tools.py`
- Create: `tests/test_mcp_server.py`

- [ ] **Step 1: Write failing tool-registration test**

Test that the server exposes exactly:

```python
EXPECTED_TOOLS = {
    "ingest_external_source",
    "summarize_source",
    "extract_strategy_cards",
    "upsert_strategy_card",
    "search_strategy_cards",
    "get_past_experiments",
    "generate_experiment_configs",
    "run_experiment",
    "read_metrics",
    "compare_results",
    "write_experiment_result",
    "generate_experiment_report",
}
```

Start the server through the official stdio client and assert the negotiated tool list:

```python
def test_stdio_server_lists_expected_tools(tmp_path):
    async def list_tool_names():
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
            env={
                **os.environ,
                "JIAOZI_KB_ROOT": str(tmp_path / "kb"),
                "JIAOZI_WORKSPACE_ROOT": str(tmp_path / "workspace"),
                "JIAOZI_MCP_TRANSPORT": "stdio",
            },
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
        return {tool.name for tool in tools.tools}

    assert asyncio.run(list_tool_names()) == EXPECTED_TOOLS
```

- [ ] **Step 2: Register thin FastMCP wrappers**

Use:

```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Jiaozi Knowledge-Guided MLE", json_response=True)


@mcp.tool()
def search_strategy_cards(
    task: str,
    current_model: str,
    target_metric: str,
    top_k: int = 5,
) -> dict:
    """Return compact local strategy cards for one experiment-planning context."""
    return knowledge_tools.search_strategy_cards(
        task=task,
        current_model=current_model,
        target_metric=target_metric,
        top_k=top_k,
    )
```

Register all 12 tools with typed arguments and docstrings. `main()` reads `JIAOZI_MCP_TRANSPORT`, accepts only `stdio` or `streamable-http`, and calls `mcp.run(transport=transport)`.

- [ ] **Step 3: Ensure stdio-safe logging**

Configure `logging.basicConfig(stream=sys.stderr, level=logging.INFO)`. Remove or redirect `print()` from MCP call paths. Tool errors should raise `ValueError` with concise messages rather than printing.

- [ ] **Step 4: Run direct tests and Inspector smoke**

Run:

```bash
uv run --with "mcp[cli]>=1.2.0" --with pytest python -m pytest tests/test_mcp_server.py -q
```

Expected: PASS.

Then run:

```bash
npx -y @modelcontextprotocol/inspector uv run jiaozi-mcp
```

Expected: Inspector lists all 12 tools; call `search_strategy_cards` and receive at most `top_k` compact cards.

- [ ] **Step 5: Commit**

```bash
git add mcp_server tests/test_mcp_server.py
git commit -m "feat: expose knowledge and experiment MCP tools"
```

### Task 9: Low-Token MLE Experiment Agent

**Files:**
- Create: `agents/mle_experiment_agent.py`
- Create: `tests/test_mle_experiment_agent.py`
- Modify: `mcp_server/context.py`

- [ ] **Step 1: Add failing plan-only loop test**

```python
def test_agent_plan_only_never_runs_training(fake_services):
    agent = MLEExperimentAgent(services=fake_services)
    result = agent.run(
        task="cassava_leaf_disease",
        baseline=BASELINE,
        fingerprint=FINGERPRINT,
        dataset_id="cassava",
        project_dir="/workspace/cassava",
        target_metric="macro_f1",
        rounds=1,
        execute=False,
    )
    assert len(result["rounds"][0]["proposals"]) <= 3
    assert fake_services.run_calls == []
```

Add an execute-mode test with mocked run records proving the best result becomes the next baseline and every completed experiment is written back.

- [ ] **Step 2: Implement one bounded round**

Require the baseline JSON shape:

```json
{
  "experiment_name": "baseline_b3",
  "config": {
    "backbone": "efficientnet_b3",
    "augmentation": "basic",
    "loss": "cross_entropy_loss",
    "optimizer": "adamw",
    "scheduler": "cosine"
  },
  "metrics": {
    "accuracy": 0.8621,
    "macro_f1": 0.7558,
    "val_loss": 0.48
  }
}
```

Each round must:

1. call `search_strategy_cards(top_k=5)`,
2. call `get_past_experiments(top_k=10)`,
3. call `generate_experiment_configs(baseline=baseline["config"], max_experiments=3, max_changed_variables=2)`,
4. return immediately in plan-only mode,
5. otherwise run each proposal, read metrics, and write the result,
6. compare results against the current baseline,
7. promote the best successful experiment only when it improves the target metric.

Stop early on no proposals, no successful runs, or no improvement. Hard-limit `rounds` to `1..10`.

- [ ] **Step 3: Add compact context accounting**

Return:

```json
{
  "context_stats": {
    "strategy_cards": 5,
    "past_experiments": 10,
    "serialized_context_chars": 0
  }
}
```

Compute `serialized_context_chars` from only baseline, compact cards, and compact history. Do not include raw source text or source summaries in the MLE loop.

- [ ] **Step 4: Add CLI**

Support:

```text
python -m agents.mle_experiment_agent
  --baseline baseline.json
  --fingerprint fingerprint.json
  --project-dir workspace/cassava/module4_code
  --dataset-id cassava
  --task cassava_leaf_disease
  --target-metric macro_f1
  --rounds 3
  [--execute]
```

The CLI prints JSON to stdout only after completion; operational logs go to stderr.

- [ ] **Step 5: Run agent tests**

Run: `uv run --no-project --with pytest python -m pytest tests/test_mle_experiment_agent.py -q`

Expected: PASS with no real training.

- [ ] **Step 6: Commit**

```bash
git add agents/mle_experiment_agent.py tests/test_mle_experiment_agent.py mcp_server/context.py
git commit -m "feat: add bounded low-token MLE experiment agent"
```

### Task 10: Reports, Documentation, And Demo Flow

**Files:**
- Modify: `recommender/report.py`
- Modify: `README.md`
- Create: `docs/MCP_KNOWLEDGE_AGENT.md`
- Modify: `docs/recommender.md`
- Modify: `mcp_server/tools/report_tools.py`
- Test: `recommender/test_recommender.py`

- [ ] **Step 1: Add failing report-column test**

Assert report rows include:

```python
{
    "experiment_id",
    "parent_experiment_id",
    "strategy_card_ids",
    "metric",
    "metric_delta",
    "tokens",
    "epochs",
    "wall_clock_sec",
}
```

- [ ] **Step 2: Extend the report service**

`generate_experiment_report(memory_path=None, output_path=None)` must return Markdown containing:

```text
| Experiment | Parent | Strategies | Metric | Delta | Tokens | Epochs | Wall clock |
```

Keep the existing console table and optional scatter plot working.

- [ ] **Step 3: Document the exact demo**

Document:

```bash
# 1. Install
uv sync

# 2. Learn one local source
uv run jiaozi-knowledge-learn \
  --file tests/fixtures/cassava_solution_note.md \
  --source-name cassava_solution_001 \
  --source-type note

# 3. Inspect MCP tools
npx -y @modelcontextprotocol/inspector uv run jiaozi-mcp

# 4. Produce a low-token plan without training
uv run jiaozi-mle-agent \
  --baseline workspace/cassava/baseline.json \
  --fingerprint workspace/cassava/fingerprint.json \
  --project-dir workspace/cassava/module4_code \
  --dataset-id cassava \
  --task cassava_leaf_disease \
  --target-metric macro_f1 \
  --rounds 1

# 5. Explicitly execute
uv run jiaozi-mle-agent \
  --baseline workspace/cassava/baseline.json \
  --fingerprint workspace/cassava/fingerprint.json \
  --project-dir workspace/cassava/module4_code \
  --dataset-id cassava \
  --task cassava_leaf_disease \
  --target-metric macro_f1 \
  --rounds 3 \
  --execute
```

State accurately: "a simplified MLE-STAR-inspired workflow with MCP and a compressed local knowledge base", not a full MLE-STAR reproduction.

- [ ] **Step 4: Run report tests**

Run: `uv run --no-project --with pytest python -m pytest recommender/test_recommender.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add recommender README.md docs mcp_server/tools/report_tools.py
git commit -m "docs: add MCP knowledge agent workflow"
```

### Task 11: End-To-End Verification

**Files:**
- Modify only files required by failures found in this task.

- [ ] **Step 1: Run fast unit tests**

```bash
uv run --with pytest python -m pytest \
  tests \
  recommender/test_recommender.py \
  module4_agent/tests/test_spec_builder.py \
  module4_agent/tests/test_code_generator.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run the existing project suites**

```bash
uv run --with pytest python -m pytest module4_agent/tests retrieval/test_rag_retrieval.py retrieval/test_golden.py -q
uv run python -m unittest test_pipeline.py -v
```

Expected: PASS. Record pre-existing environment-only failures separately; do not weaken tests.

- [ ] **Step 3: Run a temporary knowledge-learning smoke**

Use a temporary `JIAOZI_KB_ROOT`, mock/fake provider output, ingest `tests/fixtures/cassava_solution_note.md`, rebuild the index, and verify `search_strategy_cards(task="classification", current_model="efficientnet_b3", target_metric="macro_f1", top_k=5)` never returns raw source content.

- [ ] **Step 4: Run a fake generated-project experiment smoke**

Create a temporary project containing a `run.py` that prints a deterministic summary. Verify plan -> run -> read -> compare -> write-back creates one outcome row with card IDs, config hash, metric delta, and cost.

- [ ] **Step 5: Run static and repository checks**

```bash
python3 -m compileall agents knowledge mcp_server recommender module4_agent
git diff --check
git status --short
```

Expected: compilation succeeds, no whitespace errors, and only intended files are modified.

- [ ] **Step 6: Commit final verification fixes**

```bash
git add -A
git commit -m "test: verify MCP knowledge experiment loop"
```

## Acceptance Criteria

- The Knowledge Learner can turn one local note into validated, merged strategy cards.
- The MLE Experiment Agent never reads raw source text.
- Strategy retrieval returns at most `top_k` compact cards with deterministic reasons and priorities.
- Proposal generation creates at most three experiments and changes at most two variables each.
- Duplicate and previously failed configs are not proposed.
- Real execution cannot run arbitrary commands or escape the configured workspace root.
- RandAugment, MixUp, CutMix, label smoothing, cosine scheduling, TTA, and backbone templates are executable by generated Module 4 projects.
- Every completed experiment writes config, metrics, strategy IDs, cost, and delta into `OutcomeMemory`.
- The next round promotes only a genuinely improved result.
- MCP Inspector can list and call all 12 tools.
- Existing recommender and Module 4 behavior remains backward compatible.

## Implementation References

- Official MCP Python SDK: https://github.com/modelcontextprotocol/python-sdk
- Official MCP server guide: https://modelcontextprotocol.io/docs/develop/build-server
- Official MCP Inspector guide: https://modelcontextprotocol.io/docs/tools/inspector

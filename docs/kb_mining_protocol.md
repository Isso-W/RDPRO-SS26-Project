# kb_mining: Kaggle winning solution mining and KB enhanced protocol

> This document documents reproducible collection, derived statistics, and manual review protocols. The current implementation is located at
> `kb_mining/`; the running command and publishing boundaries are based on the root `README.md`.
>
> **One-time project**: No cron, no increment, no notebook AST analysis, no new backbone
> Node, no set-aside verification and short-term training A/B (the verification part has been clearly postponed), **Do not change the scoring constant**
> (The `w_vector` / bonus value will only be submitted separately after reviewing the consensus form).

---

## 0. Overview of goals and outputs

Mine top-solution write-ups from Kaggle computer-vision competitions that ended in or after 2021. Aggregate evidence for "dataset traits → component choice", produce a concise consensus table, and classify suggested KB changes with the five-tier decision process. The pipeline does not edit `EDGES`, `EDGE_CONDITIONS`, or node fields automatically; a reviewer applies approved suggestions separately.

The only KB code change implemented with this project is Phase B: let `_select_components` consume the `preferred_when` edges (now dead data) between loss nodes, and replace the existing hard coding with KB data loss The first priority of the rule chain.

```
kb_mining/
  __init__.py
  catalog.py        # Competition list + feature card, architecture release schedule, component alias table (pure data)
  harvest.py        # Meta Kaggle dump → data/posts.jsonl
  extract.py        # LLM extraction → data/facts.jsonl
  aggregate.py      # facts × feature card → data/consensus.{json,md} + three side tables
  decide.py         # consensus × existing KB → data/proposals.md (five-tier decision)
  tests/
    fixtures/       # Mini CSV, canned LLM response, sample facts
    test_harvest.py
    test_extract.py
    test_aggregate.py
    test_decide.py
kb_mining/data/     # All products are here (gitignore original dump, product jsonl in storage)
```

Each stage of the pipeline is idempotent and can be rerun independently; communication between stages is only through files under `kb_mining/data/`.

---

## 0.5 Pre-step 0 - Data source foundation verification (half a day, prior to catalog)

**Single point of risk for the entire link**: Competitions → Forum → Whether the JOIN of the post is real and feasible. The forum-related fields of Meta Kaggle have changed historically, and `Competitions.csv` does not guarantee a stable `ForumId` column. **Before writing any other code** verify/false:

1. Download `Competitions.csv` (small file, full size available) + `ForumTopics.csv` header and sampling lines
   + `ForumMessages.csv` Only the header and the first few lines (`head`, not fully downloaded);
2. Confirm that the JOIN key actually exists: Slug in Competitions is associated with the forum column (`ForumId` or equivalent fields; if ForumTopics has a column directly connected to the competition, the link will be shorter);
3. **End-to-end verification of a known record**: Use cassava to compete with the known existing "1st place solution" post, and actually retrieve the text along the link, and confirm with the naked eye that it is the full text of write-up rather than the abstract/truncation;
4. Output `kb_mining/data/source_check.md`: Record the actual column names, sample JOIN results, ForumMessages actual format of the text (markdown/HTML). **harvest.py is encoded according to this document, not according to the "expected link" of this plan. **

**Alternatives for broken links** (according to priority): ① ForumTopics If there are other competition-related columns, change the link; ② Use the solution post index maintained by the community (such as farid.one/kaggle-solutions, aggregate the solution posts URL by competition) to get the post list, and then get the text one by one; ③ If all else fails, the data source plan for this project needs to be renegotiated - **Stop and report to the decision-maker at this time, do not write a hard-coded crawler**.

---

## 1. catalog.py: pure data, no logic

### 1.1 Competition List + Feature Card `COMPETITIONS`

```python
COMPETITIONS: dict[str, dict] = {
    "<slug>": {
        "slug": str, # kaggle Competition slug
        "title": str,
        "start": "YYYY-MM", # Start time (for coexistence judgment)
        "end": "YYYY-MM",
        "task_type": "classification",
        "traits": {             # Feature card: key = legal condition key removed "=True"
            "fine_grained": bool,
            "class_imbalance": bool,
            "medical": bool,
            "data_size": "small" | "medium" | "large",
        },
        "traits_verified": bool, # Set True for manual verification; aggregate for unverified ones warning
        "notes": str, # Special remarks (multiple labels, metric-learning flavor, etc.)
    },
}
```

Initial candidate pool (verify one by one during implementation: ① It will end after 2021-01 ② The discussion area has ≥ 5 articles solution write-up with ranking ③ Characteristic card value; unsatisfied ones are deleted from the list, each key feature `fine_grained` / `class_imbalance` / `medical` Keep at least 3 contests):

| slug | years | initial judgment characteristics |
|---|---|---|
| cassava-leaf-disease-classification | 2021 | fine_grained, mild imbalance, medium |
| plant-pathology-2021-fgvc8 | 2021 | fine_grained (note: multiple tags, notes tag) |
| herbarium-2022-fgvc9 | 2022 | fine_grained, long tail imbalance, large |
| sorghum-id-fgvc-9 | 2022 | fine_grained |
| paddy-disease-classification | 2022 | fine_grained, small |
| happy-whale-and-dolphin | 2022 | fine_grained (metric-learning heavy flavor, notes mark) |
| mayo-clinic-strip-ai | 2022 | medical, small |
| rsna-breast-cancer-detection | 2023 | medical, extreme imbalance |
| ubc-ocean | 2023 | medical, imbalance |
| isic-2024-challenge | 2024 | medical, extreme imbalance, large |
| hms-harmful-brain-activity-classification | 2024 | medical, imbalance (input is an image rendered by electroencephalogram spectrogram, notes marks an unnatural image) |
| rsna-2024-lumbar-spine-degenerative-classification | 2024 | medical, imbalance (MRI multi-part grading, notes marked multi-output) |
| fathomnet-2025 | 2025 | fine_grained (marine species level classification; the community is small, verify whether the number of write-up is ≥5) |
| rsna-intracranial-aneurysm-detection | 2025 | medical, imbalance, large (CT/MR, 1100+ team, write-up sufficient; notes tag contains positioning subtask) |

**Mechanical enumeration for 2026 and future competitions** (to make up for the timeliness blind spot of manual lists): `catalog.py` comes with an auxiliary function `list_recent_cv_candidates(dump_dir) -> list[dict]`, scan Meta Kaggle `Competitions.csv`, filter ①The deadline is after 2025-01 ②The tag/title contains CV classification signal (tag "Computer Vision", title containing classification/detection, etc.) ③ Number of participating teams ≥ 300 (guaranteed there are enough write-up), output candidate rows (slug, title, start and end time, number of teams) for manual selection and then add `COMPETITIONS`. This function only does enumeration and does not automatically enter the list - the feature card is still checked manually. harvest's CLI plus `--list-recent` switch calls it.

Feature card filling method: Competition Overview/Data page description + write-up cross, LLM can be used to assist in initial filling, but `traits_verified=True` must be manually reviewed (a total of more than ten lines, the cost is negligible).

### 1.2 Architecture release schedule `FAMILY_RELEASE` (for coexistence filtering)

Only 14 backbone families covering KB (id are exactly the same as `retrieval/rag_retrieval.py` `COMPONENTS`):

```python
FAMILY_RELEASE: dict[str, str] = {   # family_id -> "YYYY-MM"
    "resnet": "2015-12", "efficientnet": "2019-05", "mobilenet_v3": "2019-05",
    "vit": "2020-10", "swin_transformer": "2021-03", "convnext": "2022-01",
    "yolov8": "2023-01", "detr": "2020-05", "rt_detr": "2023-04",
    "segformer": "2021-05", "mask2former": "2021-12", "unet": "2015-05",
    "dinov2": "2023-04", "clip_vit": "2021-01",
}
```

(When implementing, check the paper/publication time one by one, with a monthly granularity.)

### 1.3 Component alias table `MODEL_ALIASES` / `LOSS_ALIASES`

Raw string → KB Mapping of id, sequentially matched list of `(regex, family_id)`, all case-insensitive; none hits → `"unknown"`:

```python
MODEL_ALIASES: list[tuple[str, str]] = [
    (r"(tf_)?efficientnet(v2)?", "efficientnet"),
    (r"convnext", "convnext"),
    (r"swin", "swin_transformer"),
    (r"(deit|beit|^vit|_vit|vit_)", "vit"),
    (r"dinov2", "dinov2"),
    (r"clip", "clip_vit"),
    (r"(resnet|resnext|resnest|se_?resnext)", "resnet"),
    (r"mobilenet", "mobilenet_v3"),
    # yolov8/detr/rt_detr/segformer/mask2former/unet Classification competition appears and is mapped.
    # However, if task_type does not match, it will be automatically blocked by the 0 check in the decide stage.
]
LOSS_ALIASES: list[tuple[str, str]] = [
    (r"focal", "focal_loss"),
    (r"(weighted|class.?weight).*(ce|cross.?entropy)", "focal_loss"), # Weighted CE is incorporated into focal evidence, notes retains raw
    (r"cross.?entropy|\bce\b|label.?smooth", "cross_entropy_loss"),
    (r"bce.?dice", "bce_dice_loss"),
    (r"\bdice\b", "dice_loss"),
    (r"infonce|(?<!arc)contrastive", "infonce_loss"),
    (r"arcface|cosface|triplet|metric.?learning", "unknown"), # metric-learning losses are not consolidated and entered into the side table
    (r"hungarian|matching", "hungarian_matching_loss"),
]
```

**loss merge discipline** (loss consensus is exactly the consumer object of Phase B, and the pollution cost is the highest):

1. **arcface/cosface/triplet does not merge** into infonce - they are the stuff of metric-learning, and merging would inflate support from infonce. Mapping as `unknown` parallel to `unknown_components.json` side table (with `metric_learning` label);
2. **Weighted CE→focal retains the merge** (KB has no weighted CE node, and the KB action corresponding to the two is the same edge), but the row in consensus.md must** be split to display raw Count**; if weighted CE accounts for > 50% of the combined votes, add ⚠ to the row (the body of evidence is not literally focal);
3. **notes Contests marked as metric-learning flavor (such as happy-whale) exclude loss votes in their entirety** - their loss signals are not suitable for category recommendations; backbone votes are not affected.

---

## 2. harvest.py: Meta Kaggle dump → posts.jsonl

**Data source**: Kaggle official data set `kaggle/meta-kaggle` (full site CSV dump updated daily), only three files: `Competitions.csv`, `ForumTopics.csv`, `ForumMessages.csv`. Use Kaggle API to download separately by file (`api.dataset_download_file("kaggle/meta-kaggle", <name>, path=...)`), and reuse `ingestion/kaggle_loader._authenticate()`.

**Note**: `ForumMessages.csv` number GB level, **must** use `pandas.read_csv(chunksize=100_000)` flow filtering, and cannot be loaded into the entire table. The listing of CSV is based on the actual dump header (check `head -1` in the first step). The expected link is:

```
Competitions: Slug → ForumId (filtered by slug of catalog.COMPETITIONS)
ForumTopics: ForumId → Topic Id/Title (the title passed the regular filter solution post)
ForumMessages: ForumTopicId → Message (take the first message of each poster, which is the text of write-up)
```

**solution Post Judgment Regularity** (for topic title):

```python
RANK_RE = re.compile(r"\b(\d{1,3})(st|nd|rd|th)\s+place\b|\bplace\s+(\d{1,3})\b", re.I)
SOLUTION_RE = re.compile(r"solution|write.?up|summary", re.I)
# Those who hit RANK_RE will be collected; those who only hit SOLUTION_RE without ranking will also be collected, rank will be recorded as None
```

The top-ranked entries in each competition will be awarded at most `MAX_POSTS_PER_COMP = 10` (rank and None are ranked last).

**Two-level remediation and clear policy for insufficient recall** (solution post title is not guaranteed to contain place/solution - "My approach", "Gold - timm ensemble", which will be missed by regular rules):

- **Level 2 Recall**: When a competition regular hit < 5 articles, take the top 30 topic with the highest voting points (Score column) in the competition forum, and use LLM to determine whether the "ranking plan post" (enter the title + the first 500 characters of the text; use the same injection method as extract `llm_fn`, testable);
- **Policy (hard-written, no discretion)**: After the second level recall, ≥5 articles → normal; 3-4 articles → retain the competition, harvest output warning; **<3 articles → remove the competition from this mining**, and list it in the harvest summary (the feature card is reserved for future supplements).

**Output `data/posts.jsonl`**, each line:

```json
{"competition": "<slug>", "topic_id": 123, "topic_title": "1st Place Solution",
 "rank": 1, "author_message_id": 456, "text": "<raw markdown/html Text>",
 "post_date": "2021-02-20"}
```

CLI: `python -m kb_mining.harvest [--dump-dir kb_mining/data/meta_kaggle] [--force-download]` (If the dump directory already exists, the download will be skipped, the same as the cache habit of `kaggle_loader`).

---

## 3. extract.py: LLM extraction → facts.jsonl

**LLM client**: reuse `features_extraction_api._provider()` + `_client_for_provider(provider)` (OpenAI compatible interface, default qwen). `temperature=0`. **Testability Requirements**: The core function signature is

```python
def extract_post(post: dict, llm_fn: Callable[[str, str], str]) -> dict | None
# llm_fn(system_prompt, user_content) -> raw completion text;
# The production takes the real client, and the test injects the canned response.
```

**Input truncation**: When the text exceeds 12_000 characters, the first 9_000 + the last 3_000 are retained (the model configuration of write-up is mostly at the beginning, and the score table is often at the end).

**LLM outputs schema** (prompt requires pure JSON, without markdown fence; `raw` field is copied from the original text, the mapping is done by the code using an alias table, ** does not allow LLM to be output directly KB id**, reduce the hallucination surface):

```json
{
  "kind": "single" | "ensemble" | "unclear",
  "members": [{"raw_model": "tf_efficientnet_b4_ns", "image_size": 512}],
  "loss_raw": "focal loss" | null,
  "best_single_model_raw": "..." | null,
  "best_single_score": 0.899 | null,
  "used_pseudo_labeling": bool,
  "used_tta": bool,
  "citations": ["our best single model was a B4 at 512px"]
}
```

**Verification Rules** (However, the entire article will be discarded and recorded as `data/extract_rejects.jsonl` with reasons):

1. JSON can be parsed, `kind` is legal, `members` is not empty;
2. **Citation verification**: At least one of `citations` can be found in the text (just match the substring after blank normalization); the main gate to prevent LLM hallucination;
3. `members` If the number > 12, it will be considered as flying and discarded.

**Code side post-processing** (completed within `extract_post`):

- `raw_model` passes `MODEL_ALIASES` → `family`; **Family deduplication** in the same article (B4+B5 is only counted once efficientnet), `image_size` takes the mode of the family members;
- `loss_raw` passes `LOSS_ALIASES` → `loss_kb`;
- `best_single_model_raw` is also mapped → `best_single_family`.

**Output `data/facts.jsonl`**, each line = posts line + the above parsing result (the original raw is all retained). CLI: `python -m kb_mining.extract [--limit N]` (`--limit` for trial running).

---

## 4. aggregate.py: Consensus calculation (pure function, no IO dependency on LLM)

### 4.1 Voting rules (finalized, no changes)

For each fact, each family:

| situation | vote weight |
|---|---|
| `kind == "single"` | 1.0 |
| `kind == "ensemble"` and family == `best_single_family` | 1.0 |
| `kind == "ensemble"` remaining members | 0.5 |
| `kind == "unclear"` | 0.5 |

Losses use the same weights. Because loss is recorded once per write-up, use the highest applicable write-up weight. Multiply the weight by 0.8 when `used_pseudo_labeling=True` to account for uncertain attribution.

### 4.2 Eligibility Filtering (Coexistence + 2021)

A fact contributes to support for `(trait T, family A)` only when both conditions hold:

1. The competition has `end >= "2021-01"`. The catalog enforces this and the aggregation code asserts it again.
2. **Coexistence rule**: `FAMILY_RELEASE[A] < competition start`. A fact that fails this rule contributes neither a family vote nor a denominator entry.

### 4.3 Consensus line

For each `(T, A)`, T is one of the feature card's true Boolean traits: `fine_grained`, `class_imbalance`, or `medical`. Do not mine `data_size` as an independent trait because competition size is heavily confounded and has a weaker relationship with the winning backbone. Use `data_size` only in these two places:

1. **Prototype-query parameters** (§5.1): Set `data_size` to the modal size among competitions that provide evidence for the trait.
2. **Tier 1 field-fix evidence** (§5.2): For each family A, compare the data-size distribution of competitions that voted for A with the A node's `data_size` list. Emit a tier 1 suggestion when they conflict. This check aggregates by family without trait weighting.

```python
support = Σ votes(A) / Σ votes(all families)   # Only within qualifying contests with T
breadth = number of distinct competitions that voted for A
```

**Threshold constant** (top level of module, allowed to be overridden by CLI):

```python
SUPPORT_MIN = 0.50
BREADTH_MIN = 2
```

### 4.4 Output

- `data/consensus.json`: Every consensus row, including rows below the threshold, with fields `{trait, component_type, kb_id, support, breadth, votes, total_votes, n_competitions, passed: bool, evidence: [{competition, rank, raw, citation}]}`.
- `data/consensus.md`: A human-readable table grouped by trait and sorted by descending support. The `raw` column preserves merge traces; unchecked feature cards are marked with ⚠.
- Three auxiliary tables are saved but not consumed:
  - `data/unknown_components.json`: Counts unmapped raw strings for future node creation.
  - `data/recipes.json`: Stores `(family, trait) → {image_size mode and distribution}` and other recipe fields.
  - `data/ensemble_cooccurrence.json`: Stores the family co-occurrence matrix for the §7 ensemble stage.

CLI: `python -m kb_mining.aggregate [--support-min 0.5] [--breadth-min 2]`

---

## 5. `decide.py`: five-tier decision list

For every `passed=True` row in `consensus.json`, compare the consensus with the current KB and write a suggestion to `data/proposals.md`. This module never changes KB data.

### 5.1 Prototype query

```python
ARCHETYPE_QUERY = {
"fine_grained": {"task_type": "classification", "data_size": <data_size mode of the trait evidence competition>,
                        "priority": "balanced", "constraints": {"fine_grained": True},
                        "description": "fine-grained image classification"},
    "class_imbalance": {..., "constraints": {"class_imbalance": True}, ...},
    "medical": {..., "constraints": {"medical": True}, ...},
    "data_size=small": {..., "data_size": "small", "constraints": {}, ...},
    # and so on
}
```

Retrieval call (run from repository root):

```python
from retrieval.rag_retrieval import build_graph, build_vector_index, retrieve_top3_hybrid
col = build_vector_index(persist_path=str(REPO_ROOT / "retrieval" / "chroma_db_kb"))
```

(Note: `fine_grained` is not a legal constraint key now. `_matches_condition` just returns a mismatch for the unknown key - in the tier 0 check, it is naturally equivalent to "query without this signal", correct.)

### 5.2 Five-tier decision process

| tier | condition | output in `proposals.md` |
|---|---|---|
| 0 confirmed | A is already the prototype query's top-1 result (use its loss field for loss rows) | Record that no change is required; confirmed rows provide positive evidence that the KB is correct |
| 1 field-fix | A node field conflicts with evidence; currently this checks whether its `data_size` list includes the modal evidence size | Identify the node, field, and proposed value |
| 2 edge-tune | `EDGES` already has edge `preferred_when` with source A, and the condition is related to T but does not contain T | Suggested specific modifications to `EDGE_CONDITIONS` (all→any/key addition) |
| 3 new-edge | None of the above applies, and T is a legal condition key | Suggest a new edge `(A, <current-top-1>, preferred_when)` and its condition; **target = the current top-1 prototype-query result** (scoring does not consume the target; it exists only for documentation semantics) |
| 4 schema-ext | T is not a legal condition key (for example, `fine_grained`) | Add the constraint key, synchronize the Module 1 prompt, add the tier 3 edge, and list affected code |

**Conflict check**: Mark a suggestion `CONFLICT` when it reverses an existing edge under an overlapping condition or changes a golden test's expected component. State that short A/B training is required before applying it. For every tier 1-4 suggestion, trial-apply it to a graph copy, rerun the prototype query, and record the resulting top-3 diff.

**Stacking check**: After trial-applying all tier 3 and 4 suggestions, a mined edge may receive at most one bonus-condition hit. Merge multiple traits into one `any` edge. If the limit is exceeded, add a warning at the top of `proposals.md`.

CLI: `python -m kb_mining.decide`

---

## 6. Phase B: loss `preferred_when` side wiring (only search code change)

**File**: `retrieval/rag_retrieval.py` `_select_components` (about 1283 lines).

**Status quo**: loss selection is hardcoded if chain (`class_imbalance→focal`, `segmentation→dice/bce_dice`, `detr→hungarian`); loss edge between `preferred_when` nodes (`focal_loss → cross_entropy_loss`, `condition={"all": ["class_imbalance=True"]}`) is dead data.

**Change**: Note that the existing code structure is `chosen = candidates[0]` default value + independent `if ctype == "loss":` block, and within the block is the entire chain of `if class_imbalance... / elif segmentation... / elif detr...`. Edge consumption **cannot** use `elif` to join this chain - the correct structure is to wrap the entire existing chain into `else:` (indent the whole by one layer), and the default value of `chosen = candidates[0]` remains at the front:

```python
chosen = candidates[0]  # default: First compatible item (leave it as is)

if ctype == "loss":
    # preferred_when Side consumption: Preference between candidates, if the conditions match, the winner will take the top spot.
    # (backbone only uses the source + condition of the edge for scoring; this is an intra-candidate selection, and the target is meaningful)
    edge_pick = None
    for cand in candidates:
        for succ in graph.successors(cand):
            e = graph[cand][succ]
            if (e.get("relation") == "preferred_when"
                    and succ in candidates
                    and _matches_condition(e.get("condition", {}), input_json)):
                edge_pick = cand
                break
        if edge_pick:
            break
    if edge_pick is not None:
        chosen = edge_pick
    else:
        # ↓ The entire existing hardcoded if/elif chain is moved into this else unchanged, only the indentation changes
        if c.get("class_imbalance") and "focal_loss" in candidates:
            ...
```

Deterministic requirements: The order of `candidates` is the traversal order, and the first hit wins (consistent with the existing deterministic habit of `candidates[0]`). **Do not delete** Existing hard-coded rules - they cover the situation where edges have not yet been expressed (bce_dice, hungarian); they will be cleaned up after the edges in the mining output are completed.

**Test** (Add `retrieval/test_rag_retrieval.py` style use case; item 1 is a hard requirement, no alternatives are given - otherwise the matter of "changing life from death to life" itself has not been verified):

1. **Edge paths must be proven to trigger**: Construct a copy of the graph within the test, add a synthetic loss `preferred_when` edge, the selection result is different from the hard-coded fallback chain (for example, add the condition `medical=True` to `cross_entropy_loss → focal_loss` - the semantics are fictitious, only for testing), assert medical Select cross_entropy instead of fallback results under query. In reality, the focal edge in KB has the same conclusion as the hard-coded rule. Behavioral regression cannot measure whether the edge is alive, and it must be distinguished by synthetic edges;
2. Class imbalance classification query → focal_loss; no imbalance → cross_entropy_loss (behavioral regression, both paths are covered);
3. `cd retrieval && python -m pytest test_golden.py test_rag_retrieval.py -q` All tests pass (**Hard conditions for acceptance**).

---

## 7. Test plan (kb_mining/tests/)

All offline, no network, no real LLM:

- `test_harvest.py`: fixtures puts three mini CSV (2 competition, 4 topic, 6 message), verify slug filtering, title regularization (including "1st place", "Solution summary", unrelated posts), rank analysis, each competition is truncated to MAX_POSTS_PER_COMP, chunked read path (fixture also goes to chunksize=2 Force multiple chunk).
- `test_extract.py`: canned LLM responds to injection `llm_fn`, covering: normal single, ensemble+best_single, reference verification failure and rejection, JSON bad rejection, family deduplication, alias table (including `tf_efficientnetv2_m`→efficientnet, `seresnext50`→resnet, unknown→unknown).
- `test_aggregate.py`: Handwritten fact fixtures verify vote weights (including the pseudo-label discount), coexistence filtering, support/breadth precision, threshold boundaries, and the three auxiliary tables. A competition held before Swin's release must contribute neither votes nor denominator weight to Swin.
- `test_decide.py`: Use the real `build_graph()` (chroma is not required - the tier 0 check can inject the fake retrieval function `retrieve_fn`), and assert the constructed consensus line to have one hit in each of the five tiers, conflict marks, and stacked alarms.

---

## 8. Runbook (full process command)

```bash
# 0. Prefix: Kaggle certificate (~/.kaggle/kaggle.json), LLM certificate (same as Module 1's env)
# 1. Collect (first download meta-kaggle, three CSV, ForumMessages several GB, be patient)
python -m kb_mining.harvest
# 2. Extract (first --limit 5 test run to check the quality of facts, then full quantity)
python -m kb_mining.extract --limit 5
python -m kb_mining.extract
# 3. Aggregation + Decision-making
python -m kb_mining.aggregate
python -m kb_mining.decide
# 4. Read data/consensus.md + data/proposals.md, select the changes to be applied → submit after independent review
# 5. Regression (after any KB changes)
cd retrieval && python -m pytest test_golden.py test_rag_retrieval.py -q
```

---

## 9. Acceptance criteria

1. `kb_mining/tests/` is completely green and has no network dependence in the entire process;
2. Real run-through harvest→extract→aggregate→decide: ≥8 competitions, global ≥50 articles facts, each key trait (fine_grained / class_imbalance / medical) has ≥3 competitions and each contributed ≥3 articles facts, `consensus.md` / `proposals.md` are generated and readable (≥5 articles in a single competition is no longer a hard target, see the recall policy of harvest);
3. `extract_rejects.jsonl` rejection rate < 30% (higher indicates prompt or the truncation strategy needs to be adjusted);
4. Phase B combined `test_golden.py` + `test_rag_retrieval.py` all tests pass;
5. This change in collection and derived statistics **does not include** any data changes to `EDGES` / `EDGE_CONDITIONS` / node fields / scoring constants (except for code changes to Phase B).

## 10. Implementation sequence

0. **Step 0 Data source verification (§0.5) - before everything else**; if falsified, stop and reconsider the data source
1. `catalog.py` (Data verification is the main workload: competition list, release time, initial filling of feature cards)
2. `harvest.py` + tests (first developed with fixture, and finally downloaded)
3. `extract.py` + tests (The first draft of prompt is in the appendix A; **0.5 days are reserved for special project prompt debugging** - test run with --limit, press the rejection iteration of extract_rejects, this is the whole project Don't crowd out the links that are most likely to be repeated)
4. `aggregate.py` + tests (pure function, fastest)
5. `decide.py` + tests
6. Phase B Wiring + Return (standalone commit)

The estimated net workload is 4-5 days (including half a day for step 0 and half a day for prompt debugging); the actual download of harvest and the full LLM call of extract are the only two external dependencies, both of which are done at the end of their respective stages.

---

## Appendix A: extract System prompt First Draft

Use this as a starting point when coding (write-up corpus is in English, prompt is in English); modifications during debugging need to be synchronized back to this appendix:

```text
You are extracting structured facts from a Kaggle competition solution write-up.
Return pure JSON only: no markdown fences, no commentary.

Output schema:
{
  "kind": "single" | "ensemble" | "unclear",
  "members": [{"raw_model": "<model name exactly as written>", "image_size": <int|null>}],
  "loss_raw": "<loss name exactly as written>" | null,
  "best_single_model_raw": "<model name>" | null,
  "best_single_score": <float|null>,
  "used_pseudo_labeling": true | false,
  "used_tta": true | false,
  "citations": ["<verbatim sentence copied from the post>"]
}

Rules:
1. members: every distinct model architecture in the FINAL submission only:
   ignore abandoned experiments. Copy names exactly as written
   (e.g. "tf_efficientnet_b4_ns"); do NOT normalize or expand them.
2. kind: "single" if the final submission is one model; "ensemble" if it
   averages/stacks several; "unclear" if you cannot tell.
3. loss_raw: the training loss of the main model(s); null if never stated.
4. best_single_model_raw / best_single_score: only if the post explicitly
   reports a best single-model score (e.g. "our best single model scored
   0.899"); otherwise null.
5. citations: 1-3 quotes copied character-for-character from the post that
   mention the models or the loss. They are used for automatic verification;
   paraphrased quotes will cause the whole extraction to be rejected.
6. If the post is not actually a solution write-up (e.g. a question or a
   congratulations thread), return {"kind": "unclear", "members": []}.
```

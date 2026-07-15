# A/B arbitration experiment: CE vs focal loss under `class_imbalance`

> This specification defines the experiment used to resolve the only conflict produced by `kb_mining`.
> The decision criteria are fixed here and in `configs.py` before any experiment runs.

---

## 0. Question and decision rules (pre-registration)

**Question**: For classification tasks with `class_imbalance=True`, should the default loss be focal loss (the current KB edge `focal_loss → cross_entropy_loss` plus the hard-coded `_select_components` rule) or cross-entropy (the mined Kaggle consensus: support 0.71, dominance ×2.66, breadth 10)?

**Testbeds** (v3: the secondary testbed now uses an existing catalog entry; together the testbeds cover medical and agricultural domains):

- **Primary**: `siim_isic`, an extremely imbalanced medical binary-classification task with approximately 1.8% positive samples; metric: ROC-AUC.
- **Secondary**: `cassava`, an existing catalog task that already runs end to end. It has five fine-grained leaf-disease classes, moderate imbalance (the largest class is approximately 61%), and 21,000 images. The decision metric is **macro-F1**, not the competition's official accuracy, because accuracy is insensitive to minority-class performance. Accuracy is recorded only as a secondary metric. APTOS is excluded because it is not in `vision_benchmark_catalog`; Diabetic Retinopathy is excluded because its 80 GB download is too costly for this experiment.

**Experimental design (v2: paired 5-fold, replacing the original two-seed single split)**: Each testbed uses five stratified folds shared by both arms, for 2 losses × 5 folds = 10 short training runs. Everything except the loss is identical: checkpoint, image size, augmentation, learning rate and schedule, epochs, and sampler. Both arms use ordinary shuffle rather than weighted sampling; loss/sampler interactions are out of scope. The global seed is 42. Focal-loss hyperparameters use the current Module 4 defaults, so this experiment compares the system's deployed focal-loss configuration against its deployed cross-entropy configuration.

**Statistic and decision rule** (fixed in advance; v2 corrects an overly narrow ±0.002 tie band): For every fold, compute `Δ_i = metric(CE, fold_i) − metric(focal, fold_i)`, then `Δ̄ = mean(Δ)` and `SE = std(Δ)/√5`. The tie band adapts to the observed noise and is never narrower than 0.005:

- `Δ̄ ≥ max(0.005, 2×SE)` → The test bench **CE wins**;
- `Δ̄ ≤ −max(0.005, 2×SE)` → The test bench **focal wins**;
- Otherwise → **TIE**. Reversing the current default requires a result larger than the estimated noise.

**Combined verdict**: Return `CE_WINS` only if no testbed favors focal loss and at least one favors cross-entropy. Apply the symmetric rule for `FOCAL_WINS`; otherwise return `TIE`. The conclusion must state the tested scope: extreme medical binary imbalance for SIIM-ISIC and moderate agricultural fine-grained multiclass imbalance for Cassava. These two testbeds do not cover every `class_imbalance` setting.

**Secondary metrics** (recorded but excluded from the decision to prevent metric shopping): SIIM-ISIC records PR-AUC, which is more sensitive than ROC-AUC under extreme imbalance; Cassava records accuracy for comparison with the official competition metric. If the primary metric ties but a secondary metric shows a large consistent difference, note it as follow-up evidence but do not use it to change the KB.

**Expected statistical power**: With strong pretraining and fine-tuning, the difference between cross-entropy and focal loss may fall within the noise. A `TIE` is a valid outcome: no detectable difference under the deployed conditions means keeping the current default and marking the conflict `resolved-tie`.

**Cost**: At 224 px with EfficientNet-B0, SIIM-ISIC takes approximately 1-1.5 hours per run, or 2-3 Kaggle GPU sessions for 10 runs. Cassava (21,000 images, 8 epochs) takes approximately 0.5-1 hour per run, or 1-2 sessions for 10 runs. Both can run unattended.

---

## 1. File structure

```
experiments/ab_loss_imbalance/
  configs.py     # Frozen experimental matrix + ruling constant (single source of truth)
  run_ab.py      # Driver: create folds → generate Module 4 project → run every (arm, fold) pair
  collect.py     # Aggregate fold-level paired differences into testbed and combined verdicts
  README.md      # Links to this specification
results/outcomes.jsonl   # One append-only record per run; see §3
```

## 1.5 Prerequisite: Module 4 fold injection

**Current limitation**: `code_generator.py` only supports an internal `val_split`; callers cannot inject folds. Without fold injection, each arm creates a different validation split and the paired difference `Δ_i` is meaningless. Complete this prerequisite before running the experiment.

**Interface**: Add two optional keys to `model_config`:

- `"fold_file"`: fold file path. Format:
  ```json
  {"seed": 42, "n_folds": 5, "stratified": true,
   "id_column": "image_id",
   "folds": [["id1","id7",...], ...]}   // validation sample IDs for each fold
  ```
Store sample IDs rather than row numbers so that CSV row-order changes do not invalidate the folds.
- `"fold_index"`: An integer from 0 to 4. The selected fold is validation data; all other folds are training data.

**`code_generator` changes**: In templates related to `_build_local_dataloader`, use IDs from `fold_file` and bypass the internal `val_split` when the file is present. Validate that the folds are pairwise disjoint and that their union equals every CSV sample ID; reject invalid files. If the two keys are absent, preserve the existing behavior.

**Tests** (in `module4_agent/tests/`): ① A fixture CSV and `folds.json` produce exactly the requested validation IDs. ② Two projects using the same fold file but different losses produce identical validation sets. ③ Incomplete or overlapping fold files are rejected. ④ Omitting both keys preserves the old behavior.

**Related prerequisite: multi-metric export**. Module 4 normally computes only the catalog metric (Cassava accuracy or SIIM ROC-AUC), but this experiment also needs Cassava macro-F1 and SIIM PR-AUC. After training, `run.py` must export validation labels and predictions to `val_preds.json` as `{"y_true": [...], "y_score": [...]}`. `run_ab.py` then computes the metric bundle with scikit-learn and stores it in `val_metric`.

### configs.py

```python
MARGIN_FLOOR = 0.005          # Tie-band floor; actual width = max(0.005, 2*SE)
N_FOLDS = 5
GLOBAL_SEED = 42
ARMS = ("focal_loss", "cross_entropy_loss")
TESTBEDS = {
    "siim_isic": {"metric": "roc_auc", "image_size": 224, "epochs": 8,
                  "secondary_metrics": ["pr_auc"]},
    "cassava": {"metric": "macro_f1", "image_size": 224, "epochs": 8,
                  "secondary_metrics": ["accuracy"]},
}
BASE = {                      # All frozen except loss/fold
    "backbone": "efficientnet",
    # Resolve the checkpoint once through Module 3, then hard-code the ID here with the date.
    # Do not resolve it dynamically, because later KB changes must not alter this experiment.
    "pretrained": "efficientnet_b0_XXXX",
    "sampler": "shuffle", # Explicit statement: unweighted sampling
    "cv": {"n_folds": 5, "stratified": True, "shared_across_arms": True},
}
def build_matrix() -> list[dict]: ...   # 10 configs per testbed; only loss and fold_index differ
```

**Tests**: ① Each testbed matrix has 10 entries, and every value except `loss` and `fold_index` is identical. ② Both arms reference the same `fold_file`. ③ `pretrained` contains no placeholder text.

### run_ab.py

Reuse the existing pipeline rather than creating separate training code. For each testbed:

1. Download data with `ingest_benchmark(<testbed>, data_root)`.
2. Read labels from the training CSV and create `StratifiedKFold(5, shuffle=True, random_state=42)`. Write `folds_<testbed>.json` in the §1.5 format and calculate its SHA-256. Reuse an existing fold file so reruns cannot silently create new folds.
3. Generate one Module 4 project through `run_kaggle_benchmark.prepare_project`. For each run, override `loss`, `fold_file`, and `fold_index` in `model_config`, then train sequentially.
4. Append the §3 record, including `fold_file_sha256`, to `results/outcomes.jsonl`. On restart, skip completed `(arm, fold)` pairs.

CLI: `python -m experiments.ab_loss_imbalance.run_ab [--testbed cassava] [--data-root ...] [--only focal_loss:3]` (`--only arm:fold` runs one arm/fold combination)

### collect.py

Read `outcomes.jsonl` and produce a five-row paired-difference table, `Δ̄ ± SE`, a verdict for each testbed, and the combined verdict. Unit tests must cover the three testbed verdicts, values exactly on either side of `max(0.005, 2×SE)`, and every combined-verdict case. This logic is a pure function and requires no network access.

## 3. outcomes.jsonl record schema

```json
{"experiment": "ab_loss_imbalance", "benchmark": "siim_isic",
 "arm": "cross_entropy_loss", "fold": 3, "seed": 42,
 "config": {"backbone": "...", "pretrained": "...", "image_size": 224,
            "epochs": 8, "sampler": "shuffle"},
 "val_metric": {"roc_auc": 0.912, "pr_auc": 0.231}, "best_epoch": 6,
 "fold_file_sha256": "…", "kb_version": "<git sha>", "date": "2026-07-xx"}
```

## 4. Verdict-specific KB actions

| verdict | KB Action |
|---|---|
| CE_WINS | ① Flip the edge: delete `focal_loss→cross_entropy_loss` and add `cross_entropy_loss→focal_loss` (conditions remain unchanged); ② Update the hard-coded imbalance rule in `_select_components`; ③ Update the corresponding golden-test assertion; ④ Add traceability comments to every change (`# ab_loss_imbalance 2026-07: paired 5-fold, siim Δ̄=+x.xxx±SE / cassava Δ̄=…; scope=extreme medical imbalance + moderate agricultural imbalance`) |
| FOCAL_WINS | KB does not move; add annotation "Kaggle consensus (0.71) contrary to this, siim_isic A/B empirical defense (focal +x.xxx)"; proposals's CONFLICT Mark resolved-rejected |
| TIE | KB does not move; the side annotation records "A/B no significant difference, maintain the status quo"; CONFLICT marks resolved-tie |

After applying any verdict-specific action, `cd retrieval && python -m pytest test_golden.py test_rag_retrieval.py -q` must pass.

## 5. Acceptance criteria

1. Offline tests pass for configuration freezing, paired folds, placeholder rejection, and verdict aggregation.
2. Both testbeds complete 10 runs each, and `outcomes.jsonl` contains 20 complete records.
3. `collect` produces both testbed verdicts and the combined verdict; the §4 action is applied; golden tests pass.
4. The documentation contains a concise result summary with the paired-difference table, verdict, scope, and KB action.

## 6. Implementation sequence

0. **§1.5 fold-injection mechanism + Four Tests (Half Day) - Do not proceed to any subsequent steps until completed**
1. configs.py + three freeze tests, collect.py + verdict test (half a day, purely offline)
2. run_ab.py connects to calculation/generation/overwriting/continuation (for half a day, first use `--only focal_loss:0` single-arm single fold 1 epoch to verify the link on cassava)
3. Full run: cassava 10 times (1-2 session) first to obtain its testbed verdict, then siim_isic 10 times (2-3 session)
4. Combined verdict → §4 Action → golden return → experiment summary (half day)

**Out of scope** (clearly not done): weighted sampling interaction, focal hyperparameter search, KB changed according to secondary indicators.

# Recipe layer: hyperparameter recommendation design

> This document describes the recipe layer implemented and tested in `recipe/`.
> The graph selects components; the recipe layer configures them.
> It recommends model-dependent hyperparameters (`image_size`, `learning_rate`, `epochs`, and `augmentation`)
> plus optional early-stopping metadata. Module 4 implements early stopping during training.
> Execution-level hyperparameters (batch_size / num_workers, etc.) are still managed by Module 4.

---

## 0. Current status (why this layer is needed)

Hyperparameters are currently scattered in three places with no unified source:

- `pipeline.py::derive_recommended_epochs` is the only signal-to-hyperparameter function and is injected after the main pipeline logic through `setdefault`.
- `image_size` and `learning_rate` are fixed Module 4 template defaults that ignore the data and backbone.
- No augmentation recommendation exists.

The recipe layer consolidates these decisions in a deterministic Module 3 component and records provenance for each output. It uses rules rather than an LLM and can be unit-tested offline.

---

## 1. Responsibility and call site

Use an independent module so it can be tested and changed without coupling it to retrieval:

```
recipe/
  __init__.py
  layer.py       # Orchestration: build_recipe(config, input_json, backbone_facts, data_stats)
  tables.py      # Frozen lookup tables (epochs, base LR, augmentation tiers, vetoes, image constraints)
  augment.py     # Three-dimensional augmentation parser
  tests/
```

Call `build_recipe` once at the end of `rag_retrieval.build_task_list`, after `model_config` has its backbone, checkpoint, loss, and fine-tuning strategy. Merge the result into `model_config`. This keeps configuration generation with the recommendation and removes duplicated post-processing in callers such as `pipeline` and `run_kaggle_benchmark`.

**Signature**:

```python
def build_recipe(
    config: dict,          # Assembled model_config
    input_json: dict,      # Module 3 input
    backbone_facts: dict,  # Selected graph-node properties and image constraints
    data_stats: dict | None = None,  # Module 2 resolution and color statistics
) -> tuple[dict, dict]: # (recipe, provenance)
```

`build_task_list` accepts and forwards an optional `data_stats`. When it is absent, the recipe uses conservative fallback values.

---

## 2. Pre-step 0 - connect Module 2 statistics (signal foundation of image_size / grayscale veto)

**Current limitation**: `merge_modules` retains only `data_size`, `num_classes`, and `class_imbalance`. It drops Module 2 resolution statistics, `mode_distribution`, and `format_distribution`, leaving the recipe without signals for resolution-aware image sizing or grayscale vetoes.

**Changes** (derive_* pattern along improvements §3):

```python
def derive_resolution_tier(stats) -> str:  # low (<256), medium, or high (>=768), based on average shorter side
def derive_color_mode(stats) -> str:       # rgb or grayscale, based on mode_distribution
```

`merge_modules` should pass both fields to Module 3 in a separate `data_stats` object rather than adding them to retrieval constraints. Without these fields, the recipe still runs but uses the backbone's default image size and permissive augmentation defaults; provenance records `signal_missing`.

---

## 3. Four sub-decisions (v0 scope: classification task)

### 3.1 epochs

Move `derive_recommended_epochs` and `_RECOMMENDED_EPOCHS` into `recipe/tables.py`. Keep a re-export in `pipeline.py` so existing imports continue to work. The table key is `(data_size, mode)`, where `mode ∈ {head_only, finetune, scratch}` comes from `finetune_strategy` and `use_pretrained`. Provenance: `"epochs_table[{data_size},{mode}]"`.

### 3.2 image_size ("convergence point" of improvements §4)

The three inputs are combined and solved **in order**:

1. **Baseline**: Prefer the selected checkpoint's expected input resolution (for example, EfficientNet-B0 uses 224). Otherwise use the family default from `_FAMILY_IMAGE_DEFAULT`.
2. **Resolution + fine-grained up-scaling**: `data_stats.resolution_tier == "high"` **and** `constraints.fine_grained` (fine-grained requires details) → one step up (224→384); `priority == "speed"` or `data_size == "large"` → no up-scaling (high resolution is slower);
3. **Hard-constraint snapping**: Some backbones require divisible input sizes (DINOv2 by 14, Swin by 32, ViT by 16). Store verified divisors in `_IMAGE_DIVISOR` and snap the result to the nearest valid value. Apply this safety constraint last.

If `data_stats` is missing, use the backbone default and still apply divisor snapping. Provenance: `"ckpt_default=224 | fine_grained+high_res bump→384 | snapped /14→392"`.

### 3.3 learning_rate (coupled with finetune_strategy)

Use `_LR_BASE[(family_class, mode)]`, where `family_class ∈ {cnn, transformer}` and transformers use a lower learning rate. `mode` is the same value used for epochs:

| | head_only | finetune | scratch |
|---|---|---|---|
| cnn | 1e-3 | 1e-4 | 5e-4 |
| transformer | 1e-3 | 3e-5 | 3e-4 |

(The value is v0 by default, pending recipes.json / A/B calibration.) warmup / schedule types **are not in this layer** - they require runtime feedback and fall under Module 4 (§7). Provenance: `"lr_base[cnn,finetune]"`.

### 3.4 augmentation (3D: Strength ⊗ Invariance ⊗ Schedule, see augment.py §4)

---

## 4. `augment.py`: three-dimensional augmentation policy

Output structure:

```python
{"tier": "medium",
 "invariance": {"hflip": True, "vflip": False, "rot90": False, "color": True,
                "crop_scale_min": 0.8},
 "schedule": "taper_last_20pct"}
```

### Dimension 1: Strength level (tier) - "Add how much"

Rules (in order, the latter overrides the tier adjustment of the former):

```
data_size=small  → heavy; medium → medium; large → light
finetune_strategy=head_only → reduce by one tier (a frozen backbone adapts poorly to strong distortion)
constraints.few_shot=True → heavy with mandatory RandAugment
```

Tier contents use torchvision v2 without new dependencies: `none` = resize + normalization; `light` = RRC(0.8-1.0) + horizontal flip; `medium` adds rotation, translation, ColorJitter, and RandomErasing; `heavy` adds RandAugment and MixUp/CutMix.

### Dimension 2: Invariance mask: which transforms are safe

Vetoes override the transform switches selected by the strength tier:

| Signal | source | veto |
|---|---|---|
| color_mode=grayscale | Module 2 (`data_stats`) | `color=False` (color jitter is meaningless for grayscale images) |
| domain ∈ {satellite, aerial, pathology, microscopy} | Module 1 semantics (v0 is missing, see below) | `vflip=True, rot90=True` (no fixed orientation) |
| domain ∈ {document, digit, ocr} | Module 1 semantics | `hflip=False, vflip=False, rot90=False` (flipping can change the label) |
| domain=medical and orientation-sensitive (such as chest X-rays) | Module 1 semantics | `hflip=False` |
| constraints.fine_grained | Existing constraint | Raise `crop_scale_min` to 0.5 so crops retain discriminative features |

**v0 limitation**: Module 1 does not yet extract a domain field. In v0, only the grayscale veto and the existing `fine_grained` crop veto receive real signals. Other domain-based vetoes use permissive defaults until Module 1 adds domain extraction; provenance records `domain_signal_missing`.

### Dimension 3: Schedule (schedule) - "When to add"

v0 emits one static label: `data_size in {small,medium}` uses `"taper_last_20pct"`, reducing augmentation during the final 20% of epochs; `large` uses `"constant"`. Dynamic scheduling based on the train/validation gap requires runtime feedback and belongs in Module 4 v1.

---

## 5. Output with provenance

Parallel to `model_config`:

```python
config["recipe"] = {"image_size":384, "learning_rate":1e-4, "epochs":20,
                    "augmentation": {...}}
config["recipe_provenance"] = {"image_size": "...", "learning_rate": "...",
                    "epochs": "...", "augmentation": "..."}
```

Each provenance field contains one sentence describing the rule path and any missing signal. This makes every recommended hyperparameter explainable in the final report.

**Calibration hook (not automated in v0)**: Comments beside defaults in `tables.py` identify values that later `kb_mining` recipe data or A/B results may override. v0 uses manually selected defaults.

---

## 6. Module 4 consumption

Module 4 reads `config["recipe"]` and uses `image_size`, `learning_rate`, and `epochs` directly. At dataloader construction, it converts the augmentation tier, invariance switches, and schedule into a torchvision v2 transform pipeline. If `recipe` is absent, Module 4 preserves its existing defaults. The recipe layer produces structured configuration, not code.

---

## 7. Scope discipline (HP attribution in §2 of improvements is explicitly not done)

- **Owned by Module 4**: batch size, OOM retry and gradient accumulation, workers, mixed precision, gradient clipping, early stopping, learning-rate warmup, and dynamic augmentation scheduling.
- **Excluded from v0**: detection and segmentation recipes, scheduler selection, per-checkpoint image-size search, and Module 1 domain extraction.

---

## 8. Test plan (recipe/tests/, fully offline pure function)

- **Decision unit tests**: Epoch values remain unchanged after migration; DINOv2 image sizes are divisible by 14; learning-rate lookups hit the expected row; each augmentation rule is covered.
- **Invariant tests**: The selected image size satisfies the backbone divisor; `head_only` never receives `heavy`; grayscale always sets `color=False`; `fine_grained` always sets `crop_scale_min>=0.5`; missing `data_stats` follows the documented fallback without failing.
- **golden-style end-to-end**: several representative inputs (small data fine-grained / big data speed first / grayscale medicine / few_shot) → assert complete recipe.
- **Integration regression**: `cd retrieval && python -m pytest test_golden.py test_rag_retrieval.py` passes after `build_task_list` adds the new `recipe` field.

---

## 9. Implementation sequence

0. §2 Module 2 Statistics access merge_modules + derive_* + test (half a day; if not done, recipe will run in fallback mode and can be used as a parallel item, but image_size/ grayscale veto will run idling)
1. tables.py (migrate epochs + image/lr/augment default table) + layer.py orchestration + epochs regression test (half a day)
2. image_size + learning_rate Sub-decision + Invariant Test (half day)
3. augment.py 3D analysis + test (half a day, the most complex)
4. Access build_task_list + integrated regression; Module 4 template consumption recipe (§6) + backward compatibility (half a day)

Estimated net workload 2-3 days. §2 (Module 2 access) and domain extraction of Module 1 are two signal source fronts that can be advanced independently - without blocking the recipe skeleton, the invariance dimension is complete after being connected.

---

## 10. One sentence overview

The recipe layer deterministically maps the selected backbone and data signals to `image_size`, learning rate, epochs, and augmentation, with provenance for every value. v0 covers classification. The graph provides component facts, the recipe configures them, and Module 4 turns the configuration into training code.

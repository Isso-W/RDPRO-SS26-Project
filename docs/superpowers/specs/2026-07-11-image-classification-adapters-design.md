# Image Classification Adapters Design

**Date:** 2026-07-11
**Branch:** codex/mlestar-kaggle-benchmarks
**Status:** Approved for implementation planning

## Goal

Extend the standalone MLE-STAR reproduction so it can train and evaluate on
six more benchmark-catalog tasks (in addition to the already-working
`leaf_classification`):

- `plant_pathology_2020` -- image_multilabel, metric `roc_auc`
- `aptos_2019` -- image_ordinal, metric `qwk`
- `dog_breed` -- image_multiclass, metric `multiclass_log_loss`
- `aerial_cactus` -- image_binary, metric `roc_auc`
- `dogs_vs_cats` -- image_binary, metric `log_loss`
- `histopathologic_cancer` -- image_binary, metric `roc_auc`

The three remaining catalog entries (`global_wheat` / object detection,
`ultrasound_nerve` / segmentation, `denoising_dirty_documents` / image
denoising) are architecturally distinct (different model families, different
label formats) and are explicitly **out of scope** for this design. They stay
`NotImplementedError` until a future design covers them.

## Why this scope

`mlestar/metrics.py` and `mlestar/contracts.py` already implement and
validate every metric these six tasks need (`roc_auc`, `log_loss`,
`multiclass_log_loss`, `qwk`) plus the metric-direction contract in
`MetricSpec`. `benchmarks/catalog.py` already registers `TaskSpec` entries for
all six (competition slug, modality, metric, id/prediction columns). Nothing
in the scoring or contract layer needs to change. What's missing is purely
the **data adapter layer**: reading each competition's images/labels off
disk, training a model, producing OOF + test predictions, and packaging them
into the same `ExperimentReceipt` shape `LeafClassificationAdapter` already
produces.

## Architecture

```
mlestar/adapters/
  tabular.py              # existing, untouched
  vision.py                # NEW: ImageClassificationAdapter base class
```

`ImageClassificationAdapter` is an abstract base implementing the full
`CandidateEvaluator` protocol (`evaluate`, `merge`, `run`) exactly like
`LeafClassificationAdapter`, so `initialization.py`, `refinement.py`, and
`ensemble.py` require **no changes**. It owns:

- fixed fold splitting via plain `KFold` (reusing `task.fold.n_splits`,
  deliberately not `StratifiedKFold` -- `plant_pathology_2020`'s labels are
  multi-label indicator vectors, which `StratifiedKFold` cannot take as `y`
  at all, and the tiny synthetic fixtures have too few members per class
  for stratification's per-class minimum-count requirement even on the
  single-label tasks; plain index-based `KFold` has neither constraint)
- timm model construction, per-fold fine-tuning, OOF + test prediction
- modality dispatch (loss function, output activation, prediction shaping)
- calling `score_metric(self.task.metric, y_true, oof, ...)`
- writing `oof.csv` / `test_predictions.csv` / `folds.csv` / `submission.csv`
  via `RunArtifacts`, matching `LeafRun`'s contract

Six thin subclasses each implement exactly one method:

```python
class ImageClassificationAdapter:
    def _load_dataset(self, data_root: Path) -> tuple[list[Path], np.ndarray, list[str]]:
        """Return (image_paths, labels, ids). Subclasses implement this."""
        raise NotImplementedError

class PlantPathologyAdapter(ImageClassificationAdapter): ...
class AptosAdapter(ImageClassificationAdapter): ...
class DogBreedAdapter(ImageClassificationAdapter): ...
class AerialCactusAdapter(ImageClassificationAdapter): ...
class DogsVsCatsAdapter(ImageClassificationAdapter): ...
class HistopathologicCancerAdapter(ImageClassificationAdapter): ...
```

Each subclass lives in `mlestar/adapters/vision.py` alongside the base class
(six small classes; split into a separate file only if the file grows
unwieldy during implementation).

### Why a base class + subclasses instead of one branching class

The six competitions' raw file layouts are genuinely different (see table
below) -- unlike modality (which is a clean 4-way branch), the data-loading
logic per competition is bespoke enough that folding it into one method with
six branches would make each competition's assumptions harder to audit in
isolation and easier to accidentally cross-contaminate. Keeping loaders as
separate small classes/methods means changing one competition's parsing
cannot silently affect another's, and each loader is unit-testable on its
own fixture.

## Per-competition data layout

This is the part most likely to need correction once real data is
downloaded and inspected -- the loaders must fail loudly (clear
`FileNotFoundError`/`ValueError`, matching `LeafClassificationAdapter`'s
style) rather than silently guess, if the on-disk layout doesn't match.

| Task | Label source | Image path pattern | Notes |
|---|---|---|---|
| `plant_pathology_2020` | `train.csv` (`image_id`, `healthy`, `multiple_diseases`, `rust`, `scab`) | `images/{image_id}.jpg` | train/test images share one flat directory |
| `aptos_2019` | `train.csv` (`id_code`, `diagnosis` 0-4) | `train_images/{id_code}.png` | |
| `dog_breed` | `labels.csv` (**not** `train.csv`; `id`, `breed`) | `train/{id}.jpg` | class names come from `sample_submission.csv` header (`prediction_from_sample=True` in the task contract) |
| `aerial_cactus` | `train.csv` (`id`, `has_cactus`) | `train/{id}` | `id` already includes the file extension |
| `dogs_vs_cats` | **no CSV** -- label is encoded in the filename (`cat.N.jpg` / `dog.N.jpg`) | `train/` | parse label from filename prefix |
| `histopathologic_cancer` | `train_labels.csv` (`id`, `label`) | `train/{id}.tif` | |

## Training details

- Model: `timm.create_model(name, pretrained=..., num_classes=...)`, default
  `resnet18`.
- Images resized to 128x128 by default (configurable), Adam optimizer,
  default 3 epochs/fold (configurable).
- Modality -> loss/output head:
  - `image_binary` / `image_multiclass`: CrossEntropyLoss + softmax
  - `image_multilabel`: BCEWithLogitsLoss + independent sigmoid per label
  - `image_ordinal`: regression head (MSELoss), predictions rounded to the
    nearest integer class before scoring with `qwk`
- **Large-dataset subsampling**: competitions with large training sets
  (`histopathologic_cancer` ~220k images, `dogs_vs_cats` ~25k) default to a
  `max_train_samples` cap of 2000 images per fold (stratified sample),
  overridable via adapter/CLI parameter, so all six tasks x 5 seeds stays
  tractable in a free Colab session. This only bounds training data; OOF/test
  scoring still runs over the full held-out fold.

## CLI / experiment.py changes

`experiment.py::compare()` currently hard-fails for every benchmark except
`leaf_classification`:

```python
if benchmark != "leaf_classification":
    raise NotImplementedError(...)
```

Replace this with a `benchmark -> adapter class` registry covering all seven
implemented tasks; unregistered benchmarks (the three out-of-scope ones)
keep raising `NotImplementedError` with the same message. `_candidate()` /
the baseline-vs-initial-vs-refined-vs-ensemble flow in `compare()` stays
structurally the same for every adapter, since they all implement the same
`CandidateEvaluator` protocol.

## Notebook changes

The Leaf Classification download cell (cell 3) currently hardcodes the
`leaf-classification` competition slug and the nested-zip nuance for that one
competition. This becomes a reusable helper (competition slug + `DATA_ROOT`
parameterized, still using `KAGGLE_API_TOKEN` from the environment, still
extracting the outer zip then any nested `*.csv.zip`/`*.zip` members, still
raising loudly if the expected marker file is missing afterward), and one new
notebook cell per new task calls it with that task's competition slug and
data root, followed by a `mlestar.cli compare --benchmark <key> ...` cell.

## Testing strategy

- Six new `examples/synthetic_<task>/` fixture directories, each with a
  handful of tiny (e.g. 8x8) generated images plus the matching label
  CSV/filename convention for that competition -- mirrors
  `examples/synthetic_leaf/`.
- Tests instantiate each adapter with `pretrained=False` (randomly
  initialized timm model, no network access, no weight download) so the
  suite stays fast and offline, matching the current ~6.5s/28-test runtime.
  The notebook's real runs use `pretrained=True`.
- One test per adapter subclass asserting `_load_dataset` parses its
  fixture into the expected `(paths, labels, ids)` shape, plus one
  end-to-end `run()` test per adapter (small fixture, `pretrained=False`,
  few epochs) asserting a non-`None` `metric_value` and correctly shaped
  `oof.csv`/`test_predictions.csv`.
- Extend `tests/test_standalone_layout.py`-style isolation checks are not
  needed here (no new top-level directories).

## Error handling

Same philosophy as `LeafClassificationAdapter.run()`: wrap the whole
per-candidate run in `try/except Exception`, converting failures into an
`ExperimentReceipt` with `metric_value=None` and a populated `error` field
rather than crashing the whole `compare()` call -- but (learning from the
`choose_best` debugging session on this branch) `experiment.py` should also
be checked to make sure a failed baseline/candidate's `receipt.error` is
surfaced somewhere the user can see it (e.g. included in `choose_best`'s
raised message or printed to notebook output) rather than only discarded
into a never-reached `receipts.jsonl`.

## Out of scope (explicitly deferred)

- `global_wheat` (object detection), `ultrasound_nerve` (segmentation),
  `denoising_dirty_documents` (image-to-image denoising) -- different model
  families, deferred to future designs.
- Full-dataset (non-subsampled) training runs -- possible via the
  `max_train_samples` override, but not the default and not exercised by
  this design's test suite.
- Actual submission to Kaggle for these six tasks (the existing `SUBMIT`
  gate and `--no-submit` default already cover this; no new behavior
  needed).

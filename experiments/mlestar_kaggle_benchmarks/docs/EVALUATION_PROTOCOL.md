# Evaluation protocol

The target protocol has four arms: deterministic modality baseline,
search-informed initial solution, targeted refinement, and OOF-only ensemble.
It calls for identical folds and seeds, explicit data/code manifests, comparable
resource budgets, per-fold values, uncertainty, failures, refinement counts,
and elapsed time.

The current `mlestar compare` runner implements the four arms with deterministic
fold construction, identical seeds, OOF-based selection, per-seed comparison
rows, summary mean/SEM, wins, failures, and JSONL receipts. Its default seeds
are 13, 29, and 47. It does not enforce equal wall-clock budgets or generate a
data/environment/Git manifest for general benchmark runs. The scripted smoke
entry point adds those manifests for the committed synthetic Leaf fixture only;
resource-budget enforcement and full-study provenance remain pending.

OOF predictions are required for all selection: model choice, ordinal
thresholds, detection confidence/NMS settings, segmentation post-processing and
ensemble weights. Test predictions are only used after an arm has been selected
from its OOF result.

Public Kaggle leaderboard scores are not substituted for OOF metrics. This
package has no automatic submission path: an externally submitted result is
recorded only after Kaggle accepts and processes it, with the receipt kept
separate from every comparison run.

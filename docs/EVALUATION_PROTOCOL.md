# Evaluation protocol

Each benchmark comparison has four arms: deterministic modality baseline,
search-informed initial solution, targeted refinement, and OOF-only ensemble.
Every arm uses the identical persisted fold assignment, seeds, data manifest and
wall-clock budget. The default study runs seeds 13, 29 and 47 and reports
per-fold values, mean, standard error, run failures, search/refinement counts
and elapsed time.

OOF predictions are required for all selection: model choice, ordinal
thresholds, detection confidence/NMS settings, segmentation post-processing and
ensemble weights. Test predictions are only used after an arm has been selected
from its OOF result.

Public Kaggle leaderboard scores are not substituted for OOF metrics. A score
is recorded only when the API accepts and processes an explicit submission; the
submission command is opt-in and separate from every comparison run.

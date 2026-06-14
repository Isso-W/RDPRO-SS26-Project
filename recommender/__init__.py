"""Accumulating, explainable recommender layer for Module 3.

Turns Module 3's constrained candidate shortlist into a ranked, *explained*
recommendation that improves as an outcome memory accumulates — the things an
autonomous, black-box, per-task agent (e.g. MLE-STAR) structurally can't do.
"""

from .fingerprint import dataset_fingerprint, fingerprint_distance
from .logme import logme_score
from .outcome_memory import OutcomeMemory
from .ranker import rank_candidates, recommend, log_run, log_from_summary

__all__ = [
    "dataset_fingerprint",
    "fingerprint_distance",
    "logme_score",
    "OutcomeMemory",
    "rank_candidates",
    "recommend",
    "log_run",
    "log_from_summary",
]

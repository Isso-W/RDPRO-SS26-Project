"""Typed Kaggle benchmark contracts used by the MLE-STAR runner."""

from .catalog import get_benchmark, list_benchmarks
from .contracts import BenchmarkContract, FoldContract, SubmissionContract

__all__ = ["BenchmarkContract", "FoldContract", "SubmissionContract", "get_benchmark", "list_benchmarks"]

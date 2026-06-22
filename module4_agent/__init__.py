"""Module 4 local code-generation agent for CV Auto-DL."""

from .spec_builder import build_training_specs
from .workflow import run_workflow

__all__ = ["build_training_specs", "run_workflow"]


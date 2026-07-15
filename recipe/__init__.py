"""Deterministic recipe layer for model-coupled hyperparameters."""

from .layer import build_recipe
from .tables import derive_recommended_epochs

__all__ = ["build_recipe", "derive_recommended_epochs"]

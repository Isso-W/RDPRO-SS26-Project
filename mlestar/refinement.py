"""Component-scoped source patches used by targeted MLE-STAR refinement."""

from __future__ import annotations

from typing import Mapping

from .contracts import COMPONENT_NAMES


def component_body(source: str, component: str) -> str:
    """Return the bytes between one required component marker pair."""

    start, end = _markers(component)
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError(f"Source must contain one marker pair for {component!r}.")
    left, remainder = source.split(start, 1)
    body, right = remainder.split(end, 1)
    del left, right
    return body


def apply_component_patch(source: str, component: str, body: str) -> str:
    """Replace exactly one component body, preserving every other byte."""

    start, end = _markers(component)
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError(f"Source must contain one marker pair for {component!r}.")
    prefix, remainder = source.split(start, 1)
    _, suffix = remainder.split(end, 1)
    return f"{prefix}{start}\n{body.strip()}\n{end}{suffix}"


def _markers(component: str) -> tuple[str, str]:
    if component not in COMPONENT_NAMES:
        raise ValueError(f"Unsupported component {component!r}.")
    return f"# MLESTAR_COMPONENT:{component}:START", f"# MLESTAR_COMPONENT:{component}:END"

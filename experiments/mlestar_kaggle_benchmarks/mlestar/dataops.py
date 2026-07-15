"""Small, serializable DataOps graphs for isolated MLE-STAR runs.

The graph deliberately transports metadata and artifact paths only.  Adapters
write their own artifacts; models, arrays, and data frames never become graph
state, which keeps DataOps evaluation deterministic and safe to serialize.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
import re
from typing import Any

import skrub


_PHASE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_STATE_KEYS = frozenset({"run_context", "component_trace", "artifacts"})


def build_run_graph(run_dir: Path, phases: tuple[str, ...]):
    """Build a named skrub graph that evaluates each requested phase once.

    The returned graph accepts an input mapping named ``run_context``.  Its
    value must be JSON metadata and should contain ``run_dir`` pointing at the
    same directory supplied here.  Each graph node produces a new state rather
    than mutating the preceding one.
    """

    root = _run_root(run_dir)
    phase_names = tuple(_validate_phase_name(phase) for phase in phases)
    if len(set(phase_names)) != len(phase_names):
        raise ValueError("phases must be unique so every component runs once")

    state = skrub.var("run_context")
    for phase in phase_names:
        state = skrub.deferred(run_phase)(state, phase, str(root)).skb.set_name(
            phase
        )
    return state


def run_phase(state: Mapping[str, Any], phase: str, run_dir: str | Path) -> dict[str, Any]:
    """Return a validated, copied state after recording one execution phase.

    ``state`` is restricted to JSON-compatible run metadata, an ordered phase
    trace, and artifact paths below ``run_dir``.  The synthetic phase artifact
    is metadata for the adapter output that phase is expected to write; no data
    object is retained by the DataOps graph.
    """

    root = _run_root(run_dir)
    phase_name = _validate_phase_name(phase)
    copied = _copy_json_metadata(state, label="state")
    if not isinstance(copied, dict):
        raise TypeError("state must be a JSON object")

    # ``skrub.var("run_context")`` evaluates to the variable value itself,
    # rather than a mapping containing that variable name.  Treat that first
    # value as the run context; subsequent nodes use the complete state below.
    if "run_context" not in copied:
        copied = {"run_context": copied}

    unexpected = set(copied).difference(_STATE_KEYS)
    if unexpected:
        names = ", ".join(sorted(unexpected))
        raise ValueError(f"state may contain only run metadata and artifact paths: {names}")

    context = copied.get("run_context", {})
    if not isinstance(context, dict):
        raise TypeError("run_context must be a JSON object")
    _validate_context_run_dir(context, root)
    context["run_dir"] = str(root)

    trace = copied.get("component_trace", [])
    if not isinstance(trace, list) or not all(isinstance(item, str) for item in trace):
        raise TypeError("component_trace must be a JSON list of phase names")
    if phase_name in trace:
        raise ValueError(f"phase {phase_name!r} has already been evaluated")

    artifacts = copied.get("artifacts", {})
    if not isinstance(artifacts, dict):
        raise TypeError("artifacts must be a JSON object mapping names to paths")
    validated_artifacts = {
        _validate_artifact_name(name): _resolve_artifact_path(path, root)
        for name, path in artifacts.items()
    }
    validated_artifacts[phase_name] = str(root / f"{phase_name}.json")

    return {
        "run_context": context,
        "component_trace": [*trace, phase_name],
        "artifacts": validated_artifacts,
    }


def _run_root(run_dir: str | Path) -> Path:
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"run directory must exist and be a directory: {root}")
    return root


def _validate_phase_name(phase: str) -> str:
    if not isinstance(phase, str) or not _PHASE_NAME.fullmatch(phase):
        raise ValueError(
            "phase names must contain only letters, numbers, dots, underscores, or hyphens"
        )
    return phase


def _validate_artifact_name(name: object) -> str:
    if not isinstance(name, str) or not name:
        raise TypeError("artifact names must be non-empty strings")
    return name


def _validate_context_run_dir(context: dict[str, Any], root: Path) -> None:
    declared = context.get("run_dir")
    if declared is None:
        return
    if not isinstance(declared, str):
        raise TypeError("run_context.run_dir must be a string path")
    if Path(declared).expanduser().resolve() != root:
        raise ValueError("run_context.run_dir must match the graph run directory")


def _resolve_artifact_path(value: object, root: Path) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError("artifact paths must be non-empty strings")
    candidate = Path(value).expanduser()
    resolved = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("artifact paths must remain inside the run directory")
    return str(resolved)


def _copy_json_metadata(value: Any, *, label: str) -> Any:
    """Deep-copy JSON metadata while rejecting models, tensors, and paths."""

    try:
        encoded = json.dumps(deepcopy(value), allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must contain only JSON metadata and artifact paths") from error
    return json.loads(encoded)

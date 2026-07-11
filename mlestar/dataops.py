"""The skrub DataOps boundary used by generated MLE-STAR projects.

DataOps keeps the entire generated workflow lazy until the executor explicitly
supplies a run context.  The project itself provides only five component
functions in a local ``pipeline.py``; the framework owns the immutable run
context and component trace.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from hashlib import sha256
from pathlib import Path
from types import ModuleType
from typing import Any, Literal, Mapping

import skrub

from .contracts import COMPONENT_NAMES, TaskContract


ProjectMode = Literal["validate", "predict", "submit"]
COMPONENT_FUNCTIONS = {
    "data_loading": "load_data",
    "data_preparation": "prepare_data",
    "model": "build_model",
    "training": "train_model",
    "prediction": "predict_or_submit",
}
_PROJECT_MODULES: dict[Path, tuple[tuple[int, int], ModuleType]] = {}


class ProjectProtocolError(ValueError):
    """A generated project does not meet the DataOps component protocol."""


def build_dataops_plan(
    contract: TaskContract,
    project_dir: str | Path,
    mode: ProjectMode = "validate",
    *,
    workspace_root: str | Path | None = None,
):
    """Build, but do not evaluate, a five-component skrub DataOps DAG."""

    project, workspace = _resolve_project_and_workspace(project_dir, workspace_root)
    _validate_mode(mode)
    run_context = skrub.var("run_context")
    node = run_context
    for component in COMPONENT_NAMES:
        node = skrub.deferred(_run_component)(node, str(project), component)
        # DataOps are immutable: set_name returns the renamed copy.
        node = node.skb.set_name(component)
    return node


def run_dataops_project(
    contract: TaskContract,
    project_dir: str | Path,
    mode: ProjectMode = "validate",
    *,
    workspace_root: str | Path | None = None,
    include_full_report: bool = False,
) -> dict[str, Any]:
    """Evaluate a generated project once and save inspectable DataOps metadata.

    ``full_report`` evaluates a DataOps plan again. It is therefore opt-in and
    disabled for training by default, preventing a diagnostic request from
    accidentally repeating a model-training run.
    """

    project, workspace = _resolve_project_and_workspace(project_dir, workspace_root)
    plan = build_dataops_plan(contract, project, mode, workspace_root=workspace)
    initial_state = _initial_state(contract, project, workspace, mode)
    description = plan.skb.describe_steps()
    terminal = plan.skb.eval({"run_context": initial_state})
    report: dict[str, Any] = {
        "skrub_version": getattr(skrub, "__version__", "unknown"),
        "description": description,
        "full_report": {"status": "skipped_to_prevent_second_evaluation"},
    }
    if include_full_report:
        try:
            result = plan.skb.full_report(
                {"run_context": initial_state},
                open=False,
                output_dir=project / "dataops_full_report",
                overwrite=True,
            )
            report["full_report"] = {
                "status": "complete",
                "report_path": _json_value(result.get("report_path")),
                "error": _json_value(result.get("error")),
                "result": _json_value(result.get("result")),
            }
        except Exception as exc:  # diagnostics must not hide a successful primary run
            report["full_report"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    _write_json(project / "dataops_report.json", report)
    return terminal


def _initial_state(contract: TaskContract, project: Path, workspace: Path, mode: ProjectMode) -> dict[str, Any]:
    return {
        "run_context": {
            "contract": contract.to_dict(),
            "project_dir": str(project),
            "workspace_root": str(workspace),
        },
        "mode": mode,
        "component_trace": [],
        "artifacts": {},
        "previous": None,
    }


def _run_component(state: Mapping[str, Any], project_path: str, component: str) -> dict[str, Any]:
    """Invoke exactly one project component and enforce framework-owned fields."""

    if component not in COMPONENT_NAMES:
        raise ProjectProtocolError(f"Unsupported component {component!r}.")
    if not isinstance(state, Mapping):
        raise ProjectProtocolError("Each DataOps component must receive a mapping state.")
    # Generated functions receive their own copy. A shallow copy would let a
    # component mutate the framework-owned trace list in place before we can
    # compare it with the input state.
    before = copy.deepcopy(dict(state))
    _validate_framework_state(before)
    _validate_closed_project_context(before["run_context"], Path(project_path))
    module = _load_project_module(Path(project_path))
    function_name = COMPONENT_FUNCTIONS[component]
    function = getattr(module, function_name, None)
    if not callable(function):
        raise ProjectProtocolError(f"pipeline.py must define callable {function_name}(state).")
    result = function(copy.deepcopy(before))
    if not isinstance(result, Mapping):
        raise ProjectProtocolError(f"{component}() must return a mapping state.")
    after = dict(result)
    if after.get("run_context") != before["run_context"]:
        raise ProjectProtocolError(f"{component}() may not modify run_context.")
    if after.get("mode") != before["mode"]:
        raise ProjectProtocolError(f"{component}() may not modify mode.")
    if after.get("component_trace") != before["component_trace"]:
        raise ProjectProtocolError(f"{component}() may not modify component_trace.")
    after["component_trace"] = [*before["component_trace"], component]
    # Preserve the actual output of this node for the next component without
    # nesting the former ``previous`` value indefinitely.
    after["previous"] = {key: value for key, value in after.items() if key != "previous"}
    _ensure_json_compatible(after)
    return after


def _load_project_module(project: Path) -> ModuleType:
    pipeline = project / "pipeline.py"
    if not pipeline.is_file():
        raise ProjectProtocolError(f"Generated project is missing {pipeline}.")
    signature = (pipeline.stat().st_mtime_ns, pipeline.stat().st_size)
    cached = _PROJECT_MODULES.get(pipeline)
    if cached is not None and cached[0] == signature:
        return cached[1]
    module_name = f"_mlestar_project_{sha256(str(pipeline).encode('utf-8')).hexdigest()}"
    spec = importlib.util.spec_from_file_location(module_name, pipeline)
    if spec is None or spec.loader is None:
        raise ProjectProtocolError(f"Cannot load generated pipeline {pipeline}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _PROJECT_MODULES[pipeline] = (signature, module)
    return module


def _resolve_project_and_workspace(project_dir: str | Path, workspace_root: str | Path | None) -> tuple[Path, Path]:
    project = Path(project_dir).resolve()
    workspace = Path(workspace_root).resolve() if workspace_root is not None else project.parent.resolve()
    if not project.is_dir():
        raise FileNotFoundError(f"Project directory does not exist: {project}")
    try:
        project.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"Project directory {project} must be inside workspace root {workspace}.") from exc
    return project, workspace


def _validate_mode(mode: str) -> None:
    if mode not in {"validate", "predict", "submit"}:
        raise ValueError(f"Unsupported DataOps mode {mode!r}.")


def _validate_framework_state(state: dict[str, Any]) -> None:
    required = {"run_context", "mode", "component_trace", "artifacts", "previous"}
    missing = sorted(required.difference(state))
    if missing:
        raise ProjectProtocolError(f"Component state is missing required keys: {', '.join(missing)}.")
    if not isinstance(state["component_trace"], list):
        raise ProjectProtocolError("component_trace must be a list owned by the framework.")


def _validate_closed_project_context(run_context: object, project: Path) -> None:
    if not isinstance(run_context, Mapping):
        raise ProjectProtocolError("run_context must be a mapping owned by the framework.")
    if Path(str(run_context.get("project_dir") or "")).resolve() != project.resolve():
        raise ProjectProtocolError("run_context project_dir does not match the bound generated project.")
    workspace = Path(str(run_context.get("workspace_root") or "")).resolve()
    try:
        project.resolve().relative_to(workspace)
    except ValueError as exc:
        raise ProjectProtocolError("run_context workspace_root does not contain the generated project.") from exc


def _ensure_json_compatible(value: Any) -> None:
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ProjectProtocolError("Component output must be JSON-compatible.") from exc


def _json_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

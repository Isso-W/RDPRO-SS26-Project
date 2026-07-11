"""Private entry point used to evaluate one generated MLE-STAR project.

The outer executor deliberately starts a fresh Python process for generated
code.  This module is the small, deterministic bridge in that process: it
loads the project entry point, evaluates the DataOps plan exactly once, and
emits a compact terminal result.  In particular, project log messages can
never corrupt the JSON protocol consumed by :mod:`mlestar.executor`.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence

from .contracts import TaskContract
from .dataops import run_dataops_project


TERMINAL_KEYS = frozenset(
    {
        "metric_name",
        "metric_value",
        "oof_path",
        "prediction_path",
        "submission_path",
        "component_trace",
    }
)


class GeneratedRunnerError(RuntimeError):
    """The generated project did not honour the execution protocol."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one generated MLE-STAR DataOps project.")
    parser.add_argument("--project", required=True, help="generated project directory")
    parser.add_argument("--task-contract", required=True, help="temporary TaskContract JSON inside the project")
    parser.add_argument("--workspace-root", required=True, help="run workspace containing the generated project")
    parser.add_argument("--mode", choices=("validate", "predict", "submit"), default="validate")
    return parser


def _resolve_contained(path: str | Path, root: str | Path, *, description: str) -> Path:
    resolved = Path(path).resolve()
    root_path = Path(root).resolve()
    try:
        resolved.relative_to(root_path)
    except ValueError as exc:
        raise GeneratedRunnerError(f"{description} must be inside {root_path}.") from exc
    return resolved


def _load_contract(contract_path: Path) -> TaskContract:
    try:
        payload = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GeneratedRunnerError(f"Cannot read task contract: {exc}") from exc
    if not isinstance(payload, dict):
        raise GeneratedRunnerError("Task contract must be a JSON object.")
    try:
        return TaskContract.from_dict(payload)
    except (TypeError, ValueError) as exc:
        raise GeneratedRunnerError(f"Invalid task contract: {exc}") from exc


def _load_project_module(project: Path) -> ModuleType:
    """Load the sole generated entry point, never a project-selected runner."""

    entrypoint = project / "project.py"
    if not entrypoint.is_file():
        raise GeneratedRunnerError(f"Generated project is missing {entrypoint.name}.")
    spec = importlib.util.spec_from_file_location("_mlestar_generated_project", entrypoint)
    if spec is None or spec.loader is None:
        raise GeneratedRunnerError(f"Cannot load generated project {entrypoint}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _as_optional_path(value: object, key: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise GeneratedRunnerError(f"Terminal field {key!r} must be a non-empty string or null.")
    return value


def _terminal_result(value: object, contract: TaskContract) -> dict[str, Any]:
    """Validate and reduce a project result to the stable wire schema."""

    if not isinstance(value, Mapping):
        raise GeneratedRunnerError("DataOps evaluation must return a mapping terminal result.")
    keys = set(value)
    missing = sorted(TERMINAL_KEYS.difference(keys))
    # DataOps carries framework state and component artifacts alongside the
    # terminal fields.  Its implementation details are deliberately not part
    # of the subprocess protocol: we reduce the result to the exact public
    # schema below.  Only a missing public field is therefore an error here.
    if missing:
        raise GeneratedRunnerError("Invalid terminal result schema: missing " + ", ".join(missing) + ".")

    metric_name = value["metric_name"]
    if not isinstance(metric_name, str) or not metric_name.strip():
        raise GeneratedRunnerError("Terminal field 'metric_name' must be a non-empty string.")
    if metric_name != contract.metric.name:
        raise GeneratedRunnerError(
            f"Terminal metric {metric_name!r} does not match the task metric {contract.metric.name!r}."
        )

    metric_value = value["metric_value"]
    if isinstance(metric_value, bool) or not isinstance(metric_value, (int, float)):
        raise GeneratedRunnerError("Terminal field 'metric_value' must be a finite number.")
    metric = float(metric_value)
    if not math.isfinite(metric):
        raise GeneratedRunnerError("Terminal field 'metric_value' must be a finite number.")

    oof_path = _as_optional_path(value["oof_path"], "oof_path")
    if oof_path is None:
        raise GeneratedRunnerError("Terminal field 'oof_path' must be a non-empty string.")
    trace = value["component_trace"]
    if not isinstance(trace, list) or any(not isinstance(item, str) for item in trace):
        raise GeneratedRunnerError("Terminal field 'component_trace' must be a list of strings.")
    if tuple(trace) != contract.component_names:
        raise GeneratedRunnerError("Terminal component_trace must contain the task components exactly once in order.")

    return {
        "metric_name": metric_name,
        "metric_value": metric,
        "oof_path": oof_path,
        "prediction_path": _as_optional_path(value["prediction_path"], "prediction_path"),
        "submission_path": _as_optional_path(value["submission_path"], "submission_path"),
        "component_trace": trace,
    }


def _evaluate(project: Path, task_contract: Path, workspace_root: Path, mode: str) -> dict[str, Any]:
    project = _resolve_contained(project, workspace_root, description="Project directory")
    if not project.is_dir():
        raise GeneratedRunnerError(f"Project directory does not exist: {project}")
    task_contract = _resolve_contained(task_contract, project, description="Task contract")
    if not task_contract.is_file():
        raise GeneratedRunnerError(f"Task contract does not exist: {task_contract}")
    contract = _load_contract(task_contract)
    # New projects use the generated pipeline.py component protocol. Keep the
    # project.py bridge below for projects produced during the earlier Module 4
    # migration, so executor receipts remain backward compatible.
    if (project / "pipeline.py").is_file():
        return _terminal_result(
            run_dataops_project(contract, project, mode, workspace_root=workspace_root),
            contract,
        )
    module = _load_project_module(project)
    builder = getattr(module, "build_dataops_plan", None)
    if not callable(builder):
        raise GeneratedRunnerError("project.py must define build_dataops_plan(config_path, mode).")
    plan = builder(str(task_contract), mode)
    skb = getattr(plan, "skb", None)
    evaluate = getattr(skb, "eval", None)
    if not callable(evaluate):
        raise GeneratedRunnerError("build_dataops_plan() must return an object exposing skb.eval().")
    return _terminal_result(evaluate(), contract)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    captured_stdout = io.StringIO()
    try:
        # Generated code (and third-party estimators) often prints progress.
        # Preserve that diagnostic stream on stderr while reserving stdout for
        # one protocol JSON object.
        with contextlib.redirect_stdout(captured_stdout):
            terminal = _evaluate(
                Path(args.project),
                Path(args.task_contract),
                Path(args.workspace_root),
                args.mode,
            )
    except Exception as exc:  # the parent converts every generated-code failure into a receipt
        log = captured_stdout.getvalue()
        if log:
            print(log, file=sys.stderr, end="" if log.endswith("\n") else "\n")
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, sort_keys=True), flush=True)
        return 1

    log = captured_stdout.getvalue()
    if log:
        print(log, file=sys.stderr, end="" if log.endswith("\n") else "\n")
    print(json.dumps(terminal, sort_keys=True, separators=(",", ":"), allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through subprocess execution.
    raise SystemExit(main())

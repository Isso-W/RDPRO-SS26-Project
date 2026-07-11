"""Constrained subprocess execution for generated MLE-STAR projects.

Generated programs never run in the orchestrator interpreter.  They receive a
one-shot task-contract file and communicate their terminal metric through the
JSON-only protocol implemented by :mod:`mlestar.generated_runner`.
"""

from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Literal, Mapping
from uuid import uuid4

from .contracts import ExperimentReceipt, TaskContract
from .generated_runner import TERMINAL_KEYS


ProjectMode = Literal["validate", "predict", "submit"]
DEFAULT_TIMEOUT_SECONDS = 30 * 60
_STDOUT_FILE = "executor_stdout.txt"
_STDERR_FILE = "executor_stderr.txt"
_RECEIPT_FILE = "execution_receipt.json"


def execute_project(
    project_dir: str | Path,
    contract: TaskContract,
    *,
    workspace_root: str | Path,
    mode: ProjectMode = "validate",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    candidate_id: str | None = None,
    experiment_id: str | None = None,
    component: str = "prediction",
    stage: str = "initial",
    data_fingerprint: str = "unknown",
    parent_experiment_id: str | None = None,
    fold: int | None = None,
    seed: int | None = None,
) -> ExperimentReceipt:
    """Run a generated project and always persist an :class:`ExperimentReceipt`.

    Invalid caller paths are rejected before any subprocess is started.  A
    project import, training, timeout, or terminal-protocol error is instead a
    failed receipt, allowing the search/refinement workflow to continue.
    """

    project, workspace = _resolve_project_and_workspace(project_dir, workspace_root)
    if not isinstance(contract, TaskContract):
        raise TypeError("contract must be a TaskContract.")
    if mode not in {"validate", "predict", "submit"}:
        raise ValueError(f"Unsupported execution mode {mode!r}.")
    if not isinstance(timeout_seconds, (int, float)) or isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be a positive number.")
    if component not in contract.component_names:
        raise ValueError(f"component must be one of the task components, got {component!r}.")
    if not isinstance(stage, str) or not stage.strip():
        raise ValueError("stage must be a non-empty string.")
    if not isinstance(data_fingerprint, str) or not data_fingerprint.strip():
        raise ValueError("data_fingerprint must be a non-empty string.")

    candidate = candidate_id or project.name
    if not candidate.strip():
        raise ValueError("candidate_id must be non-empty.")
    receipt_id = experiment_id or f"{candidate}-{uuid4().hex}"
    source_sha = _project_sha256(project)
    started = time.monotonic()
    contract_path = _write_temporary_contract(project, contract)
    stdout = ""
    stderr = ""

    try:
        command = [
            sys.executable,
            "-m",
            "mlestar.generated_runner",
            "--project",
            str(project),
            "--task-contract",
            str(contract_path),
            "--workspace-root",
            str(workspace),
            "--mode",
            mode,
        ]
        completed = subprocess.run(
            command,
            cwd=project,
            env=_sanitized_environment(project),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=float(timeout_seconds),
            check=False,
        )
        stdout = completed.stdout
        stderr = completed.stderr
        elapsed = time.monotonic() - started
        if completed.returncode != 0:
            receipt = _failed_receipt(
                receipt_id,
                candidate,
                component,
                stage,
                contract,
                elapsed,
                source_sha,
                data_fingerprint,
                _process_error(completed.returncode, stderr, stdout),
                parent_experiment_id=parent_experiment_id,
                fold=fold,
                seed=seed,
            )
        else:
            try:
                terminal = _parse_terminal_json(stdout, contract, workspace)
            except ValueError as exc:
                receipt = _failed_receipt(
                    receipt_id,
                    candidate,
                    component,
                    stage,
                    contract,
                    elapsed,
                    source_sha,
                    data_fingerprint,
                    str(exc),
                    parent_experiment_id=parent_experiment_id,
                    fold=fold,
                    seed=seed,
                )
            else:
                receipt = ExperimentReceipt(
                    experiment_id=receipt_id,
                    candidate_id=candidate,
                    component=component,
                    stage=stage,
                    metric_name=terminal["metric_name"],
                    greater_is_better=contract.metric.greater_is_better,
                    metric_value=terminal["metric_value"],
                    elapsed_seconds=elapsed,
                    status="success",
                    code_sha256=source_sha,
                    data_fingerprint=data_fingerprint,
                    oof_path=terminal["oof_path"],
                    prediction_path=terminal["prediction_path"],
                    submission_path=terminal["submission_path"],
                    parent_experiment_id=parent_experiment_id,
                    fold=fold,
                    seed=seed,
                )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        stdout = _coerce_process_output(exc.stdout)
        stderr = _coerce_process_output(exc.stderr)
        receipt = _failed_receipt(
            receipt_id,
            candidate,
            component,
            stage,
            contract,
            elapsed,
            source_sha,
            data_fingerprint,
            f"Execution timed out after {timeout_seconds} seconds.",
            parent_experiment_id=parent_experiment_id,
            fold=fold,
            seed=seed,
        )
    except OSError as exc:
        elapsed = time.monotonic() - started
        receipt = _failed_receipt(
            receipt_id,
            candidate,
            component,
            stage,
            contract,
            elapsed,
            source_sha,
            data_fingerprint,
            f"Could not start generated runner: {type(exc).__name__}: {exc}",
            parent_experiment_id=parent_experiment_id,
            fold=fold,
            seed=seed,
        )
    finally:
        contract_path.unlink(missing_ok=True)

    _write_execution_artifacts(project, stdout, stderr, receipt)
    return receipt


def _resolve_project_and_workspace(project_dir: str | Path, workspace_root: str | Path) -> tuple[Path, Path]:
    project = Path(project_dir).resolve()
    workspace = Path(workspace_root).resolve()
    if not workspace.is_dir():
        raise FileNotFoundError(f"Workspace root does not exist: {workspace}")
    if not project.is_dir():
        raise FileNotFoundError(f"Project directory does not exist: {project}")
    try:
        project.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"Project directory {project} must be inside workspace root {workspace}.") from exc
    return project, workspace


def _project_sha256(project: Path) -> str:
    """Hash the generated entry point that the runner is permitted to load."""

    entrypoint = project / "pipeline.py"
    if not entrypoint.is_file():
        entrypoint = project / "project.py"
    try:
        content = entrypoint.read_bytes()
    except OSError:
        content = b"<missing mlestar pipeline.py>"
    return sha256(content).hexdigest()


def _write_temporary_contract(project: Path, contract: TaskContract) -> Path:
    """Persist a short-lived contract in the project with owner-only permissions."""

    path = project / f".mlestar-task-contract-{uuid4().hex}.json"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(contract.to_dict(), handle, sort_keys=True)
            handle.write("\n")
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _sanitized_environment(project: Path) -> dict[str, str]:
    """Return an allowlisted environment with no inherited credentials.

    The source checkout is placed on ``PYTHONPATH`` so this works from a
    source tree as well as from an installed editable package.  All writable
    home and temporary locations stay within the generated project.
    """

    package_root = Path(__file__).resolve().parent.parent
    home = project / ".mlestar-home"
    temp = project / ".mlestar-tmp"
    home.mkdir(exist_ok=True)
    temp.mkdir(exist_ok=True)
    return {
        "PATH": os.pathsep.join((str(Path(sys.executable).resolve().parent), "/usr/bin", "/bin")),
        "PYTHONPATH": str(package_root),
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": str(home),
        "TMPDIR": str(temp),
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
    }


def _parse_terminal_json(stdout: str, contract: TaskContract, workspace: Path) -> dict[str, Any]:
    """Parse exactly one runner JSON object and validate all receipt fields."""

    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Runner stdout must be exactly one JSON object: {exc.msg}.") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Runner stdout must be exactly one JSON object.")
    keys = set(payload)
    if keys != TERMINAL_KEYS:
        missing = sorted(TERMINAL_KEYS.difference(keys))
        extra = sorted(keys.difference(TERMINAL_KEYS))
        parts: list[str] = []
        if missing:
            parts.append("missing " + ", ".join(missing))
        if extra:
            parts.append("unexpected " + ", ".join(extra))
        raise ValueError("Runner terminal schema is invalid: " + "; ".join(parts) + ".")
    metric_name = payload["metric_name"]
    metric_value = payload["metric_value"]
    if metric_name != contract.metric.name or not isinstance(metric_name, str):
        raise ValueError("Runner metric_name does not match the task contract.")
    if isinstance(metric_value, bool) or not isinstance(metric_value, (int, float)):
        raise ValueError("Runner metric_value must be a finite number.")
    metric = float(metric_value)
    if not _is_finite(metric):
        raise ValueError("Runner metric_value must be a finite number.")
    oof_path = _contained_artifact_path(payload["oof_path"], workspace, "oof_path", required=True)
    prediction_path = _contained_artifact_path(payload["prediction_path"], workspace, "prediction_path")
    submission_path = _contained_artifact_path(payload["submission_path"], workspace, "submission_path")
    trace = payload["component_trace"]
    if not isinstance(trace, list) or tuple(trace) != contract.component_names:
        raise ValueError("Runner component_trace must contain task components exactly once in order.")
    if any(not isinstance(item, str) for item in trace):
        raise ValueError("Runner component_trace must be a list of strings.")
    return {
        "metric_name": metric_name,
        "metric_value": metric,
        "oof_path": oof_path,
        "prediction_path": prediction_path,
        "submission_path": submission_path,
        "component_trace": trace,
    }


def _is_finite(value: float) -> bool:
    # Avoid importing the runner's private validation helpers into the parent
    # process protocol implementation.
    return value != float("inf") and value != float("-inf") and value == value


def _contained_artifact_path(value: object, workspace: Path, field: str, *, required: bool = False) -> str | None:
    if value is None:
        if required:
            raise ValueError(f"Runner {field} must be a non-empty string.")
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Runner {field} must be a non-empty string or null.")
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (workspace / candidate).resolve()
    try:
        resolved.relative_to(workspace)
    except ValueError as exc:
        raise ValueError(f"Runner {field} must stay inside the workspace root.") from exc
    return value


def _failed_receipt(
    experiment_id: str,
    candidate_id: str,
    component: str,
    stage: str,
    contract: TaskContract,
    elapsed_seconds: float,
    code_sha256: str,
    data_fingerprint: str,
    error_text: str,
    *,
    parent_experiment_id: str | None,
    fold: int | None,
    seed: int | None,
) -> ExperimentReceipt:
    return ExperimentReceipt(
        experiment_id=experiment_id,
        candidate_id=candidate_id,
        component=component,
        stage=stage,
        metric_name=contract.metric.name,
        greater_is_better=contract.metric.greater_is_better,
        metric_value=None,
        elapsed_seconds=elapsed_seconds,
        status="failed",
        code_sha256=code_sha256,
        data_fingerprint=data_fingerprint,
        oof_path=None,
        parent_experiment_id=parent_experiment_id,
        fold=fold,
        seed=seed,
        error_text=error_text,
    )


def _coerce_process_output(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _process_error(returncode: int, stderr: str, stdout: str) -> str:
    details = stderr.strip() or stdout.strip()
    suffix = f": {details}" if details else ""
    return f"Generated runner exited with status {returncode}{suffix}"


def _write_execution_artifacts(project: Path, stdout: str, stderr: str, receipt: ExperimentReceipt) -> None:
    (project / _STDOUT_FILE).write_text(stdout, encoding="utf-8")
    (project / _STDERR_FILE).write_text(stderr, encoding="utf-8")
    (project / _RECEIPT_FILE).write_text(
        json.dumps(receipt.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

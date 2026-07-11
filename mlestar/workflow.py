"""Runnable evidence -> generated DataOps project -> receipt workflow."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .audits import audit_project, has_audit_errors
from .contracts import ExperimentReceipt, SearchEvidence, TaskContract
from .dataset import inspect_dataset
from .executor import execute_project
from .generation import GenerationProvider, generate_project
from .search import SearchProvider, retrieve_model_evidence


def run_mlestar(
    *,
    task_path: str | Path,
    data_root: str | Path,
    run_dir: str | Path,
    search: SearchProvider | None = None,
    generator: GenerationProvider | None = None,
    initial_candidates: int = 1,
    plan_only: bool = False,
    timeout_seconds: float = 30 * 60,
) -> dict[str, Any]:
    """Execute a baseline candidate wave and persist every decision artifact."""

    if initial_candidates < 1:
        raise ValueError("initial_candidates must be positive.")
    task = _load_task(task_path)
    root = Path(run_dir).resolve()
    root.mkdir(parents=True, exist_ok=True)
    inventory = inspect_dataset(data_root, output_path=root / "inventory.json")
    _write_json(root / "task.json", task.to_dict())
    evidence = _retrieve_evidence(task, search, root)
    candidates: list[dict[str, Any]] = []
    receipts: list[ExperimentReceipt] = []
    all_findings: list[dict[str, Any]] = []
    for index in range(initial_candidates):
        candidate_id = f"candidate_{index + 1}"
        project = root / "projects" / candidate_id
        generated = generate_project(project, task, candidate_id=candidate_id, provider=generator, evidence=evidence)
        candidates.append({
            "candidate": generated.candidate.to_dict(), "used_fallback": generated.used_fallback,
            "fallback_reason": generated.fallback_reason, "rationale": generated.rationale,
            "assumptions": list(generated.assumptions),
        })
        findings = audit_project(project, task, inventory, output_path=project / "audit.jsonl")
        all_findings.extend({"candidate_id": candidate_id, **finding.to_dict()} for finding in findings)
        if plan_only:
            continue
        if has_audit_errors(findings):
            receipts.append(_audit_failed_receipt(candidate_id, generated.candidate.code_sha256, task, inventory.fingerprint))
            continue
        receipts.append(execute_project(
            project, task, workspace_root=root, timeout_seconds=timeout_seconds,
            candidate_id=candidate_id, experiment_id=f"{candidate_id}-initial", data_fingerprint=inventory.fingerprint,
        ))
    _write_json(root / "candidates.json", candidates)
    _write_jsonl(root / "audit.jsonl", all_findings)
    _write_jsonl(root / "experiments.jsonl", [item.to_dict() for item in receipts])
    successful = [receipt for receipt in receipts if receipt.success]
    best = _select_best(successful, task) if successful else None
    report = {
        "status": "planned" if plan_only else "success" if best else "failed",
        "task_id": task.task_id, "run_dir": str(root), "data_fingerprint": inventory.fingerprint,
        "search_evidence_count": len(evidence), "candidate_count": len(candidates),
        "best_experiment": best.to_dict() if best else None, "experiment_count": len(receipts),
    }
    _write_json(root / "final_report.json", report)
    return report


def _load_task(path: str | Path) -> TaskContract:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Task contract JSON must be an object.")
    return TaskContract.from_dict(payload)


def _retrieve_evidence(task: TaskContract, search: SearchProvider | None, root: Path) -> list[SearchEvidence]:
    if search is None:
        _write_json(root / "search_evidence.json", [])
        return []
    return retrieve_model_evidence(task, search, output_path=root / "search_evidence.json")


def _audit_failed_receipt(candidate_id: str, code_sha256: str, task: TaskContract, fingerprint: str) -> ExperimentReceipt:
    return ExperimentReceipt(
        experiment_id=f"{candidate_id}-audit-blocked", candidate_id=candidate_id, component="prediction",
        stage="initial", metric_name=task.metric.name, greater_is_better=task.metric.greater_is_better,
        metric_value=None, elapsed_seconds=0.0, status="failed", code_sha256=code_sha256,
        data_fingerprint=fingerprint, oof_path=None, error_text="Execution blocked by audit errors.",
    )


def _select_best(receipts: Sequence[ExperimentReceipt], task: TaskContract) -> ExperimentReceipt:
    if task.metric.greater_is_better:
        return max(receipts, key=lambda item: float(item.metric_value))
    return min(receipts, key=lambda item: float(item.metric_value))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows), encoding="utf-8")

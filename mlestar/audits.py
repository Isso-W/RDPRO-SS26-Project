"""Static safety audits for generated MLE-STAR projects."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Iterable

from .contracts import AuditFinding, DatasetInventory, TaskContract


def audit_project(
    project_dir: str | Path,
    task: TaskContract,
    inventory: DatasetInventory | dict[str, Any],
    *,
    output_path: str | Path | None = None,
) -> list[AuditFinding]:
    """Audit a project before execution and persist JSONL findings."""

    del task
    project = Path(project_dir)
    source_path = project / "pipeline.py"
    source = source_path.read_text(encoding="utf-8") if source_path.is_file() else ""
    findings: list[AuditFinding] = []
    if not source:
        findings.append(_finding("project_source", "error", "pipeline.py is missing.", "missing_pipeline", "error"))
    else:
        findings.extend(_audit_source(source))
        findings.extend(_audit_data_usage(source, inventory))
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(item.to_dict(), sort_keys=True) + "\n" for item in findings), encoding="utf-8")
    return findings


def has_audit_errors(findings: Iterable[AuditFinding]) -> bool:
    return any(finding.severity == "error" for finding in findings)


def _audit_source(source: str) -> list[AuditFinding]:
    findings: list[AuditFinding] = []
    try:
        tree = ast.parse(source, filename="pipeline.py")
    except SyntaxError as exc:
        return [_finding("syntax", "error", str(exc), "syntax_error", "error")]
    text = source.lower().replace(" ", "")
    if "pd.concat([train,test]" in text or "pd.concat((train,test)" in text:
        findings.append(_finding("leakage", "error", "Train and test are concatenated before preprocessing.", "test_statistics", "error"))
    if any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in {"eval", "exec"} for node in ast.walk(tree)):
        findings.append(_finding("unsafe_code", "error", "Generated source calls eval or exec.", "dynamic_execution", "error"))
    imported = {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
    imported.update(node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module)
    if any(name == "subprocess" or name.startswith("subprocess.") for name in imported):
        findings.append(_finding("unsafe_code", "error", "Generated source imports subprocess.", "subprocess_import", "error"))
    if "random_state" not in text and "manual_seed" not in text and "seed" not in text:
        findings.append(_finding("reproducibility", "warning", "No explicit random seed was found.", "missing_seed", "warning"))
    return findings


def _audit_data_usage(source: str, inventory: DatasetInventory | dict[str, Any]) -> list[AuditFinding]:
    payload = inventory.to_dict() if isinstance(inventory, DatasetInventory) else inventory
    files = payload.get("files") or []
    paths = [str(item.get("path") if isinstance(item, dict) else item) for item in files]
    referenced = source.replace("\\", "/")
    findings: list[AuditFinding] = []
    for path in paths:
        name = Path(path).name
        # CSV/image files are only mandatory when their name implies an input
        # source other than the expected test sample/submission artifact.
        if not name or name in {"sample_submission.csv", "test.csv"}:
            continue
        if ("mask" in path.lower() or path.lower().endswith((".csv", ".jpg", ".jpeg", ".png"))) and name not in referenced and path not in referenced:
            findings.append(_finding("data_usage", "warning", f"Inventory data source is not referenced: {path}", "unused_data_source", "warning"))
    return findings


def _finding(audit_name: str, status: str, message: str, code: str, severity: str) -> AuditFinding:
    return AuditFinding(audit_name=audit_name, status=status, message=message, code=code, severity=severity)

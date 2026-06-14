"""Compact experiment report generation."""

from __future__ import annotations

import json
from pathlib import Path


def generate_experiment_report_service(
    context,
    report: dict,
    output_path: str,
) -> dict:
    destination = Path(output_path).expanduser().resolve()
    allowed_roots = [context.workspace_root, context.store.root.resolve()]
    if not any(
        destination == root or root in destination.parents
        for root in allowed_roots
    ):
        raise ValueError("Report path must be inside the workspace or knowledge-base root.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"report_path": str(destination), "bytes": destination.stat().st_size}

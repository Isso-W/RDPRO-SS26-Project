from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from .schemas import ExecutionResult, GeneratedProject


class GeneratedProjectExecutor:
    def run(
        self,
        project: GeneratedProject,
        spec_path: str | Path,
        run_dir: str | Path,
        execution_mode: str = "simulate",
    ) -> ExecutionResult:
        run_path = Path(run_dir).resolve()
        run_path.mkdir(parents=True, exist_ok=True)
        spec_file = Path(spec_path).resolve()

        command = [
            sys.executable,
            project.train_script_path,
            "--manifest",
            project.manifest_path,
            "--spec",
            str(spec_file),
            "--output-dir",
            str(run_path),
            "--execution-mode",
            execution_mode,
        ]
        start = time.time()
        result = subprocess.run(
            command,
            cwd=Path(project.project_dir).resolve(),
            capture_output=True,
            text=True,
            check=False,
        )
        runtime_sec = round(time.time() - start, 4)
        payload = self._extract_payload(result.stdout)
        if payload is None:
            return ExecutionResult(
                status="failed",
                stdout=result.stdout,
                stderr=result.stderr or "Could not parse training result payload",
                runtime_sec=runtime_sec,
                primary_metric_name="unknown",
                primary_metric_value=None,
                checkpoint_path=None,
                artifacts_path=str(run_path),
            )
        return ExecutionResult(
            status=payload.get("status", "failed"),
            stdout=result.stdout,
            stderr=payload.get("stderr", result.stderr),
            runtime_sec=payload.get("runtime_sec", runtime_sec),
            primary_metric_name=payload.get("primary_metric_name", "unknown"),
            primary_metric_value=payload.get("primary_metric_value"),
            checkpoint_path=payload.get("checkpoint_path"),
            artifacts_path=payload.get("artifacts_path", str(run_path)),
        )

    @staticmethod
    def _extract_payload(stdout: str) -> dict[str, object] | None:
        for line in reversed(stdout.splitlines()):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and "status" in value:
                return value
        return None

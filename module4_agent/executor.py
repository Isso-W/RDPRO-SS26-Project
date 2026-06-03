"""Subprocess executor for generated Module 4 projects."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from .schemas import CommandResult, GeneratedFiles, SmokeResult


def write_generated_files(generated: GeneratedFiles, output_dir: str | Path) -> list[Path]:
    """Write generated files into an output directory."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for relative_path, content in generated.files.items():
        path = output_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        written.append(path)
    return written


def run_command(command: list[str], cwd: str | Path, timeout: int = 60) -> CommandResult:
    """Run one subprocess command with captured output."""

    start = time.time()
    try:
        env = os.environ.copy()
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        env.setdefault("KMP_INIT_AT_FORK", "FALSE")
        env.setdefault("KMP_AFFINITY", "disabled")
        env.setdefault("OMP_PROC_BIND", "FALSE")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return CommandResult(
            command=command,
            return_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            runtime_sec=round(time.time() - start, 4),
            timed_out=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
        stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace")
        return CommandResult(
            command=command,
            return_code=124,
            stdout=stdout,
            stderr=stderr or f"Timed out after {timeout} seconds.",
            runtime_sec=round(time.time() - start, 4),
            timed_out=True,
        )


def run_smoke(output_dir: str | Path, timeout: int = 60) -> SmokeResult:
    """Run generated single-config and all-candidate smoke drivers."""

    start = time.time()
    output_path = Path(output_dir)
    commands = [
        [sys.executable, "run.py"],
        [sys.executable, "run_experiments.py"],
    ]
    results: list[CommandResult] = []
    for command in commands:
        result = run_command(command, cwd=output_path, timeout=timeout)
        results.append(result)
        if not result.success:
            break
    return SmokeResult(
        success=all(result.success for result in results),
        command_results=results,
        runtime_sec=round(time.time() - start, 4),
    )

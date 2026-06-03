"""Subprocess executor for generated Module 4 projects."""

from __future__ import annotations

import contextlib
import io
import os
import runpy
import subprocess
import sys
import time
import traceback
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
        env.setdefault("KMP_USE_SHM", "0")
        env.setdefault("KMP_BLOCKTIME", "0")
        env.setdefault("OMP_WAIT_POLICY", "PASSIVE")
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
        if completed.returncode != 0 and _is_openmp_shm_error(completed.stderr):
            return _run_python_script_inprocess(command, cwd, start)
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


def _is_openmp_shm_error(stderr: str) -> bool:
    return "OMP: Error #179" in stderr or "Can't open SHM failed" in stderr


def _run_python_script_inprocess(command: list[str], cwd: str | Path, start: float) -> CommandResult:
    if len(command) < 2 or not command[1].endswith(".py"):
        return CommandResult(
            command=command,
            return_code=1,
            stdout="",
            stderr="OpenMP SHM fallback only supports Python script commands.",
            runtime_sec=round(time.time() - start, 4),
        )

    cwd_path = Path(cwd)
    script_path = cwd_path / command[1]
    saved_cwd = Path.cwd()
    saved_argv = list(sys.argv)
    saved_path = list(sys.path)
    module_names = ("model", "utils", "smoke_data", "train", "evaluate", "infer", "run", "run_experiments")
    saved_modules = {name: sys.modules.get(name) for name in module_names}
    stdout = io.StringIO()
    stderr = io.StringIO()
    try:
        for name in module_names:
            sys.modules.pop(name, None)
        os.chdir(cwd_path)
        sys.argv = [command[1], *command[2:]]
        sys.path.insert(0, str(cwd_path))
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            runpy.run_path(str(script_path), run_name="__main__")
        return_code = 0
    except Exception:
        return_code = 1
        stderr.write(traceback.format_exc())
    finally:
        os.chdir(saved_cwd)
        sys.argv = saved_argv
        sys.path[:] = saved_path
        for name, saved in saved_modules.items():
            if saved is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = saved
    return CommandResult(
        command=command,
        return_code=return_code,
        stdout=stdout.getvalue(),
        stderr=stderr.getvalue(),
        runtime_sec=round(time.time() - start, 4),
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

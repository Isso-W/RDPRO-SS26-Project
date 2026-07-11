"""Execute adapter commands with credentials removed from the child process."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import os
from pathlib import Path
import subprocess


_EXACT_SECRET_NAMES = frozenset(
    {
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "HF_TOKEN",
        "HUGGINGFACEHUB_API_TOKEN",
        "KAGGLE_API_TOKEN",
        "KAGGLE_KEY",
        "KAGGLE_USERNAME",
        "OPENAI_API_KEY",
    }
)
_SECRET_SUFFIXES = (
    "_ACCESS_TOKEN",
    "_API_KEY",
    "_API_TOKEN",
    "_PRIVATE_KEY",
    "_SECRET",
    "_SECRET_KEY",
    "_SESSION_TOKEN",
    "_TOKEN",
)


def is_secret_environment_name(name: str) -> bool:
    """Return whether an environment variable may carry a provider credential."""

    normalized = name.upper()
    return (
        normalized in _EXACT_SECRET_NAMES
        or normalized.startswith("KAGGLE_")
        or normalized.endswith(_SECRET_SUFFIXES)
    )


def sanitized_child_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Copy the ambient environment, applying overrides without any secrets.

    The returned environment intentionally drops Kaggle credentials, OpenAI
    credentials, and generic provider-key/token/secret variables.  This keeps
    adapter subprocesses offline and ensures credentials remain at the CLI or
    notebook boundary rather than in an experiment worker.
    """

    merged = dict(os.environ)
    if environment is not None:
        merged.update(environment)

    cleaned: dict[str, str] = {}
    for name, value in merged.items():
        if not isinstance(name, str) or not isinstance(value, str):
            raise TypeError("child environment names and values must be strings")
        if not is_secret_environment_name(name):
            cleaned[name] = value
    return cleaned


def clean_child_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Alias for :func:`sanitized_child_environment` for explicit call sites."""

    return sanitized_child_environment(environment)


def execute_adapter(
    command: Sequence[str | Path],
    run_dir: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
    timeout: float | None = None,
    check: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run one adapter command in ``run_dir`` with a credential-free environment.

    A shell command string is rejected so callers cannot accidentally bypass
    argument handling.  Standard output and standard error are captured for an
    ``ExperimentReceipt`` to persist later without retaining child objects.
    """

    if isinstance(command, (str, bytes)) or not command:
        raise TypeError("command must be a non-empty sequence of executable arguments")
    arguments = [str(argument) for argument in command]
    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"run directory must exist and be a directory: {root}")

    return subprocess.run(
        arguments,
        cwd=root,
        env=sanitized_child_environment(environment),
        text=True,
        capture_output=True,
        timeout=timeout,
        check=check,
    )

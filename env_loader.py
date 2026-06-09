from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path | None = None) -> bool:
    """Load simple KEY=VALUE pairs from .env without overwriting existing env vars."""

    env_path = Path(path) if path is not None else Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return False

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)
    return True

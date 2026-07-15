"""Safe, run-local persistence helpers.

All code-generated paths go through :class:`RunArtifacts`, so a candidate
cannot accidentally (or deliberately) write outside its allocated run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class RunArtifacts:
    """Resolve and persist artifacts below exactly one run directory."""

    def __init__(self, run_directory: str | Path) -> None:
        self.run_directory = Path(run_directory).expanduser().resolve()

    def resolve(self, relative_path: str | Path) -> Path:
        """Return a contained artifact path or reject a directory escape."""

        requested = Path(relative_path)
        destination = (
            requested.resolve()
            if requested.is_absolute()
            else (self.run_directory / requested).resolve()
        )
        try:
            destination.relative_to(self.run_directory)
        except ValueError as error:
            raise ValueError(
                f"artifact path must remain inside run directory: {relative_path!s}"
            ) from error
        return destination

    def relative(self, path: str | Path) -> str:
        """Return a portable run-relative path after enforcing containment."""

        return self.resolve(path).relative_to(self.run_directory).as_posix()

    def mkdir(self, relative_path: str | Path = ".") -> Path:
        directory = self.resolve(relative_path)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def write_json(self, relative_path: str | Path, value: Any) -> Path:
        destination = self.resolve(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
        return destination

    def write_csv(self, relative_path: str | Path, frame: Any) -> Path:
        """Persist a dataframe-like object inside the run without index leakage."""

        destination = self.resolve(relative_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
        return destination

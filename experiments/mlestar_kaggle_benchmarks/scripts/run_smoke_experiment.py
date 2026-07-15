#!/usr/bin/env python3
"""Run the deterministic, offline synthetic benchmark and record provenance."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np

from mlestar import __version__
from mlestar.experiment import compare


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "examples" / "synthetic_leaf"
SMOKE_SEED = 13
BENCHMARK = "leaf_classification"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _entries(paths: Sequence[Path], *, relative_to: Path) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(paths, key=lambda item: item.relative_to(relative_to).as_posix()):
        entries.append(
            {
                "path": path.relative_to(relative_to).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return entries


def _entries_hash(entries: Sequence[dict[str, Any]]) -> str:
    payload = json.dumps(list(entries), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _data_manifest() -> dict[str, Any]:
    files = [path for path in DATA_ROOT.rglob("*") if path.is_file()]
    entries = _entries(files, relative_to=DATA_ROOT)
    return {
        "fixture": DATA_ROOT.relative_to(PROJECT_ROOT).as_posix(),
        "files": entries,
        "tree_sha256": _entries_hash(entries),
    }


def _source_hash() -> str:
    paths = [PROJECT_ROOT / "pyproject.toml", Path(__file__).resolve()]
    for package in ("benchmarks", "mlestar"):
        paths.extend(path for path in (PROJECT_ROOT / package).rglob("*.py") if path.is_file())
    return _entries_hash(_entries(paths, relative_to=PROJECT_ROOT))


def _git_output(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _git_manifest() -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return {
        "commit": _git_output("rev-parse", "HEAD"),
        "ref": _git_output("branch", "--show-current"),
        "tracked_files_modified": bool(status.stdout.strip()) if status.returncode == 0 else None,
    }


def _package_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _environment_manifest() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.system(),
        "machine": platform.machine(),
        "packages": {
            name: _package_version(name)
            for name in ("numpy", "pandas", "scikit-learn", "skrub")
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_smoke(output_dir: str | Path) -> dict[str, Any]:
    """Run one fixed-seed comparison and write only non-submission outputs."""

    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    destination.mkdir(parents=True)

    random.seed(SMOKE_SEED)
    np.random.seed(SMOKE_SEED)
    os.environ.setdefault("PYTHONHASHSEED", str(SMOKE_SEED))

    with tempfile.TemporaryDirectory(prefix="mlestar-smoke-") as temporary:
        temporary_root = Path(temporary)
        report = compare(
            benchmark=BENCHMARK,
            data_root=DATA_ROOT,
            run_root=temporary_root,
            seeds=(SMOKE_SEED,),
            outer_rounds=1,
            inner_rounds=1,
        )
        if list(temporary_root.rglob("submission*.csv")):
            raise RuntimeError("offline comparison unexpectedly wrote a submission artifact")
        shutil.copyfile(temporary_root / "comparison.csv", destination / "comparison.csv")

    result = {
        "report": report,
        "protocol": {
            "offline_only": True,
            "submission_attempted": False,
            "submission_artifacts_in_output": False,
        },
    }
    result_path = destination / "result.json"
    _write_json(result_path, result)

    config = {
        "benchmark": BENCHMARK,
        "seed": SMOKE_SEED,
        "outer_rounds": 1,
        "inner_rounds": 1,
        "submission_enabled": False,
        "data_fixture": DATA_ROOT.relative_to(PROJECT_ROOT).as_posix(),
    }
    comparison_path = destination / "comparison.csv"
    manifest = {
        "schema_version": 1,
        "config": config,
        "data": _data_manifest(),
        "environment": _environment_manifest(),
        "git": _git_manifest(),
        "version": {
            "distribution": "mlestar-dataops",
            "project": __version__,
            "python_requires": ">=3.11",
        },
        "hashes": {
            "algorithm": "sha256",
            "source_tree": _source_hash(),
            "comparison.csv": _sha256(comparison_path),
            "result.json": _sha256(result_path),
        },
    }
    _write_json(destination / "manifest.json", manifest)

    unexpected = list(destination.rglob("submission*.csv"))
    if unexpected:
        raise RuntimeError("smoke output unexpectedly contains a submission artifact")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the fixed-seed offline synthetic MLE-STAR smoke experiment."
    )
    parser.add_argument("--output-dir", required=True, help="new directory for manifests and results")
    arguments = parser.parse_args(argv)
    run_smoke(arguments.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

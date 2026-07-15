"""Protect the experiment package's dependency and licensing boundary."""

from pathlib import Path
import tomllib


def test_experiment_tree_has_no_root_runtime_modules() -> None:
    root = Path(__file__).parents[1]
    assert (root / "mlestar").is_dir()
    assert not any(
        (root / name).exists()
        for name in ("cv_autodl_agent", "retrieval", "module1_agent", "module4_agent", "pipeline.py")
    )


def test_experiment_keeps_its_python_cli_and_license_contract() -> None:
    root = Path(__file__).parents[1]
    metadata = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))

    assert metadata["project"]["requires-python"] == ">=3.11"
    assert metadata["project"]["scripts"]["mlestar"] == "mlestar.cli:main"
    assert metadata["project"]["license"] == "Apache-2.0"
    assert metadata["project"]["license-files"] == ["LICENSE"]
    assert "Apache License" in (root / "LICENSE").read_text(encoding="utf-8")

"""Prevent accidental reintroduction of Jiaozi runtime code."""

from pathlib import Path


def test_standalone_tree_has_no_jiaozi_modules() -> None:
    root = Path(__file__).parents[1]
    assert (root / "mlestar").is_dir()
    assert not any(
        (root / name).exists()
        for name in ("cv_autodl_agent", "retrieval", "module1_agent", "module4_agent", "pipeline.py")
    )

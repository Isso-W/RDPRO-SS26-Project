import json
from pathlib import Path


def test_notebook_uses_single_token_secret_and_no_submit_default() -> None:
    notebook = json.loads(Path("notebooks/mlestar_kaggle_experiments.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "KAGGLE_API_TOKEN" in source
    assert "KAGGLE_USERNAME" not in source
    assert "KAGGLE_KEY" not in source
    assert "SUBMIT = False" in source


def test_notebook_downloads_leaf_classification_data_when_missing() -> None:
    notebook = json.loads(Path("notebooks/mlestar_kaggle_experiments.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    # Auto-fetch must run before the compare command references DATA_ROOT files.
    assert source.index("kaggle competitions download") < source.index("mlestar.cli compare")
    assert "leaf-classification" in source
    assert "kaggle.json" in source
    # Skip re-downloading when the user already staged train.csv manually.
    assert "train.csv" in source
    # Still no separate username/key secrets introduced by the download step.
    assert "KAGGLE_USERNAME" not in source
    assert "KAGGLE_KEY" not in source

import json
from pathlib import Path


def test_notebook_uses_single_token_secret_and_no_submit_default() -> None:
    notebook = json.loads(Path("notebooks/mlestar_kaggle_experiments.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
    assert "KAGGLE_API_TOKEN" in source
    assert "KAGGLE_USERNAME" not in source
    assert "KAGGLE_KEY" not in source
    assert "SUBMIT = False" in source

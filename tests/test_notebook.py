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
    # Skip re-downloading when the user already staged train.csv manually.
    assert "train.csv" in source
    # Still no separate username/key secrets introduced by the download step.
    assert "KAGGLE_USERNAME" not in source
    assert "KAGGLE_KEY" not in source
    # Modern Kaggle tokens (KGAT_...) are a bearer string, not kaggle.json's
    # {"username":..., "key":...} shape. Writing the raw token into a file
    # named kaggle.json makes the CLI try (and fail) to parse it as JSON.
    # The `kaggle` CLI reads KAGGLE_API_TOKEN directly from the environment,
    # which is already set by the previous cell -- nothing else to write.
    assert "kaggle.json" not in source


def test_notebook_download_cell_fails_loudly_if_data_still_missing() -> None:
    notebook = json.loads(Path("notebooks/mlestar_kaggle_experiments.ipynb").read_text(encoding="utf-8"))
    cells_source = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    download_cell = next(src for src in cells_source if "kaggle competitions download" in src)
    # Shell magics (!cmd) don't raise on non-zero exit, so a failed Kaggle
    # download/unzip would otherwise continue silently. The cell must check
    # the outcome itself and raise before control reaches the compare cell.
    extract_index = download_cell.index("extractall")
    postcheck_index = download_cell.index("train.csv", extract_index)
    raise_index = download_cell.index("raise", postcheck_index)
    assert postcheck_index < raise_index


def test_notebook_extracts_nested_leaf_csv_archives() -> None:
    notebook = json.loads(Path("notebooks/mlestar_kaggle_experiments.ipynb").read_text(encoding="utf-8"))
    cells_source = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    download_cell = next(src for src in cells_source if "kaggle competitions download" in src)
    # The leaf-classification competition zip contains train.csv.zip,
    # test.csv.zip, and sample_submission.csv.zip nested one level deeper --
    # extracting only the outer archive leaves train.csv.zip on disk, not
    # train.csv, so the adapter's FileNotFoundError check still fires.
    assert "zipfile" in download_cell
    assert "*.csv.zip" in download_cell or "csv.zip" in download_cell

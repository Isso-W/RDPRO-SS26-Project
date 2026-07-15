import json
from pathlib import Path


NOTEBOOK_DIR = Path("notebooks")
EXPERIMENT_NOTEBOOK = NOTEBOOK_DIR / "mlestar_kaggle_experiments.ipynb"
NOTEBOOKS = (
    NOTEBOOK_DIR / "jiaozi_kaggle_competition.ipynb",
    EXPERIMENT_NOTEBOOK,
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _source(path: Path) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in _load(path)["cells"])


def test_notebooks_clone_jiaozi_main_and_enter_experiment_directory() -> None:
    for path in NOTEBOOKS:
        source = _source(path)
        assert "https://github.com/Isso-W/Jiaozi.git" in source
        assert "experiments/mlestar_kaggle_benchmarks" in source
        assert "--branch main" in source or "'--branch', 'main'" in source
        assert "PROJECT_DIR" in source
        assert "%cd {PROJECT_DIR}" in source or "os.chdir(PROJECT_DIR)" in source


def test_notebooks_are_saved_without_execution_state() -> None:
    for path in NOTEBOOKS:
        notebook = _load(path)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                assert cell.get("execution_count") is None
                assert cell.get("outputs", []) == []


def test_notebooks_use_one_colab_token_without_printing_or_persisting_it() -> None:
    for path in NOTEBOOKS:
        source = _source(path)
        assert "userdata.get('KAGGLE_API_TOKEN')" in source
        assert "KAGGLE_USERNAME" not in source
        assert "KAGGLE_KEY" not in source
        assert "KAGGLE_JSON" not in source
        assert "kaggle.json" not in source
        assert "access_token" not in source
        assert "getpass" not in source
        assert "print(kaggle_token" not in source
        assert "write_text(kaggle_token" not in source
        assert "SUBMIT = False" in source or "SUBMIT      = False" in source


def test_experiment_notebook_downloads_leaf_classification_data_when_missing() -> None:
    source = _source(EXPERIMENT_NOTEBOOK)
    # Auto-fetch must run before the compare command references DATA_ROOT files.
    assert source.index("kaggle competitions download") < source.index("mlestar.cli compare")
    assert "leaf-classification" in source
    # Skip re-downloading when the user already staged train.csv manually.
    assert "train.csv" in source


def test_experiment_notebook_download_cell_fails_loudly_if_data_still_missing() -> None:
    notebook = _load(EXPERIMENT_NOTEBOOK)
    cells_source = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    helper_cell = next(src for src in cells_source if "def fetch_kaggle_competition" in src)
    # Shell magics do not raise on non-zero exit, so the helper must verify the
    # extracted marker before allowing an experiment to start.
    extract_index = helper_cell.index("extractall")
    postcheck_index = helper_cell.index("marker_file", extract_index)
    raise_index = helper_cell.index("raise", postcheck_index)
    assert postcheck_index < raise_index


def test_experiment_notebook_extracts_nested_leaf_csv_archives() -> None:
    notebook = _load(EXPERIMENT_NOTEBOOK)
    cells_source = ["".join(cell.get("source", [])) for cell in notebook["cells"]]
    helper_cell = next(src for src in cells_source if "def fetch_kaggle_competition" in src)
    assert "zipfile" in helper_cell
    assert "*.csv.zip" in helper_cell or "csv.zip" in helper_cell


def test_experiment_notebook_reuses_download_helper_for_all_seven_tasks() -> None:
    source = _source(EXPERIMENT_NOTEBOOK)
    assert "def fetch_kaggle_competition" in source
    for slug in (
        "leaf-classification",
        "plant-pathology-2020-fgvc7",
        "aptos2019-blindness-detection",
        "dog-breed-identification",
        "aerial-cactus-identification",
        "dogs-vs-cats-redux-kernels-edition",
        "histopathologic-cancer-detection",
    ):
        assert slug in source, f"missing competition slug: {slug}"
    for benchmark in (
        "leaf_classification",
        "plant_pathology_2020",
        "aptos_2019",
        "dog_breed",
        "aerial_cactus",
        "dogs_vs_cats",
        "histopathologic_cancer",
    ):
        assert f"--benchmark {benchmark}" in source, f"missing compare cell for: {benchmark}"

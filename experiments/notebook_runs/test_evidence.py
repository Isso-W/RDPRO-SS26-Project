import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def test_manifest_notebooks_and_cell_logs_are_consistent() -> None:
    records = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    assert len(records) == 21
    assert len({record["notebook"] for record in records}) == len(records)

    for record in records:
        notebook_path = ROOT / record["notebook"]
        log_path = ROOT / record["cell_log"]
        assert notebook_path.is_file()
        assert log_path.is_file()
        assert _sha256(notebook_path) == record["notebook_sha256"]

        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        code_cells = [
            (index, cell)
            for index, cell in enumerate(notebook["cells"])
            if cell.get("cell_type") == "code"
        ]
        assert len(code_cells) == record["code_cells"]
        assert sum(cell.get("execution_count") is not None for _, cell in code_cells) == record[
            "executed_code_cells"
        ]
        assert sum(bool(cell.get("outputs")) for _, cell in code_cells) == record["output_cells"]

        log = log_path.read_text(encoding="utf-8")
        for index, _ in code_cells:
            assert f"=== Cell {index} |" in log

        for cell in notebook["cells"]:
            for output in cell.get("outputs", []):
                data = output.get("data", {})
                assert not any(mime.startswith("image/") for mime in data)
                assert "video/mp4" not in data


def test_templates_are_not_reported_as_completed_results() -> None:
    records = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    templates = [record for record in records if record["status"].startswith("template_")]
    assert len(templates) == 2
    assert all(record["executed_code_cells"] == 0 for record in templates)
    assert all(record["output_cells"] == 0 for record in templates)

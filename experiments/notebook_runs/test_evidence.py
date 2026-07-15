import csv
import hashlib
import json
from pathlib import Path

from experiments.notebook_runs.export_evidence import ExportSpec, render_cell_log


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
        destination = str(Path(record["notebook"]).relative_to("notebooks"))
        spec = ExportSpec(
            workflow=record["workflow"],
            competition=record["competition"],
            source=notebook_path,
            destination=destination,
            archive=record["source_archive"],
            status=record["status"],
        )
        assert log == render_cell_log(notebook, spec, record["source_sha256"])
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


def test_supplemental_leaderboard_scores_are_well_formed() -> None:
    score_path = ROOT / "results" / "rdpro_experiment_v2_scores.csv"
    manifest = json.loads(
        (ROOT / "results" / "source_manifest.json").read_text(encoding="utf-8")
    )
    with score_path.open(encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))

    assert len(records) == manifest["normalized_score_records"] == 19
    keys = {(record["workflow"], record["competition"]) for record in records}
    assert len(keys) == len(records)

    for record in records:
        assert float(record["public_score"]) >= 0
        assert float(record["private_score"]) >= 0
        assert 1 <= int(record["rank"]) <= int(record["leaderboard_size"])
        assert 0 <= int(record["reported_rank_percent"]) <= 100
        assert 18 <= int(record["source_csv_row"]) <= 28
        if record["notebook_log"]:
            assert (ROOT / record["notebook_log"]).is_file()

    by_key = {
        (record["workflow"], record["competition"]): record for record in records
    }
    assert by_key[("Jiaozi", "APTOS 2019 Blindness Detection")]["private_score"] == "0.865298"
    assert by_key[("MLE-STAR", "TGS Salt Identification Challenge")]["rank"] == "2141"
    assert by_key[("MLE-STAR", "Leaf Classification")]["evidence_status"] == (
        "supplemental_leaderboard_only"
    )

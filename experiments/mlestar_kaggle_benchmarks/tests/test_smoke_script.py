from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import run_smoke_experiment as smoke


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_smoke_runner_writes_reproducibility_manifest_without_submission(
    tmp_path, monkeypatch
) -> None:
    def fake_compare(*, benchmark, data_root, run_root, seeds, outer_rounds, inner_rounds):
        assert benchmark == "leaf_classification"
        assert Path(data_root).is_dir()
        assert seeds == (13,)
        assert (outer_rounds, inner_rounds) == (1, 1)
        run_root = Path(run_root)
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "comparison.csv").write_text(
            "seed,arm,metric_value\n13,baseline,1.0\n", encoding="utf-8"
        )
        return {
            "benchmark": benchmark,
            "seeds": [13],
            "status": "offline_oof_complete",
        }

    monkeypatch.setattr(smoke, "compare", fake_compare)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "must-not-appear")
    output = tmp_path / "smoke"

    result = smoke.run_smoke(output)

    assert result["protocol"] == {
        "offline_only": True,
        "submission_attempted": False,
        "submission_artifacts_in_output": False,
    }
    assert sorted(path.name for path in output.iterdir()) == [
        "comparison.csv",
        "manifest.json",
        "result.json",
    ]
    assert not list(output.rglob("submission*.csv"))

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["seed"] == 13
    assert manifest["config"]["submission_enabled"] is False
    assert manifest["data"]["files"]
    assert manifest["git"]["commit"]
    assert manifest["version"]["python_requires"] == ">=3.11"
    assert manifest["hashes"]["source_tree"]
    assert manifest["hashes"]["comparison.csv"] == _sha256(output / "comparison.csv")
    assert manifest["hashes"]["result.json"] == _sha256(output / "result.json")
    assert "must-not-appear" not in json.dumps(manifest)


def test_smoke_runner_rejects_any_submission_artifact(tmp_path, monkeypatch) -> None:
    def unsafe_compare(**arguments):
        run_root = Path(arguments["run_root"])
        (run_root / "comparison.csv").write_text(
            "seed,arm,metric_value\n13,baseline,1.0\n", encoding="utf-8"
        )
        (run_root / "submission.csv").write_text("id,target\n", encoding="utf-8")
        return {"status": "offline_oof_complete"}

    monkeypatch.setattr(smoke, "compare", unsafe_compare)

    with pytest.raises(RuntimeError, match="submission artifact"):
        smoke.run_smoke(tmp_path / "unsafe-smoke")

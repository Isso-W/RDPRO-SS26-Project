#!/usr/bin/env python3
"""Export sanitized, executed notebooks and one text log section per code cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


REPOSITORY_SETUP = """# Reproducible repository setup for Colab.
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY = "https://github.com/Isso-W/Jiaozi.git"
BRANCH = "main"
REPOSITORY_ROOT = Path("/content/Jiaozi")
EXPERIMENT_ROOT = REPOSITORY_ROOT / "experiments" / "mlestar_kaggle_benchmarks"

if REPOSITORY_ROOT.exists():
    shutil.rmtree(REPOSITORY_ROOT)
subprocess.run(
    ["git", "clone", "--depth", "1", "--branch", BRANCH, REPOSITORY, str(REPOSITORY_ROOT)],
    check=True,
)
actual_commit = subprocess.check_output(
    ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"], text=True
).strip()
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", f"{EXPERIMENT_ROOT}[vision,llm,kaggle,dev]"],
    check=True,
)
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--upgrade", "kaggle==2.2.2"],
    check=True,
)
os.chdir(EXPERIMENT_ROOT)
if str(EXPERIMENT_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_ROOT))
print("Repository commit:", actual_commit)
subprocess.run([sys.executable, "-m", "mlestar.cli", "benchmarks"], check=True)
"""

ANSI_ESCAPE = re.compile(r"\x1b(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
MEDIA_MIMES = {"image/png", "image/jpeg", "image/gif", "image/svg+xml", "video/mp4"}
SKIPPED_RICH_MIMES = {"text/html", "application/javascript"}
SOURCE_ARCHIVE_SHA256 = {
    "jiaozi.zip": "4d578802aaa3f1f56f00fd8360b3efde768dc33caacfbc2c8b9fedacb5aba3c0",
    "mle.zip": "c4f7c6e575be03d034e6819311ffd1e15f373da6e18dcf45e3bb3d465793a0a8",
    "mle_final_submission.zip": "66126db1149262fcd4f59a3824a52e16aa557491838733ea2758af4e4db8dd8a",
    "agent_final_submission.zip": "17e393c460781267c89f2df86636676e1f421d55536aa4d9413166de690235ae",
}


@dataclass(frozen=True)
class ExportSpec:
    workflow: str
    competition: str
    source: Path
    destination: str
    archive: str
    status: str
    cells: tuple[int, ...] | None = None
    rewrite_setup: bool = False


def text_lines(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    if not lines:
        return []
    if not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rewrite_jiaozi_sources(notebook: dict) -> None:
    replacements = {
        "https://github.com/muzhi-hac/Jiaozi-rag-wang.git": "https://github.com/Isso-W/Jiaozi.git",
        "rag_wang": "main",
        "integration-update": "main",
    }
    for cell in notebook.get("cells", []):
        source = "".join(cell.get("source", []))
        for old, new in replacements.items():
            source = source.replace(old, new)
        cell["source"] = text_lines(source)


def strip_embedded_media(notebook: dict) -> None:
    for cell in notebook.get("cells", []):
        for output in cell.get("outputs", []):
            data = output.get("data")
            if not isinstance(data, dict):
                continue
            removed = [mime for mime in MEDIA_MIMES if mime in data]
            for mime in removed:
                data.pop(mime, None)
            if removed and "text/plain" not in data:
                data["text/plain"] = [
                    "[Embedded rich-media output removed from the public evidence copy; "
                    "the textual cell log is retained.]"
                ]


def evidence_header(spec: ExportSpec, source_hash: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text_lines(
            f"""# Stored execution evidence — {spec.competition}

This public evidence copy preserves every textual output cell supplied in `{spec.archive}`.
The original notebook SHA-256 is `{source_hash}`. Embedded binary rich media is removed so
that competition data and generated images are not redistributed. The repository setup cell
now targets `Isso-W/Jiaozi` at `main`; stored metrics and submission receipts remain historical
outputs and are not presented as a fresh rerun of the current branch.
"""
        ),
    }


def sanitize_notebook(spec: ExportSpec) -> tuple[dict, str]:
    source_hash = sha256(spec.source)
    notebook = json.loads(spec.source.read_text(encoding="utf-8"))
    if spec.cells is not None:
        notebook["cells"] = [notebook["cells"][index] for index in spec.cells]
    if spec.workflow == "jiaozi":
        rewrite_jiaozi_sources(notebook)
    if spec.rewrite_setup:
        if len(notebook.get("cells", [])) < 2 or notebook["cells"][1].get("cell_type") != "code":
            raise ValueError(f"Expected setup code cell at index 1: {spec.source}")
        notebook["cells"][1]["source"] = text_lines(REPOSITORY_SETUP)
    strip_embedded_media(notebook)
    notebook["cells"].insert(0, evidence_header(spec, source_hash))
    notebook.setdefault("metadata", {})["jiaozi_evidence"] = {
        "competition": spec.competition,
        "source_archive": spec.archive,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256[spec.archive],
        "source_sha256": source_hash,
        "stored_output_status": spec.status,
    }
    return notebook, source_hash


def output_text(output: dict) -> Iterable[str]:
    output_type = output.get("output_type", "unknown")
    if output_type == "stream":
        yield f"[{output_type}:{output.get('name', 'unknown')}]\n"
        yield "".join(output.get("text", []))
        return
    if output_type == "error":
        yield f"[error:{output.get('ename', 'Error')}] {output.get('evalue', '')}\n"
        traceback = output.get("traceback", [])
        if traceback:
            yield "\n".join(traceback) + "\n"
        return
    data = output.get("data", {})
    yielded = False
    for mime in ("text/plain", "application/json"):
        if mime not in data:
            continue
        yielded = True
        value = data[mime]
        if not isinstance(value, str):
            value = "".join(value) if isinstance(value, list) else json.dumps(value, ensure_ascii=False)
        yield f"[{output_type}:{mime}]\n{value}\n"
    skipped = sorted(mime for mime in data if mime in SKIPPED_RICH_MIMES)
    if skipped:
        yielded = True
        yield f"[{output_type}:rich output omitted from text log: {', '.join(skipped)}]\n"
    if not yielded:
        yield f"[{output_type}:no textual payload]\n"


def render_cell_log(notebook: dict, spec: ExportSpec, source_hash: str) -> str:
    sections = [
        f"Notebook: {spec.destination}\n",
        f"Competition: {spec.competition}\n",
        f"Workflow: {spec.workflow}\n",
        f"Source archive: {spec.archive}\n",
        f"Source archive SHA-256: {SOURCE_ARCHIVE_SHA256[spec.archive]}\n",
        f"Source notebook SHA-256: {source_hash}\n",
        f"Stored output status: {spec.status}\n",
        "\n",
    ]
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue
        outputs = cell.get("outputs", [])
        sections.append(
            f"=== Cell {index} | execution_count={cell.get('execution_count')} | outputs={len(outputs)} ===\n"
        )
        if not outputs:
            sections.append("[no stored output]\n\n")
            continue
        for output_index, output in enumerate(outputs):
            sections.append(f"--- output {output_index} ---\n")
            sections.extend(output_text(output))
        sections.append("\n")
    cleaned = ANSI_ESCAPE.sub("", "".join(sections))
    return "\n".join(line.rstrip() for line in cleaned.splitlines()).rstrip() + "\n"


def write_cell_log(notebook: dict, path: Path, spec: ExportSpec, source_hash: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_cell_log(notebook, spec, source_hash), encoding="utf-8")


def refresh_current_evidence(output: Path) -> None:
    """Refresh logs and derived metadata after text-only edits to public notebooks."""

    manifest_path = output / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for record in manifest:
        notebook_path = output / record["notebook"]
        log_path = output / record["cell_log"]
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        destination = str(Path(record["notebook"]).relative_to("notebooks"))
        spec = ExportSpec(
            workflow=record["workflow"],
            competition=record["competition"],
            source=notebook_path,
            destination=destination,
            archive=record["source_archive"],
            status=record["status"],
        )
        write_cell_log(notebook, log_path, spec, record["source_sha256"])

        code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
        record["notebook_sha256"] = sha256(notebook_path)
        record["code_cells"] = len(code_cells)
        record["executed_code_cells"] = sum(
            cell.get("execution_count") is not None for cell in code_cells
        )
        record["output_cells"] = sum(bool(cell.get("outputs")) for cell in code_cells)
        record["error_cells"] = sum(
            any(output.get("output_type") == "error" for output in cell.get("outputs", []))
            for cell in code_cells
        )

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Refreshed {len(manifest)} notebook records in {output}")


def specs(new_root: Path, prior_root: Path) -> list[ExportSpec]:
    new_jiaozi = new_root / "jiaozi" / "jiaozi"
    new_mle = new_root / "mle" / "mle"
    prior_agent = prior_root / "agent" / "agent_final_submission"
    prior_mle = prior_root / "mle" / "mle_final_submission"
    combined = new_mle / "mlestar_all_3_APTOS_DogBreed_Cactus_DogsCats_HistopathologicCancer.ipynb"
    return [
        ExportSpec("jiaozi", "APTOS 2019 Blindness Detection", new_jiaozi / "integration_update_APTOS.ipynb", "jiaozi/aptos_2019.ipynb", "jiaozi.zip", "validation_complete_submission_blocked"),
        ExportSpec("jiaozi", "Global Wheat Detection", new_jiaozi / "integration_update_Global_Wheat.ipynb", "jiaozi/global_wheat_detection.ipynb", "jiaozi.zip", "validation_complete_submission_blocked"),
        ExportSpec("jiaozi", "Dogs vs. Cats Redux", new_jiaozi / "integration_update_dogsVScats_epoch15.ipynb", "jiaozi/dogs_vs_cats_redux.ipynb", "jiaozi.zip", "scored"),
        ExportSpec("jiaozi", "Histopathologic Cancer Detection", new_jiaozi / "integration_update_Histopathologic_Cancer_final.ipynb", "jiaozi/histopathologic_cancer.ipynb", "jiaozi.zip", "scored_with_backbone_warning"),
        ExportSpec("jiaozi", "Plant Pathology 2020", new_jiaozi / "integration_update_Plant.ipynb", "jiaozi/plant_pathology_2020.ipynb", "jiaozi.zip", "validation_complete_submission_not_executed"),
        ExportSpec("jiaozi", "Aerial Cactus Identification", prior_agent / "integration_update_colab_Aerial_Cactus_Identification_Kaggle_Assets_Packager.ipynb", "jiaozi/aerial_cactus.ipynb", "agent_final_submission.zip", "validation_complete_assets_packaged"),
        ExportSpec("jiaozi", "Dog Breed Identification", prior_agent / "integration_update_colab_Dog_Breed_Identification.ipynb", "jiaozi/dog_breed.ipynb", "agent_final_submission.zip", "validation_complete_after_interrupted_cell"),
        ExportSpec("jiaozi", "RANZCR CLiP", prior_agent / "integration_update_colab_Ranzcr_Clip_Kaggle_Assets_Packager.ipynb", "jiaozi/ranzcr_clip.ipynb", "agent_final_submission.zip", "validation_complete_assets_packaged"),
        ExportSpec("jiaozi", "TGS Salt Identification Challenge", prior_agent / "integration_update_colab_TGS_SALT_Kaggle_Assets_Packager.ipynb", "jiaozi/tgs_salt.ipynb", "agent_final_submission.zip", "scored"),
        ExportSpec("jiaozi", "Ultrasound Nerve Segmentation", prior_agent / "integration_update_colab_Ultrasound_Nerve_Segmentation.ipynb", "jiaozi/ultrasound_nerve_segmentation.ipynb", "agent_final_submission.zip", "scored"),
        ExportSpec("jiaozi", "Aerial Cactus Identification (inference)", prior_agent / "jiaozi-agent-aerial-cactus-kaggle-inference.ipynb", "jiaozi/aerial_cactus_inference.ipynb", "agent_final_submission.zip", "submission_file_ready"),
        ExportSpec("jiaozi", "RANZCR CLiP (inference)", prior_agent / "jiaozi-agent-ranzcr-clip-kaggle-inference.ipynb", "jiaozi/ranzcr_clip_inference.ipynb", "agent_final_submission.zip", "submission_file_ready"),
        ExportSpec("mlestar", "APTOS 2019 Blindness Detection", combined, "mlestar/aptos_2019.ipynb", "mle.zip", "validation_complete_submission_blocked", (0, 1, 2, 3, 4, 5, 6, 7, 10), True),
        ExportSpec("mlestar", "Dog Breed Identification", combined, "mlestar/dog_breed.ipynb", "mle.zip", "scored", (0, 1, 2, 3, 4, 5, 6, 7, 11), True),
        ExportSpec("mlestar", "Aerial Cactus Identification", combined, "mlestar/aerial_cactus.ipynb", "mle.zip", "validation_complete_submission_blocked", (0, 1, 2, 3, 4, 5, 6, 7, 12), True),
        ExportSpec("mlestar", "Dogs vs. Cats Redux", combined, "mlestar/dogs_vs_cats_redux.ipynb", "mle.zip", "scored", (0, 1, 2, 3, 4, 5, 6, 7, 13), True),
        ExportSpec("mlestar", "Histopathologic Cancer Detection", combined, "mlestar/histopathologic_cancer.ipynb", "mle.zip", "scored", (0, 1, 2, 3, 4, 5, 6, 7, 14), True),
        ExportSpec("mlestar", "Plant Pathology 2020", new_mle / "mlestar_all_3_Plant.ipynb", "mlestar/plant_pathology_2020.ipynb", "mle.zip", "scored", (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13), True),
        ExportSpec("mlestar", "RANZCR CLiP", prior_mle / "MLE_STAR_RANZCR_CLiP_Colab_Train_and_Package.ipynb", "mlestar/ranzcr_clip.ipynb", "mle_final_submission.zip", "validation_complete_assets_packaged", None, True),
        ExportSpec("mlestar", "RANZCR CLiP (inference template)", prior_mle / "MLE_STAR_RANZCR_CLiP_Kaggle_Inference.ipynb", "mlestar/ranzcr_clip_inference_template.ipynb", "mle_final_submission.zip", "template_no_stored_output"),
        ExportSpec("mlestar", "TGS Salt Identification Challenge (template)", prior_mle / "MLE_STAR_TGS_Salt_Colab_Train_and_Submit.ipynb", "mlestar/tgs_salt_template.ipynb", "mle_final_submission.zip", "template_no_stored_output"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-root", type=Path)
    parser.add_argument("--prior-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--refresh-current",
        action="store_true",
        help="refresh logs and manifest metadata from notebooks already under --output",
    )
    args = parser.parse_args()

    if args.refresh_current:
        refresh_current_evidence(args.output)
        return
    if args.new_root is None or args.prior_root is None:
        parser.error("--new-root and --prior-root are required unless --refresh-current is used")

    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for spec in specs(args.new_root, args.prior_root):
        if not spec.source.is_file():
            raise FileNotFoundError(spec.source)
        notebook, source_hash = sanitize_notebook(spec)
        destination = args.output / "notebooks" / spec.destination
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        log_path = args.output / "logs" / Path(spec.destination).with_suffix(".log")
        write_cell_log(notebook, log_path, spec, source_hash)
        code_cells = [cell for cell in notebook["cells"] if cell.get("cell_type") == "code"]
        manifest.append(
            {
                "competition": spec.competition,
                "workflow": spec.workflow,
                "status": spec.status,
                "source_archive": spec.archive,
                "source_archive_sha256": SOURCE_ARCHIVE_SHA256[spec.archive],
                "source_sha256": source_hash,
                "notebook": str(destination.relative_to(args.output)),
                "notebook_sha256": sha256(destination),
                "cell_log": str(log_path.relative_to(args.output)),
                "code_cells": len(code_cells),
                "executed_code_cells": sum(cell.get("execution_count") is not None for cell in code_cells),
                "output_cells": sum(bool(cell.get("outputs")) for cell in code_cells),
                "error_cells": sum(
                    any(output.get("output_type") == "error" for output in cell.get("outputs", []))
                    for cell in code_cells
                ),
            }
        )
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {len(manifest)} notebook records to {args.output}")


if __name__ == "__main__":
    main()

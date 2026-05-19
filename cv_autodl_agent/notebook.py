from __future__ import annotations

import json
from pathlib import Path

from .schemas import DatasetManifest, TrainingSpec


class NotebookExporter:
    def export(
        self,
        project_dir: str | Path,
        manifest: DatasetManifest,
        spec: TrainingSpec,
        active_spec_file: str,
        best_run_dir: str,
        execution_mode: str = "simulate",
    ) -> Path:
        root = Path(project_dir)
        notebook_path = root / "notebook.ipynb"
        cells = [
            self._markdown_cell(
                f"# CV Auto-DL Agent\n\nTask family: `{manifest.task_family}`\n\nModel: `{spec.selected_model_id}`"
            ),
            self._code_cell("!pip install -r requirements.txt"),
            self._code_cell(
                "DATA_DIR = '/content/data'\nOUTPUT_DIR = '/content/outputs'\nMANIFEST_PATH = 'manifest.json'\nSPEC_PATH = '"
                + active_spec_file
                + "'\nEXECUTION_MODE = '"
                + execution_mode
                + "'"
            ),
            self._code_cell(
                "from pathlib import Path\nPath(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)\nprint('Upload or mount your dataset under', DATA_DIR)"
            ),
            self._code_cell(
                "!python train.py --manifest {manifest} --spec {spec} --output-dir {out} --execution-mode {mode}".format(
                    manifest="{MANIFEST_PATH}",
                    spec="{SPEC_PATH}",
                    out="{OUTPUT_DIR}/train_run",
                    mode="{EXECUTION_MODE}",
                )
            ),
            self._code_cell(
                "import json\nfrom pathlib import Path\nmetrics_path = Path(OUTPUT_DIR) / 'train_run' / 'metrics.json'\nprint(json.dumps(json.loads(metrics_path.read_text()), indent=2)) if metrics_path.exists() else print('metrics.json not found yet')"
            ),
            self._code_cell("!python inference.py --checkpoint {OUTPUT_DIR}/train_run/checkpoints/best.pt --input sample_image.jpg"),
        ]
        notebook = {
            "cells": cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.11"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        notebook_path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
        return notebook_path

    @staticmethod
    def _code_cell(source: str) -> dict[str, object]:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in source.splitlines()],
        }

    @staticmethod
    def _markdown_cell(source: str) -> dict[str, object]:
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in source.splitlines()],
        }

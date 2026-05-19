from __future__ import annotations

from pathlib import Path

from .io_utils import write_json
from .schemas import DatasetManifest, GeneratedProject, TrainingSpec
from .templates import dataset_script_template, inference_script_template, train_script_template


class CodeGenerator:
    def generate(self, project_dir: str | Path, manifest: DatasetManifest, spec: TrainingSpec) -> GeneratedProject:
        root = Path(project_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)

        manifest_path = write_json(root / "manifest.json", manifest)
        spec_path = write_json(root / "training_spec.json", spec)

        train_script_path = root / "train.py"
        train_script_path.write_text(train_script_template(spec.task_family), encoding="utf-8")

        dataset_script_path = root / "dataset.py"
        dataset_script_path.write_text(dataset_script_template(manifest, spec), encoding="utf-8")

        inference_script_path = root / "inference.py"
        inference_script_path.write_text(inference_script_template(spec.task_family), encoding="utf-8")

        requirements_path = root / "requirements.txt"
        requirements_path.write_text(self._requirements(manifest, spec), encoding="utf-8")

        return GeneratedProject(
            project_dir=str(root),
            manifest_path=str(manifest_path),
            spec_path=str(spec_path),
            train_script_path=str(train_script_path),
            dataset_script_path=str(dataset_script_path),
            inference_script_path=str(inference_script_path),
            requirements_path=str(requirements_path),
        )

    @staticmethod
    def _requirements(manifest: DatasetManifest, spec: TrainingSpec) -> str:
        packages = [
            "torch>=2.2.0",
            "torchvision>=0.17.0",
            "pillow>=10.0.0",
        ]
        if manifest.hf_dataset_id:
            packages.append("datasets>=2.19.0")
        model_id = spec.selected_model_id.lower()
        if "hf" in model_id or "huggingface" in model_id or "transformers" in model_id:
            packages.append("transformers>=4.40.0")
        if manifest.task_family == "detection":
            packages.append("pycocotools>=2.0.7")
        return "\n".join(dict.fromkeys(packages)) + "\n"

from __future__ import annotations

from textwrap import dedent

from .schemas import DatasetManifest, TrainingSpec



def train_script_template(task_family: str) -> str:
    return dedent(
        f'''\
        from __future__ import annotations

        import argparse
        import hashlib
        import json
        import os
        import time
        from pathlib import Path

        TASK_FAMILY = "{task_family}"
        DEFAULT_EXECUTION_MODE = os.getenv("CV_AUTO_DL_EXECUTION_MODE", "simulate")


        def load_json(path: str | Path) -> dict:
            with Path(path).open("r", encoding="utf-8") as handle:
                return json.load(handle)


        def ensure_output_dir(path: str | Path) -> Path:
            out = Path(path)
            out.mkdir(parents=True, exist_ok=True)
            return out


        def validate_manifest(manifest: dict) -> None:
            required = [
                "dataset_name",
                "task_family",
                "train_path",
                "val_path",
                "test_path",
                "annotation_format",
                "recommended_metric",
            ]
            for key in required:
                if not manifest.get(key):
                    raise ValueError(f"manifest missing {{key}}")
            if manifest["task_family"] != TASK_FAMILY:
                raise ValueError("manifest task_family does not match generated template")


        def validate_spec(spec: dict) -> None:
            required = [
                "selected_model_id",
                "template_id",
                "task_family",
                "dataset_loader_strategy",
                "transforms",
                "loss_fn",
                "metric",
                "optimizer",
                "scheduler",
                "epochs",
                "batch_size",
                "freeze_strategy",
                "checkpoint_policy",
                "early_stopping",
            ]
            for key in required:
                if key not in spec:
                    raise ValueError(f"training spec missing {{key}}")
            if spec["task_family"] != TASK_FAMILY:
                raise ValueError("training spec task_family does not match generated template")


        def _hash_fraction(value: str) -> float:
            digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
            return int(digest[:8], 16) / 0xFFFFFFFF


        def compute_simulated_metric(spec: dict) -> float:
            base = {{
                "classification": 0.66,
                "segmentation": 0.49,
                "detection": 0.41,
            }}[TASK_FAMILY]
            transforms = [item.lower() for item in spec.get("transforms", [])]
            optimizer = spec.get("optimizer", {{}})
            scheduler = spec.get("scheduler", {{}})
            lr = float(optimizer.get("learning_rate", 3e-4))
            batch_size = int(spec.get("batch_size", 8))
            freeze = str(spec.get("freeze_strategy", "train_all")).lower()
            loss_fn = str(spec.get("loss_fn", "")).lower()

            score = base
            if TASK_FAMILY == "classification":
                if "randaugment" in transforms:
                    score += 0.05
                if "horizontal_flip" in transforms:
                    score += 0.02
                if freeze == "freeze_backbone":
                    score += 0.03
                if scheduler.get("name") == "cosine":
                    score += 0.02
                if optimizer.get("name") == "adamw":
                    score += 0.04
                score += max(0.0, 0.025 - abs(lr - 3e-4) * 35)
                if 16 <= batch_size <= 32:
                    score += 0.02
            elif TASK_FAMILY == "segmentation":
                if "random_crop" in transforms:
                    score += 0.03
                if freeze == "freeze_backbone":
                    score += 0.02
                if "dice" in loss_fn:
                    score += 0.06
                if scheduler.get("name") == "poly":
                    score += 0.03
                score += max(0.0, 0.02 - abs(lr - 2e-4) * 40)
                if batch_size in (4, 8):
                    score += 0.015
            else:
                if optimizer.get("score_threshold") == 0.25:
                    score += 0.025
                if freeze == "freeze_backbone":
                    score += 0.015
                if scheduler.get("name") == "step":
                    score += 0.02
                score += max(0.0, 0.02 - abs(lr - 1e-4) * 60)
                if batch_size in (2, 4):
                    score += 0.01

            score += _hash_fraction(json.dumps(spec, sort_keys=True)) * 0.005
            return round(min(score, 0.99), 4)


        def run_real_training(manifest: dict, spec: dict, output_dir: Path) -> dict:
            if TASK_FAMILY != "classification":
                raise NotImplementedError("Real training is implemented for classification templates first.")
            if spec.get("dataset_loader_strategy") != "huggingface_dataset":
                raise NotImplementedError("Real training currently expects a Hugging Face image dataset manifest.")

            import torch
            from datasets import load_dataset
            from PIL import Image
            from torch import nn
            from torch.utils.data import DataLoader, Dataset
            from torchvision import models as tv_models
            from torchvision import transforms as tv_transforms

            seed = int(os.getenv("CV_AUTO_DL_SEED", "42"))
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            dataset_id = manifest.get("hf_dataset_id")
            if not dataset_id:
                raise ValueError("manifest.hf_dataset_id is required for real Hugging Face training")

            config_name = manifest.get("hf_config_name") or None
            image_column = manifest.get("image_column") or "image"
            label_column = manifest.get("label_column") or "label"
            train_split = manifest.get("train_split") or "train"
            val_split = manifest.get("val_split") or manifest.get("test_split") or "test"
            max_train_samples = int(os.getenv("MAX_TRAIN_SAMPLES", str(manifest.get("max_train_samples") or 1024)))
            max_val_samples = int(os.getenv("MAX_VAL_SAMPLES", str(manifest.get("max_val_samples") or 256)))
            max_epochs = int(os.getenv("MAX_EPOCHS", str(manifest.get("max_epochs") or spec.get("epochs", 1))))
            num_classes = int(manifest.get("num_classes") or 0)
            if num_classes <= 0:
                raise ValueError("manifest.num_classes must be positive")

            def load_split(split_name: str):
                if config_name:
                    return load_dataset(dataset_id, config_name, split=split_name)
                return load_dataset(dataset_id, split=split_name)

            train_ds = load_split(train_split).shuffle(seed=seed)
            val_ds = load_split(val_split)
            if max_train_samples > 0:
                train_ds = train_ds.select(range(min(max_train_samples, len(train_ds))))
            if max_val_samples > 0:
                val_ds = val_ds.select(range(min(max_val_samples, len(val_ds))))

            image_size = 224
            for item in spec.get("transforms", []):
                if isinstance(item, str) and item.startswith("resize:"):
                    image_size = int(item.split(":", 1)[1])
                    break

            train_transform = tv_transforms.Compose([
                tv_transforms.Resize((image_size, image_size)),
                tv_transforms.RandomHorizontalFlip(),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            val_transform = tv_transforms.Compose([
                tv_transforms.Resize((image_size, image_size)),
                tv_transforms.ToTensor(),
                tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            class HuggingFaceImageDataset(Dataset):
                def __init__(self, hf_dataset, transform):
                    self.hf_dataset = hf_dataset
                    self.transform = transform

                def __len__(self):
                    return len(self.hf_dataset)

                def __getitem__(self, index):
                    row = self.hf_dataset[index]
                    image = row[image_column]
                    if not isinstance(image, Image.Image):
                        image = Image.open(image)
                    image = image.convert("RGB")
                    return self.transform(image), int(row[label_column])

            batch_size = int(spec.get("batch_size", 16))
            train_loader = DataLoader(
                HuggingFaceImageDataset(train_ds, train_transform),
                batch_size=batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=torch.cuda.is_available(),
            )
            val_loader = DataLoader(
                HuggingFaceImageDataset(val_ds, val_transform),
                batch_size=batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=torch.cuda.is_available(),
            )

            def normalize_model_name(value: str) -> str:
                name = value.lower().replace("/", "-")
                if name.startswith("timm-"):
                    name = name[len("timm-"):]
                if name.startswith("torchvision-"):
                    name = name[len("torchvision-"):]
                if name.startswith("hf-"):
                    return "resnet18"
                return name.replace("-", "_")

            def create_torchvision_classifier(model_name: str, class_count: int):
                name = normalize_model_name(model_name)
                if name == "resnet18":
                    model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
                    model.fc = nn.Linear(model.fc.in_features, class_count)
                    return model, name
                if name == "resnet34":
                    model = tv_models.resnet34(weights=tv_models.ResNet34_Weights.DEFAULT)
                    model.fc = nn.Linear(model.fc.in_features, class_count)
                    return model, name
                if name == "resnet50":
                    model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.DEFAULT)
                    model.fc = nn.Linear(model.fc.in_features, class_count)
                    return model, name
                if name == "mobilenet_v3_small":
                    model = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT)
                    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_count)
                    return model, name
                if name == "efficientnet_b0":
                    model = tv_models.efficientnet_b0(weights=tv_models.EfficientNet_B0_Weights.DEFAULT)
                    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, class_count)
                    return model, name
                model = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
                model.fc = nn.Linear(model.fc.in_features, class_count)
                return model, "resnet18"

            model, model_name = create_torchvision_classifier(str(spec.get("selected_model_id", "resnet18")), num_classes)

            if spec.get("freeze_strategy") == "freeze_backbone":
                for name, parameter in model.named_parameters():
                    if not any(head in name.lower() for head in ("head", "classifier", "fc")):
                        parameter.requires_grad = False

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            optimizer_config = spec.get("optimizer", {{}})
            optimizer = torch.optim.AdamW(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                lr=float(optimizer_config.get("learning_rate", 1e-4)),
                weight_decay=float(optimizer_config.get("weight_decay", 1e-4)),
            )
            criterion = nn.CrossEntropyLoss()

            best_accuracy = 0.0
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            best_checkpoint = checkpoint_dir / "best.pt"
            for epoch in range(max_epochs):
                model.train()
                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    loss = criterion(model(images), labels)
                    loss.backward()
                    optimizer.step()

                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in val_loader:
                        images = images.to(device)
                        labels = labels.to(device)
                        predictions = model(images).argmax(dim=1)
                        correct += int((predictions == labels).sum().item())
                        total += int(labels.numel())
                accuracy = correct / max(total, 1)
                print(f"epoch={{epoch + 1}} val_accuracy={{accuracy:.4f}}")
                if accuracy >= best_accuracy:
                    best_accuracy = accuracy
                    torch.save(
                        {{
                            "model_state_dict": model.state_dict(),
                            "model_name": model_name,
                            "num_classes": num_classes,
                            "manifest": manifest,
                            "spec": spec,
                        }},
                        best_checkpoint,
                    )

            payload = {{
                "status": "success",
                "primary_metric_name": spec.get("metric", "accuracy"),
                "primary_metric_value": round(float(best_accuracy), 4),
                "checkpoint_path": str(best_checkpoint),
                "artifacts_path": str(output_dir),
            }}
            (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return payload


        def run_simulated_training(manifest: dict, spec: dict, output_dir: Path) -> dict:
            metric_name = spec["metric"]
            metric_value = compute_simulated_metric(spec)
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / "best.pt"
            checkpoint_path.write_text("simulated-checkpoint\\n", encoding="utf-8")
            metrics_path = output_dir / "metrics.json"
            payload = {{
                "status": "success",
                "primary_metric_name": metric_name,
                "primary_metric_value": metric_value,
                "checkpoint_path": str(checkpoint_path),
                "artifacts_path": str(output_dir),
            }}
            metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            return payload


        def main() -> int:
            parser = argparse.ArgumentParser()
            parser.add_argument("--manifest", default="manifest.json")
            parser.add_argument("--spec", default="training_spec.json")
            parser.add_argument("--output-dir", default="runs/default")
            parser.add_argument("--execution-mode", default=DEFAULT_EXECUTION_MODE)
            args = parser.parse_args()

            start = time.time()
            manifest = load_json(args.manifest)
            spec = load_json(args.spec)
            validate_manifest(manifest)
            validate_spec(spec)
            output_dir = ensure_output_dir(args.output_dir)

            try:
                if args.execution_mode == "real":
                    payload = run_real_training(manifest, spec, output_dir)
                else:
                    payload = run_simulated_training(manifest, spec, output_dir)
                payload["stdout"] = ""
                payload["stderr"] = ""
                payload["runtime_sec"] = round(time.time() - start, 4)
                print(json.dumps(payload))
                return 0
            except Exception as exc:  # pragma: no cover - surfaced through executor
                payload = {{
                    "status": "failed",
                    "primary_metric_name": spec.get("metric", "unknown"),
                    "primary_metric_value": None,
                    "checkpoint_path": None,
                    "artifacts_path": str(output_dir),
                    "stdout": "",
                    "stderr": str(exc),
                    "runtime_sec": round(time.time() - start, 4),
                }}
                print(json.dumps(payload))
                return 1


        if __name__ == "__main__":
            raise SystemExit(main())
        '''
    ).lstrip()



def dataset_script_template(manifest: DatasetManifest, spec: TrainingSpec) -> str:
    return dedent(
        f'''\
        from __future__ import annotations

        from pathlib import Path

        TASK_FAMILY = "{manifest.task_family}"
        DATASET_LOADER_STRATEGY = "{spec.dataset_loader_strategy}"


        def describe_dataset_layout() -> dict[str, str]:
            return {{
                "train_path": "{manifest.train_path}",
                "val_path": "{manifest.val_path}",
                "test_path": "{manifest.test_path}",
                "annotation_format": "{manifest.annotation_format}",
            }}


        def resolve_under(root: str | Path) -> dict[str, str]:
            root_path = Path(root)
            layout = describe_dataset_layout()
            return {{key: str(root_path / value) for key, value in layout.items() if key.endswith("_path")}}
        '''
    ).lstrip()



def inference_script_template(task_family: str) -> str:
    return dedent(
        f'''\
        from __future__ import annotations

        import argparse
        import json
        from pathlib import Path

        TASK_FAMILY = "{task_family}"


        def main() -> int:
            parser = argparse.ArgumentParser()
            parser.add_argument("--checkpoint", required=False, default="runs/default/checkpoints/best.pt")
            parser.add_argument("--input", required=False, default="sample_input")
            args = parser.parse_args()
            payload = {{
                "task_family": TASK_FAMILY,
                "checkpoint": str(Path(args.checkpoint)),
                "input": args.input,
                "prediction": "simulated_prediction",
            }}
            print(json.dumps(payload))
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
        '''
    ).lstrip()

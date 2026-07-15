from email.mime import image
import os
import json
import csv

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights


class ImageFeatureExtractor:

    def __init__(
        self,
        dataset_dir: str,
        output_dir: str,
        batch_size: int = 32,
        device: str = (
            "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    ):
        # Initialize extractor configuration

        self._dataset_dir = dataset_dir

        self._output_dir = output_dir

        self._batch_size = batch_size

        self._device = device

        os.makedirs(
            output_dir,
            exist_ok=True
        )

        # Build model and preprocessing pipeline at construction time
        # so extract() does not pay loading cost on repeated calls.
        self._model = self._load_model()

        self._transform = self._build_transform()

    def extract(self) -> dict:
        # Main entry point

        return self._extract_features()

    def _extract_features(self) -> dict:
        # Walk dataset_dir, run model in batches, persist outputs

        image_records = self._collect_image_paths()

        # Sort class names alphabetically so the integer encoding is
        # deterministic and reproducible across different machines or runs.
        class_names = sorted(set(
            record["class_name"]
            for record in image_records
        ))

        class_to_idx = {
            name: idx
            for idx, name in enumerate(class_names)
        }

        all_features: list[np.ndarray] = []

        all_labels: list[int] = []

        all_paths: list[str] = []

        total = len(image_records)

        processed = 0

        for batch_start in range(0, total, self._batch_size):

            batch_records = image_records[
                batch_start : batch_start + self._batch_size
            ]

            batch_paths = [
                record["path"]
                for record in batch_records
            ]

            batch_features = self._process_batch(
                batch_paths
            )

            all_features.append(batch_features)

            for record in batch_records:

                all_labels.append(
                    class_to_idx[record["class_name"]]
                )

                all_paths.append(record["path"])

            prev = processed

            processed += len(batch_records)

            # Log each time a 1000-image boundary is crossed,
            # regardless of batch size.
            if processed // 1000 > prev // 1000:

                print(
                    f"Extracted features for "
                    f"{processed}/{total} images..."
                )

        features_array = np.concatenate(
            all_features,
            axis=0
        )

        labels_array = np.array(
            all_labels,
            dtype=np.int64
        )

        self._save_features(
            features=features_array,
            labels=labels_array,
            image_paths=all_paths
        )

        self._save_class_mapping(
            class_to_idx
        )

        return {

            "total_images":
                total,

            "feature_dimension":
                int(features_array.shape[1]),

            "output_directory":
                self._output_dir
        }

    def _load_model(self) -> nn.Module:
        # Load ResNet50 pretrained on ImageNet and remove the classification head

        # IMAGENET1K_V1 reproduces the original He et al. ResNet50 training
        # recipe. Newer V2 weights exist but V1 is more widely expected in
        # downstream benchmarks and research comparisons.
        base = models.resnet50(
            weights=ResNet50_Weights.IMAGENET1K_V2
        )

        # ResNet50 children in order:
        #   conv1 → bn1 → relu → maxpool →
        #   layer1 → layer2 → layer3 → layer4 →
        #   avgpool → fc
        #
        # Slicing off the last child (fc) leaves avgpool as the final
        # operation. Its output shape is (batch, 2048, 1, 1); the spatial
        # dimensions are squeezed away in _process_batch to yield the
        # 2048-dimensional embedding vector.
        encoder = nn.Sequential(
            *list(base.children())[:-1]
        )

        # eval() disables dropout and batch-norm running-stat updates,
        # which is mandatory for deterministic inference.
        encoder.eval()

        encoder.to(self._device)

        return encoder

    def _build_transform(self) -> transforms.Compose:
        # Build the ImageNet preprocessing pipeline

        # Images are already 224×224 RGB JPEGs so no spatial transform is
        # needed. The two steps below mirror exactly what ResNet50 expects:
        #
        #   ToTensor()  — converts a PIL Image (H×W×C uint8, range 0–255)
        #                 to a float32 tensor (C×H×W, range 0.0–1.0).
        #
        #   Normalize() — shifts each channel to zero mean and unit variance
        #                 using the per-channel statistics of ImageNet.
        #                 Deviating from these exact values moves the input
        #                 distribution away from what the weights were trained
        #                 on and degrades feature quality.
        return transforms.Compose([

            transforms.ToTensor(),

            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def _collect_image_paths(self) -> list[dict]:
        # Walk dataset_dir / split / class_name to collect all JPEG records

        image_records: list[dict] = []

        for split in os.listdir(
            self._dataset_dir
        ):

            split_dir = os.path.join(
                self._dataset_dir,
                split
            )

            if not os.path.isdir(split_dir):
                continue

            # Sort so class order is deterministic and consistent
            # with the integer encoding produced by class_to_idx.
            for class_name in sorted(
                os.listdir(split_dir)
            ):

                class_dir = os.path.join(
                    split_dir,
                    class_name
                )

                if not os.path.isdir(class_dir):
                    continue

                for filename in sorted(
                    os.listdir(class_dir)
                ):

                    if not filename.lower().endswith(".jpg"):
                        continue

                    image_path = os.path.join(
                        class_dir,
                        filename
                    )

                    image = Image.open(image_path)

                    image_records.append({

                        "path": image_path,

                        "class_name": class_name,

                        "split": split
                    })

        return image_records

    def _process_batch(
        self,
        image_paths: list[str]
    ) -> np.ndarray:
        # Load images, preprocess and run one forward pass through the encoder

        tensors = []

        for path in image_paths:

            # convert("RGB") is a safety net: images on disk should already
            # be RGB JPEGs from ImageStandardizer, but a JPEG saved from a
            # greyscale source is still valid JPEG and may decode as "L".
            image = Image.open(path).convert("RGB")

            tensors.append(
                self._transform(image)
            )

        # Stack list of (3, 224, 224) tensors → (batch, 3, 224, 224)
        # and move the whole batch to the target device in one transfer.
        batch_tensor = torch.stack(tensors).to(
            self._device
        )

        with torch.no_grad():

            # Encoder output shape: (batch, 2048, 1, 1)
            output = self._model(batch_tensor)

        # Remove the trailing spatial dimensions to get (batch, 2048).
        # Two explicit squeeze calls are used instead of squeeze() to avoid
        # accidentally collapsing the batch dimension when batch_size == 1.
        features = output.squeeze(-1).squeeze(-1)

        return features.cpu().numpy()

    def _save_features(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        image_paths: list[str]
    ) -> None:
        # Persist feature matrix, label vector and path list to output_dir

        # features.npy — float32 array of shape (N, 2048)
        np.save(
            os.path.join(self._output_dir, "features.npy"),
            features
        )

        # labels.npy — int64 array of shape (N,), values are class indices
        # that map to class names via the feature_index column in metadata.csv
        np.save(
            os.path.join(self._output_dir, "labels.npy"),
            labels
        )

        # image_paths.json — ordered list of absolute file paths;
        # index i corresponds to row i in features.npy and labels.npy
        paths_file = os.path.join(
            self._output_dir,
            "image_paths.json"
        )

        with open(
            paths_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                image_paths,
                f,
                indent=2
            )

    def _save_class_mapping(
        self,
        class_to_idx: dict[str, int]
    ) -> None:

        filepath = os.path.join(
            self._output_dir,
            "class_mapping.json"
        )

        with open(
            filepath,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                class_to_idx,
                f,
                indent=4
            )
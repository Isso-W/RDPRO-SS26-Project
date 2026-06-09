import os

import pandas as pd


class ImageMetadataExtractor:

    def __init__(
        self,
        output_dir: str
    ):
        # Initialize metadata extractor

        self._output_dir = output_dir

        os.makedirs(
            output_dir,
            exist_ok=True
        )

    def extract(
        self,
        dataset
    ) -> dict:
        # Main entry point

        return self._extract_metadata(dataset)

    def _extract_metadata(
        self,
        dataset
    ) -> dict:
        # Collect rows from every split, save CSV, return summary

        rows = []

        for split in dataset.keys():

            rows.extend(
                self._extract_split_metadata(
                    dataset,
                    split
                )
            )

        metadata_path = self._save_metadata(rows)

        return {

            "metadata_rows":
                len(rows),

            "metadata_file":
                metadata_path
        }

    def _extract_split_metadata(
        self,
        dataset,
        split: str
    ) -> list[dict]:
        # Extract one row per image in the given split

        rows = []

        # Per-class index counter, reset for each split.
        # Must match ImageStandardizer's naming convention exactly
        # so that image_path values align with the saved JPEG files.
        label_indices = {}

        # ClassLabel feature resolves integer label_ids to human-readable
        # class names, e.g. 0 → "apple_pie" for Food101.
        label_feature = (
            dataset[split]
            .features["label"]
        )

        for sample in dataset[split]:
            
            image_key = (
                "image"
                if "image" in sample
                else "img"
            )

            image = sample[image_key]

            label_id = sample["label"]

            class_name = label_feature.int2str(
                label_id
            )

            # Mirror the ImageStandardizer counter so the generated
            # path matches the file that will be written later.
            index = label_indices.get(
                class_name,
                0
            )

            # Relative path from the workspace directory.
            # Format: processed_dataset/<split>/<class_name>/<index>.jpg
            image_path = os.path.join(
                "processed_dataset",
                split,
                class_name,
                f"{index:06d}.jpg"
            )

            label_indices[class_name] = index + 1

            # None is the PIL default when format cannot be determined
            # (e.g. images created in memory); normalise to "Unknown".
            fmt = image.format or "Unknown"

            rows.append({

                "image_path":
                    image_path,

                "class_name":
                    class_name,

                "label_id":
                    label_id,

                "split":
                    split,

                "original_width":
                    image.width,

                "original_height":
                    image.height,

                "original_aspect_ratio":
                    round(image.width / image.height, 4),

                "original_mode":
                    image.mode,

                "original_format":
                    fmt
            })

        return rows

    def _save_metadata(
        self,
        rows: list[dict]
    ) -> str:
        # Build DataFrame and write CSV to output_dir

        df = pd.DataFrame(rows)

        metadata_path = os.path.join(
            self._output_dir,
            "metadata.csv"
        )

        df.to_csv(
            metadata_path,
            index=False
        )

        return metadata_path

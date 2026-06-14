from pathlib import Path

import pandas as pd
from PIL import Image

from ingestion.kaggle_loader import analyze_benchmark


def test_dog_breed_module2_reads_csv_and_images(tmp_path):
    image_dir = tmp_path / "train"
    image_dir.mkdir()
    rows = []
    for index, breed in enumerate(["akita", "beagle", "akita"]):
        image_id = f"img{index}"
        Image.new("RGB", (32 + index, 24 + index)).save(image_dir / f"{image_id}.jpg")
        rows.append({"id": image_id, "breed": breed})
    csv_path = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    report = analyze_benchmark(
        {
            "train_csv": str(csv_path),
            "image_dir": str(image_dir),
            "image_column": "id",
            "label_column": "breed",
            "image_path_template": "{image}",
            "image_extension": ".jpg",
        }
    )
    assert report["num_classes"] == 2
    assert report["class_distribution"] == {"akita": 2, "beagle": 1}
    assert report["metadata_sample_size"] == 3

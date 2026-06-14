"""Kaggle competition ingestion — download competition data via the Kaggle API.

Entry point for using real Kaggle competition datasets (and their hidden test sets)
in the Jiaozi pipeline. Downloads + extracts a competition, then locates the train
CSV and image directory so the result plugs straight into Module 4's local CSV
dataloader (`_build_local_dataloader`, which already reads `train_csv` / `image_dir` /
`image_column` / `label_column`).

Credentials (one of):
  - `~/.kaggle/kaggle.json` containing {"username": ..., "key": ...}, or
  - env vars `KAGGLE_USERNAME` and `KAGGLE_KEY`.
You must also accept the competition rules on its Kaggle page once, or downloads 403.

Typical use (e.g. on Colab):

    from ingestion.kaggle_loader import ingest_benchmark
    info = ingest_benchmark("cassava", data_root="/content/drive/MyDrive/Jiaozi/kaggle")
    # info -> {train_csv, image_dir, image_column, label_column, ...}  ready for training
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any

from PIL import Image


def _authenticate():
    """Return an authenticated KaggleApi (imported lazily so the module loads without it)."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise RuntimeError(
            "The 'kaggle' package is not installed. Run: pip install kaggle"
        ) from exc
    api = KaggleApi()
    api.authenticate()  # reads ~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY
    return api


def _extract_all_zips(root: Path) -> None:
    """Recursively unzip every .zip under root (competitions often nest zips)."""
    seen: set[Path] = set()
    while True:
        zips = [z for z in root.rglob("*.zip") if z not in seen]
        if not zips:
            break
        for archive in zips:
            seen.add(archive)
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(archive.parent)


def download_competition(competition: str, dest_dir: str | Path, force: bool = False) -> Path:
    """Download + extract a Kaggle competition's files into dest_dir. Returns dest_dir."""
    dest = Path(dest_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    # Skip the download if it already looks populated (e.g. cached on Drive).
    already = any(p.suffix.lower() in {".csv", ".jpg", ".jpeg", ".png"} for p in dest.rglob("*"))
    if already and not force:
        print(f"[kaggle] {dest} already populated; skipping download (force=True to re-fetch).")
        return dest

    api = _authenticate()
    print(f"[kaggle] Downloading competition '{competition}' -> {dest} ...")
    api.competition_download_files(competition, path=str(dest), force=force, quiet=False)
    _extract_all_zips(dest)
    print(f"[kaggle] Done. Extracted under {dest}")
    return dest


def _locate(root: Path, globs: list[str], *, want_dir: bool) -> Path | None:
    """Return the first path under root matching any of the glob patterns."""
    for pattern in globs or []:
        for match in sorted(root.glob(pattern)):
            if want_dir and match.is_dir():
                return match
            if not want_dir and match.is_file():
                return match
    return None


def ingest_benchmark(
    benchmark_key: str,
    data_root: str | Path,
    force: bool = False,
) -> dict:
    """Download a catalog Kaggle benchmark and return paths ready for Module 4 training.

    The returned dict mirrors the config keys consumed by Module 4's local dataloader:
    train_csv, image_dir, image_column, label_column, image_path_template,
    image_extension — plus test_dir / sample_submission when present.
    """
    from vision_benchmark_catalog import get_benchmark

    benchmark = get_benchmark(benchmark_key)
    if benchmark.get("source") != "kaggle":
        raise ValueError(
            f"Benchmark {benchmark_key!r} is source={benchmark.get('source')!r}, not 'kaggle'. "
            "Use the HuggingFace path (pipeline --dataset) for non-Kaggle benchmarks."
        )

    competition = benchmark["competition"]
    dest = download_competition(competition, Path(data_root) / benchmark_key, force=force)

    train_csv = _locate(dest, benchmark.get("csv_globs", []), want_dir=False)
    image_dir = _locate(dest, benchmark.get("image_dir_globs", []), want_dir=True)
    if train_csv is None or image_dir is None:
        available = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*"))[:40]
        raise FileNotFoundError(
            f"Could not locate train_csv ({benchmark.get('csv_globs')}) or image_dir "
            f"({benchmark.get('image_dir_globs')}) under {dest}.\nFirst entries: {available}"
        )

    # Best-effort test-set discovery (hidden labels — used for prediction + submission).
    test_dir = _locate(dest, ["**/test_images", "**/test"], want_dir=True)
    sample_submission = _locate(dest, ["**/sample_submission.csv"], want_dir=False)

    info = {
        "benchmark": benchmark_key,
        "competition": competition,
        "train_csv": str(train_csv),
        "image_dir": str(image_dir),
        "image_column": benchmark.get("image_column", "image"),
        "label_column": benchmark.get("label_column", "label"),
        "image_path_template": benchmark.get("image_path_template", "{image}"),
        "image_extension": benchmark.get("image_extension", ""),
        "num_classes": benchmark.get("num_classes"),
        "metric": benchmark.get("metric"),
        "test_dir": str(test_dir) if test_dir else None,
        "sample_submission": str(sample_submission) if sample_submission else None,
    }
    print(f"[kaggle] train_csv={info['train_csv']}")
    print(f"[kaggle] image_dir={info['image_dir']}")
    if info["test_dir"]:
        print(f"[kaggle] test_dir={info['test_dir']} (labels hidden — predict + submit to score)")
    return info


def read_class_stats(train_csv: str | Path, label_column: str) -> tuple[int, dict, int]:
    """Read labels from the competition CSV (no image decoding).

    Returns (num_classes, class_distribution, total_rows).
    """
    import pandas as pd
    from collections import Counter

    frame = pd.read_csv(train_csv)
    if label_column not in frame.columns:
        raise ValueError(
            f"label_column {label_column!r} not in {train_csv} columns: {list(frame.columns)}"
        )
    labels = frame[label_column].tolist()
    distribution = dict(Counter(labels))
    return len(distribution), distribution, len(labels)


class _LocalImageSplit:
    """Minimal Dataset-like split used by the existing Module 2 analyzer."""

    column_names = ["image", "label"]

    def __init__(self, image_paths: list[Path], labels: list[Any]):
        self.image_paths = image_paths
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, key):
        if key == "label":
            return self.labels
        if key == "image":
            return [Image.open(path).convert("RGB") for path in self.image_paths]
        path = self.image_paths[int(key)]
        with Image.open(path) as image:
            loaded = image.convert("RGB")
        return {"image": loaded, "label": self.labels[int(key)]}


def build_module2_dataset(info: dict) -> dict[str, _LocalImageSplit]:
    """Build a lazy local dataset so Dog Breed runs the real Module 2 analyzer."""
    import pandas as pd

    frame = pd.read_csv(info["train_csv"])
    image_column = info["image_column"]
    label_column = info["label_column"]
    missing = {image_column, label_column} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")
    root = Path(info["image_dir"])
    template = info.get("image_path_template", "{image}")
    extension = info.get("image_extension", "")
    paths = []
    for _, row in frame.iterrows():
        relative = template.format(
            image=row[image_column],
            label=row[label_column],
        )
        path = root / relative
        if extension and not str(path).lower().endswith(extension.lower()):
            path = Path(f"{path}{extension}")
        if not path.is_file():
            raise FileNotFoundError(f"Training image referenced by CSV is missing: {path}")
        paths.append(path)
    return {"train": _LocalImageSplit(paths, frame[label_column].tolist())}


def analyze_benchmark(info: dict) -> dict:
    """Run the standard ImageStatisticsAnalyzer on Kaggle CSV + real image samples."""
    from analyzer.image_statistics import ImageStatisticsAnalyzer

    return ImageStatisticsAnalyzer().analyze(build_module2_dataset(info))


def build_module3_input(
    info: dict,
    priority: str = "balanced",
    description: str | None = None,
    constraints: dict | None = None,
) -> dict:
    """Turn an ``ingest_benchmark`` result into a Module 3 retrieval input.

    `data_size` and `class_imbalance` are derived from the CSV labels (cheap), so the
    Kaggle flow still goes through our own Module 3 selection rather than a hard-coded
    backbone. `priority` defaults to "balanced" (favours a finetuneable CNN — a good fit
    for fine-grained Kaggle classification).
    """
    from pipeline import derive_data_size, derive_class_imbalance

    num_classes, class_distribution, total = read_class_stats(
        info["train_csv"], info["label_column"]
    )
    data_size = derive_data_size(total, num_classes=num_classes, task_type="classification")

    merged_constraints = dict(constraints or {})
    merged_constraints["class_imbalance"] = (
        merged_constraints.get("class_imbalance", False)
        or derive_class_imbalance(class_distribution)
    )

    return {
        "task_type": "classification",
        "data_size": data_size,
        "priority": priority,
        "constraints": merged_constraints,
        "description": description or f"{info.get('benchmark', 'kaggle')} image classification",
        "num_classes": num_classes,
    }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Download a Kaggle benchmark from the catalog.")
    parser.add_argument("benchmark", help="Catalog key, e.g. cassava / state_farm / siim_isic")
    parser.add_argument("--data-root", default="./kaggle_data", help="Where to download/extract.")
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    args = parser.parse_args()

    result = ingest_benchmark(args.benchmark, args.data_root, force=args.force)
    print(json.dumps(result, indent=2, ensure_ascii=False))

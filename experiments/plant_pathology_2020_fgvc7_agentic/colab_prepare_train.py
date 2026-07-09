"""Colab helper for the Jiaozi Plant Pathology 2020 FGVC7 project.

Run from the repository root or from this experiment folder after cloning the
generated code on Colab. The script downloads Kaggle data, materializes the
one-hot labels through Jiaozi ingestion, writes stable stratified folds, patches
the generated configs with the actual Colab paths, and optionally trains.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


COMPETITION = "plant-pathology-2020-fgvc7"
LABEL_COLUMNS = ["healthy", "multiple_diseases", "rust", "scab"]


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "run_kaggle_benchmark.py").exists() and (candidate / "ingestion").exists():
            return candidate
    raise RuntimeError("Could not find Jiaozi repository root.")


EXPERIMENT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = _find_repo_root(EXPERIMENT_ROOT)
MODULE4_DIR = EXPERIMENT_ROOT / "module4_code"


def _ensure_import_paths() -> None:
    for path in (REPO_ROOT, MODULE4_DIR):
        raw = str(path)
        if raw not in sys.path:
            sys.path.insert(0, raw)


def _make_stratified_folds(
    train_csv: Path,
    *,
    image_column: str,
    label_column: str,
    output: Path,
    n_splits: int,
    seed: int,
) -> Path:
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold

    frame = pd.read_csv(train_csv)
    if image_column not in frame.columns:
        raise ValueError(f"{image_column!r} not found in {train_csv}")
    if label_column not in frame.columns:
        raise ValueError(f"{label_column!r} not found in {train_csv}")

    labels = frame[label_column].astype(str)
    min_count = int(labels.value_counts().min())
    splits = max(2, min(n_splits, min_count))
    splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)

    folds: list[list[str]] = []
    for _, val_idx in splitter.split(frame[image_column].astype(str), labels):
        folds.append(frame.iloc[val_idx][image_column].astype(str).tolist())

    payload = {
        "competition": COMPETITION,
        "strategy": "StratifiedKFold",
        "n_folds": len(folds),
        "seed": seed,
        "id_column": image_column,
        "label_column": label_column,
        "label_distribution": labels.value_counts().to_dict(),
        "folds": folds,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def _patch_config_item(item: dict, info: dict, fold_file: Path, args: argparse.Namespace) -> None:
    patch = {
        "train_csv": info["train_csv"],
        "image_dir": info["image_dir"],
        "image_column": info["image_column"],
        "label_column": info["label_column"],
        "label_columns": LABEL_COLUMNS,
        "image_path_template": info["image_path_template"],
        "image_extension": info["image_extension"],
        "evaluation_metric": "roc_auc",
        "offline_smoke": False,
        "fold_file": str(fold_file),
        "fold_index": args.fold_index,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "image_size": args.image_size,
        "use_class_weights": True,
        "class_weight_power": args.class_weight_power,
    }
    for key, value in patch.items():
        item[key] = value
    model_config = item.setdefault("model_config", {})
    if isinstance(model_config, dict):
        for key, value in patch.items():
            model_config[key] = value


def _patch_configs(config_path: Path, info: dict, fold_file: Path, args: argparse.Namespace) -> None:
    configs = json.loads(config_path.read_text(encoding="utf-8"))
    if isinstance(configs, dict):
        configs = [configs]
    if not isinstance(configs, list) or not configs:
        raise ValueError(f"Expected a non-empty config list in {config_path}")

    for item in configs:
        if isinstance(item, dict):
            _patch_config_item(item, info, fold_file, args)

    config_path.write_text(json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8")


def _run(cmd: list[str], cwd: Path) -> None:
    print("[colab]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare and optionally train Plant Pathology 2020 on Colab.")
    parser.add_argument("--data-root", default="/content/kaggle_data")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--class-weight-power", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train", action="store_true", help="Train the first config after preparation.")
    parser.add_argument("--train-all", action="store_true", help="Train every generated candidate config.")
    parser.add_argument("--make-submission", action="store_true", help="Write submission.csv after training.")
    parser.add_argument("--submit", action="store_true", help="Submit submission.csv to Kaggle.")
    parser.add_argument("--log-memory", action="store_true", help="Log public score to OutcomeMemory when available.")
    args = parser.parse_args()

    _ensure_import_paths()

    from ingestion.kaggle_loader import ingest_benchmark

    info = ingest_benchmark(COMPETITION, args.data_root, force=args.force_download)
    train_csv = Path(info["train_csv"])
    fold_file = _make_stratified_folds(
        train_csv,
        image_column=info["image_column"],
        label_column=info["label_column"],
        output=Path(args.data_root) / COMPETITION / "folds.json",
        n_splits=args.folds,
        seed=args.seed,
    )
    _patch_configs(MODULE4_DIR / "configs.json", info, fold_file, args)

    print(f"[colab] prepared configs: {MODULE4_DIR / 'configs.json'}")
    print(f"[colab] folds: {fold_file}")

    if args.train_all:
        _run(
            [sys.executable, "run_experiments.py", "--input", "configs.json", "--epochs", str(args.epochs)],
            MODULE4_DIR,
        )
    elif args.train:
        _run(
            [sys.executable, "run.py", "--config", "configs.json", "--epochs", str(args.epochs)],
            MODULE4_DIR,
        )

    if args.make_submission or args.submit:
        submit_cmd = [
            sys.executable,
            "kaggle_submit.py",
            COMPETITION,
            "--project",
            str(MODULE4_DIR),
            "--data-root",
            args.data_root,
            "--receipt-out",
            str(EXPERIMENT_ROOT / "submission_receipt.json"),
            "--run-manifest",
            str(EXPERIMENT_ROOT / "kaggle_run_manifest.json"),
        ]
        if args.submit:
            submit_cmd.append("--submit")
        if args.log_memory:
            submit_cmd.append("--log-memory")
        _run(submit_cmd, REPO_ROOT)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

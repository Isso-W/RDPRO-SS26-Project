"""Predict on a Kaggle competition test set and (optionally) submit.

The competition test labels are hidden, so "running on the test set" means: load the
trained model from a generated project, run inference over the test images, write
predictions in the sample_submission format, and submit to Kaggle for scoring.

Usage:
    # write submission.csv only
    python kaggle_submit.py cassava --project ./kaggle_run/module4_code --data-root ./kaggle_data
    # write and submit
    python kaggle_submit.py cassava --project ./kaggle_run/module4_code --data-root ./kaggle_data \
        --submit --message "Jiaozi efficientnet baseline"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _flatten_config(config: dict) -> dict:
    merged = dict(config)
    model_config = config.get("model_config")
    if isinstance(model_config, dict):
        for key, value in model_config.items():
            if value is not None or key not in merged:
                merged[key] = value
    return merged


def load_model(project_dir: str | Path, config_path: str | Path | None = None):
    """Load the trained model + eval transform from a generated project."""
    import json

    import torch

    project_dir = Path(project_dir).resolve()
    if str(project_dir) not in sys.path:
        sys.path.insert(0, str(project_dir))
    cwd = os.getcwd()
    os.chdir(project_dir)
    try:
        from model import build_model
        from train import _build_image_transform

        source = Path(config_path).resolve() if config_path else Path("configs.json")
        raw = json.loads(source.read_text(encoding="utf-8"))
        config = _flatten_config(raw[0] if isinstance(raw, list) else raw)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = build_model(config)

        ckpt_path = Path(str(config.get("checkpoint_dir", "checkpoints"))) / "best_model.pt"
        if not ckpt_path.exists():
            ckpt_path = Path("checkpoints") / "best_model.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No best_model.pt found ({ckpt_path}). Train the project first: "
                f"cd {project_dir} && python -u run.py --epochs N"
            )
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device).eval()
        transform = _build_image_transform(config, "test")
        return model, transform, config, device
    finally:
        os.chdir(cwd)


def predict_directory(
    model,
    transform,
    device,
    image_dir: str | Path,
    batch_size: int = 64,
    *,
    tta_horizontal_flip: bool = False,
):
    """Run inference and return ``[(filename, probability_vector)]``."""
    import torch
    from PIL import Image

    files = sorted(p for p in Path(image_dir).rglob("*") if p.suffix.lower() in _IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f"No images found under {image_dir}")

    results: list[tuple[str, list[float]]] = []
    batch: list = []
    names: list[str] = []

    def _flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(model(x), dim=1)
            if tta_horizontal_flip:
                flipped = torch.softmax(model(torch.flip(x, dims=[3])), dim=1)
                probabilities = (probabilities + flipped) / 2.0
        for name, values in zip(names, probabilities.cpu().tolist()):
            results.append((name, [float(value) for value in values]))

    for path in files:
        image = Image.open(path).convert("RGB")
        batch.append(transform(image))
        names.append(path.name)
        if len(batch) >= batch_size:
            _flush()
            batch, names = [], []
    _flush()
    print(f"[submit] Predicted {len(results)} test images from {image_dir}")
    return results


def index_to_label_map(train_csv: str | Path, label_column: str) -> dict:
    """Reconstruct the model's class-index -> original-label map used at training time.

    Mirrors `_build_local_dataloader`: labels are encoded by `sorted(set, key=str)`.
    """
    import pandas as pd

    labels = pd.read_csv(train_csv)[label_column].tolist()
    unique = sorted(set(labels), key=lambda value: str(value))
    return {index: value for index, value in enumerate(unique)}


def apply_calibrated_imagenet_prior(
    project_dir: str | Path,
    config: dict,
    image_dir: str | Path,
    learned_predictions,
    *,
    batch_size: int,
):
    """Apply the validation-selected ImageNet breed-prior blend to test predictions."""
    value = config.get("imagenet_prior_blend", False)
    enabled = (
        value.strip().lower() in {"auto", "true", "yes", "on", "1"}
        if isinstance(value, str)
        else bool(value)
    )
    if not enabled:
        return learned_predictions, None

    import numpy as np

    checkpoint_dir = Path(str(config.get("checkpoint_dir", "checkpoints")))
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = Path(project_dir).resolve() / checkpoint_dir
    artifact_path = checkpoint_dir / "validation_probabilities.npz"
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"ImageNet-prior calibration artifact not found: {artifact_path}"
        )
    artifact = np.load(artifact_path, allow_pickle=False)
    if "prior_alpha" not in artifact:
        raise ValueError(
            f"Validation artifact does not contain prior_alpha: {artifact_path}"
        )
    alpha = float(np.asarray(artifact["prior_alpha"]).item())
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"Invalid ImageNet-prior alpha: {alpha}")
    if alpha <= 0.0:
        return learned_predictions, {
            "alpha": 0.0,
            "validation_artifact": str(artifact_path),
            "prior_model": str(config.get("imagenet_prior_model", "")),
        }

    project = Path(project_dir).resolve()
    if str(project) not in sys.path:
        sys.path.insert(0, str(project))
    from ensemble import combine_prediction_sets
    from imagenet_prior import predict_prior_directory

    prior_predictions, metadata = predict_prior_directory(
        config,
        image_dir,
        batch_size=batch_size,
    )
    combined = combine_prediction_sets(
        [learned_predictions, prior_predictions],
        [1.0 - alpha, alpha],
    )
    return combined, {
        **metadata,
        "alpha": alpha,
        "validation_artifact": str(artifact_path),
    }


def write_submission(predictions, index_to_label, sample_submission, out_path):
    """Write calibrated probabilities using the sample's exact class-column order."""
    import pandas as pd
    import numpy as np

    sample = pd.read_csv(sample_submission)
    columns = list(sample.columns)
    id_column, class_columns = columns[0], columns[1:]
    if len(class_columns) == 1 and class_columns[0] not in {
        str(label) for label in index_to_label.values()
    }:
        target_column = class_columns[0]
        by_key = {}
        for filename, probabilities in predictions:
            label = index_to_label[int(max(range(len(probabilities)), key=probabilities.__getitem__))]
            by_key[filename] = label
            by_key[Path(filename).stem] = label
        values = []
        for raw_id in sample[id_column].tolist():
            value = by_key.get(str(raw_id), by_key.get(Path(str(raw_id)).stem))
            if value is None:
                raise ValueError(f"Sample ID {raw_id} has no prediction.")
            values.append(value)
        sample[target_column] = values
        sample.to_csv(out_path, index=False)
        print(f"[submit] Wrote {out_path} (id='{id_column}', target='{target_column}')")
        return out_path
    if len(class_columns) != len(index_to_label):
        raise ValueError(
            f"Sample has {len(class_columns)} class columns but training has "
            f"{len(index_to_label)} labels."
        )
    expected_labels = [str(index_to_label[index]) for index in range(len(index_to_label))]
    if set(class_columns) != set(expected_labels):
        raise ValueError("Sample submission class columns do not match the training labels.")

    by_key: dict[str, list[float]] = {}
    for filename, probabilities in predictions:
        by_key[filename] = probabilities
        by_key[Path(filename).stem] = probabilities

    def _lookup(raw_id):
        key = str(raw_id)
        return by_key.get(key, by_key.get(Path(key).stem))

    rows = []
    missing = []
    label_to_index = {str(label): index for index, label in index_to_label.items()}
    for raw_id in sample[id_column].tolist():
        probabilities = _lookup(raw_id)
        if probabilities is None:
            missing.append(str(raw_id))
            continue
        if len(probabilities) != len(index_to_label):
            raise ValueError(f"Prediction for {raw_id} has the wrong class count.")
        rows.append([probabilities[label_to_index[column]] for column in class_columns])
    if missing:
        raise ValueError(f"{len(missing)} sample IDs have no prediction; first: {missing[:3]}")
    matrix = np.asarray(rows, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Submission probabilities contain NaN or infinity.")
    if (matrix < 0).any() or (matrix > 1).any():
        raise ValueError("Submission probabilities must be in [0, 1].")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=1.0e-4):
        raise ValueError("Each submission probability row must sum to 1.")
    sample.loc[:, class_columns] = matrix
    sample.to_csv(out_path, index=False)
    print(f"[submit] Wrote {out_path} (id='{id_column}', classes={len(class_columns)})")
    return out_path


def _submission_status(value) -> str:
    name = getattr(value, "name", None)
    if name:
        return str(name).lower()
    raw = getattr(value, "value", value)
    return str(raw or "").rsplit(".", 1)[-1].lower()


def _submission_value(item, *names):
    for name in names:
        value = getattr(item, name, None)
        if value is not None:
            return value
    return None


def submit_and_poll(
    competition: str,
    submission_csv: str | Path,
    message: str,
    *,
    timeout_sec: int = 1800,
    poll_interval_sec: int = 20,
) -> dict:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"[submit] Submitting {submission_csv} to '{competition}' ...")
    api.competition_submit(str(submission_csv), message, competition)
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        submissions = api.competition_submissions(competition)
        if submissions:
            latest = max(
                submissions,
                key=lambda item: str(getattr(item, "date", "") or getattr(item, "submittedAt", "")),
            )
            status = _submission_status(getattr(latest, "status", ""))
            result = {
                "status": status or "unknown",
                "public_score": _submission_value(
                    latest, "publicScore", "public_score", "_public_score"
                ),
                "private_score": _submission_value(
                    latest, "privateScore", "private_score", "_private_score"
                ),
                "submitted_at": str(
                    getattr(latest, "date", "") or getattr(latest, "submittedAt", "")
                ),
                "description": getattr(latest, "description", message),
            }
            print(f"[submit] status={result['status']} public={result['public_score']}")
            if status in {"complete", "error", "failed", "cancelled"}:
                return result
        time.sleep(max(1, poll_interval_sec))
    return {
        "status": "timeout",
        "public_score": None,
        "private_score": None,
        "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def predict_and_submit(
    benchmark_key: str,
    project_dir: str | Path,
    data_root: str | Path,
    out_path: str | Path | None = None,
    message: str | None = None,
    do_submit: bool = False,
    batch_size: int = 64,
    config_path: str | Path | None = None,
    score_timeout_sec: int = 1800,
    metadata_path: str | Path | None = None,
    selected_experiment: str = "baseline",
    ensemble_members: list[dict] | None = None,
) -> dict:
    from ingestion.kaggle_loader import ingest_benchmark

    # Re-locate the competition files (download is skipped if already present).
    info = ingest_benchmark(benchmark_key, data_root)
    if not info.get("test_dir"):
        raise FileNotFoundError(
            f"No test image directory found for {benchmark_key!r}; cannot predict the test set."
        )

    ensemble_metadata = None
    prior_metadata = None
    if ensemble_members:
        import torch

        from ensemble import combine_prediction_sets, save_probability_artifact

        prediction_sets = []
        weights = []
        artifact_dir = Path(project_dir) / ".jiaozi_ensemble"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        saved_members = []
        for index, member in enumerate(ensemble_members):
            member_config_path = member["config_path"]
            model, transform, config, device = load_model(
                project_dir,
                config_path=member_config_path,
            )
            member_predictions = predict_directory(
                model,
                transform,
                device,
                info["test_dir"],
                batch_size=batch_size,
                tta_horizontal_flip=bool(config.get("tta_horizontal_flip", False)),
            )
            member_predictions, member_prior = apply_calibrated_imagenet_prior(
                project_dir,
                config,
                info["test_dir"],
                member_predictions,
                batch_size=batch_size,
            )
            prediction_sets.append(member_predictions)
            weights.append(float(member["weight"]))
            artifact_path = save_probability_artifact(
                artifact_dir / f"member_{index}.npz",
                ids=[name for name, _ in member_predictions],
                probabilities=[values for _, values in member_predictions],
            )
            saved_members.append(
                {
                    **member,
                    "test_artifact": str(artifact_path),
                    "imagenet_prior": member_prior,
                }
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        predictions = combine_prediction_sets(prediction_sets, weights)
        ensemble_metadata = {
            "members": saved_members,
            "weights": weights,
        }
    else:
        model, transform, config, device = load_model(project_dir, config_path=config_path)
        predictions = predict_directory(
            model,
            transform,
            device,
            info["test_dir"],
            batch_size=batch_size,
            tta_horizontal_flip=bool(config.get("tta_horizontal_flip", False)),
        )
        predictions, prior_metadata = apply_calibrated_imagenet_prior(
            project_dir,
            config,
            info["test_dir"],
            predictions,
            batch_size=batch_size,
        )
    idx_to_label = index_to_label_map(info["train_csv"], info["label_column"])

    out_path = Path(out_path) if out_path else Path(project_dir) / "submission.csv"
    sample = info.get("sample_submission")
    if not sample:
        raise FileNotFoundError(
            f"No sample_submission.csv found for {benchmark_key!r}; cannot format a submission."
        )
    write_submission(predictions, idx_to_label, sample, out_path)

    score = None
    if do_submit:
        score = submit_and_poll(
            info["competition"],
            out_path,
            message or f"Jiaozi {benchmark_key} submission",
            timeout_sec=score_timeout_sec,
        )
    try:
        import subprocess

        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=False,
        ).stdout.strip()
    except Exception:
        commit = ""
    result = {
        "submission": str(out_path),
        "competition": info["competition"],
        "submitted": do_submit,
        "score": score,
        "git_commit": commit,
        "selected_experiment": selected_experiment,
        "ensemble": ensemble_metadata,
        "imagenet_prior": prior_metadata,
        "config_path": str(config_path) if config_path else str(Path(project_dir) / "configs.json"),
    }
    destination = Path(metadata_path) if metadata_path else Path(project_dir) / "submission_result.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["metadata_path"] = str(destination)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict a Kaggle test set and optionally submit.")
    parser.add_argument("benchmark", help="Catalog key, e.g. cassava")
    parser.add_argument("--project", required=True, help="Generated project dir (with checkpoints/best_model.pt).")
    parser.add_argument("--data-root", default="./kaggle_data", help="Where the competition data lives.")
    parser.add_argument("--out", default=None, help="Submission CSV path (default: <project>/submission.csv).")
    parser.add_argument("--submit", action="store_true", help="Submit to Kaggle after writing the CSV.")
    parser.add_argument("--message", default=None, help="Submission message.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--config", default=None, help="Selected experiment config JSON.")
    parser.add_argument("--score-timeout", type=int, default=1800)
    parser.add_argument("--selected-experiment", default="baseline")
    args = parser.parse_args()

    result = predict_and_submit(
        args.benchmark,
        args.project,
        args.data_root,
        out_path=args.out,
        message=args.message,
        do_submit=args.submit,
        batch_size=args.batch_size,
        config_path=args.config,
        score_timeout_sec=args.score_timeout,
        selected_experiment=args.selected_experiment,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

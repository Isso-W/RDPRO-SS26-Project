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
import os
import sys
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


def load_model(project_dir: str | Path):
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

        config = _flatten_config(json.loads(Path("configs.json").read_text(encoding="utf-8"))[0])
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


def predict_directory(model, transform, device, image_dir: str | Path, batch_size: int = 64):
    """Run inference over every image in image_dir.

    Returns [(filename, class_index, probabilities)] where probabilities is a
    softmax list for classification outputs. Older callers that only use the
    first two tuple positions remain compatible.
    """
    import torch
    from PIL import Image

    files = sorted(p for p in Path(image_dir).rglob("*") if p.suffix.lower() in _IMAGE_EXTS)
    if not files:
        raise FileNotFoundError(f"No images found under {image_dir}")

    results: list[tuple[str, int, list[float]]] = []
    batch: list = []
    names: list[str] = []

    def _flush():
        if not batch:
            return
        x = torch.stack(batch).to(device)
        with torch.no_grad():
            logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu()
        for name, idx, prob in zip(names, probs.argmax(dim=1).tolist(), probs.tolist()):
            results.append((name, int(idx), [float(value) for value in prob]))

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


def write_submission(predictions, index_to_label, sample_submission, out_path, label_columns=None,
                     positive_class=None):
    """Fill the sample_submission with predictions, matching ids by filename or stem.

    ``positive_class`` (for single-target binary competitions scored on a
    probability, e.g. ROC AUC / log loss) makes the target column receive the
    probability of that class rather than the argmax label.
    """
    import pandas as pd

    sample = pd.read_csv(sample_submission)
    columns = list(sample.columns)
    id_column, target_column = columns[0], columns[-1]
    one_hot_columns = [column for column in (label_columns or []) if column in sample.columns]
    # Auto-detect probability-per-class submissions (e.g. Dog Breed's 120 columns,
    # Leaf's 99) when no label_columns were configured: every non-id column is a
    # class probability, matched to predictions by class name.
    if not one_hot_columns and positive_class is None and len(columns) > 2:
        one_hot_columns = columns[1:]

    positive_index = None
    if positive_class is not None:
        label_to_index = {str(label): index for index, label in index_to_label.items()}
        positive_index = label_to_index.get(str(positive_class))

    by_key: dict[str, object] = {}
    probabilities_by_key: dict[str, list[float]] = {}
    for item in predictions:
        filename, index = item[0], item[1]
        probabilities = item[2] if len(item) > 2 else None
        label = index_to_label.get(index, index)
        by_key[filename] = label
        by_key[Path(filename).stem] = label
        if probabilities is not None:
            probabilities_by_key[filename] = probabilities
            probabilities_by_key[Path(filename).stem] = probabilities

    def _lookup(raw_id):
        key = str(raw_id)
        if positive_index is not None:
            probs = probabilities_by_key.get(key, probabilities_by_key.get(Path(key).stem))
            if probs is not None and positive_index < len(probs):
                return probs[positive_index]
        return by_key.get(key, by_key.get(Path(key).stem))

    if one_hot_columns:
        # Probability columns may be int-typed placeholders in the sample file.
        for column in one_hot_columns:
            sample[column] = sample[column].astype(float)
        missing = 0
        for position, raw_id in enumerate(sample[id_column].tolist()):
            key = str(raw_id)
            value = by_key.get(key, by_key.get(Path(key).stem))
            probabilities = probabilities_by_key.get(key, probabilities_by_key.get(Path(key).stem))
            if value is None and probabilities is None:
                missing += 1
                continue
            if probabilities is not None:
                for index, probability in enumerate(probabilities):
                    label = str(index_to_label.get(index, index))
                    if label in one_hot_columns:
                        sample.at[position, label] = probability
            else:
                label = str(value)
                for column in one_hot_columns:
                    sample.at[position, column] = 1.0 if column == label else 0.0
        sample.to_csv(out_path, index=False)
        if missing:
            print(f"[submit] WARNING: {missing} sample ids had no matching prediction (kept sample defaults).")
        print(f"[submit] Wrote {out_path}  (id='{id_column}', targets={one_hot_columns})")
        return out_path

    filled = []
    missing = 0
    for position, raw_id in enumerate(sample[id_column].tolist()):
        value = _lookup(raw_id)
        if value is None:
            missing += 1
            value = sample.iloc[position][target_column]
        filled.append(value)
    sample[target_column] = filled
    sample.to_csv(out_path, index=False)
    if missing:
        print(f"[submit] WARNING: {missing} sample ids had no matching prediction (kept sample default).")
    print(f"[submit] Wrote {out_path}  (id='{id_column}', target='{target_column}')")
    return out_path


def _get_attr(obj, *names):
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


def submit(competition: str, submission_csv: str | Path, message: str) -> dict:
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    print(f"[submit] Submitting {submission_csv} to '{competition}' ...")
    response = api.competition_submit(str(submission_csv), message, competition)
    details = {"api_response": str(response) if response is not None else None}

    try:
        submissions = api.competition_submissions(competition)
    except Exception as exc:  # pragma: no cover - depends on Kaggle service/auth state
        details["submission_history_error"] = f"{type(exc).__name__}: {exc}"
        print("[submit] Submitted. Could not retrieve submission history yet.")
        return {"status": "submitted", "public_score": None, "details": details}

    latest = submissions[0] if submissions else None
    status = _get_attr(latest, "status", "Status") or "submitted"
    public_score = _get_attr(latest, "publicScore", "public_score", "score", "Score")
    details["latest_submission"] = str(latest) if latest is not None else None
    print(f"[submit] Submitted. status={status} public_score={public_score}")
    return {"status": status, "public_score": public_score, "details": details}


def predict_and_submit(
    benchmark_key: str,
    project_dir: str | Path,
    data_root: str | Path,
    out_path: str | Path | None = None,
    message: str | None = None,
    do_submit: bool = False,
    batch_size: int = 64,
    receipt_out: str | Path | None = None,
    log_memory: bool = False,
    memory_path: str | Path | None = None,
    run_manifest_path: str | Path | None = None,
) -> dict:
    from ingestion.kaggle_loader import ingest_benchmark
    from kaggle_orchestrator import log_kaggle_outcome_if_scored, write_submission_receipt

    # Re-locate the competition files (download is skipped if already present).
    info = ingest_benchmark(benchmark_key, data_root)
    if not info.get("test_dir"):
        raise FileNotFoundError(
            f"No test image directory found for {benchmark_key!r}; cannot predict the test set."
        )

    model, transform, _config, device = load_model(project_dir)
    predictions = predict_directory(model, transform, device, info["test_dir"], batch_size=batch_size)
    idx_to_label = index_to_label_map(info["train_csv"], info["label_column"])

    out_path = Path(out_path) if out_path else Path(project_dir) / "submission.csv"
    sample = info.get("sample_submission")
    if not sample:
        raise FileNotFoundError(
            f"No sample_submission.csv found for {benchmark_key!r}; cannot format a submission."
        )
    write_submission(
        predictions,
        idx_to_label,
        sample,
        out_path,
        label_columns=info.get("label_columns"),
        positive_class=info.get("submission_positive_class"),
    )

    submit_result = {"status": "not_submitted", "public_score": None, "details": {}}
    submission_message = message or f"Jiaozi {benchmark_key} submission"
    if do_submit:
        submit_result = submit(info["competition"], out_path, submission_message)

    project_dir = Path(project_dir)
    receipt_path = Path(receipt_out) if receipt_out else project_dir / "submission_receipt.json"
    write_submission_receipt(
        receipt_path,
        benchmark_key=benchmark_key,
        competition=info["competition"],
        submission_csv=out_path,
        submitted=do_submit,
        message=submission_message if do_submit else None,
        status=submit_result.get("status"),
        public_score=submit_result.get("public_score"),
        details=submit_result.get("details"),
    )

    memory_log = None
    if log_memory:
        manifest = (
            Path(run_manifest_path)
            if run_manifest_path
            else project_dir.parent / "kaggle_run_manifest.json"
        )
        memory_log = log_kaggle_outcome_if_scored(
            receipt_path,
            run_manifest_path=manifest,
            project_dir=project_dir,
            memory_path=memory_path,
        )

    return {
        "submission": str(out_path),
        "competition": info["competition"],
        "submitted": do_submit,
        "receipt": str(receipt_path),
        "public_score": submit_result.get("public_score"),
        "memory_log": memory_log,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Predict a Kaggle test set and optionally submit.")
    parser.add_argument("benchmark", help="Catalog key, e.g. cassava")
    parser.add_argument("--project", required=True, help="Generated project dir (with checkpoints/best_model.pt).")
    parser.add_argument("--data-root", default="./kaggle_data", help="Where the competition data lives.")
    parser.add_argument("--out", default=None, help="Submission CSV path (default: <project>/submission.csv).")
    parser.add_argument("--submit", action="store_true", help="Submit to Kaggle after writing the CSV.")
    parser.add_argument("--message", default=None, help="Submission message.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--receipt-out", default=None,
                        help="Where to write submission_receipt.json (default: <project>/submission_receipt.json).")
    parser.add_argument("--log-memory", action="store_true",
                        help="Append the Kaggle public score to outcome memory when the receipt has a score.")
    parser.add_argument("--memory", default=None,
                        help="Outcome-memory JSONL path (default: recommender/outcomes.jsonl).")
    parser.add_argument("--run-manifest", default=None,
                        help="Path to kaggle_run_manifest.json (default: <project>/../kaggle_run_manifest.json).")
    args = parser.parse_args()

    result = predict_and_submit(
        args.benchmark,
        args.project,
        args.data_root,
        out_path=args.out,
        message=args.message,
        do_submit=args.submit,
        batch_size=args.batch_size,
        receipt_out=args.receipt_out,
        log_memory=args.log_memory,
        memory_path=args.memory,
        run_manifest_path=args.run_manifest,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

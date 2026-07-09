"""Train candidate producers and export OOF/test probability artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image

from ensemble_artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    candidate_id,
    clipped_probabilities,
    mean_column_auc,
    now_iso,
    probability_columns,
    truth_columns,
    write_json,
)
from train import _build_image_transform, _classification_logits, _fold_split_indices, train_model
from utils import as_int, compact_config_summary, get_value, load_configs, set_seed


DEFAULT_LABEL_COLUMNS = ["healthy", "multiple_diseases", "rust", "scab"]


def _parse_candidates(raw: str, total: int) -> list[int]:
    if raw.strip().lower() in {"", "all", "*"}:
        return list(range(1, total + 1))
    selected = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 1 or value > total:
            raise ValueError(f"candidate index {value} is outside 1..{total}")
        selected.append(value)
    return sorted(set(selected))


def _label_order(frame: pd.DataFrame, label_column: str, configured: list[str]) -> list[str]:
    training_order = sorted(set(frame[label_column].astype(str).tolist()), key=str)
    if configured and training_order != configured:
        if set(training_order) != set(configured):
            raise ValueError(
                f"Configured label_columns={configured} do not match observed labels={training_order}"
            )
        print(
            "[producer] WARNING: configured label order differs from training encoder order; "
            f"using training order {training_order}."
        )
    return training_order


def _resolved_label_columns(config: dict[str, Any]) -> list[str]:
    configured = list(get_value(config, "label_columns", DEFAULT_LABEL_COLUMNS) or DEFAULT_LABEL_COLUMNS)
    train_csv = Path(str(get_value(config, "train_csv", "") or "")).expanduser()
    label_column = str(get_value(config, "label_column", "label") or "label")
    if train_csv.exists():
        frame = pd.read_csv(train_csv)
        if label_column in frame.columns:
            return _label_order(frame, label_column, configured)
    return configured


def _image_path(
    image_id: str,
    *,
    base_dir: Path,
    image_extension: str,
    image_path_template: str,
    label: str = "",
) -> Path:
    relative = image_path_template.format(
        image=image_id,
        stem=Path(image_id).stem,
        label=label,
    )
    path = base_dir / relative
    if image_extension and not path.suffix:
        path = path.with_suffix(image_extension)
    return path


class _PredictionDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        *,
        image_column: str,
        target_indices: list[int] | None,
        label_values: list[str] | None,
        base_dir: Path,
        image_extension: str,
        image_path_template: str,
        transform,
    ) -> None:
        self.rows = rows.reset_index(drop=True)
        self.image_column = image_column
        self.target_indices = target_indices
        self.label_values = label_values
        self.base_dir = base_dir
        self.image_extension = image_extension
        self.image_path_template = image_path_template
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        image_id = str(row[self.image_column])
        label = self.label_values[index] if self.label_values is not None else ""
        path = _image_path(
            image_id,
            base_dir=self.base_dir,
            image_extension=self.image_extension,
            image_path_template=self.image_path_template,
            label=label,
        )
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        target = -1 if self.target_indices is None else int(self.target_indices[index])
        return tensor, image_id, target


def _predict_probabilities(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    *,
    config: dict[str, Any],
    batch_size: int,
) -> tuple[list[str], list[int], list[list[float]]]:
    workers = as_int(get_value(config, "num_workers", 2), 2)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
    device = next(model.parameters()).device
    model.eval()
    image_ids: list[str] = []
    targets: list[int] = []
    probabilities: list[list[float]] = []
    with torch.no_grad():
        for x, batch_ids, batch_targets in dataloader:
            x = x.to(device, non_blocking=True)
            logits = _classification_logits(model, x, config)
            probs = torch.softmax(logits, dim=1).cpu()
            image_ids.extend(str(value) for value in batch_ids)
            targets.extend(int(value) for value in batch_targets.tolist())
            probabilities.extend(clipped_probabilities(probs.numpy()).tolist())
    return image_ids, targets, probabilities


def _training_rows_for_fold(config: dict[str, Any], fold_index: int) -> tuple[pd.DataFrame, pd.DataFrame, list[int], list[str]]:
    train_csv = Path(str(get_value(config, "train_csv", "") or "")).expanduser()
    if not train_csv.exists():
        raise FileNotFoundError(f"train_csv does not exist: {train_csv}")
    frame = pd.read_csv(train_csv)
    image_column = str(get_value(config, "image_column", "image") or "image")
    label_column = str(get_value(config, "label_column", "label") or "label")
    configured_labels = list(get_value(config, "label_columns", DEFAULT_LABEL_COLUMNS) or DEFAULT_LABEL_COLUMNS)
    if image_column not in frame.columns or label_column not in frame.columns:
        raise ValueError(f"CSV must contain {image_column!r} and {label_column!r}: {train_csv}")

    label_columns = _label_order(frame, label_column, configured_labels)
    label_to_index = {label: index for index, label in enumerate(label_columns)}
    encoded = [label_to_index[str(value)] for value in frame[label_column].tolist()]
    fold_file = str(get_value(config, "fold_file", "") or "").strip()
    if fold_file:
        _, validation_indices = _fold_split_indices(frame, image_column, fold_file, fold_index)
    else:
        from train import _split_indices

        seed = as_int(get_value(config, "seed", 42), 42)
        train_indices, validation_indices = _split_indices(encoded, 0.2, seed)
        _ = train_indices
    validation_frame = frame.iloc[validation_indices].reset_index(drop=True)
    validation_targets = [encoded[index] for index in validation_indices]
    validation_labels = [str(frame.iloc[index][label_column]) for index in validation_indices]
    return frame, validation_frame, validation_targets, validation_labels


def _write_oof(
    model: torch.nn.Module,
    config: dict[str, Any],
    *,
    fold_index: int,
    output_dir: Path,
    batch_size: int,
) -> dict[str, Any]:
    full_frame, validation_frame, validation_targets, validation_labels = _training_rows_for_fold(config, fold_index)
    image_column = str(get_value(config, "image_column", "image") or "image")
    label_column = str(get_value(config, "label_column", "label") or "label")
    label_columns = _label_order(
        full_frame,
        label_column,
        list(get_value(config, "label_columns", DEFAULT_LABEL_COLUMNS) or DEFAULT_LABEL_COLUMNS),
    )
    image_dir = Path(str(get_value(config, "image_dir", "") or "")).expanduser()
    if not image_dir.exists():
        raise FileNotFoundError(f"image_dir does not exist: {image_dir}")
    transform = _build_image_transform(config, "test")
    dataset = _PredictionDataset(
        validation_frame,
        image_column=image_column,
        target_indices=validation_targets,
        label_values=validation_labels,
        base_dir=image_dir,
        image_extension=str(get_value(config, "image_extension", "") or ""),
        image_path_template=str(get_value(config, "image_path_template", "{image}") or "{image}"),
        transform=transform,
    )
    image_ids, targets, probabilities = _predict_probabilities(
        model,
        dataset,
        config=config,
        batch_size=batch_size,
    )

    prob_cols = probability_columns(label_columns)
    true_cols = truth_columns(label_columns)
    rows: dict[str, Any] = {
        image_column: image_ids,
        "fold": [fold_index] * len(image_ids),
        "target_index": targets,
        "target_label": [label_columns[index] for index in targets],
    }
    for index, column in enumerate(true_cols):
        rows[column] = [1.0 if target == index else 0.0 for target in targets]
    for index, column in enumerate(prob_cols):
        rows[column] = [float(prob[index]) for prob in probabilities]

    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    oof_path = output_dir / "oof.csv"
    frame.to_csv(oof_path, index=False)
    metric = mean_column_auc(frame[true_cols], frame[prob_cols], label_columns)
    return {
        "path": str(oof_path),
        "rows": int(len(frame)),
        "metric": metric,
    }


def _sample_rows(sample_submission: Path) -> pd.DataFrame:
    sample = pd.read_csv(sample_submission)
    if sample.empty:
        raise ValueError(f"sample submission is empty: {sample_submission}")
    return sample[[sample.columns[0]]].rename(columns={sample.columns[0]: "__sample_id"})


def _write_test_probs(
    model: torch.nn.Module,
    config: dict[str, Any],
    *,
    fold_index: int,
    output_dir: Path,
    sample_submission: Path,
    test_dir: Path | None,
    batch_size: int,
) -> dict[str, Any]:
    label_columns = _resolved_label_columns(config)
    base_dir = test_dir or Path(str(get_value(config, "image_dir", "") or "")).expanduser()
    if not base_dir.exists():
        raise FileNotFoundError(f"test image directory does not exist: {base_dir}")
    sample_ids = _sample_rows(sample_submission)
    transform = _build_image_transform(config, "test")
    dataset = _PredictionDataset(
        sample_ids,
        image_column="__sample_id",
        target_indices=None,
        label_values=None,
        base_dir=base_dir,
        image_extension=str(get_value(config, "image_extension", "") or ""),
        image_path_template=str(get_value(config, "image_path_template", "{image}") or "{image}"),
        transform=transform,
    )
    image_ids, _, probabilities = _predict_probabilities(
        model,
        dataset,
        config=config,
        batch_size=batch_size,
    )
    prob_cols = probability_columns(label_columns)
    rows: dict[str, Any] = {
        "image_id": image_ids,
        "fold": [fold_index] * len(image_ids),
    }
    for index, column in enumerate(prob_cols):
        rows[column] = [float(prob[index]) for prob in probabilities]
    frame = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_path = output_dir / "test_probs.csv"
    frame.to_csv(test_path, index=False)
    return {
        "path": str(test_path),
        "rows": int(len(frame)),
    }


def _configure_for_artifacts(
    config: dict[str, Any],
    *,
    index: int,
    fold_index: int,
    seed: int,
    artifact_root: Path,
) -> tuple[dict[str, Any], Path, Path]:
    cid = candidate_id(index)
    run_dir = artifact_root / cid / f"fold_{fold_index}"
    checkpoint_dir = run_dir / "checkpoints"
    patched = dict(config)
    patched["seed"] = seed
    patched["fold_index"] = fold_index
    patched["checkpoint_dir"] = str(checkpoint_dir)
    patched["export_preds_path"] = str(run_dir / "legacy_val_predictions.json")
    return patched, run_dir, checkpoint_dir


def run_producers(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs = load_configs(args.input, [])
    selected = _parse_candidates(args.candidates, len(configs))
    artifact_root = Path(args.artifact_root)
    sample_submission = Path(args.sample_submission).expanduser() if args.sample_submission else None
    test_dir = Path(args.test_dir).expanduser() if args.test_dir else None
    summaries = []

    for index in selected:
        base_config = configs[index - 1]
        fold_index = args.fold_index
        if fold_index is None:
            fold_index = int(get_value(base_config, "fold_index", 0) or 0)
        config, run_dir, checkpoint_dir = _configure_for_artifacts(
            base_config,
            index=index,
            fold_index=fold_index,
            seed=args.seed,
            artifact_root=artifact_root,
        )
        set_seed(args.seed)
        epochs = args.epochs
        if epochs is None:
            epochs = as_int(get_value(config, "recommended_epochs", 10), 10)
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(run_dir / "config.json", config)
        print(f"[producer] training {candidate_id(index)} fold={fold_index} -> {run_dir}")
        model, train_result = train_model(config, epochs=epochs, max_steps=0, save_dir=str(checkpoint_dir))
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_checkpoint = checkpoint_dir / "best_model.pt"
        if not best_checkpoint.exists():
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "train": train_result,
                    "fallback_saved_by": "producer.py",
                },
                best_checkpoint,
            )
        predict_batch_size = args.predict_batch_size or as_int(
            get_value(config, "eval_batch_size", get_value(config, "batch_size", 16)),
            16,
        )
        oof = _write_oof(
            model,
            config,
            fold_index=fold_index,
            output_dir=run_dir,
            batch_size=predict_batch_size,
        )
        test_probs = None
        if sample_submission is not None:
            test_probs = _write_test_probs(
                model,
                config,
                fold_index=fold_index,
                output_dir=run_dir,
                sample_submission=sample_submission,
                test_dir=test_dir,
                batch_size=predict_batch_size,
            )
        label_columns = _resolved_label_columns(config)
        manifest = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "created_at": now_iso(),
            "candidate_index": index,
            "candidate_id": candidate_id(index),
            "fold_index": fold_index,
            "label_columns": label_columns,
            "prediction_columns": probability_columns(label_columns),
            "config_summary": compact_config_summary(config, rank_default=index),
            "paths": {
                "run_dir": str(run_dir),
                "checkpoint_dir": str(checkpoint_dir),
                "best_checkpoint": str(checkpoint_dir / "best_model.pt"),
                "oof": oof["path"],
                "test_probs": test_probs["path"] if test_probs else None,
            },
            "train": train_result,
            "oof": oof,
            "test_probs": test_probs,
        }
        write_json(run_dir / "producer_manifest.json", manifest)
        summaries.append(manifest)
        print(
            "[producer] done "
            f"{candidate_id(index)} fold={fold_index} "
            f"oof_auc={oof['metric'].get('metric_value')}"
        )
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description="Train candidate producers and export OOF/test probabilities.")
    parser.add_argument("--input", default="configs.json")
    parser.add_argument("--artifact-root", default="producer_artifacts")
    parser.add_argument("--candidates", default="all", help="1-based list such as '1,3', or 'all'.")
    parser.add_argument("--fold-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--predict-batch-size", type=int, default=None)
    parser.add_argument("--sample-submission", default=None)
    parser.add_argument("--test-dir", default=None)
    args = parser.parse_args()
    result = run_producers(args)
    print(json.dumps({"status": "success", "runs": len(result)}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

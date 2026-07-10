"""Blend producer artifacts into the final Plant Pathology submission."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ensemble_artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    clipped_probabilities,
    mean_column_auc,
    now_iso,
    probability_columns,
    rank_normalize,
    read_json,
    truth_columns,
    weight_grid,
    write_json,
)


DEFAULT_LABEL_COLUMNS = ["healthy", "multiple_diseases", "rust", "scab"]


def _apply_blend_space(arrays: list[np.ndarray], space: str) -> list[np.ndarray]:
    """Map each candidate's per-column scores into the blending space.

    ``"prob"`` blends raw probabilities. ``"rank"`` (default) rank-normalises
    each candidate's columns first, which suits the mean column-wise ROC AUC
    metric — only ordering matters, and it stops a narrow-band model from
    being drowned out by a wide-range one.
    """

    if space == "prob":
        return list(arrays)
    if space == "rank":
        return [rank_normalize(array) for array in arrays]
    raise ValueError(f"unknown blend space: {space!r}")


def _find_repo_root(start: Path) -> Path | None:
    for candidate in [start, *start.parents]:
        if (candidate / "kaggle_submit.py").exists() and (candidate / "kaggle_orchestrator.py").exists():
            return candidate
    return None


def _add_repo_root_to_path() -> None:
    root = _find_repo_root(Path(__file__).resolve())
    if root is not None and str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _manifest_paths(artifact_root: Path) -> list[Path]:
    return sorted(artifact_root.glob("candidate_*/fold_*/producer_manifest.json"))


def _load_manifests(artifact_root: Path) -> list[dict[str, Any]]:
    manifests = []
    for path in _manifest_paths(artifact_root):
        manifest = read_json(path)
        manifest["_manifest_path"] = str(path)
        if int(manifest.get("schema_version", 0)) != ARTIFACT_SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema_version in {path}: {manifest.get('schema_version')}")
        manifests.append(manifest)
    if not manifests:
        raise FileNotFoundError(f"No producer manifests found under {artifact_root}")
    return manifests


def _validate_columns(frame: pd.DataFrame, columns: list[str], path: str | Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")


def _combine_candidate(
    candidate_id: str,
    manifests: list[dict[str, Any]],
    label_columns: list[str],
) -> dict[str, Any]:
    prob_cols = probability_columns(label_columns)
    true_cols = truth_columns(label_columns)
    oof_frames = []
    test_frames = []
    fold_indices = []
    manifest_paths = []

    for manifest in sorted(manifests, key=lambda item: int(item["fold_index"])):
        paths = manifest.get("paths") or {}
        fold_indices.append(int(manifest["fold_index"]))
        manifest_paths.append(manifest["_manifest_path"])
        oof_path = paths.get("oof")
        if not oof_path:
            raise ValueError(f"{candidate_id} fold {manifest['fold_index']} has no OOF path")
        oof = pd.read_csv(oof_path)
        _validate_columns(oof, ["image_id", "fold", *true_cols, *prob_cols], oof_path)
        oof_frames.append(oof)

        test_path = paths.get("test_probs")
        if test_path:
            test = pd.read_csv(test_path)
            _validate_columns(test, ["image_id", "fold", *prob_cols], test_path)
            test_frames.append(test)

    oof_all = pd.concat(oof_frames, ignore_index=True)
    duplicates = oof_all["image_id"].duplicated()
    if duplicates.any():
        duplicate_ids = sorted(oof_all.loc[duplicates, "image_id"].astype(str).unique().tolist())[:10]
        raise ValueError(f"{candidate_id} has duplicate OOF image_ids: {duplicate_ids}")
    oof_all = oof_all.sort_values("image_id").reset_index(drop=True)
    metric = mean_column_auc(oof_all[true_cols], oof_all[prob_cols], label_columns)

    test_mean = None
    if test_frames:
        test_all = pd.concat(test_frames, ignore_index=True)
        grouped = test_all.groupby("image_id", sort=False)[prob_cols].mean().reset_index()
        test_mean = grouped

    return {
        "candidate_id": candidate_id,
        "fold_indices": fold_indices,
        "manifest_paths": manifest_paths,
        "oof": oof_all,
        "test": test_mean,
        "metric": metric,
    }


def _load_candidates(manifests: list[dict[str, Any]], label_columns: list[str]) -> list[dict[str, Any]]:
    by_candidate: dict[str, list[dict[str, Any]]] = {}
    for manifest in manifests:
        if list(manifest.get("label_columns") or []) != label_columns:
            raise ValueError(
                f"label_columns mismatch in {manifest['_manifest_path']}: {manifest.get('label_columns')}"
            )
        by_candidate.setdefault(str(manifest["candidate_id"]), []).append(manifest)
    return [
        _combine_candidate(candidate_id, items, label_columns)
        for candidate_id, items in sorted(by_candidate.items())
    ]


def _validate_oof_alignment(candidates: list[dict[str, Any]], label_columns: list[str]) -> list[str]:
    true_cols = truth_columns(label_columns)
    base_ids = candidates[0]["oof"]["image_id"].astype(str).tolist()
    base_truth = candidates[0]["oof"][true_cols].to_numpy(dtype=float)
    for candidate in candidates[1:]:
        ids = candidate["oof"]["image_id"].astype(str).tolist()
        if ids != base_ids:
            raise ValueError(
                f"OOF image_id coverage/order mismatch: {candidates[0]['candidate_id']} vs {candidate['candidate_id']}"
            )
        truth = candidate["oof"][true_cols].to_numpy(dtype=float)
        if not np.array_equal(base_truth, truth):
            raise ValueError(f"OOF truth columns mismatch for {candidate['candidate_id']}")
    return base_ids


def _validate_test_alignment(
    candidates: list[dict[str, Any]],
    sample_submission: Path,
) -> list[str]:
    sample = pd.read_csv(sample_submission)
    id_column = sample.columns[0]
    sample_ids = sample[id_column].astype(str).tolist()
    for candidate in candidates:
        test = candidate.get("test")
        if test is None:
            raise ValueError(f"{candidate['candidate_id']} has no test_probs artifact")
        test_ids = test["image_id"].astype(str).tolist()
        if sorted(test_ids) != sorted(sample_ids):
            missing = sorted(set(sample_ids) - set(test_ids))[:10]
            extra = sorted(set(test_ids) - set(sample_ids))[:10]
            raise ValueError(
                f"{candidate['candidate_id']} test IDs do not match sample submission; "
                f"missing={missing} extra={extra}"
            )
    return sample_ids


def _candidate_correlation(candidates: list[dict[str, Any]], label_columns: list[str]) -> dict[str, dict[str, float | None]]:
    prob_cols = probability_columns(label_columns)
    flattened = {
        candidate["candidate_id"]: candidate["oof"][prob_cols].to_numpy(dtype=float).reshape(-1)
        for candidate in candidates
    }
    report: dict[str, dict[str, float | None]] = {}
    for left_id, left_values in flattened.items():
        report[left_id] = {}
        for right_id, right_values in flattened.items():
            if np.std(left_values) == 0 or np.std(right_values) == 0:
                report[left_id][right_id] = None
            else:
                report[left_id][right_id] = float(np.corrcoef(left_values, right_values)[0, 1])
    return report


def _score_weights(
    weights: list[float],
    candidate_arrays: list[np.ndarray],
    y_true: np.ndarray,
    label_columns: list[str],
) -> dict[str, Any]:
    blended = np.zeros_like(candidate_arrays[0], dtype=float)
    for weight, values in zip(weights, candidate_arrays):
        blended += float(weight) * values
    return mean_column_auc(y_true, blended, label_columns)


def _select_weights(
    candidates: list[dict[str, Any]],
    label_columns: list[str],
    *,
    step: float,
    blend_space: str = "rank",
) -> dict[str, Any]:
    prob_cols = probability_columns(label_columns)
    true_cols = truth_columns(label_columns)
    arrays = _apply_blend_space(
        [candidate["oof"][prob_cols].to_numpy(dtype=float) for candidate in candidates],
        blend_space,
    )
    y_true = candidates[0]["oof"][true_cols].to_numpy(dtype=float)

    candidate_ids = [candidate["candidate_id"] for candidate in candidates]
    equal_weights = [1.0 / len(candidates)] * len(candidates)
    equal_metric = _score_weights(equal_weights, arrays, y_true, label_columns)
    best = {
        "method": "equal_average",
        "weights": dict(zip(candidate_ids, equal_weights)),
        "metric": equal_metric,
    }

    for weights in weight_grid(len(candidates), step):
        metric = _score_weights(weights, arrays, y_true, label_columns)
        value = metric.get("metric_value")
        best_value = best["metric"].get("metric_value")
        if value is not None and (best_value is None or float(value) > float(best_value)):
            best = {
                "method": "oof_weight_grid",
                "weights": dict(zip(candidate_ids, weights)),
                "metric": metric,
            }
    return {
        "selected": best,
        "equal_average": {
            "weights": dict(zip(candidate_ids, equal_weights)),
            "metric": equal_metric,
        },
        "grid_step": step,
        "blend_space": blend_space,
    }


def _blend_frames(
    frames: list[pd.DataFrame],
    weights: dict[str, float],
    candidates: list[dict[str, Any]],
    label_columns: list[str],
    *,
    ids: list[str],
    include_truth: bool,
    blend_space: str = "rank",
) -> pd.DataFrame:
    prob_cols = probability_columns(label_columns)
    ordered_arrays = _apply_blend_space(
        [
            frame.set_index("image_id").loc[ids][prob_cols].to_numpy(dtype=float)
            for frame in frames
        ],
        blend_space,
    )
    blended = np.zeros((len(ids), len(prob_cols)), dtype=float)
    for values, candidate in zip(ordered_arrays, candidates):
        weight = float(weights[candidate["candidate_id"]])
        blended += weight * values

    rows: dict[str, Any] = {"image_id": ids}
    if include_truth:
        true_cols = truth_columns(label_columns)
        ordered_truth = frames[0].set_index("image_id").loc[ids]
        for column in true_cols:
            rows[column] = ordered_truth[column].to_numpy(dtype=float)
    blended = clipped_probabilities(blended)
    for index, column in enumerate(prob_cols):
        rows[column] = blended[:, index]
    return pd.DataFrame(rows)


def _write_submission(
    test_probs: pd.DataFrame,
    *,
    sample_submission: Path,
    label_columns: list[str],
    output_path: Path,
) -> Path:
    sample = pd.read_csv(sample_submission)
    id_column = sample.columns[0]
    prob_cols = probability_columns(label_columns)
    ordered = test_probs.set_index("image_id").loc[sample[id_column].astype(str).tolist()]
    for label, prob_col in zip(label_columns, prob_cols):
        if label not in sample.columns:
            raise ValueError(f"sample submission missing target column {label!r}")
        sample[label] = ordered[prob_col].to_numpy(dtype=float)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output_path, index=False)
    return output_path


def _write_receipt(
    *,
    receipt_path: Path,
    benchmark_key: str,
    competition: str,
    submission_csv: Path,
    submitted: bool,
    message: str | None,
    status: str | None,
    public_score: Any,
    details: dict[str, Any],
) -> Path:
    _add_repo_root_to_path()
    try:
        from kaggle_orchestrator import write_submission_receipt

        return write_submission_receipt(
            receipt_path,
            benchmark_key=benchmark_key,
            competition=competition,
            submission_csv=submission_csv,
            submitted=submitted,
            message=message,
            status=status,
            public_score=public_score,
            details=details,
        )
    except Exception as exc:
        receipt = {
            "created_at": now_iso(),
            "benchmark_key": benchmark_key,
            "competition": competition,
            "submission_csv": str(submission_csv),
            "submitted": submitted,
            "message": message,
            "status": status or ("submitted" if submitted else "not_submitted"),
            "public_score": public_score,
            "details": {
                **details,
                "receipt_writer_fallback": f"{type(exc).__name__}: {exc}",
            },
        }
        write_json(receipt_path, receipt)
        return receipt_path


def _submit_to_kaggle(competition: str, submission_csv: Path, message: str) -> dict[str, Any]:
    _add_repo_root_to_path()
    from kaggle_submit import submit

    return submit(competition, submission_csv, message)


def run_consumer(args: argparse.Namespace) -> dict[str, Any]:
    artifact_root = Path(args.artifact_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_submission = Path(args.sample_submission).expanduser()
    label_columns = args.label_columns.split(",") if args.label_columns else DEFAULT_LABEL_COLUMNS

    manifests = _load_manifests(artifact_root)
    candidates = _load_candidates(manifests, label_columns)
    _validate_oof_alignment(candidates, label_columns)
    sample_ids = _validate_test_alignment(candidates, sample_submission)

    blend_space = getattr(args, "blend_space", "rank")
    weights = _select_weights(
        candidates, label_columns, step=args.weight_grid_step, blend_space=blend_space
    )
    selected_weights = weights["selected"]["weights"]
    prob_cols = probability_columns(label_columns)
    true_cols = truth_columns(label_columns)

    oof_ids = candidates[0]["oof"]["image_id"].astype(str).tolist()
    ensemble_oof = _blend_frames(
        [candidate["oof"] for candidate in candidates],
        selected_weights,
        candidates,
        label_columns,
        ids=oof_ids,
        include_truth=True,
        blend_space=blend_space,
    )
    ensemble_oof_path = output_dir / "ensemble_oof.csv"
    ensemble_oof.to_csv(ensemble_oof_path, index=False)

    test_blend = _blend_frames(
        [candidate["test"] for candidate in candidates],
        selected_weights,
        candidates,
        label_columns,
        ids=sample_ids,
        include_truth=False,
        blend_space=blend_space,
    )
    test_probs_path = output_dir / "test_probs.csv"
    test_blend.to_csv(test_probs_path, index=False)

    submission_path = Path(args.submission_out) if args.submission_out else output_dir / "submission.csv"
    _write_submission(
        test_blend,
        sample_submission=sample_submission,
        label_columns=label_columns,
        output_path=submission_path,
    )

    report = {
        "created_at": now_iso(),
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "label_columns": label_columns,
        "prediction_columns": prob_cols,
        "truth_columns": true_cols,
        "blend_space": blend_space,
        "candidate_metrics": {
            candidate["candidate_id"]: {
                "fold_indices": candidate["fold_indices"],
                "metric": candidate["metric"],
                "manifest_paths": candidate["manifest_paths"],
            }
            for candidate in candidates
        },
        "correlation": _candidate_correlation(candidates, label_columns),
        "weights": weights,
        "ensemble_oof_metric": mean_column_auc(ensemble_oof[true_cols], ensemble_oof[prob_cols], label_columns),
        "paths": {
            "artifact_root": str(artifact_root),
            "ensemble_oof": str(ensemble_oof_path),
            "test_probs": str(test_probs_path),
            "submission": str(submission_path),
        },
    }
    report_path = write_json(output_dir / "blend_report.json", report)
    pipeline_manifest = {
        "created_at": now_iso(),
        "competition": args.competition,
        "benchmark_key": args.benchmark_key,
        "stage": "consumer_final_blend",
        "producer_manifests": [manifest["_manifest_path"] for manifest in manifests],
        "blend_report": str(report_path),
        "submission_csv": str(submission_path),
        "selected_weights": selected_weights,
        "local_oof_metric": report["ensemble_oof_metric"],
    }
    pipeline_manifest_path = write_json(output_dir / "pipeline_manifest.json", pipeline_manifest)

    submission_message = args.message or f"Jiaozi producer-consumer ensemble {args.benchmark_key}"
    submit_result = {"status": "not_submitted", "public_score": None, "details": {}}
    if args.submit:
        submit_result = _submit_to_kaggle(args.competition, submission_path, submission_message)

    receipt_path = Path(args.receipt_out) if args.receipt_out else output_dir / "submission_receipt.json"
    _write_receipt(
        receipt_path=receipt_path,
        benchmark_key=args.benchmark_key,
        competition=args.competition,
        submission_csv=submission_path,
        submitted=args.submit,
        message=submission_message if args.submit else None,
        status=submit_result.get("status"),
        public_score=submit_result.get("public_score"),
        details={
            **(submit_result.get("details") or {}),
            "blend_report": str(report_path),
            "pipeline_manifest": str(pipeline_manifest_path),
            "local_oof_metric": report["ensemble_oof_metric"],
        },
    )

    memory_log = None
    if args.log_memory:
        _add_repo_root_to_path()
        from kaggle_orchestrator import log_kaggle_outcome_if_scored

        memory_log = log_kaggle_outcome_if_scored(
            receipt_path,
            run_manifest_path=args.run_manifest,
            project_dir=Path(__file__).resolve().parent,
            memory_path=args.memory,
        )

    return {
        "status": "success",
        "submission": str(submission_path),
        "blend_report": str(report_path),
        "pipeline_manifest": str(pipeline_manifest_path),
        "receipt": str(receipt_path),
        "public_score": submit_result.get("public_score"),
        "local_oof_metric": report["ensemble_oof_metric"],
        "memory_log": memory_log,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Blend producer artifacts and write the final submission.")
    parser.add_argument("--artifact-root", default="producer_artifacts")
    parser.add_argument("--output-dir", default="ensembles/final_blend")
    parser.add_argument("--sample-submission", required=True)
    parser.add_argument("--submission-out", default=None)
    parser.add_argument("--label-columns", default=",".join(DEFAULT_LABEL_COLUMNS))
    parser.add_argument("--weight-grid-step", type=float, default=0.05)
    parser.add_argument(
        "--blend-space",
        choices=["rank", "prob"],
        default="rank",
        help="Blend candidates in rank space (default, suits ROC AUC) or raw probability space.",
    )
    parser.add_argument("--benchmark-key", default="plant-pathology-2020-fgvc7")
    parser.add_argument("--competition", default="plant-pathology-2020-fgvc7")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--message", default=None)
    parser.add_argument("--receipt-out", default=None)
    parser.add_argument("--log-memory", action="store_true")
    parser.add_argument("--memory", default=None)
    parser.add_argument("--run-manifest", default=None)
    args = parser.parse_args()
    result = run_consumer(args)
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

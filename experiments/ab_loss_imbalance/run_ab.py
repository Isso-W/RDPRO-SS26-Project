"""Driver for the CE-vs-focal A/B run.

Real training requires a GPU and downloaded Kaggle data. Pure logic such as
fold generation and metric bundling is factored into offline-testable helpers.
The orchestration path calls prepare_project and runs the generated run.py in a
subprocess.

CLI: python -m experiments.ab_loss_imbalance.run_ab --testbed cassava [--data-root ...]
     [--output ...] [--only focal_loss:3]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from experiments.ab_loss_imbalance import configs

RESULTS = Path("experiments/ab_loss_imbalance/results")
OUTCOMES = RESULTS / "outcomes.jsonl"


# Pure logic: stratified folds.
def compute_folds(labels: list, ids: list, n_folds: int = configs.N_FOLDS,
                  seed: int = configs.GLOBAL_SEED) -> list[list[str]]:
    """Return validation sample ids for each stratified fold."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds: list[list[str]] = [[] for _ in range(n_folds)]
    for i, (_, val_idx) in enumerate(skf.split(list(ids), list(labels))):
        folds[i] = [str(ids[j]) for j in val_idx]
    return folds


def fold_spec(
    labels,
    ids,
    n_folds=configs.N_FOLDS,
    seed=configs.GLOBAL_SEED,
    id_column: str | None = None,
) -> dict:
    return {"seed": seed, "n_folds": n_folds, "stratified": True,
            "id_column": id_column,
            "folds": compute_folds(labels, ids, n_folds, seed)}


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# Pure logic: metric bundle from validation predictions.
def metric_bundle(y_true: list, y_prob: list, metrics: list[str]) -> dict:
    """Compute requested metrics from validation predictions."""
    import numpy as np
    from sklearn.metrics import (accuracy_score, average_precision_score,
                                 cohen_kappa_score, f1_score, roc_auc_score)
    yt = np.asarray(y_true)
    yp = np.asarray(y_prob)
    preds = yp.argmax(axis=1)
    binary = yp.shape[1] == 2
    out: dict[str, float | None] = {}
    for m in metrics:
        try:
            if m == "accuracy":
                out[m] = float(accuracy_score(yt, preds))
            elif m == "macro_f1":
                out[m] = float(f1_score(yt, preds, average="macro"))
            elif m == "roc_auc":
                out[m] = float(roc_auc_score(yt, yp[:, 1]) if binary
                               else roc_auc_score(yt, yp, multi_class="ovr"))
            elif m == "pr_auc":
                # PR-AUC is binary-first; leave it empty for multiclass runs.
                out[m] = float(average_precision_score(yt, yp[:, 1])) if binary else None
            elif m == "qwk":
                out[m] = float(cohen_kappa_score(yt, preds, weights="quadratic"))
            else:
                out[m] = None
        except (ValueError, IndexError):
            out[m] = None
    return out


def metrics_for(testbed: str) -> list[str]:
    tb = configs.TESTBEDS[testbed]
    return [tb["metric"], *tb["secondary_metrics"]]


# Resume support: skip completed (arm, fold) pairs.
def completed_pairs(outcomes_path: Path, testbed: str) -> set[tuple]:
    done = set()
    if outcomes_path.exists():
        for line in outcomes_path.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("benchmark") == testbed and "val_metric" in r:
                done.add((r["arm"], r["fold"]))
    return done


def kb_version() -> str:
    repo_root = Path(__file__).resolve().parents[2]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def parse_run_summary(output: str) -> dict:
    """Extract the final JSON summary printed by generated run.py."""
    decoder = json.JSONDecoder()
    summary = {}
    for index, char in enumerate(output):
        if char != "{":
            continue
        try:
            candidate, _end = decoder.raw_decode(output[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and candidate.get("status") == "success" and "train" in candidate:
            summary = candidate
    return summary


def run_training_command(command: list[str], cwd: Path) -> str:
    proc = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    lines: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    return_code = proc.wait()
    output = "".join(lines)
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command, output=output)
    return output


# Orchestration path; requires GPU/data and is not used by offline unit tests.
def run_matrix(testbed: str, data_root: str, output_dir: str,
               only: tuple | None = None) -> None:
    from run_kaggle_benchmark import prepare_project

    RESULTS.mkdir(parents=True, exist_ok=True)
    res = prepare_project(testbed, data_root, output_dir)
    info = res["info"]
    project = Path(output_dir) / "module4_code"

    # 1) Compute folds once and reuse them.
    import pandas as pd
    frame = pd.read_csv(info["train_csv"])
    ids = frame[info["image_column"]].tolist()
    labels = frame[info["label_column"]].tolist()
    fold_path = project / configs.fold_file_name(testbed)
    if not fold_path.exists():
        fold_path.write_text(
            json.dumps(fold_spec(labels, ids, id_column=info["image_column"])),
            encoding="utf-8",
        )
    fold_sha = sha256_of(fold_path)

    # 2) Freeze the base config from configs.BASE; no dynamic Module 3 selection.
    base_cfg = _frozen_config(testbed, info, fold_path)
    done = completed_pairs(OUTCOMES, testbed)

    for loss in configs.ARMS:
        for fold_index in range(configs.N_FOLDS):
            if only and (loss, fold_index) != only:
                continue
            if (loss, fold_index) in done:
                print(f"[run_ab] skip done {loss}:{fold_index}")
                continue
            _run_one(testbed, project, base_cfg, loss, fold_index, fold_sha, info)


def _frozen_config(testbed, info, fold_path) -> dict:
    tb = configs.TESTBEDS[testbed]
    return {
        **configs.BASE,
        "task_type": "classification",
        "num_classes": info.get("num_classes"),
        "train_csv": info["train_csv"], "image_dir": info["image_dir"],
        "image_column": info["image_column"], "label_column": info["label_column"],
        "image_path_template": info["image_path_template"],
        "image_extension": info["image_extension"],
        "image_size": tb["image_size"], "offline_smoke": False,
        "evaluation_metric": tb["metric"],
        "recommended_epochs": tb["epochs"],
        "epochs": tb["epochs"],
        "fold_file": str(fold_path),
    }


def _run_one(testbed, project, base_cfg, loss, fold_index, fold_sha, info) -> None:
    tb = configs.TESTBEDS[testbed]
    preds_path = project / f"val_preds_{loss}_{fold_index}.json"
    cfg = {**base_cfg, "loss": loss, "fold_index": fold_index,
           "export_preds_path": str(preds_path)}
    cfg_path = project / f"config_{loss}_{fold_index}.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    print(f"[run_ab] train {testbed} {loss} fold {fold_index} ...", flush=True)
    output = run_training_command(
        [
            sys.executable,
            "-u",
            "run.py",
            "--config",
            cfg_path.name,
            "--epochs",
            str(tb["epochs"]),
            "--seed",
            str(configs.GLOBAL_SEED),
        ],
        project,
    )
    summary = parse_run_summary(output)
    preds = json.loads(preds_path.read_text(encoding="utf-8"))
    y_score = preds.get("y_prob", preds.get("y_score"))
    if y_score is None:
        raise ValueError(f"{preds_path} missing y_prob/y_score")
    bundle = metric_bundle(preds["y_true"], y_score, metrics_for(testbed))
    _append_outcome(
        testbed,
        loss,
        fold_index,
        bundle,
        fold_sha,
        base_cfg,
        best_epoch=(summary.get("train") or {}).get("best_epoch"),
    )


def _append_outcome(testbed, loss, fold_index, bundle, fold_sha, base_cfg, best_epoch=None) -> None:
    import datetime
    rec = {
        "experiment": "ab_loss_imbalance", "benchmark": testbed,
        "arm": loss, "fold": fold_index, "seed": configs.GLOBAL_SEED,
        "config": {
            k: base_cfg[k]
            for k in ("backbone", "pretrained", "image_size", "epochs", "sampler")
        },
        "val_metric": bundle, "fold_file_sha256": fold_sha,
        "best_epoch": best_epoch,
        "kb_version": kb_version(),
        "date": datetime.date.today().isoformat(),
    }
    with OUTCOMES.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _parse_only(s: str | None) -> tuple | None:
    if not s:
        return None
    arm, fold = s.split(":")
    return (arm, int(fold))


def _cli() -> None:
    ap = argparse.ArgumentParser(description="A/B run driver")
    ap.add_argument("--testbed", required=True, choices=list(configs.TESTBEDS))
    ap.add_argument("--data-root", default="./kaggle_data")
    ap.add_argument("--output", default="./ab_runs")
    ap.add_argument("--only", default=None, help="arm:fold, for example focal_loss:3")
    args = ap.parse_args()
    run_matrix(args.testbed, args.data_root, f"{args.output}/{args.testbed}",
               only=_parse_only(args.only))


if __name__ == "__main__":
    _cli()

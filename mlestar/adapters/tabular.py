"""Fixed-fold Leaf Classification adapter with real OOF artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Literal
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from skrub import TableVectorizer

from ..artifacts import RunArtifacts
from ..contracts import ExperimentReceipt, TaskSpec
from ..initialization import CandidateSpec
from ..metrics import score_metric


@dataclass(frozen=True)
class LeafRun:
    receipt: ExperimentReceipt
    y_true: np.ndarray
    oof: np.ndarray
    test_prediction: np.ndarray
    class_columns: tuple[str, ...]


class LeafClassificationAdapter:
    """Execute Leaf Classification candidates using fixed stratified folds.

    This adapter deliberately keeps learned encoders inside each fold pipeline.
    skrub's TableVectorizer is fit only on the training partition; OOF rows are
    never seen by that fold's preprocessing or model.
    """

    def __init__(self, data_root: str | Path, run_dir: str | Path, task: TaskSpec) -> None:
        self.data_root = Path(data_root).resolve()
        self.artifacts = RunArtifacts(run_dir)
        self.task = task
        if task.key != "leaf_classification":
            raise ValueError("LeafClassificationAdapter requires the leaf_classification task contract.")

    def evaluate(
        self,
        candidate: CandidateSpec,
        *,
        phase: str,
        seed: int,
        parent_experiment_id: str | None = None,
    ) -> ExperimentReceipt:
        return self.run(candidate, phase=phase, seed=seed, parent_experiment_id=parent_experiment_id).receipt

    def merge(self, incumbent: CandidateSpec, addition: CandidateSpec) -> CandidateSpec:
        """A merged candidate is represented as an OOF ensemble instruction."""

        models = f"ensemble:{incumbent.block('model')}+{addition.block('model')}"
        blocks = dict(incumbent.blocks)
        blocks["model"] = models
        return CandidateSpec(
            candidate_id=f"{incumbent.candidate_id}+{addition.candidate_id}",
            blocks=tuple(blocks.items()),
            evidence_urls=tuple(sorted(set(incumbent.evidence_urls + addition.evidence_urls))),
        )

    def run(
        self,
        candidate: CandidateSpec,
        *,
        phase: str,
        seed: int,
        parent_experiment_id: str | None = None,
    ) -> LeafRun:
        started = perf_counter()
        try:
            train, test, sample = self._read_data()
            id_column = self.task.submission.id_columns[0]
            target = self.task.target_columns[0]
            if target not in train or id_column not in train or id_column not in test:
                raise ValueError(f"Expected {target!r} in train and {id_column!r} in train/test.")
            y_labels = train[target].astype(str).to_numpy()
            class_columns = tuple(sample.columns[1:]) if sample is not None else tuple(sorted(set(y_labels)))
            if set(class_columns) != set(y_labels):
                class_columns = tuple(sorted(set(y_labels)))
            x_train = train.drop(columns=[target])
            x_test = test.copy()
            oof, test_prediction, folds = self._cross_validate(
                x_train, y_labels, x_test, class_columns, candidate.block("model"), seed
            )
            metric = score_metric(self.task.metric, y_labels, oof, labels=list(class_columns))
            receipt_id = f"{phase}-{candidate.candidate_id}-{uuid4().hex[:12]}"
            stem = f"{phase}_{candidate.candidate_id}_{seed}".replace("/", "_")
            oof_frame = pd.DataFrame(oof, columns=class_columns)
            oof_frame.insert(0, id_column, train[id_column].to_numpy())
            test_frame = pd.DataFrame(test_prediction, columns=class_columns)
            test_frame.insert(0, id_column, test[id_column].to_numpy())
            oof_path = self.artifacts.write_csv(f"{stem}/oof.csv", oof_frame)
            test_path = self.artifacts.write_csv(f"{stem}/test_predictions.csv", test_frame)
            self.artifacts.write_csv(f"{stem}/folds.csv", folds)
            self.artifacts.write_csv(f"{stem}/submission.csv", test_frame)
            return LeafRun(
                receipt=ExperimentReceipt(
                    experiment_id=receipt_id,
                    parent_experiment_id=parent_experiment_id,
                    phase=phase,
                    candidate_id=candidate.candidate_id,
                    metric_value=metric.value,
                    fold_scores=tuple(float(value) for value in self._fold_scores(folds, y_labels, oof, class_columns)),
                    seed=seed,
                    oof_path=self.artifacts.relative(oof_path),
                    test_path=self.artifacts.relative(test_path),
                    error=None,
                ),
                y_true=y_labels,
                oof=oof,
                test_prediction=test_prediction,
                class_columns=class_columns,
            )
        except Exception as error:
            return LeafRun(
                receipt=ExperimentReceipt(
                    experiment_id=f"{phase}-{candidate.candidate_id}-{uuid4().hex[:12]}",
                    parent_experiment_id=parent_experiment_id,
                    phase=phase,
                    candidate_id=candidate.candidate_id,
                    metric_value=None,
                    fold_scores=(),
                    seed=seed,
                    oof_path=None,
                    test_path=None,
                    error=f"{type(error).__name__}: {error}",
                ),
                y_true=np.array([], dtype=str),
                oof=np.empty((0, 0)),
                test_prediction=np.empty((0, 0)),
                class_columns=(),
            )
        finally:
            self.artifacts.write_json("runtime.json", {"elapsed_seconds": perf_counter() - started})

    def _read_data(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
        train_path = self.data_root / "train.csv"
        test_path = self.data_root / "test.csv"
        if not train_path.is_file() or not test_path.is_file():
            raise FileNotFoundError("Leaf Classification needs train.csv and test.csv in data_root.")
        sample_path = self.data_root / "sample_submission.csv"
        return pd.read_csv(train_path), pd.read_csv(test_path), pd.read_csv(sample_path) if sample_path.is_file() else None

    def _cross_validate(
        self, x_train: pd.DataFrame, y: np.ndarray, x_test: pd.DataFrame, classes: tuple[str, ...], model: str, seed: int
    ) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        splitter = StratifiedKFold(n_splits=self.task.fold.n_splits, shuffle=True, random_state=seed)
        oof = np.zeros((len(x_train), len(classes)), dtype=float)
        test_prediction = np.zeros((len(x_test), len(classes)), dtype=float)
        fold_ids = np.empty(len(x_train), dtype=int)
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(x_train, y)):
            vectorizer = TableVectorizer(cardinality_threshold=40)
            train_features = vectorizer.fit_transform(x_train.iloc[train_idx])
            valid_features = vectorizer.transform(x_train.iloc[valid_idx])
            test_features = vectorizer.transform(x_test)
            classifier = self._build_model(model, seed + fold)
            classifier.fit(train_features, y[train_idx])
            oof[valid_idx] = self._align(classifier.predict_proba(valid_features), classifier.classes_, classes)
            test_prediction += self._align(classifier.predict_proba(test_features), classifier.classes_, classes) / self.task.fold.n_splits
            fold_ids[valid_idx] = fold
        folds = pd.DataFrame({"row_index": np.arange(len(x_train)), "fold": fold_ids})
        return oof, test_prediction, folds

    @staticmethod
    def _align(prediction: np.ndarray, observed: np.ndarray, expected: tuple[str, ...]) -> np.ndarray:
        output = np.zeros((len(prediction), len(expected)), dtype=float)
        for index, label in enumerate(observed.astype(str)):
            output[:, expected.index(label)] = prediction[:, index]
        return output

    @staticmethod
    def _build_model(model: str, seed: int):
        if model in {"extra_trees", "ensemble:extra_trees+random_forest", "ensemble:random_forest+extra_trees"}:
            return ExtraTreesClassifier(n_estimators=200, min_samples_leaf=1, n_jobs=-1, random_state=seed)
        if model == "random_forest":
            return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed)
        if model == "pass":
            return DummyClassifier(strategy="prior")
        raise ValueError(f"Unsupported Leaf candidate model {model!r}.")

    def _fold_scores(self, folds: pd.DataFrame, y: np.ndarray, oof: np.ndarray, classes: tuple[str, ...]) -> list[float]:
        return [
            score_metric(self.task.metric, y[folds["fold"].to_numpy() == fold], oof[folds["fold"].to_numpy() == fold], labels=list(classes)).value
            for fold in range(self.task.fold.n_splits)
        ]

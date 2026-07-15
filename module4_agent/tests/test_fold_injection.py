"""Generated-code tests for paired fold injection and prediction export.

The tests import generated train.py / evaluate.py and exercise their pure helper
functions directly.
"""

from __future__ import annotations

import importlib
import json
import sys

import pytest

from module4_agent.code_generator import generate_files
from module4_agent.tests.test_code_generator import _specs


def _import_generated(tmp_path, monkeypatch, module_name):
    generated = generate_files(_specs(), llm_provider="none")
    for name, content in generated.files.items():
        (tmp_path / name).write_text(content, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    for mod in ("train", "evaluate", "model", "smoke_data", "utils"):
        sys.modules.pop(mod, None)
    return importlib.import_module(module_name)


def _frame():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame({
        "image_id": [f"id{i}" for i in range(10)],
        "label": [i % 2 for i in range(10)],
    })


def _write_folds(tmp_path, folds):
    p = tmp_path / "folds.json"
    p.write_text(
        json.dumps({"n_folds": len(folds), "id_column": "image_id", "folds": folds}),
        encoding="utf-8",
    )
    return str(p)


# Fold injection by sample id.
def test_fold_split_matches_specified_ids(tmp_path, monkeypatch):
    train = _import_generated(tmp_path, monkeypatch, "train")
    frame = _frame()
    ff = _write_folds(tmp_path, [[f"id{i}" for i in range(5)],
                                 [f"id{i}" for i in range(5, 10)]])
    tr, val = train._fold_split_indices(frame, "image_id", ff, 0)
    assert val == [0, 1, 2, 3, 4]           # fold 0 maps exactly to these row positions
    assert tr == [5, 6, 7, 8, 9]


def test_paired_same_foldfile_same_val(tmp_path, monkeypatch):
    # Same fold_file + fold_index gives the same validation set for paired runs.
    train = _import_generated(tmp_path, monkeypatch, "train")
    frame = _frame()
    ff = _write_folds(tmp_path, [[f"id{i}" for i in range(5)],
                                 [f"id{i}" for i in range(5, 10)]])
    _, val_a = train._fold_split_indices(frame, "image_id", ff, 1)
    _, val_b = train._fold_split_indices(frame, "image_id", ff, 1)
    assert val_a == val_b == [5, 6, 7, 8, 9]
    _, val_other = train._fold_split_indices(frame, "image_id", ff, 0)
    assert set(val_a).isdisjoint(val_other)   # different folds do not overlap


def test_incomplete_folds_rejected(tmp_path, monkeypatch):
    train = _import_generated(tmp_path, monkeypatch, "train")
    frame = _frame()
    ff = _write_folds(tmp_path, [[f"id{i}" for i in range(5)]])   # missing id5..id9
    with pytest.raises(ValueError, match="do not match"):
        train._fold_split_indices(frame, "image_id", ff, 0)


def test_overlapping_folds_rejected(tmp_path, monkeypatch):
    train = _import_generated(tmp_path, monkeypatch, "train")
    frame = _frame()
    ff = _write_folds(tmp_path, [[f"id{i}" for i in range(6)],       # id5 overlaps
                                 [f"id{i}" for i in range(5, 10)]])
    with pytest.raises(ValueError, match="overlap"):
        train._fold_split_indices(frame, "image_id", ff, 0)


def test_fold_id_column_mismatch_rejected(tmp_path, monkeypatch):
    train = _import_generated(tmp_path, monkeypatch, "train")
    frame = _frame()
    ff = tmp_path / "folds.json"
    ff.write_text(
        json.dumps({
            "n_folds": 2,
            "id_column": "wrong_id",
            "folds": [[f"id{i}" for i in range(5)], [f"id{i}" for i in range(5, 10)]],
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="id_column"):
        train._fold_split_indices(frame, "image_id", str(ff), 0)


def test_fold_count_mismatch_rejected(tmp_path, monkeypatch):
    train = _import_generated(tmp_path, monkeypatch, "train")
    frame = _frame()
    ff = tmp_path / "folds.json"
    ff.write_text(
        json.dumps({
            "n_folds": 3,
            "id_column": "image_id",
            "folds": [[f"id{i}" for i in range(5)], [f"id{i}" for i in range(5, 10)]],
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="n_folds"):
        train._fold_split_indices(frame, "image_id", str(ff), 0)


def test_backward_compatible_without_fold_keys(tmp_path, monkeypatch):
    # Without fold_file, the original internal _split_indices path still works.
    train = _import_generated(tmp_path, monkeypatch, "train")
    tr, val = train._split_indices([0, 1] * 10, validation_fraction=0.2, seed=42)
    assert len(tr) + len(val) == 20 and set(tr).isdisjoint(val)


# Prediction export from evaluation.
def test_eval_exports_predictions(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    import torch.nn as nn
    evaluate = _import_generated(tmp_path, monkeypatch, "evaluate")

    class _M(nn.Module):
        def __init__(self):
            super().__init__()
            self.l = nn.Linear(3 * 16 * 16, 2)

        def forward(self, x):
            return self.l(x.flatten(1))

    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(8, 3, 16, 16),
                                       torch.randint(0, 2, (8,))),
        batch_size=4,
    )
    out = tmp_path / "val_preds.json"
    config = {"task_type": "classification", "num_classes": 2,
              "evaluation_metric": "roc_auc", "export_preds_path": str(out)}
    result = evaluate._eval_on_dataloader(_M(), loader, config)
    assert result["status"] == "success"
    assert out.exists()
    preds = json.loads(out.read_text(encoding="utf-8"))
    assert len(preds["y_true"]) == 8
    assert len(preds["y_prob"]) == 8 and len(preds["y_prob"][0]) == 2
    assert preds["y_score"] == preds["y_prob"]

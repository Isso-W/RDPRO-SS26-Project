"""Build the canonical Colab entrypoint without embedding training hyperparameters."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    markdown(
        """# Jiaozi Dog Breed MCP Low-Token Experiment Agent

This notebook is the fixed training entrypoint for the `mcp_knowledge` branch.

It runs:

1. Fixed-source Knowledge Learner through MCP.
2. Full Jiaozi Module 1 -> 2 -> 3 -> accumulating recommender -> Module 4.
3. One AutoPipeline baseline and at most three controlled MCP experiments.
4. Validation-selected ImageNet breed-prior calibration and selected-config folds.
5. Fold-ensemble probability prediction, Kaggle submission, and official score polling.

InceptionV3 + a basic classification head is reported only as the official comparison template. It is not forced into training. This notebook deliberately exposes no backbone, optimizer, learning rate, batch size, image size, augmentation, scheduler, or epoch controls.
"""
    ),
    code(
        """%pip install -q "mcp[cli]>=1.2.0" "kaggle>=1.6.0"
"""
    ),
    code(
        """import os
import shutil
import subprocess
from pathlib import Path

REPO_DIR = Path("/content/Jiaozi")
os.chdir("/content")
if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)
subprocess.run(
    [
        "git", "clone", "--branch", "mcp_knowledge", "--single-branch",
        "https://github.com/Isso-W/Jiaozi.git", str(REPO_DIR),
    ],
    check=True,
)
os.chdir(REPO_DIR)
subprocess.run(["python", "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)
print(subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True).stdout.strip())
"""
    ),
    code(
        """import json
from google.colab import drive, userdata

DRIVE_AVAILABLE = False
try:
    drive.mount("/content/drive")
    DRIVE_AVAILABLE = Path("/content/drive/MyDrive").exists()
except Exception as exc:
    print(f"Drive mount unavailable; using local Colab storage: {exc}")

def optional_secret(name):
    try:
        return userdata.get(name)
    except Exception:
        return None

dashscope_key = optional_secret("JIAOZI_DASHSCOPE_API_KEY")
openai_key = optional_secret("OPENAI_API_KEY")
if dashscope_key:
    os.environ["JIAOZI_DASHSCOPE_API_KEY"] = dashscope_key
    os.environ["JIAOZI_LLM_PROVIDER"] = "qwen"
    os.environ["KNOWLEDGE_LLM_PROVIDER"] = "qwen"
elif openai_key:
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["JIAOZI_LLM_PROVIDER"] = "openai"
    os.environ["KNOWLEDGE_LLM_PROVIDER"] = "openai"
else:
    raise RuntimeError("Add JIAOZI_DASHSCOPE_API_KEY or OPENAI_API_KEY to Colab Secrets.")

kaggle_dir = Path.home() / ".kaggle"
kaggle_dir.mkdir(parents=True, exist_ok=True)
kaggle_api_token = optional_secret("KAGGLE_API_TOKEN")
kaggle_json = optional_secret("KAGGLE_JSON")
if kaggle_api_token:
    os.environ["KAGGLE_API_TOKEN"] = kaggle_api_token
    (kaggle_dir / "access_token").write_text(kaggle_api_token, encoding="utf-8")
    (kaggle_dir / "access_token").chmod(0o600)
elif kaggle_json:
    credentials = json.loads(kaggle_json)
    (kaggle_dir / "kaggle.json").write_text(json.dumps(credentials), encoding="utf-8")
    (kaggle_dir / "kaggle.json").chmod(0o600)
else:
    credentials = {
        "username": optional_secret("KAGGLE_USERNAME"),
        "key": optional_secret("KAGGLE_KEY"),
    }
    if not credentials.get("username") or not credentials.get("key"):
        raise RuntimeError(
            "Add KAGGLE_API_TOKEN, KAGGLE_JSON, or KAGGLE_USERNAME + KAGGLE_KEY "
            "to Colab Secrets."
        )
    (kaggle_dir / "kaggle.json").write_text(json.dumps(credentials), encoding="utf-8")
    (kaggle_dir / "kaggle.json").chmod(0o600)
"""
    ),
    code(
        """DRIVE_ROOT = (
    Path("/content/drive/MyDrive/Jiaozi")
    if DRIVE_AVAILABLE
    else Path("/content/Jiaozi_runtime")
)
KB_ROOT = DRIVE_ROOT / "knowledge_base"
WORKSPACE_ROOT = DRIVE_ROOT / "workspace"
DATA_ROOT = DRIVE_ROOT / "kaggle_data"
REPORT_PATH = DRIVE_ROOT / "reports" / "dog_breed_mcp_report.json"

for path in (KB_ROOT, WORKSPACE_ROOT, DATA_ROOT, REPORT_PATH.parent):
    path.mkdir(parents=True, exist_ok=True)

os.environ["JIAOZI_KB_ROOT"] = str(KB_ROOT)
os.environ["JIAOZI_WORKSPACE_ROOT"] = str(WORKSPACE_ROOT)
os.environ["JIAOZI_OUTCOME_MEMORY"] = str(KB_ROOT / "experiments" / "outcomes.jsonl")
os.environ["JIAOZI_MCP_TRANSPORT"] = "stdio"
print({"knowledge_base": str(KB_ROOT), "workspace": str(WORKSPACE_ROOT)})
"""
    ),
    code(
        """from dog_breed_workflow import execute_dog_breed_workflow

report = await execute_dog_breed_workflow(
    data_root=DATA_ROOT,
    workspace_root=WORKSPACE_ROOT,
    report_path=REPORT_PATH,
    run_knowledge_learner=True,
    submit_to_kaggle=True,
)
print(f"Saved report: {REPORT_PATH}")
"""
    ),
    code(
        """import pandas as pd
from IPython.display import JSON, display

cards = report["experiment_loop"]["strategy_cards"]
display(pd.DataFrame(cards)[["id", "strategy_name", "component", "priority", "summary"]])

probe_rows = report["candidate_calibration"]["trials"]
display(pd.DataFrame(probe_rows)[[
    "candidate_index", "status", "metric_name", "metric_value",
    "accuracy", "macro_f1", "runtime_sec", "probe_epochs",
]])

baseline = report["baseline_config"]
diff_rows = []
for proposal in report["experiment_loop"]["proposals"]:
    diff_rows.append({
        "experiment": proposal["experiment_name"],
        "strategy_cards": ", ".join(proposal["strategy_card_ids"]),
        "changed_fields": ", ".join(proposal["changed_fields"]),
        "changes": {
            field: {"from": baseline.get(field), "to": proposal["config"].get(field)}
            for field in proposal["changed_fields"]
        },
    })
display(pd.DataFrame(diff_rows))

metric_rows = [report["baseline_metrics"], *report["experiment_loop"]["metrics"]]
display(pd.DataFrame(metric_rows))
fold_rows = report["fold_ensemble"]["runs"]
display(pd.DataFrame(fold_rows)[[
    "name", "status", "metric_name", "metric_value", "accuracy", "macro_f1",
    "prior_alpha", "prior_model", "best_epoch", "runtime_sec",
]])
display(JSON({
    "mcp_calls": report["mcp_calls"],
    "comparison": report["experiment_loop"]["comparison"],
    "validation_ensemble": report["ensemble"],
    "selected_config_fold_ensemble": report["fold_ensemble"],
}))
"""
    ),
    code(
        """display(JSON({
    "autopipeline_selected": report["autopipeline_selected"],
    "standard_reference_only": report["standard_reference_only"],
    "selected_experiment": report["selected_experiment"],
    "submission_selection": report["submission_selection"],
    "candidate_calibration": report["candidate_calibration"],
    "validation_ensemble": report["ensemble"],
    "selected_config_fold_ensemble": report["fold_ensemble"],
    "kaggle_submission": report["submission"],
    "token_and_training_cost": report["cost"],
    "report_path": str(REPORT_PATH),
}))
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"name": "integration_update_colab.ipynb", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

Path("integration_update_colab.ipynb").write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1) + "\n",
    encoding="utf-8",
)

"""Build the direct-run Colab notebook for the vision benchmark catalog."""

from __future__ import annotations

import json
from pathlib import Path


def markdown(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source.strip().splitlines(keepends=True),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip().splitlines(keepends=True),
    }


cells = [
    markdown(
        """
# Jiaozi: GPT-5.5 生成并训练 14 个视觉任务

这个 notebook 只会删除 Colab 中旧的 `/content/Jiaozi` 仓库副本，拉取最新的
`codex/integration-update-colab` 分支；不会删除已有的生成目录或 checkpoint。然后完成：

1. 选择一个比赛或公开数据集。
2. 使用配置的 `gpt-5.5` 生成 `model.py`。
3. 下载 Kaggle 数据或加载 Hugging Face 数据。
4. 在 GPU 上训练、评估并保存 checkpoint。

四个指定 Kaggle 比赛：

- `cassava`: Cassava Leaf Disease Classification
- `state_farm`: State Farm Distracted Driver Detection
- `siim_isic`: SIIM-ISIC Melanoma Classification
- `diabetic_retinopathy`: Diabetic Retinopathy Detection

另外十个可直接加载的数据集：

- `cifar10`, `cifar100`, `food101`, `beans`, `cats_vs_dogs`
- `stanford_cars`, `caltech101`, `eurosat`, `mnist`, `oxford_pets`

只需在下一格修改 `BENCHMARK_KEY`。Kaggle 比赛必须先在网页上接受比赛规则。
State Farm、SIIM-ISIC 和 Diabetic Retinopathy 数据较大，免费 Colab 可能需要较长下载时间和更多磁盘空间。

生成代码和模型会持久化到 Google Drive。重复运行第一格不会再覆盖模型。
"""
    ),
    code(
        r"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

# 只改这里即可切换任务。
BENCHMARK_KEY = "cassava"

# 正式训练默认使用完整数据。临时调试时可改为 False。
FORMAL_TRAINING = True
EPOCHS = 15 if FORMAL_TRAINING else 2
MAX_TRAIN_SAMPLES = 0 if FORMAL_TRAINING else 2000
MAX_EVAL_SAMPLES = 0 if FORMAL_TRAINING else 500
SAVE_TO_GOOGLE_DRIVE = True
RESUME_TRAINING = True

DEFAULT_REPO_URL = "https://github.com/Isso-W/Jiaozi.git"
REPO_REF = "codex/integration-update-colab"
REPO_DIR = Path("/content/Jiaozi")
DATA_ROOT = Path("/content/jiaozi_data")
OUTPUT_DIR = Path("/content/jiaozi_generated_training")


def normalize_repo_url(value: str) -> str:
    value = (value or "").strip()
    match = re.search(r"\]\((https?://[^)]+)\)", value)
    if match:
        value = match.group(1)
    match = re.search(
        r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+(?:\.git)?",
        value,
    )
    return match.group(0) if match else ""


REPO_URL = normalize_repo_url(DEFAULT_REPO_URL)
if not REPO_URL:
    raise RuntimeError(f"Invalid repository URL: {DEFAULT_REPO_URL!r}")
if os.getenv("JIAOZI_REPO_URL") or os.getenv("JIAOZI_REPO_REF"):
    print(
        "Ignoring JIAOZI_REPO_URL/JIAOZI_REPO_REF from the old Colab runtime; "
        "this notebook always pulls the tested branch."
    )

os.chdir("/content")
if REPO_DIR.exists():
    shutil.rmtree(REPO_DIR)
if OUTPUT_DIR.exists():
    print("Preserving existing generated project and local checkpoints:", OUTPUT_DIR)


def checkout_repository() -> None:
    clone_command = [
        "git",
        "clone",
        "--depth",
        "1",
        "--branch",
        REPO_REF,
        REPO_URL,
        str(REPO_DIR),
    ]
    print("Repository URL:", REPO_URL)
    print("Repository ref:", REPO_REF)
    for attempt in range(1, 4):
        if REPO_DIR.exists():
            shutil.rmtree(REPO_DIR)
        completed = subprocess.run(clone_command, capture_output=True, text=True)
        if completed.returncode == 0:
            print(completed.stdout or completed.stderr)
            return
        print(f"git clone attempt {attempt}/3 failed with exit {completed.returncode}")
        print(completed.stdout)
        print(completed.stderr)
        time.sleep(attempt)

    print("Falling back to the GitHub branch ZIP archive.")
    archive_path = Path("/content/jiaozi_branch.zip")
    extract_root = Path("/content/jiaozi_branch_extract")
    if archive_path.exists():
        archive_path.unlink()
    if extract_root.exists():
        shutil.rmtree(extract_root)
    extract_root.mkdir(parents=True)
    archive_url = (
        "https://codeload.github.com/Isso-W/Jiaozi/zip/refs/heads/"
        + REPO_REF
    )
    print("Archive URL:", archive_url)
    urllib.request.urlretrieve(archive_url, archive_path)
    with zipfile.ZipFile(archive_path) as handle:
        handle.extractall(extract_root)
    extracted_dirs = [path for path in extract_root.iterdir() if path.is_dir()]
    if len(extracted_dirs) != 1:
        raise RuntimeError(f"Unexpected GitHub archive contents: {extracted_dirs}")
    shutil.move(str(extracted_dirs[0]), str(REPO_DIR))


checkout_repository()
os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR))

subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-r",
        "requirements.txt",
        "kaggle>=1.7.4.5",
    ],
    check=True,
)

from vision_benchmark_catalog import BENCHMARKS, get_benchmark

if BENCHMARK_KEY not in BENCHMARKS:
    raise KeyError(f"Unknown BENCHMARK_KEY={BENCHMARK_KEY!r}; choose from {sorted(BENCHMARKS)}")

benchmark = get_benchmark(BENCHMARK_KEY)
print("Repository:", REPO_DIR)
print("Branch:", REPO_REF)
print("Selected:", BENCHMARK_KEY, "-", benchmark["name"])
print("Source:", benchmark["source"])
print("Metric:", benchmark["metric"])

try:
    import torch
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Warning: select Runtime > Change runtime type > GPU before real training.")
except Exception as exc:
    print("Torch GPU check failed:", exc)

print("\nAvailable benchmark keys:")
for key, item in BENCHMARKS.items():
    print(f"  {key:22s} {item['source']:11s} {item['name']}")
"""
    ),
    markdown(
        """
## 配置密钥

在 Colab 左侧 **Secrets** 中添加：

- `OPENAI_API_KEY`
- `KAGGLE_API_TOKEN`（选择四个 Kaggle 比赛时需要）
- `OPENAI_BASE_URL`（可选；默认使用 `https://yybb.codes`）

密钥只写入当前 Colab 运行时的环境变量，不会写入 Git 仓库或 notebook。
之前在聊天中发送过的密钥应视为已经暴露，建议运行成功后立即撤销并重新生成。
"""
    ),
    code(
        r"""
from getpass import getpass

try:
    from google.colab import userdata
except Exception:
    userdata = None


def read_secret(name: str, required: bool = False) -> str:
    value = ""
    if userdata is not None:
        try:
            value = userdata.get(name) or ""
        except Exception:
            value = ""
    if not value and required:
        value = getpass(f"{name} (hidden input): ").strip()
    return value.strip()


openai_api_key = read_secret("OPENAI_API_KEY", required=True)
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY is required.")

os.environ["OPENAI_API_KEY"] = openai_api_key
os.environ["OPENAI_BASE_URL"] = (
    read_secret("OPENAI_BASE_URL", required=False)
    or os.getenv("OPENAI_BASE_URL", "").strip()
    or "https://yybb.codes"
)
os.environ["M4_LLM_PROVIDER"] = "openai"
os.environ["M4_OPENAI_MODEL"] = "gpt-5.5"
os.environ["M4_OPENAI_WIRE_API"] = "responses"

if benchmark["source"] == "kaggle":
    kaggle_token = read_secret("KAGGLE_API_TOKEN", required=True)
    if not kaggle_token:
        raise RuntimeError("KAGGLE_API_TOKEN is required for Kaggle competitions.")
    os.environ["KAGGLE_API_TOKEN"] = kaggle_token
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_dir.mkdir(parents=True, exist_ok=True)
    token_path = kaggle_dir / "access_token"
    token_path.write_text(kaggle_token, encoding="utf-8")
    token_path.chmod(0o600)
    del kaggle_token

del openai_api_key
print("LLM provider: openai")
print("Configured model: gpt-5.5")
print("Wire API: responses")
print("Custom OpenAI-compatible endpoint configured:", bool(os.environ.get("OPENAI_BASE_URL")))
print("Kaggle token configured:", bool(os.environ.get("KAGGLE_API_TOKEN")))
"""
    ),
    markdown(
        """
## 准备数据

Kaggle 路径会自动下载比赛文件并递归解压 ZIP。若出现 `403`，先打开比赛页面接受规则；
若出现磁盘不足，请换用 Colab Pro、连接 Google Drive，或选择较小的数据集。
"""
    ),
    code(
        r"""
import zipfile


def run_visible(command: list[str], cwd: Path | None = None) -> None:
    print("$", " ".join(command))
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if completed.stdout:
        print(completed.stdout)
    if completed.stderr:
        print(completed.stderr)
    completed.check_returncode()


def extract_zip_archives(root: Path) -> None:
    extracted: set[Path] = set()
    while True:
        archives = [
            path
            for path in root.rglob("*.zip")
            if path.resolve() not in extracted and zipfile.is_zipfile(path)
        ]
        if not archives:
            return
        for archive in archives:
            print("Extracting:", archive)
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(archive.parent)
            extracted.add(archive.resolve())


def find_first(root: Path, patterns: list[str], expect_directory: bool) -> Path:
    matches: list[Path] = []
    for pattern in patterns:
        for path in root.glob(pattern):
            if path.is_dir() == expect_directory:
                matches.append(path)
    if not matches:
        kind = "directory" if expect_directory else "file"
        raise FileNotFoundError(
            f"Could not find expected {kind} under {root}. Patterns: {patterns}"
        )
    return sorted(matches, key=lambda path: (len(path.parts), str(path)))[0]


runtime_data: dict[str, object] = {}
if benchmark["source"] == "kaggle":
    competition_dir = DATA_ROOT / BENCHMARK_KEY
    competition_dir.mkdir(parents=True, exist_ok=True)

    extract_zip_archives(competition_dir)
    try:
        train_csv = find_first(competition_dir, benchmark["csv_globs"], False)
        image_dir = find_first(competition_dir, benchmark["image_dir_globs"], True)
        print("Reusing existing Kaggle data:", competition_dir)
    except FileNotFoundError:
        try:
            run_visible(
                [
                    "kaggle",
                    "competitions",
                    "download",
                    "-c",
                    benchmark["competition"],
                    "-p",
                    str(competition_dir),
                ]
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "Kaggle download failed. Confirm KAGGLE_API_TOKEN, accept the competition "
                f"rules, and check free disk space. Competition: {benchmark['competition']}"
            ) from exc
        extract_zip_archives(competition_dir)
        train_csv = find_first(competition_dir, benchmark["csv_globs"], False)
        image_dir = find_first(competition_dir, benchmark["image_dir_globs"], True)
    runtime_data.update(
        {
            "train_csv": str(train_csv),
            "image_dir": str(image_dir),
            "image_column": benchmark["image_column"],
            "label_column": benchmark["label_column"],
            "image_path_template": benchmark["image_path_template"],
            "image_extension": benchmark["image_extension"],
        }
    )
    print("Training CSV:", train_csv)
    print("Image directory:", image_dir)
    run_visible(["du", "-sh", str(competition_dir)])
else:
    runtime_data["dataset_id"] = benchmark["dataset_id"]
    print("Hugging Face dataset:", benchmark["dataset_id"])

print(json.dumps(runtime_data, indent=2, ensure_ascii=False))
"""
    ),
    markdown(
        """
## 使用 GPT-5.5 生成训练项目

这一步直接调用 Module 4，并使用代理配置要求的 Responses API。
生成后会检查 `generation_info.json`。如果代理返回 HTML、登录页或无效 Python，
系统会记录失败原因并自动使用经过 smoke test 的模板继续训练，不会把无效内容写进 `model.py`。
"""
    ),
    code(
        r"""
from module4_agent.workflow import run_workflow

candidate = {
    "rank": 1,
    "score": 1.0,
    "model_config": {
        "task_type": "classification",
        "backbone": benchmark["backbone"],
        "head": "classification_head",
        "loss": benchmark["loss"],
        "optimizer": "adamw",
        "learning_rate": 0.0003,
        "num_classes": benchmark["num_classes"],
        "image_size": 224,
        "offline_smoke": True,
        "use_pretrained": True,
        "finetune_strategy": "full",
        "freeze_backbone": False,
        "evaluation_metric": benchmark["metric"],
        "benchmark_name": benchmark["name"],
        **runtime_data,
    },
}

module3_input = Path("/content/jiaozi_selected_candidate.json")
module3_input.write_text(
    json.dumps([candidate], indent=2, ensure_ascii=False),
    encoding="utf-8",
)

result = run_workflow(
    module3_input,
    OUTPUT_DIR,
    max_iter=2,
    timeout=180,
    skip_smoke=False,
    llm_provider="openai",
)
if not result.is_approved:
    raise RuntimeError(
        "Generated project did not pass smoke/review checks:\n"
        + json.dumps(result.to_summary(), indent=2, ensure_ascii=False)
    )

generation_info_path = OUTPUT_DIR / "generation_info.json"
generation_info = json.loads(generation_info_path.read_text(encoding="utf-8"))
print(json.dumps(generation_info, indent=2, ensure_ascii=False))

if not generation_info.get("llm_used"):
    print(
        "\nWARNING: GPT-5.5 was attempted but did not return valid Python. "
        "Training will continue with the validated template model.py."
    )
    print("Fallback reason:", generation_info.get("fallback_reason") or "unknown")
if generation_info.get("llm_model") != "gpt-5.5":
    raise RuntimeError(f"Unexpected generation model: {generation_info.get('llm_model')!r}")

generated_configs = json.loads((OUTPUT_DIR / "configs.json").read_text(encoding="utf-8"))
runtime_config = dict(generated_configs[0])
runtime_config.update(runtime_config.pop("model_config", {}) or {})

checkpoint_dir = OUTPUT_DIR / "checkpoints"
if SAVE_TO_GOOGLE_DRIVE:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        checkpoint_dir = (
            Path("/content/drive/MyDrive/Jiaozi/formal_training")
            / BENCHMARK_KEY
        )
    except Exception as exc:
        print("Google Drive mount failed; checkpoints will stay in /content:", exc)
checkpoint_dir.mkdir(parents=True, exist_ok=True)

PERSISTENT_ARTIFACTS = (
    "README_generated.md",
    "configs.json",
    "evaluate.py",
    "generation_info.json",
    "infer.py",
    "model.py",
    "model_utils.py",
    "module4_summary.json",
    "requirements.txt",
    "run.py",
    "run_experiments.py",
    "smoke_data.py",
    "train.py",
    "utils.py",
)

runtime_config.update(
    {
        "offline_smoke": False,
        "use_pretrained": True,
        "finetune_strategy": "full",
        "freeze_backbone": False,
        "evaluation_metric": benchmark["metric"],
        "recommended_epochs": EPOCHS,
        "image_size": 300 if benchmark["backbone"] == "efficientnet_b3" else 224,
        "batch_size": 16 if benchmark["backbone"] == "efficientnet_b3" else 32,
        "eval_batch_size": 32 if benchmark["backbone"] == "efficientnet_b3" else 64,
        "num_workers": 2,
        "validation_fraction": 0.2,
        "max_train_samples": MAX_TRAIN_SAMPLES,
        "max_eval_samples": MAX_EVAL_SAMPLES,
        "augmentation": "strong" if FORMAL_TRAINING else "basic",
        "use_class_weights": FORMAL_TRAINING,
        "class_weight_power": 0.5,
        "label_smoothing": 0.1 if FORMAL_TRAINING else 0.0,
        "learning_rate": 0.0002 if FORMAL_TRAINING else 0.0003,
        "scheduler": "cosine",
        "min_learning_rate": 0.000001,
        "mixed_precision": True,
        "gradient_clip_norm": 1.0,
        "early_stopping_patience": 4 if FORMAL_TRAINING else 0,
        "save_every_epoch": False,
        "checkpoint_dir": str(checkpoint_dir),
        "resume_checkpoint": "auto" if RESUME_TRAINING else "",
        "seed": 42,
        **runtime_data,
    }
)
(OUTPUT_DIR / "configs.json").write_text(
    json.dumps(runtime_config, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

for artifact_name in PERSISTENT_ARTIFACTS:
    source_path = OUTPUT_DIR / artifact_name
    if source_path.exists():
        shutil.copy2(source_path, checkpoint_dir / artifact_name)

print("Generated project:", OUTPUT_DIR)
print("Generated project backup:", checkpoint_dir)
print("Real-training config:")
print(json.dumps(runtime_config, indent=2, ensure_ascii=False))
"""
    ),
    markdown(
        """
## 开始真实训练

正式模式默认使用完整数据训练最多 15 个 epoch，并在验证指标连续 4 轮没有提升时提前停止。
`best_model.pt` 与 `last_checkpoint.pt` 会保存到 Google Drive，可在 Colab 中断后自动续训。
"""
    ),
    code(
        r"""
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"],
    cwd=OUTPUT_DIR,
    check=True,
)

training_command = [
    sys.executable,
    "run.py",
    "--config",
    "configs.json",
    "--epochs",
    str(EPOCHS),
]
print("$", " ".join(training_command))

process = subprocess.Popen(
    training_command,
    cwd=OUTPUT_DIR,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)
training_lines: list[str] = []
assert process.stdout is not None
for line in process.stdout:
    print(line, end="")
    training_lines.append(line)
return_code = process.wait()

training_log = OUTPUT_DIR / "training_output.txt"
training_log.write_text("".join(training_lines), encoding="utf-8")
if return_code != 0:
    raise subprocess.CalledProcessError(return_code, training_command)

checkpoint_dir = Path(runtime_config["checkpoint_dir"])
shutil.copy2(training_log, checkpoint_dir / training_log.name)

print("\nTraining log:", training_log)
print("Persistent training directory:", checkpoint_dir)
print("Checkpoints:")
for path in sorted(checkpoint_dir.glob("*.pt")):
    print(" ", path, f"({path.stat().st_size / 1024 / 1024:.1f} MB)")
"""
    ),
    markdown(
        """
## 测试已保存模型

这格会从 Google Drive 恢复生成代码，加载 `best_model.pt`（没有时使用
`last_checkpoint.pt`），并在验证集上计算比赛指标、Accuracy 和 Macro-F1。
如果 Colab 运行时刚刚重启，请先运行到“准备真实数据”一节，确保验证数据存在。
"""
    ),
    code(
        r"""
import importlib

import torch

try:
    from google.colab import drive
    drive.mount("/content/drive")
except Exception as exc:
    print("Google Drive mount status:", exc)

benchmark_key = globals().get("BENCHMARK_KEY", "cassava")
persistent_dir = Path("/content/drive/MyDrive/Jiaozi/formal_training") / benchmark_key
local_project = Path("/content/jiaozi_generated_training")
local_project.mkdir(parents=True, exist_ok=True)

required_project_files = (
    "configs.json",
    "evaluate.py",
    "model.py",
    "model_utils.py",
    "smoke_data.py",
    "train.py",
    "utils.py",
)
for artifact_name in required_project_files:
    source_path = persistent_dir / artifact_name
    destination_path = local_project / artifact_name
    if source_path.exists():
        shutil.copy2(source_path, destination_path)

missing_files = [
    name for name in required_project_files
    if not (local_project / name).exists()
]
if missing_files:
    raise FileNotFoundError(
        "Missing generated project files: "
        + ", ".join(missing_files)
        + ". Run the GPT generation cell once to recreate and back them up."
    )

checkpoint_candidates = (
    persistent_dir / "best_model.pt",
    persistent_dir / "last_checkpoint.pt",
    local_project / "checkpoints" / "best_model.pt",
    local_project / "checkpoints" / "last_checkpoint.pt",
)
checkpoint_path = next((path for path in checkpoint_candidates if path.exists()), None)
if checkpoint_path is None:
    raise FileNotFoundError(
        "No saved model checkpoint was found. Start the training cell; "
        "the repaired notebook saves checkpoints to Google Drive after every epoch."
    )

training_config = json.loads(
    (local_project / "configs.json").read_text(encoding="utf-8")
)
training_config["checkpoint_dir"] = str(persistent_dir)
training_config["offline_smoke"] = False

if str(local_project) not in sys.path:
    sys.path.insert(0, str(local_project))
os.chdir(local_project)
importlib.invalidate_caches()

from evaluate import evaluate
from model import build_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint.get("model_state_dict", checkpoint)
model = build_model(training_config).to(device)
model.load_state_dict(state_dict)

test_result = evaluate(model, training_config)
print("Checkpoint:", checkpoint_path)
print("Device:", device)
print(json.dumps(test_result, indent=2, ensure_ascii=False))
"""
    ),
    markdown(
        """
## 检查生成与训练结果

最后一格显示 Module 4 审核摘要、GPT 使用记录、实际配置和生成文件。
"""
    ),
    code(
        r"""
summary = json.loads((OUTPUT_DIR / "module4_summary.json").read_text(encoding="utf-8"))
generation_info = json.loads((OUTPUT_DIR / "generation_info.json").read_text(encoding="utf-8"))
training_config = json.loads((OUTPUT_DIR / "configs.json").read_text(encoding="utf-8"))

print("=== Module 4 summary ===")
print(json.dumps(summary, indent=2, ensure_ascii=False))
print("\n=== GPT generation record ===")
print(json.dumps(generation_info, indent=2, ensure_ascii=False))
print("\n=== Training config ===")
print(json.dumps(training_config, indent=2, ensure_ascii=False))
print("\n=== Generated files ===")
for path in sorted(OUTPUT_DIR.iterdir()):
    print(path.name)
"""
    ),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "name": "vision_benchmarks_colab.ipynb",
            "provenance": [],
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

output_path = Path(__file__).with_name("vision_benchmarks_colab.ipynb")
output_path.write_text(
    json.dumps(notebook, indent=2, ensure_ascii=False) + "\n",
    encoding="utf-8",
)
print(output_path)

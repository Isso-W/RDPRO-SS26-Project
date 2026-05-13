下面是一版**技术文档草案**。我把你们原始项目目标和 PDF 中的路线整合了，并修正了一个关键范围问题：你们 PDF 里有不少内容偏向图像分类、ResNet/EfficientNet、LoRA，但原始任务明确要求 agent 生成 **基于 skrub DataOps 的 Python ML pipeline**。skrub 官方定位是面向 tabular dataframes 的机器学习库，并提供 scikit-learn 兼容 API；DataOps 是对 scikit-learn pipeline 的泛化。因此建议 MVP 聚焦 **tabular ML pipeline synthesis**，把图像/深度学习部分放到 future extension。

---

# Technical Design Document

## MLE-STAR-style Machine Learning Engineering Agent for skrub DataOps Pipelines

### 1. Project Overview

本项目目标是实现一个 Machine Learning Engineering Agent，用于根据用户给定的机器学习任务和数据集，自动生成、执行、评估并迭代优化 Python 机器学习 pipeline。

该项目参考 MLE-STAR: Machine Learning Engineering Agent via Search and Targeted Refinement 的思想，即通过搜索、实验反馈和 targeted refinement 来改进 pipeline。与原论文不同，本项目的核心约束是：agent 生成的代码必须使用 **skrub DataOps abstraction**，而不是普通的手写 pandas/scikit-learn pipeline。

你们现有 PDF 中已经定义了整体方向，包括 agent 工作流、RAG、ablation-based refinement、实验追踪、评估计划和主要挑战。特别是 PDF 中提到的核心挑战包括 controlled code modification、ablation design、experiment efficiency、evaluation reliability、automated debugging、search stability、experiment tracking 和 scope management，这些将作为本文档的设计约束。

---

## 2. Scope

### 2.1 MVP Scope

MVP 只支持 **tabular supervised learning**：

| 类型          | 支持范围                                                                |
| ----------- | ------------------------------------------------------------------- |
| 输入数据        | CSV / Parquet / pandas DataFrame                                    |
| 任务类型        | classification / regression                                         |
| 特征类型        | numerical, categorical, text-like categorical                       |
| 缺失值         | 支持                                                                  |
| 类别不平衡       | 支持基础检测与 class weight 策略                                             |
| pipeline 框架 | skrub DataOps + scikit-learn estimator                              |
| 优化方式        | RAG-guided baseline generation + ablation-based targeted refinement |
| 输出          | 可执行 Python 脚本、实验日志、最佳 pipeline 配置                                   |

skrub 的 TableVectorizer 目标是把 dataframe 转换成适合机器学习算法使用的数值特征表示，因此适合作为 agent 默认生成 tabular pipeline 的核心组件。

### 2.2 Out of Scope for MVP

以下内容不建议放入第一版：

| 内容                              | 原因 |
| ------------------------------- | -- |
| ResNet / EfficientNet / CNN     |    |
| LoRA fine-tuning                |    |
| object detection / segmentation |    |
| Hugging Face vision models      |    |
| ensemble of deep models         |    |

这些内容可以放入 future extension。你们 PDF 中提到图像数据集、ResNet、EfficientNet、LoRA 和多任务视觉数据集，但这与 skrub DataOps 的 tabular ML 定位不完全一致。为避免 scope 失控，MVP 应先完成 tabular DataOps agent，再扩展到 vision 或 multimodal。

---

## 3. System Goals

### 3.1 Functional Goals

系统需要完成以下功能：

1. 接收用户任务描述和数据集路径。
2. 自动分析数据集结构。
3. 判断任务类型：classification 或 regression。
4. 生成基于 skrub DataOps 的初始 pipeline。
5. 执行 pipeline 并记录实验结果。
6. 基于实验结果选择需要优化的模块。
7. 对选定模块进行 controlled code modification。
8. 重复执行、评估和 refinement。
9. 输出最佳 pipeline、实验日志和报告。

### 3.2 Non-functional Goals

| 目标   | 要求                                                   |
| ---- | ---------------------------------------------------- |
| 可复现性 | fixed train/validation/test split, fixed random seed |
| 可控性  | 每次 refinement 只能修改一个 pipeline component              |
| 安全性  | LLM 生成代码不能直接无限制执行                                    |
| 可追踪性 | 记录 config、score、runtime、error log                    |
| 效率   | 限制候选模型数量和实验次数                                        |
| 可解释性 | 每次修改必须有 reason 和 expected effect                     |

你们 PDF 的 Evaluation Plan 已经提出固定数据划分、固定 seed、限制候选模型和训练次数，并记录实验配置、score、runtime 和 error log，这些内容应直接纳入系统设计。

---

# 4. Overall Architecture

系统由七个核心模块组成：

```text
User Task + Dataset
        |
        v
[1] Task & Dataset Analyzer
        |
        v
[2] RAG Knowledge Retriever
        |
        v
[3] Pipeline Planner
        |
        v
[4] skrub Code Generator
        |
        v
[5] Sandbox Executor
        |
        v
[6] Experiment Evaluator
        |
        v
[7] Ablation Analyzer + Targeted Refiner
        |
        +---- repeat until budget exhausted
```

---

## 5. Core Modules

## 5.1 Task & Dataset Analyzer

### Responsibility

分析输入数据集，生成 structured dataset profile。

### Input

```json
{
  "dataset_path": "data/train.csv",
  "target_column": "Survived",
  "task_hint": "classification"
}
```

### Output

```json
{
  "task_type": "classification",
  "n_rows": 891,
  "n_columns": 12,
  "target_column": "Survived",
  "target_type": "binary",
  "missing_columns": ["Age", "Cabin", "Embarked"],
  "categorical_columns": ["Sex", "Embarked", "Cabin"],
  "numerical_columns": ["Age", "Fare", "Pclass"],
  "class_balance": {
    "0": 549,
    "1": 342
  },
  "recommended_metric": "f1"
}
```

### Implementation Notes

该模块不需要 LLM。使用 pandas / polars 完成即可。

核心逻辑：

```python
def analyze_dataset(df, target_column):
    profile = {}
    profile["n_rows"] = len(df)
    profile["n_columns"] = df.shape[1]
    profile["missing_rate"] = df.isna().mean().to_dict()
    profile["dtypes"] = df.dtypes.astype(str).to_dict()
    profile["target_column"] = target_column
    profile["target_unique_values"] = df[target_column].nunique()
    return profile
```

---

## 5.2 RAG Knowledge Retriever

### Responsibility

根据 dataset profile 和 task type 检索 pipeline 设计知识。

### Knowledge Base Content

RAG 知识库不应一开始做得太复杂。建议先用本地 JSON / Markdown 文件。

```text
knowledge_base/
  classification_baselines.md
  regression_baselines.md
  missing_values.md
  categorical_features.md
  imbalanced_classification.md
  skrub_dataops_examples.md
```

### Example Retrieval Query

```text
Task: binary classification
Dataset: mixed numerical and categorical columns
Missing values: yes
Class imbalance: moderate
Need: skrub DataOps pipeline
```

### Output

```json
{
  "recommended_preprocessing": [
    "Use TableVectorizer for heterogeneous tabular features",
    "Use class_weight='balanced' if class imbalance is detected"
  ],
  "candidate_models": [
    "LogisticRegression",
    "HistGradientBoostingClassifier",
    "RandomForestClassifier"
  ],
  "candidate_metrics": [
    "accuracy",
    "f1",
    "roc_auc"
  ]
}
```

你们 PDF 中的 RAG 设计提到输入用户需求关键词，输出模型名称、用途和特点。这里建议把它改成更适合 skrub 的版本：输入 dataset profile，输出 tabular pipeline 设计建议。

---

## 5.3 Pipeline Planner

### Responsibility

把任务分析结果和 RAG 结果转成 pipeline plan。

### Output Schema

```json
{
  "pipeline_id": "baseline_001",
  "task_type": "classification",
  "data_ops": {
    "feature_selection": "all_except_target",
    "vectorizer": "TableVectorizer",
    "missing_value_strategy": "default"
  },
  "model": {
    "name": "HistGradientBoostingClassifier",
    "params": {
      "max_iter": 100,
      "random_state": 42
    }
  },
  "metric": "f1",
  "split": {
    "train_size": 0.7,
    "valid_size": 0.15,
    "test_size": 0.15,
    "random_state": 42
  }
}
```

---

## 5.4 skrub Code Generator

### Responsibility

根据 pipeline plan 生成可执行 Python 脚本。

### Required Properties

生成代码必须满足：

1. 使用 skrub DataOps。
2. 能独立运行。
3. 有固定 random seed。
4. 有统一输出格式。
5. 不允许在 refinement 中修改无关模块。
6. 所有实验结果写入 `experiments.jsonl`。

### Example Code Skeleton

```python
import json
import time
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

import skrub


RANDOM_STATE = 42
DATA_PATH = "data/train.csv"
TARGET_COLUMN = "target"


def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def build_pipeline():
    # This function should be generated by the agent.
    # It must use skrub components for tabular preprocessing.
    from skrub import TableVectorizer
    from sklearn.pipeline import make_pipeline

    model = HistGradientBoostingClassifier(
        max_iter=100,
        random_state=RANDOM_STATE
    )

    pipeline = make_pipeline(
        TableVectorizer(),
        model
    )
    return pipeline


def evaluate():
    start_time = time.time()

    X, y = load_data(DATA_PATH)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_valid)

    score = f1_score(y_valid, y_pred, average="macro")
    runtime = time.time() - start_time

    result = {
        "score": score,
        "metric": "macro_f1",
        "runtime": runtime,
        "status": "success"
    }

    print(json.dumps(result))
    return result


if __name__ == "__main__":
    evaluate()
```

说明：实际最终版本应进一步替换为更严格的 DataOps plan。skrub DataOps 提供 `.skb.apply`、`.skb.train_test_split`、`.skb.cross_validate`、`.skb.make_grid_search` 等接口，可用于把数据处理、验证和搜索组织为 DataOps workflow。

---

## 5.5 Sandbox Executor

### Responsibility

安全执行 agent 生成的 Python 文件。

### Execution Command

```bash
python generated_pipeline.py
```

### Captured Outputs

```json
{
  "experiment_id": "exp_0003",
  "pipeline_id": "refine_model_001",
  "status": "success",
  "score": 0.812,
  "metric": "macro_f1",
  "runtime": 18.4,
  "stdout": "...",
  "stderr": "",
  "error_type": null
}
```

### Error Categories

| Error Type   | Meaning                  |
| ------------ | ------------------------ |
| syntax_error | Python 语法错误              |
| import_error | 缺少依赖或错误导入                |
| data_error   | target column 不存在、数据格式错误 |
| fit_error    | pipeline.fit 失败          |
| metric_error | metric 与任务类型不匹配          |
| timeout      | 超过运行时间限制                 |

---

## 5.6 Experiment Evaluator

### Responsibility

比较不同实验结果，维护 leaderboard。

### Leaderboard Schema

```json
[
  {
    "rank": 1,
    "experiment_id": "exp_0007",
    "pipeline_id": "refine_model_003",
    "score": 0.842,
    "metric": "macro_f1",
    "runtime": 21.3,
    "status": "success",
    "modified_component": "model"
  }
]
```

### Selection Rule

```python
best_experiment = max(
    successful_experiments,
    key=lambda x: x["score"]
)
```

对于 regression，选择规则应改为最小化 RMSE / MAE。

---

## 5.7 Ablation Analyzer

### Responsibility

识别哪个 pipeline component 对性能影响最大。

### Components

```text
1. preprocessing
2. model
3. hyperparameters
4. imbalance handling
5. feature selection
```

### Ablation Strategy

每次只改一个 component。

Example:

| Experiment           | Preprocessing | Model | Hyperparameters | Score |
| -------------------- | ------------- | ----- | --------------- | ----- |
| baseline             | A             | A     | A               | 0.76  |
| ablate preprocessing | B             | A     | A               | 0.78  |
| ablate model         | A             | B     | A               | 0.83  |
| ablate hyperparams   | A             | A     | B               | 0.77  |

结论：

```json
{
  "most_impactful_component": "model",
  "reason": "Changing the model produced the largest validation improvement."
}
```

你们 PDF 已经把 ablation design 作为核心挑战之一，并提出对比 baseline、random refinement 和 ablation-based targeted refinement。技术实现上，ablation analyzer 就是连接实验结果和下一轮 refinement 的核心模块。

---

# 6. Targeted Refinement Design

## 6.1 Refinement Loop

```text
1. Run baseline pipeline
2. Evaluate validation score
3. Generate candidate modifications
4. Run ablation experiments
5. Identify most impactful component
6. Modify only that component
7. Re-run experiment
8. Update leaderboard
9. Stop when budget is exhausted
```

## 6.2 Controlled Code Modification

每次 refinement 必须绑定一个 component。

```json
{
  "target_component": "model",
  "allowed_edit_region": "build_pipeline.model",
  "forbidden_regions": [
    "load_data",
    "train_valid_split",
    "metric",
    "logging"
  ]
}
```

### Example Prompt to LLM

```text
You are modifying a Python ML pipeline.

You may only modify the model definition inside build_pipeline().
Do not change data loading.
Do not change train/validation split.
Do not change metric computation.
Do not change logging format.

Current model:
HistGradientBoostingClassifier(max_iter=100, random_state=42)

Dataset profile:
- binary classification
- mixed numerical and categorical features
- missing values present
- moderate class imbalance

Suggest one model replacement using scikit-learn.
Return only the modified build_pipeline() function.
```

---

# 7. DataOps Pipeline Contract

每个生成 pipeline 必须满足统一 contract。

## 7.1 Input Contract

```json
{
  "dataset_path": "string",
  "target_column": "string",
  "task_type": "classification | regression",
  "metric": "string",
  "random_seed": 42
}
```

## 7.2 Output Contract

```json
{
  "status": "success | failed",
  "score": "float | null",
  "metric": "string",
  "runtime": "float",
  "error": "string | null",
  "pipeline_config": "object"
}
```

## 7.3 Reproducibility Contract

所有实验必须固定：

```python
RANDOM_STATE = 42
```

并且：

```python
train_test_split(..., random_state=RANDOM_STATE)
```

在 skrub DataOps 中，也可以使用 `DataOp.skb.train_test_split` 来组织 train/test 环境；官方文档说明其返回 train/test environment 以及 X_train、X_test、y_train、y_test 等键。

---

# 8. Evaluation Plan

你们 PDF 的评估部分已经提出三类设置：fixed baseline、RAG-guided agent、ablation version，并要求记录性能、效率、鲁棒性和消融结果。这里建议改成更贴合 skrub 的版本。

## 8.1 Datasets

建议使用 2–3 个 tabular dataset：

| Dataset      | Task                  | Reason                   |
| ------------ | --------------------- | ------------------------ |
| Titanic      | binary classification | 小、易调试、有缺失值和类别特征          |
| Adult Income | binary classification | 类别特征多，适合 TableVectorizer |
| Ames Housing | regression            | 测试 regression pipeline   |

第一阶段不要使用 CIFAR、Flowers、Food-101，因为这些是图像数据集，不适合展示 skrub DataOps 的核心价值。

## 8.2 Baselines

| Setting                            | Description                       |
| ---------------------------------- | --------------------------------- |
| Fixed baseline                     | 手写固定 skrub + sklearn pipeline     |
| Random refinement                  | 随机选择 component 修改                 |
| RAG-guided refinement              | 根据 dataset profile 检索策略           |
| Ablation-based targeted refinement | 通过 ablation 选择最有价值的 component 再修改 |

## 8.3 Metrics

### Classification

| Metric   | Usage                    |
| -------- | ------------------------ |
| Accuracy | 类别平衡时使用                  |
| Macro F1 | 类别不平衡时优先使用               |
| ROC-AUC  | binary classification 可选 |

### Regression

| Metric | Usage |
| ------ | ----- |
| RMSE   | 主指标   |
| MAE    | 辅助指标  |
| R²     | 辅助解释  |

### Agent Metrics

| Metric                         | Meaning         |
| ------------------------------ | --------------- |
| best validation score          | agent 找到的最佳性能   |
| improvement over baseline      | 相对 baseline 的提升 |
| number of trials to best score | 达到最佳结果所需实验次数    |
| execution success rate         | 生成代码成功运行比例      |
| average runtime                | 平均实验运行时间        |

---

# 9. Implementation Plan

根据你们 PDF 中的时间线，可以调整为以下更可执行的版本。

## Phase 1: Project Definition & Pitch

**Goal:** 明确 scope，完成 pitch slides。

Deliverables:

```text
- project scope
- architecture diagram
- evaluation plan
- risk list
```

## Phase 2: Runnable Baseline Prototype

**Goal:** 不使用 LLM，先跑通一个固定 skrub pipeline。

Deliverables:

```text
- load CSV
- analyze dataset
- build fixed skrub pipeline
- train / validate
- output score
```

## Phase 3: Code Generation Prototype

**Goal:** 让 LLM 生成 pipeline，但先限制模板。

Deliverables:

```text
- pipeline plan JSON
- prompt template
- generated Python script
- execution result parser
```

## Phase 4: Experiment Tracking

**Goal:** 每次实验可复现、可比较。

Deliverables:

```text
- experiments.jsonl
- leaderboard.csv
- error logs
- config snapshots
```

## Phase 5: Ablation + Targeted Refinement

**Goal:** 实现项目核心创新点。

Deliverables:

```text
- component-level ablation
- most-impactful-component selector
- controlled refinement prompt
- comparison with random refinement
```

## Phase 6: Final Evaluation & Documentation

**Goal:** 多数据集评估，准备最终报告。

Deliverables:

```text
- final best pipelines
- experiment tables
- score improvement plots
- technical documentation
- README
```

---

# 10. Innovation Points

## 10.1 skrub DataOps-constrained Code Generation

普通 LLM 生成 ML pipeline 时通常会直接写 pandas + sklearn。你们的创新点是强制 agent 使用 skrub DataOps abstraction，使 pipeline 更结构化、可组合、可检查。

## 10.2 Dataset-profile-guided RAG

RAG 不只是检索模型名称，而是根据 dataset profile 检索 pipeline strategy：

```text
missing values -> TableVectorizer default handling
high-cardinality categorical features -> skrub categorical encoders
class imbalance -> macro F1 + class_weight
regression task -> RMSE + HistGradientBoostingRegressor
```

## 10.3 Ablation-based Targeted Refinement

不是随机让 LLM 改代码，而是通过 ablation 找到最值得优化的 component，然后只允许修改该部分。

## 10.4 Controlled Code Modification

为每个 pipeline component 设置可编辑区域，避免 LLM 修改数据划分、metric 或日志格式，保证实验公平性。

## 10.5 Experiment-aware Agent Memory

agent 维护历史实验结果，不重复尝试已经失败或无效的配置。

---

# 11. Repository Structure

建议项目结构如下：

```text
mle-star-skrub-agent/
  README.md
  pyproject.toml

  data/
    titanic/
    adult/
    ames/

  agent/
    __init__.py
    task_analyzer.py
    rag_retriever.py
    pipeline_planner.py
    code_generator.py
    executor.py
    evaluator.py
    ablation_analyzer.py
    targeted_refiner.py

  prompts/
    generate_baseline.txt
    refine_component.txt
    debug_error.txt

  knowledge_base/
    classification_baselines.md
    regression_baselines.md
    skrub_dataops_examples.md
    missing_values.md
    imbalanced_data.md

  generated/
    pipeline_exp_0001.py
    pipeline_exp_0002.py

  experiments/
    experiments.jsonl
    leaderboard.csv
    logs/

  tests/
    test_task_analyzer.py
    test_evaluator.py
    test_executor.py
```

---

# 12. Minimal Milestone for Demo

答辩或 pitch 阶段最小可展示版本应该是：

```text
Input:
  Titanic dataset + target column

Agent:
  1. analyzes dataset
  2. retrieves strategy
  3. generates skrub pipeline
  4. runs baseline
  5. modifies model component
  6. compares score before/after

Output:
  baseline score
  refined score
  best generated code
  experiment log
```

这个 demo 不需要一开始实现完整 MLE-STAR。只要能展示：

```text
baseline -> experiment result -> targeted refinement -> better or comparable result
```

就能说明 agent workflow 是成立的。

---

# 13. Risks and Mitigations

| Risk             | Impact | Mitigation                         |
| ---------------- | ------ | ---------------------------------- |
| LLM 生成代码无法运行     | 高      | 使用模板化 code generation              |
| agent 修改范围失控     | 高      | 使用 component-level editable region |
| 实验太慢             | 中      | 限制 trial budget，先用小数据集             |
| 指标不公平            | 高      | 固定 split、seed、metric               |
| RAG 没有明显贡献       | 中      | 对比 RAG vs no-RAG                   |
| ablation 实验成本高   | 中      | 每轮只测试少量 component                  |
| scope 偏到深度学习     | 高      | MVP 严格限制 tabular skrub pipeline    |
| final result 不稳定 | 中      | 多 seed 可作为后续扩展，MVP 先固定 seed        |

---

# 14. Final Technical Summary

本项目应实现一个面向 tabular ML 的 MLE-STAR-style agent。系统接收数据集和任务描述，先生成基于 skrub DataOps 的 baseline pipeline，然后通过实验反馈、ablation analysis 和 targeted refinement 逐步优化 pipeline。

项目核心不在于训练最强模型，而在于展示一个可控、可复现、可追踪的自动化 ML engineering workflow：

```text
Dataset -> Analyze -> Generate skrub Pipeline -> Execute -> Evaluate -> Ablate -> Refine -> Select Best
```

对于你们当前阶段，建议把技术路线明确收敛为：

```text
MVP: tabular dataset + skrub DataOps + sklearn model + RAG + ablation refinement
Future work: vision datasets + deep learning + LoRA + multimodal pipeline synthesis
```

这样既符合原始项目要求，也能保留你们 PDF 中提出的 RAG、ablation、experiment tracking 和 controlled code modification 等创新点。


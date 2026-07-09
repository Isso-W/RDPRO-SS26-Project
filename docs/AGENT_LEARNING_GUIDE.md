# Jiaozi Agent 入门学习指南

本文档面向刚开始学习 Agent 的同学。目标不是泛泛介绍 Agent，而是把 Agent 相关概念和 Jiaozi 项目里的具体代码对应起来，帮助你知道该读哪些文件、为什么读、读完应该理解什么。

## 1. 先理解本项目里的 Agent 是什么

在很多教程里，Agent 常被描述成“LLM 自己思考、自己调用工具、自己完成任务”。但在这个项目里，当前最稳定的 Agent 不是纯 LLM ReAct Agent，而是 workflow-first agent。

也就是说，项目先用确定性的 Python 工作流把任务拆成固定步骤：

```text
输入任务和候选模型
-> 选择模型
-> 生成训练配置
-> 生成训练代码
-> 执行 baseline
-> 做 ablation
-> 根据实验结果 refinement
-> review 生成项目
-> 导出 notebook
```

这条主线在 `cv_autodl_agent/workflow.py`。

这个设计很适合入门，因为你可以清楚看到 Agent 的每一步，而不是一开始就面对一团 prompt 和不稳定的 LLM 输出。

## 2. 本项目 Agent 相关代码地图

| Agent 概念 | 本项目对应文件 | 你要学什么 |
|---|---|---|
| Orchestrator / 编排器 | `cv_autodl_agent/workflow.py` | Agent 如何按步骤推进任务 |
| State / 数据合同 | `cv_autodl_agent/schemas.py` | Agent 每一步传什么数据 |
| Planner / 计划器 | `cv_autodl_agent/planner.py` | 如何从任务生成训练方案 |
| Retrieval / 知识检索 | `retrieval/rag_retrieval.py` | Agent 如何从知识库找候选方案 |
| Tool / 工具调用 | `cv_autodl_agent/codegen.py`, `executor.py`, `review.py` | Agent 如何生成、执行、检查代码 |
| Feedback loop / 反馈闭环 | `cv_autodl_agent/ablation.py` | Agent 如何根据实验结果改进 |
| Guardrails / 约束与校验 | `schemas.py`, `review.py` | 如何避免 Agent 生成无效结果 |
| LLM interface / LLM 调用 | `features_extraction_api.py` | 如何把自然语言转成结构化输入 |
| Future agent design | `docs/superpowers/plans/2026-06-14-mcp-knowledge-guided-mle-agent.md` | 下一阶段真正多 Agent 系统会怎么设计 |

## 3. Agent 入门的核心概念

### 3.1 Goal

Goal 是 Agent 要完成的目标。

在 Jiaozi 里，目标通常是：

```text
给定用户任务和数据集，推荐模型，生成训练代码，运行实验，并输出可复现的结果。
```

在代码里，这个目标被拆进 `CVAutoDLWorkflow.run()`：

```python
result = workflow.run(
    manifest,
    candidates,
    output_dir,
    execution_mode="simulate",
)
```

对应文件：

- `cv_autodl_agent/workflow.py`
- `cv_autodl_agent/__main__.py`

### 3.2 State

State 是 Agent 运行过程中保存和传递的信息。没有稳定 state 的 Agent 很难调试。

本项目用 dataclass 定义状态和接口：

- `DatasetManifest`: 数据集描述
- `RetrievedModelCandidate`: 模型候选
- `TrainingSpec`: 训练计划
- `ExecutionResult`: 执行结果
- `AblationSummary`: 消融总结
- `ReviewReport`: 代码审查结果
- `WorkflowResult`: 最终结果

对应文件：

- `cv_autodl_agent/schemas.py`

学习重点：

```text
Agent 系统不要只传自然语言。
关键步骤应该尽量传结构化对象。
结构化对象可以验证、测试、持久化、复现。
```

### 3.3 Planner

Planner 决定 Agent 下一步准备怎么做。

在本项目中，`HeuristicTrainingSpecPlanner` 根据 manifest 和 candidate 生成 `TrainingSpec`。

它会决定：

- 使用什么 transforms
- 使用什么 loss
- 使用什么 metric
- 使用什么 optimizer
- 使用什么 scheduler
- batch size 多大
- 是否 freeze backbone

对应文件：

- `cv_autodl_agent/planner.py`

当前 planner 是规则式的，不是 LLM planner。这是好事，因为它稳定、可测试。以后可以把它替换成 LLM structured output，但前提是输出仍然符合 `TrainingSpec`。

### 3.4 Tools

Agent 不能只“想”，还要调用工具做事。

本项目里的工具不是外部插件，而是普通 Python 类：

| 工具 | 文件 | 作用 |
|---|---|---|
| `CodeGenerator` | `cv_autodl_agent/codegen.py` | 生成项目文件 |
| `GeneratedProjectExecutor` | `cv_autodl_agent/executor.py` | 运行生成的训练脚本 |
| `AblationEngine` | `cv_autodl_agent/ablation.py` | 生成和执行消融实验 |
| `ProjectReviewer` | `cv_autodl_agent/review.py` | 检查生成代码和训练结果 |
| `NotebookExporter` | `cv_autodl_agent/notebook.py` | 导出 Colab notebook |

学习重点：

```text
Agent 调工具时，工具必须有清晰输入和输出。
工具输出最好也是结构化数据，而不是只靠 stdout 文本。
```

### 3.5 Observation

Observation 是工具执行后的反馈。

在本项目里，训练脚本最后会输出 JSON，`executor.py` 从 stdout 里解析最后一行 JSON，得到 `ExecutionResult`。

对应文件：

- `cv_autodl_agent/executor.py`
- `cv_autodl_agent/templates.py`

这是 Agent 闭环的关键：

```text
Agent 生成代码
-> 执行代码
-> 读取 metric / error
-> 根据结果决定下一步
```

### 3.6 Feedback Loop

Feedback loop 是 Agent 和普通脚本最大的区别之一。

普通脚本一般是：

```text
生成一次
运行一次
结束
```

Agent 更像：

```text
生成
运行
观察
比较
修改
再运行
```

本项目中最明显的 feedback loop 在 `AblationEngine`：

- baseline 先跑一次
- 修改一个 component
- 再跑多个 variant
- 比较哪个 variant 提升最大
- 把 winner 应用到 refined spec

对应文件：

- `cv_autodl_agent/ablation.py`
- `cv_autodl_agent/workflow.py`

这部分非常值得重点学习，因为它接近 MLE-Agent 的核心思想。

### 3.7 Guardrails

Guardrails 是防止 Agent 跑偏的约束。

本项目里有几类 guardrails：

1. 输入验证

   `DatasetManifest.validate()` 和 `RetrievedModelCandidate.validate()` 会检查必要字段。

2. 候选过滤

   `selectors.py` 会排除 task family 不匹配的模型。

3. 静态检查

   `review.py` 会检查生成文件是否存在、是否能 `py_compile`、metric/loss 是否适配任务。

4. fallback

   如果某个 candidate baseline 失败，workflow 会继续尝试下一个 candidate。

对应文件：

- `cv_autodl_agent/schemas.py`
- `cv_autodl_agent/selectors.py`
- `cv_autodl_agent/review.py`
- `cv_autodl_agent/workflow.py`

学习重点：

```text
Agent 越自动，越需要 guardrails。
否则它可以生成看似合理但无法运行的东西。
```

## 4. 当前最重要的 Agent 主线：CVAutoDLWorkflow

入口类：

```python
class CVAutoDLWorkflow:
    ...
```

对应文件：

- `cv_autodl_agent/workflow.py`

### 4.1 初始化

`CVAutoDLWorkflow.__init__()` 接收多个可替换组件：

```python
planner
code_generator
executor
ablation_engine
reviewer
notebook_exporter
```

这是一种很典型的 Agent 工程设计：编排器只负责流程，具体能力交给可替换组件。

好处：

- 容易测试
- 容易替换某个模块
- 以后可以把 heuristic planner 换成 LLM planner
- 以后可以把 local executor 换成远程 GPU executor

### 4.2 run() 主流程

`run()` 是最值得细读的函数。

可以按这个顺序理解：

```text
1. validate inputs
2. rank candidates
3. for each candidate:
   3.1 write selected_candidate.json
   3.2 planner 生成 TrainingSpec
   3.3 codegen 生成项目
   3.4 executor 跑 baseline
   3.5 ablation engine 构造 variants
   3.6 executor 跑 ablation variants
   3.7 summarize ablation
   3.8 apply targeted refinement
   3.9 review generated project
   3.10 export notebook
4. 如果所有 candidate 失败，抛 WorkflowExecutionError
```

这就是一个完整的 agentic loop。

### 4.3 LangGraph 预留接口

文件里还有：

```python
def build_langgraph_workflow(self):
    ...
```

它目前只是把 workflow 节点用 LangGraph 形式串起来，但每个节点都是 `passthrough`，还没有真实 LLM 推理。

你应该这样理解它：

```text
当前真实运行靠 Python workflow。
LangGraph 是未来扩展方向。
```

不要误以为项目已经有完整 LangGraph Agent。

## 5. Module 3: Retrieval Agent / Recommender

对应文件：

- `retrieval/rag_retrieval.py`
- `docs/MODULE3_API.md`
- `docs/report_module3.md`

这个模块负责推荐模型配置。它不是纯 LLM 推荐，而是混合检索：

```text
结构化规则
+ 知识图
+ 向量检索
+ 图遍历组装
```

### 5.1 知识图

`COMPONENTS` 定义模型和组件：

- backbone
- pretrained_model
- head
- loss
- optimizer

`EDGES` 定义关系：

- `compatible_with`
- `has_pretrained`
- `alternative_to`
- `preferred_when`
- `requires`

这其实是 Agent 的“外部知识”。

### 5.2 向量索引

`build_vector_index()` 使用 ChromaDB 和 sentence-transformer。

它把 backbone description 放进向量库，让自然语言需求可以找到语义接近的模型。

### 5.3 混合检索

核心函数：

```python
retrieve_top3_hybrid(input_json, graph, collection)
```

流程是：

```text
1. 根据 data_size / real_time / edge_deployment 确定 scale band
2. 找到符合 task_type 和 size 的模型
3. 用 tier 规则过滤
4. 结构化打分
5. 向量相似度打分
6. 合并分数
7. 返回 Top 3
8. 拼装 head / loss / optimizer / checkpoint
```

这是一个非常好的 Agent 入门样例，因为它说明：

```text
真正可控的 Agent 通常不是把所有判断交给 LLM。
规则和检索先缩小范围，LLM 再参与更模糊的判断。
```

### 5.4 输出给 Module 4

函数：

```python
build_task_list(result, graph, fmt="structured")
build_task_list(result, graph, fmt="nl")
```

两种输出：

- `structured`: 适合确定性代码生成
- `nl`: 适合塞进 LLM prompt

这正是 Agent 系统常见的接口设计：同一份推荐结果，可以给规则系统，也可以给 LLM。

## 6. Module 1: LLM Feature Extractor

对应文件：

- `features_extraction_api.py`
- `docs/features_extraction_api.md`

这个模块调用 Qwen，把用户自然语言转成结构化特征。

它更像 Agent 系统的入口工具，不是完整 Agent。它没有多轮 planning，也没有 tool loop。

但它体现了 Agent 的一个重要模式：

```text
用户自然语言
-> LLM 提取结构化意图
-> 下游模块用结构化字段工作
```

学习时要注意：

```text
LLM 最适合处理模糊语言。
但是下游工程系统最好消费 JSON / list / dataclass。
```

## 7. Module 2: Dataset Analyzer

对应文件：

- `analyzer.py`

这个模块分析数据集：

- 行列数
- 列名
- dtype
- missing ratio
- unique count
- outlier count
- target column
- task type

它本身不是 Agent，但可以作为 Agent 的工具。

在 Agent 视角里，它的角色是：

```text
观察环境，生成 dataset profile。
```

也就是说，Module 2 给 Agent 提供事实，避免 Agent 只靠用户描述猜测数据。

## 8. Dog Breed notebook 和项目代码的关系

你提到的 notebook：

```text
/Users/wang/Downloads/“dog_breed_identification_colab_minimal_ipynb”的副本.ipynb
```

它展示了更完整的集成思路：

```text
Module 1: 解析自然语言任务
Module 2: 分析 Kaggle 图像数据
Module 3: 检索模型配置
Recommender: 重排候选
Module 4: 生成训练代码
Executor: 运行生成的 run.py
Kaggle: 生成并提交 submission.csv
```

这个 notebook 适合理解“端到端系统怎么串起来”。

但要注意：当前仓库 tracked 源码里，最稳定可读的是：

- `cv_autodl_agent/`
- `retrieval/`
- `features_extraction_api.py`
- `analyzer.py`
- `tests/`

notebook 里提到的 `pipeline.py`、`recommender`、`module4_agent` 等，在当前分支的 tracked 源码里并不完整。学习时不要被这些名字吓到，先掌握已经实现并通过测试的主线。

## 9. Future Agent: MCP Knowledge-Guided MLE Agent

对应计划文档：

- `docs/superpowers/plans/2026-06-14-mcp-knowledge-guided-mle-agent.md`

这是未来更完整的 Agent 设计，不是当前已经全部实现的代码。

它计划包含：

- Knowledge Learner Agent
- MLE Experiment Agent
- MCP tools
- strategy cards
- experiment memory
- outcome write-back
- plan-only mode
- opt-in real execution

这个计划很适合你在入门后继续学习，因为它展示了一个更成熟的 Agent 系统应该怎么分层。

其中有几个关键思想：

### 9.1 Knowledge Learner Agent

负责从外部资料中提炼策略卡片，例如：

- RandAugment
- MixUp
- CutMix
- label smoothing
- TTA
- EfficientNet-B4

它的职责是学习知识，不直接跑训练。

### 9.2 MLE Experiment Agent

负责根据本地知识和历史实验结果提出下一轮实验。

它的职责是：

```text
读历史
找可用策略
生成实验 proposal
可选执行训练
比较结果
写回 memory
```

这才更接近完整 MLE-Agent。

### 9.3 MCP Tools

MCP 的意义是把 Agent 能调用的能力做成标准工具接口。

在这个项目规划里，MCP tools 会包装：

- knowledge ingestion
- strategy search
- experiment planning
- experiment execution
- metrics comparison
- report generation

学习重点：

```text
Agent 不应该直接乱读乱写所有东西。
应该通过受控工具访问知识库、执行器和实验记忆。
```

## 10. Agent 学习路线

建议按下面顺序学习。

### 第 1 阶段：读懂 workflow-first agent

读：

1. `cv_autodl_agent/workflow.py`
2. `cv_autodl_agent/schemas.py`
3. `tests/test_workflow.py`

目标：

```text
能画出 CVAutoDLWorkflow.run() 的完整流程。
知道每一步输入输出是什么。
```

练习：

```bash
python3 -m unittest discover -s tests -v
```

### 第 2 阶段：读懂 Agent 的状态设计

读：

1. `cv_autodl_agent/schemas.py`
2. `examples/classification_manifest.json`
3. `examples/classification_candidates.json`
4. `examples/expected_classification_summary.json`

目标：

```text
知道 manifest、candidate、spec、result 分别是什么。
知道为什么 Agent 系统要用结构化数据。
```

### 第 3 阶段：读懂计划和生成

读：

1. `cv_autodl_agent/planner.py`
2. `cv_autodl_agent/codegen.py`
3. `cv_autodl_agent/templates.py`

目标：

```text
知道 Agent 如何从抽象计划生成可运行代码。
知道当前 real training 和 simulate training 的区别。
```

重点提醒：

```text
simulate 模式不是训练真实模型。
它只是模拟 metric，用来验证 workflow。
```

### 第 4 阶段：读懂执行和反馈

读：

1. `cv_autodl_agent/executor.py`
2. `cv_autodl_agent/ablation.py`
3. `cv_autodl_agent/review.py`

目标：

```text
知道 Agent 如何执行代码。
知道 Agent 如何从 metric 中判断哪个修改有效。
知道 review 为什么是 Agent guardrail。
```

### 第 5 阶段：读懂 RAG/recommender

读：

1. `retrieval/rag_retrieval.py`
2. `retrieval/test_rag_retrieval.py`
3. `docs/MODULE3_API.md`
4. `docs/report_module3.md`

目标：

```text
知道模型推荐不是纯 prompt。
理解结构化规则、图关系、向量检索如何结合。
```

推荐测试命令：

```bash
/Users/wang/Documents/Jiaozi/.venv/bin/python retrieval/test_rag_retrieval.py
```

### 第 6 阶段：看未来 Agent 架构

读：

1. `docs/superpowers/plans/2026-06-14-mcp-knowledge-guided-mle-agent.md`

目标：

```text
理解 Knowledge Learner Agent、MLE Experiment Agent、MCP tools、experiment memory 的分工。
```

## 11. 本项目里可以动手做的小练习

### 练习 1：追踪一次 workflow

运行：

```bash
python3 -m cv_autodl_agent \
  --manifest examples/classification_manifest.json \
  --candidates examples/classification_candidates.json \
  --output-dir demo_run \
  --execution-mode simulate
```

然后看：

```text
demo_run/candidate_01_timm-resnet18/
  selected_candidate.json
  training_spec.json
  baseline_result.json
  ablation_plan.json
  ablation_trials.json
  ablation_summary.json
  refined_training_spec.json
  refined_result.json
  review_report.json
  notebook.ipynb
```

你要回答：

```text
baseline 分数是多少？
哪个 ablation variant 赢了？
refined_training_spec 改了什么？
review 为什么 pass？
```

### 练习 2：改一个 ablation variant

修改：

- `cv_autodl_agent/ablation.py`

比如给 classification 增加一个新 variant：

```python
AblationVariant(
    "learning_rate",
    "lr_1e4",
    {"optimizer.learning_rate": 1e-4},
)
```

然后重新跑测试：

```bash
python3 -m unittest discover -s tests -v
```

你要观察：

```text
新 variant 是否出现在 ablation_trials.json？
它有没有赢过 baseline？
```

### 练习 3：给 recommender 增加一个模型

修改：

- `retrieval/rag_retrieval.py`

增加一个 backbone 或 pretrained checkpoint。

你要同时考虑：

- `COMPONENTS`
- `EDGES`
- task_type
- data_size
- tier
- compatible head
- compatible loss
- optimizer

然后运行：

```bash
/Users/wang/Documents/Jiaozi/.venv/bin/python retrieval/test_rag_retrieval.py
```

### 练习 4：把 LangGraph passthrough 变成真实节点

当前：

- `cv_autodl_agent/workflow.py`
- `build_langgraph_workflow()`

里面每个节点都是 passthrough。你可以尝试先做一个简单版本：

```text
select_candidate 节点读取 state["manifest"] 和 state["candidates"]
调用 rank_candidates
把 selected_candidate 写回 state
```

这能帮助你理解 LangGraph 的 state 传递方式。

不要一开始就接 LLM。先让图真的跑通一个确定性节点。

## 12. 学 Agent 时容易误解的地方

### 误解 1：用了 LLM 才叫 Agent

不完全对。

一个系统如果能：

```text
感知状态
制定计划
调用工具
观察结果
根据反馈调整
```

它就具备 Agent 的核心结构。

本项目的 `CVAutoDLWorkflow` 就是 workflow-first agent。

### 误解 2：Agent 等于 ReAct

ReAct 是一种 Agent 模式，但不是唯一模式。

本项目更接近：

```text
deterministic workflow
+ retrieval
+ structured planning
+ execution feedback
+ targeted refinement
```

这种模式在工程上更稳定。

### 误解 3：RAG 就是向量库搜索

不完整。

本项目 Module 3 的 RAG/recommender 包含：

- 结构化过滤
- 知识图关系
- 向量相似度
- 规则打分
- 任务清单生成

这比单纯向量搜索更可控。

### 误解 4：生成代码后就结束

对 ML engineering agent 来说，生成代码只是中间步骤。

真正重要的是：

```text
生成代码
-> 运行代码
-> 读结果
-> 比较结果
-> 修改方案
```

这就是 ablation/refinement 的意义。

### 误解 5：模型新就一定分数高

不对。

比如 DINOv2 是强模型，但如果：

- 代码实际 fallback 到 ResNet18
- 只训练 linear head
- 没有 K-fold
- 没有 TTA
- 没有 ensemble
- 没有按比赛 metric 优化

分数仍然可能不好。

Agent 要解决的不只是“选模型”，而是整个实验策略。

## 13. 推荐阅读顺序

最小阅读路径：

```text
1. cv_autodl_agent/workflow.py
2. cv_autodl_agent/schemas.py
3. cv_autodl_agent/planner.py
4. cv_autodl_agent/ablation.py
5. cv_autodl_agent/executor.py
6. cv_autodl_agent/review.py
7. retrieval/rag_retrieval.py
8. docs/MODULE3_API.md
9. docs/superpowers/plans/2026-06-14-mcp-knowledge-guided-mle-agent.md
```

如果你只读一个文件，读：

```text
cv_autodl_agent/workflow.py
```

如果你只理解一个思想，理解：

```text
Agent = 状态 + 计划 + 工具 + 观察 + 反馈 + 约束
```

Jiaozi 当前最有价值的地方，就是把这个思想落到了 ML 自动实验流程里。


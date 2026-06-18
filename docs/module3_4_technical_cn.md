# Module 3 & Module 4 技术详解（技术评审 / 答辩用）

> 面向技术读者。覆盖 Module 3（模型检索/选型）与 Module 4（代码生成）的数据结构、
> 算法、每个设计选择的权衡、已知局限。最后一节"预期提问"预先准备了教授可能问的问题及回答。
>
> 对应代码：`retrieval/rag_retrieval.py`（M3）、`recommender/`（推荐器层）、
> `module4_agent/`（M4）、`pipeline.py`（编排 + M2→M3 字段映射）。

---

# Part A — Module 3：模型检索 / 选型

## A.1 知识库（Knowledge Base）双层结构

Module 3 的核心是一个**人工策展的知识库**，由两层组成：

**① NetworkX 有向图（DiGraph）—— 表达组件关系**
- 节点类型：`backbone` / `pretrained_model` / `head` / `loss` / `optimizer`
- 边类型：
  - `has_pretrained`：backbone → 它的预训练 checkpoint
  - `compatible_with`：可搭配使用（backbone↔head/loss/optimizer）
  - `requires`：集成架构（DETR/RT-DETR）的固定连接，head/loss 经 `requires` 到达后不可替换
  - `preferred_when`：条件偏好（"满足条件时优先 A 而非 B"）
  - `alternative_to`：替代关系

**② ChromaDB 向量索引 —— 语义检索**
- 只对 **backbone 的文字描述**做嵌入（embedding model：`all-MiniLM-L6-v2`）
- pretrained 节点通过图遍历到达，**不直接做向量检索**

**当前规模**：14 个 backbone、22+ 个 HF 预训练 checkpoint、7 个 head、6 个 loss、3 个 optimizer。

**设计权衡（为什么用人工 KB + 图 + 向量混合）**：
- **图**擅长表达"硬关系"（谁能配谁、集成架构的固定结构），规则清晰可控
- **向量**擅长"软匹配"（用户自由描述 ↔ 模型介绍）
- **为什么不用 agent 现场上网搜**（对比 MLE-STAR）：KB **可控、可解释、确定性**——
  能说清"为什么选"，便于审计；代价是覆盖面受人工维护限制（已列入"自动扩库"待办）。

### A.1.1 节点字段（schema）

`backbone` 节点：
- `task_type`：支持的任务列表
- `tier`：**按任务**的角色 dict，`"default"` / `"accuracy_upgrade"` / `"special_case"`
- `scratch_viable_from`：从头训练所需的最小 data_size（`small`/`medium`/`large`/`None`）
- `domain_transfer`：`strong`/`moderate`/`weak`（已采集，**暂未用于打分**——已知 gap）
- `capabilities`：如 `["zero_shot","few_shot","open_vocabulary"]`（目前只 DINOv2/CLIP 有）

`pretrained_model` 节点：
- `size_tier`：`nano`/`small`/`base`/`large`/`xlarge`
- `finetune_strategy`：`full`/`head_only`/`either`
- `freeze_viable`：bool
- `params_M`：参数量（百万）
- `flops_g`：GFLOPs@224（**新增**，成本模型用）
- `recommended_when`：dict（定义了但**暂未消费**——已知 gap）

### A.1.2 条件格式（EDGE_CONDITIONS）

条件存为**结构化 dict**，不是字符串：
```python
{"condition": {"all": ["real_time=True", "edge_deployment=True"]}}  # AND
{"condition": {"any": ["cross_modal=True", "zero_shot=True"]}}      # OR
```
`_matches_condition(condition, input_json) -> bool` 对输入求值。
合法 key 例：`real_time=True`、`edge_deployment=True`、`class_imbalance=True`、`zero_shot=True` 等。

## A.2 检索流水线（Hybrid，Scheme C）

`retrieve_top3_hybrid(input_json, graph, collection)` 六步：

**Step 1 — 规模带过滤（scale-band）**
`_determine_scale_band(input_json)` 由硬约束推出可接受的 `size_tier` 范围：
- `edge_deployment` 或 `real_time` → `{nano, small}`（硬约束）
- `data_size=small` → `{nano, small, base}`（大模型小数据过拟合风险高）
- `data_size=large` 且 `priority=accuracy` → `{base, large}`
- 其余 → 全部

`_get_eligible_pairs(...)` 产出 `(backbone_id, checkpoint_id|None)` 对。backbone 入选条件：
支持 `task_type`，且（在 scale-band 内有 checkpoint）或（`scratch_viable_from` 允许当前 data_size，
此时 checkpoint=None 表示从头训练）。

**Step 2 — tier 过滤（`_filter_by_tier`）**
- `default`：始终保留
- `accuracy_upgrade`：仅 `priority=accuracy` 时保留
- `special_case`：需要其激活约束之一（`_SPECIAL_CASE_REQUIRES`，**any-of 语义**，
  如 `clip_vit` 在 `cross_modal` 或 `zero_shot` 激活）
- `zero_shot=True`：**硬过滤**——只有 `capabilities` 含 `zero_shot` 的 backbone 通过

**Step 3 — 结构化打分（`_score_backbone`）**
data_size 匹配（0–2）+ priority vs 复杂度（0–2）+ `preferred_when` 加成（每条 +1.5）
+ `few_shot` capability 加成（+1.5）。归一化到 [0,1]。

**Step 4 — 向量打分**
输入转自然语言，与 backbone 描述做余弦相似度（all-MiniLM）。归一化到 [0,1]。

**Step 5 — 加权合并**
`structured × 0.6 + vector × 0.4` → 排序取 Top 3。

**Step 6 — 图遍历拼装**
对每个 Top-3 backbone：用 Step 1 预选的 checkpoint；经 `requires` 再 `compatible_with` 边
解析 head/loss/optimizer；`_recommend_training` 给出训练策略。

**设计权衡**：
- **60/40 权重**：结构化信号（规则）更可信，向量当辅助/打破平局。
  （已知问题：向量基于"一段话描述"，信号噪声较大，占 40% 偏高——见局限。）
- **打分是手工启发式**：data_size 3 档、priority 3 值——粗粒度导致**近似平局**
  （观察到 efficientnet 0.691 vs dinov2 0.67），Module 1 输出抖动会翻排名。已记入改进项。

## A.3 成本模型 + 约束感知（新增）

**动机**：原系统不管部署约束，可能推一个塞不进目标设备的大模型。

**成本模型**（`estimate_cost`）：
```
params = checkpoint 节点的 params_M
flops_g = flops_g@224 × (image_size / 224)²   # 随分辨率面积缩放
```
`flops_g` 集中维护在 `_CHECKPOINT_FLOPS_G` 表，`build_graph` 时注入节点。

**预算过滤**（`_within_budget`）：input 的 `constraints` 可含数值预算
`max_params_m` / `max_flops_g`（+ `image_size`）。接在 **checkpoint 候选筛选**两处
（`_select_checkpoint` 和 `_get_eligible_pairs` 的 `cps_in_band`）——
超预算的 checkpoint 被剔除，于是 **backbone 自动降级到预算内最大的变体**
（如 ResNet50→ResNet18）。无预算时全程不变（向后兼容）。

**KB 扩张**：为补足小预算档，新增 `resnet18_imagenet`（small）、`efficientnet_lite0`（边缘）。

**诚实定位**：这是 Phase 1+2（成本模型 + 预算过滤），只保证"守预算"；
"预算内按精度最优排序"是 Phase 3，需要精度信号（见推荐器/LogME）。

## A.4 训练策略解析（`_recommend_training`）

checkpoint 节点的 `finetune_strategy` 若为 `"either"`，按**上下文解析**为具体策略：
- `task_type == feature_extraction` → `head_only`（特征提取要冻结特征）
- `few_shot=True` 或 `data_size == small` → `head_only`（大 ViT 在小数据全量微调会过拟合）
- 否则 → `full`（够数据的分类用全量微调争质量）

**动机**：观察到 DINOv2 一直输给 EfficientNet——根因是 DINOv2 被固定为冻结 linear probe，
而 EfficientNet 全量微调。冻结探针在细粒度任务上天然弱于全量微调。改为上下文解析后，
DINOv2 在普通分类上也能全量微调，公平竞争（配合 Module 4 的分组 LR，见 B.5）。

## A.5 推荐器层（recommender/，核心创新）

Module 3 的检索给出候选短名单；**推荐器层**在其上做"会积累、可解释"的重排（opt-in，`use_recommender`）。

**A.5.1 数据集指纹**（`fingerprint.py`）
`dataset_fingerprint(m2_report, m3_input)` → 语义信号：task_type、num_classes、data_size、
total_images、class_imbalance、resolution_tier、color_mode。
`fingerprint_distance(a,b)`：task_type 不同 → ∞（硬门）；否则加权距离
（类别数 log 差权重 2.0、data_size 1.5、分辨率/不平衡/色彩各 0.5）。

**A.5.2 结果记忆**（`outcome_memory.py`）
JSONL 日志，每条 `(fingerprint, config, result, cost)`。
`query_similar(fingerprint, k, backbone)` 返回最相似的历史记录。
既是检索源，也是未来学习型预测器的训练数据。

**A.5.3 三层排序**（`ranker.py`）
- **memory**：相似度加权的同 backbone 历史 metric（kNN，积累）
- **logme**：在本数据集冻结特征上的 LogME 迁移性分（冷启动，数据集特异）
- **heuristic**：KB 结构化+向量分（兜底）
排序：有 memory 的（按预测 metric）> 有 logme 的（按 LogME）> 仅 heuristic（保持原序）。
每个候选附**解释字符串**（点名最近相似数据集 + 其得分 / 或 cold start 说明）。

**A.5.4 LogME**（`logme.py`）
LogME（You et al., ICML 2021）：给定冻结特征 + 标签，估计线性模型的 log 最大证据，
**不训练**即预测"微调后排名"。纯 numpy 实现，验证过单调性（可分特征分更高）。
特征提取（加载 backbone + 前向）由调用方负责，产出 `{backbone: logme}` 喂排序器。

**A.5.5 配方层**（`recipe.py`，超参推荐）
`recommend_recipe(backbone, finetune_strategy, data_size, m2_report, task_type)` 按规则给 HP：
- **硬约束**：DINOv2 patch-14 → image_size 取 14 倍数
- **惯例**：冻结 lr 1e-3；CNN 全量 3e-4；transformer 全量 1e-3(head)+`backbone_lr_scale` 0.01；
  小数据强增强；早停 patience 按 data_size
- 只产出生成代码消费的 key；v1 预留 `_llm_recipe_proposal`（LLM 提议 + 规则护栏，stub）

**设计哲学**：超参规律多为机制性、业界公认（不像选架构那么靠猜），所以**规则化可靠**；
配方 = 编码专家默认，不做超参搜索（搜索贵，是竞品主场）。

## A.6 Module 3 已知局限（主动列出）

- `preferred_when` 边在 loss/pretrained 节点上是死数据（只有 source=backbone 的被消费）
- `_select_components` 的 head/optimizer 用 `candidates[0]`（顺序依赖、脆弱）
- `domain_transfer`、`recommended_when` 已采集但未消费
- 向量索引只覆盖 backbone（head/loss 不可语义检索）
- 结构化打分粗粒度 → 近似平局（已用推荐器 + 成本/约束缓解，但根因仍在）

---

# Part B — Module 4：代码生成

## B.1 接口（Module 3 → Module 4 契约）

`build_task_list(result, graph, fmt)` / `build_all_task_lists(...)` 把检索输出转成任务清单：
- `fmt="structured"`：固定 `action` 类型（load_pretrained / train_from_scratch /
  set_finetune_strategy / configure_head/loss/optimizer）
- `fmt="nl"`：自然语言任务列表 + `model_config` 元数据 dict

`pipeline.py` 在交给 M4 前，向每个 `model_config` 注入：`num_classes`、`dataset_id`、
`evaluation_metric`、`recommended_epochs`、（opt-in）recipe 超参、`offline_smoke`。

## B.2 工作流（`workflow.run_workflow`）

```
task_lists → spec_builder → 代码生成 → reviewer（静态检查）
           → smoke harness（跑 run.py/run_experiments.py）→ refinement loop（可选）
           → 写 module4_summary.json
```

## B.3 配置流转（spec_builder + schemas）

- `build_training_specs(candidates)`：`merged = {**task_overrides, **model_config}`，
  抽出 backbone/head/loss/optimizer/finetune_strategy/lr/image_size/... 构造 `TrainingSpec`，
  **完整保留 `raw_model_config`**。
- `TrainingSpec.to_config()`：`asdict(self)` + 把 `raw_model_config` 放回 `config["model_config"]`。
- 生成的 `run.py` 用 `normalize_config` **把 model_config 拍平到顶层** →
  生成代码 `get_value(config, key, default)` 在顶层读到。

**关键**：因为有这个"model_config 透传口袋 + 拍平"，pipeline 注入的任何字段
（如 `evaluation_metric`）下游自动接住，**Module 4 无需改动**。

## B.4 代码生成（模板 vs LLM）

| 文件 | 来源 |
|---|---|
| `model.py` | LLM 生成（配了 provider 且成功）/ 否则模板 `_model_py` |
| train/evaluate/infer/utils/model_utils/run/run_experiments/smoke_data 等 | **永远确定性模板** |

**设计权衡**：把可靠性留给模板（训练骨架固定可靠），灵活性留给 LLM（仅模型架构那块）。

**LLM 路径（`llm_codegen.py`）**：
- provider 抽象：qwen / openai / vertex / none，env 切换
- `_response_text`：鲁棒提取（SDK 对象 / dict / 纯字符串 / 嵌套 choices / responses API）
- `_chat_completion`：temperature=0 调用，**被拒（如 gpt-5.x 只允许默认）自动去 temperature 重试**
- `_validate_model_python`：拒 HTML 网关页 + `ast.parse` 合法 + 必须定义 `build_model`
- `generate_model_py`：**自纠错循环**——内容错（非法 Python / 缺 build_model）把错误喂回 LLM
  重试（默认 2 次，env `M4_MODEL_PY_ATTEMPTS`）；传输错（无内容）立即退模板；
  耗尽次数退模板（**模板是可靠下限，非"一报错就放弃"**）

**LLM model.py 契约**（prompt 强制）：用 `model_utils.load_backbone`/`apply_freeze`，
导出 `build_model(config)`，forward 返回类型严格（分类裸 tensor、检测 dict、分割 [B,C,H,W]、
特征 L2 归一）。
> 已知静默坑：prompt 未强制 submodule 命名 `self.backbone`/`self.head`，
> 若 LLM 偏离，会使**特征缓存 + 分组 LR 不触发**（不报错但 DINOv2 可能回到灾难性遗忘）。已记入待办。

## B.5 生成的训练代码内部（train.py 模板）

- **真实 dataloader**：HF 数据集 + 本地 CSV（Kaggle）；检测/分割暂回退合成数据
- **冻结 backbone → 特征缓存**：检测到 backbone 全冻（head_only），**提一次特征 + 缓存到盘**，
  之后只在缓存上训 head（≈一次数据遍历替代 N 次），并用确定性预处理（标准 linear-probe 协议）
- **分组学习率**（`_build_optimizer`）：微调的 transformer backbone（vit/swin/dino/clip/...）
  用低 LR（`backbone_lr_scale` 默认 0.01，≈1e-5），head 用满 LR；CNN/冻结保持单组
  → **防止全量微调大 ViT 时灾难性遗忘**（与 A.4 配套）
- checkpoint/断点续训、val 每轮、早停、AMP、cosine scheduler、class weights、label smoothing
- `evaluate` 按 `evaluation_metric` 算（accuracy/macro_f1/roc_auc/qwk/log_loss）

## B.6 成本计量（cost_meter）

进程级累积 LLM 调用数/token、训练 runs/epochs、墙钟。在 Module 1/Module 4 的 LLM 调用点插桩
（guarded，绝不影响调用）。记入 outcome_memory 的 `cost` 字段 → 支撑"质量 vs 成本"对比。

---

# Part C — 预期教授提问 + 回答

**Q1：你的选型本质是规则 + 向量的启发式打分，凭什么比一个 LLM agent 好？**
A：在"裸分/性能"上我们不声称更好——agentic 搜索（如 MLE-STAR）会更强。我们的差异是**结构性的**：
(1) 成本——一次性推荐 vs 反复试错训练，便宜几十倍；(2) 可解释——每个推荐有证据链；
(3) 积累——outcome_memory 让它跨任务越用越准，agent 每次从零、健忘；(4) 约束感知——
在部署预算内选最优。这些是 agent 的工作方式结构上做不到的角落。

**Q2：手工知识库不会过时/覆盖窄吗？**
A：会，这是真实代价。缓解：(1) 推荐器从结果中学习，弱化对 KB 精确性的依赖；
(2) 已规划"自动扩库"——从 HF/papers-with-code 把新 backbone 灌进持久化结构图
（结合 agent 的实时性 + 我们的结构/可解释）。

**Q3：结构化打分这么粗，怎么保证选得对？**
A：承认粗粒度导致近似平局（观察到 efficientnet vs dinov2 贴脸）。三重缓解：
约束/成本过滤先收紧候选；推荐器用历史实测/LogME 重排（数据集特异，比启发式准）；
近似平局靠 cost-aware tiebreak（规划中）。根因（启发式权重未标定）已列入改进项，
计划用 catalog 评测 harness 数据驱动地标定。

**Q4：LogME 是什么、为什么用它而不是真训练或 linear probe？**
A：LogME 估计冻结特征对标签的可分性（线性模型的 log 最大证据），**不训练**即预测微调排名。
比 linear probe 更便宜（不用训探针）、且专门预测"微调后"而非"冻结精度"。它是冷启动信号
（记忆为空时），随记忆积累其权重下降但永不归零（新数据/新 backbone 总冷启动）。

**Q5：为什么 DINOv2 之前一直更差，你怎么修的？**
A：根因是它被固定为冻结 linear probe，而 CNN 全量微调——冻结探针在细粒度任务上天然弱。
修法两件套：(A) Module 3 把 `either` 按上下文解析（够数据的分类→full）；
(B) Module 4 分组 LR（微调 transformer backbone 用 ~1e-5，否则用 head 的高 LR 会灾难性遗忘）。
两者必须同时，否则只做 A 会更糟。

**Q6：评估指标怎么定的？硬编码吗？**
A：不是。Module 1 从自然语言抽 `evaluation_metric`（accuracy/macro_f1/roc_auc/qwk/log_loss，
含别名如 AUC→roc_auc），经 merge 带入 m3_input，pipeline 注入 model_config，
生成代码本就消费它。所以"按 AUC 评"/"分级一致性(QWK)"会用对的指标，而非一律 accuracy。

**Q7：超参（recipe）凭什么推荐？能到什么水平？**
A：三层依据：硬约束（backbone 事实，查表）+ 业界微调惯例（编码规则，有据）+（v1）记忆+LLM 校准。
超参规律多为机制性（如预训练需低 LR=灾难性遗忘是真现象），所以规则化可靠。
水平定位：**强默认（懂行第一猜），标准任务离精调差几个点，靠记忆越用越准，但不及完整超参搜索**。
这个能用 AIDE 对比 harness 实测（规则 vs LLM-recipe vs AIDE，质量×成本）。

**Q8：代码生成失败怎么办？可靠性如何保证？**
A：多层：模板生成训练骨架（确定性可靠）；LLM 只写 model.py 且经校验（ast.parse + 必含 build_model + 拒 HTML）；
失败先**自纠错重试**（喂回错误，有界）再退模板；模板是可靠下限。生成后还有 reviewer 静态检查 + smoke 测试。

**Q9：和 skrub DataOps 的关系？**
A：整条 pipeline（M1→M3）另有一个 skrub DataOps DAG 包装（`skrub_pipeline.py`），
可 `describe_steps()` 输出文字图、`draw_graph()` 出 SVG/PNG，满足"pipeline 用 DataOps 抽象 +
能展示计算图"的要求。每个 DAG 节点调用真实模块函数，是 pipeline 的"图视图"，目前覆盖 M1→M3。

---

## 附：关键文件索引

| 文件 | 内容 |
|---|---|
| `retrieval/rag_retrieval.py` | M3 KB + 检索 + 成本模型 + 训练策略解析 |
| `recommender/{fingerprint,outcome_memory,ranker,logme,recipe}.py` | 推荐器层 |
| `module4_agent/{workflow,spec_builder,code_generator,llm_codegen,schemas}.py` | M4 |
| `pipeline.py` | 编排 + M2→M3 字段映射 + 注入 |
| `cost_meter.py` | 成本计量 |
| `skrub_pipeline.py` | DataOps 计算图 |

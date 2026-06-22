# Module 3 — API Reference for Module 4

<!-- 模块3的输出：接收模块1的任务描述，返回最多3个模型配置，每个配置打包成任务清单供模块4使用 -->

Module 3 takes a task description from Module 1 and returns a ranked list of model configurations, each packaged as an actionable task list.

---

## Module 2 → Module 3 接口对齐（待确认）

> 本节用于与模块2负责人对齐输入格式，尚未最终确定。

### 模块3当前期望的输入

```python
{
    "task_type":   str,   # 枚举值，见下表
    "data_size":   str,   # "small" | "medium" | "large"
    "priority":    str,   # "speed" | "accuracy" | "balanced"
    "constraints": {
        "real_time":       bool,
        "edge_deployment": bool,
        "class_imbalance": bool,
        "cross_modal":     bool,
        "medical":         bool,
    },
    "description": str,   # 自由文本，用于语义检索兜底
}
```

模块3目前消费的是**结构化布尔值**。模块2如果输出的是关键词，需要在两者之间加一层映射，这层映射可以放在模块2输出侧，也可以放在模块3输入侧——需要确认由谁来做。

---

### 关键词 → 结构化字段 映射参考

#### task_type（枚举，必须精确匹配）

| 用户关键词（示例） | 映射值 |
|------------------|--------|
| 检测、定位、框出、找出、计数 | `object_detection` |
| 分类、识别、判断是什么、打标签 | `classification` |
| 分割、掩码、像素级标注、抠图 | `image_segmentation` |
| 特征、向量、检索、相似度、embedding | `feature_extraction` |

#### data_size（建议支持数量区间）

| 用户关键词（示例） | 映射值 |
|------------------|--------|
| 数据少、几百张、标注贵、样本不足 | `small` |
| 几千到几万张、自采数据 | `medium` |
| 大量数据、百万级、公开大数据集 | `large` |

数字换算由 `pipeline.derive_data_size()` 实现（2026-06-10 更新为双信号）：

**总量档位**（成本侧——总量决定标注/训练成本）：

| 任务类型 | small | medium | large |
|---------|-------|--------|-------|
| classification / feature_extraction | ≤ 3,000 | 3,001–20,000 | > 20,000 |
| object_detection / image_segmentation | ≤ 1,500 | 1,501–10,000 | > 10,000 |

**每类样本数档位**（过拟合侧，仅分类任务）：≤ 100 张/类 → small，101–1,000 → medium，> 1,000 → large。

最终取两个信号中**更保守**的一档。例：25,000 张 200 类 = 125 张/类 → 总量 large 但每类 medium → 最终 `medium`；3,000 张 2 类 = 1,500 张/类 → 每类 large 但总量 small → 最终 `small`。

#### priority

| 用户关键词（示例） | 映射值 |
|------------------|--------|
| 快、实时、低延迟、轻量、高效、跑得动 | `speed` |
| 精度高、效果好、不在乎速度、最准 | `accuracy` |
| 其他 / 未提及 | `balanced` |

#### constraints（布尔，可多选）

| 用户关键词（示例） | 字段 |
|------------------|------|
| 实时、30fps、视频流、在线推理 | `real_time` |
| 手机、移动端、嵌入式、树莓派、Jetson、低功耗 | `edge_deployment` |
| 类别不平衡、样本不均、长尾、某类数据少 | `class_imbalance` |
| 医学、医疗、CT、X光、MRI、病理、超声、内窥镜 | `medical` |
| 图文、多模态、文字搜图、语言对齐、CLIP | `cross_modal` |

---

### 当前无法映射的关键词（需要讨论）

以下关键词在用户中真实存在，但模块3的现有字段覆盖不到，需要确认处理方式：

| 关键词类型 | 示例 | 当前处理 | 建议 |
|-----------|------|---------|------|
| 特殊视角/域 | 无人机、卫星图像、工业缺陷 | 落入 `description` 兜底 | 讨论是否需要新增 constraint 字段 |
| 图像条件 | 夜间、低光照、小目标、密集 | 落入 `description` 兜底 | 暂不处理，向量检索覆盖 |
| 数据策略偏好 | 零样本、少样本、迁移学习 | 无字段 | 可映射为 `data_size=small` + 强制推荐 DINOv2/CLIP |
| 模型偏好 | "我想用YOLO"、"用Transformer" | 无字段 | 模块3不支持指定模型，建议模块2过滤掉 |
| 训练偏好 | 从头训练、不用预训练 | 无字段 | 讨论是否需要新增 `prefer_scratch` 字段 |

---

### 需要与模块2对齐的问题

1. **谁来做关键词→结构化的映射？** 模块2输出侧还是模块3输入侧？
2. **数据量会不会给具体数字？** 如果会，由谁换算成 small/medium/large？
3. **task_type 是模块2确定的枚举值，还是也是关键词？** 模块3需要精确的枚举值，无法做模糊匹配。
4. **"零样本/少样本"怎么处理？** 是直接映射 `data_size=small`，还是加新字段？
5. **用户指定模型（"我要用YOLO"）怎么处理？** 建议在模块2或上游过滤掉，不传给模块3。

---

## Quick Start

```python
from module3_kb_demo import (
    build_graph,
    build_vector_index,
    retrieve_top3_hybrid,
    build_all_task_lists,
)

G   = build_graph()       # 构建组件关系图（backbone/head/loss/optimizer）
col = build_vector_index() # 构建向量索引（用于语义检索）

# Module 1 output (passed through to Module 3)
input_json = {
    "task_type":   "object_detection",          # classification | object_detection | image_segmentation | feature_extraction
    "data_size":   "medium",                    # small | medium | large
    "priority":    "speed",                     # speed | accuracy | balanced
    "constraints": {
        "real_time":       True,   # 是否需要实时推理
        "edge_deployment": False,  # 是否部署在边缘/移动端
        "class_imbalance": False,  # 数据集是否存在类别不平衡
        "cross_modal":     False,  # 是否需要跨模态（图文对齐）特征
        "medical":         False,  # 是否为医学影像场景
    },
    "description": "Detect vehicles from traffic cameras at 30fps",
}

results    = retrieve_top3_hybrid(input_json, G, col)
task_lists = build_all_task_lists(results, G, fmt="structured")  # or fmt="nl"
```

<!-- task_lists 是列表，最多3项，按推荐分数从高到低排列，取 [0] 即为最优推荐 -->
`task_lists` is a list of up to 3 items, sorted by score descending. Use `task_lists[0]` for the top recommendation.

---

## Output Formats

### `fmt="structured"` — for deterministic code generation

<!-- 适合模板填充式的代码生成：每个 task 有固定的 action 类型，按顺序处理即可 -->

Each task has a fixed `action` type. Consume them in order.

```json
{
  "format": "structured",
  "rank": 1,
  "score": 1.0,
  "backbone": "yolov8",
  "backbone_name": "YOLOv8",
  "alternatives": [],
  "tasks": [
    {
      "id": "load_model",
      "action": "load_pretrained",
      "hf_id": "ultralytics/assets",
      "model_name": "YOLOv8-Nano / COCO",
      "params_M": 3.2,
      "finetune_base": "yolov8"
    },
    {
      "id": "train_strategy",
      "action": "set_finetune_strategy",
      "strategy": "full",
      "freeze_backbone": true,
      "scratch_viable": true
    },
    {
      "id": "head",
      "action": "configure_head",
      "type": "detection_head_anchor_free",
      "name": "Anchor-Free Detection Head"
    },
    {
      "id": "loss",
      "action": "configure_loss",
      "type": "focal_loss",
      "name": "FocalLoss"
    },
    {
      "id": "optimizer",
      "action": "configure_optimizer",
      "type": "sgd_momentum",
      "name": "SGD with Momentum"
    }
  ]
}
```

**Action types:**

| `action`                | When it appears          | Key fields                                          |
|-------------------------|--------------------------|-----------------------------------------------------|
| `load_pretrained`       | checkpoint available     | `hf_id`, `model_name`, `params_M`, `finetune_base` |
| `train_from_scratch`    | no checkpoint available  | `backbone`                                          |
| `set_finetune_strategy` | always                   | `strategy`, `freeze_backbone`, `scratch_viable`     |
| `configure_head`        | head resolved            | `type`, `name`                                      |
| `configure_loss`        | loss resolved            | `type`, `name`                                      |
| `configure_optimizer`   | optimizer resolved       | `type`, `name`                                      |

<!-- head / loss / optimizer 三个 task 不保证一定出现，取决于图中是否有兼容的组件 -->
`head` / `loss` / `optimizer` tasks may be absent if the graph has no compatible component for the task type.

---

### `fmt="nl"` — for LLM agent prompting

<!-- 适合 LLM agent：tasks 是自然语言列表，直接拼进 prompt；model_config 是结构化元数据，用于 prompt 内引用 -->

```json
{
  "format": "nl",
  "rank": 1,
  "score": 1.0,
  "model_config": {
    "backbone": "yolov8",
    "pretrained_hf_id": "ultralytics/assets",
    "pretrained_name": "YOLOv8-Nano / COCO",
    "pretrain_dataset": "COCO",
    "params_M": 3.2,
    "head": "detection_head_anchor_free",
    "loss": "focal_loss",
    "optimizer": "sgd_momentum",
    "finetune_strategy": "full",
    "freeze_backbone": true,
    "scratch_viable": true
  },
  "tasks": [
    "Load YOLOv8-Nano / COCO from ultralytics/assets (3.2M params, pretrained on COCO)",
    "Full finetune: update all backbone and head weights",
    "Use Anchor-Free Detection Head as the output head",
    "Use FocalLoss as the training loss",
    "Use SGD with Momentum as the optimizer"
  ],
  "alternatives": []
}
```

Feed `tasks` as a bullet list into your agent prompt. Use `model_config` for structured references within the prompt.

---

## Field Reference

### Top-level fields (both formats)

| Field          | Type           | Description                                              |
|----------------|----------------|----------------------------------------------------------|
| `format`       | `str`          | `"structured"` or `"nl"`                                |
| `rank`         | `int`          | 1 = best recommendation                                  |
| `score`        | `float`        | Combined retrieval score, 0–1                            |
| `backbone`     | `str`          | Backbone ID (e.g. `"yolov8"`, `"segformer"`)             |
| `backbone_name`| `str`          | Human-readable backbone name                             |
| `alternatives` | `list[str]`    | Other backbone IDs that are interchangeable              |

### `set_finetune_strategy` / `model_config` training fields

<!-- 这三个字段决定了训练方式，是模块4生成训练代码时最重要的参考 -->

| Field               | Values                              | Meaning                                                        |
|---------------------|-------------------------------------|----------------------------------------------------------------|
| `strategy`          | `"full"` / `"head_only"` / `"either"` | How to finetune the pretrained model                        |
| `freeze_backbone`   | `bool`                              | Whether to freeze backbone weights during training             |
| `scratch_viable`    | `bool`                              | Whether training from scratch is viable given the data size    |

`strategy` guide:
- `full` — update all weights (backbone + head). Required for task-specific models (YOLO, DETR, SegFormer).
- `head_only` — freeze backbone, only train the head. Typical for DINOv2, CLIP.
- `either` — both approaches work; choose based on data size and compute budget.

### Component type IDs

<!-- 这些 ID 是模块3知识库内部的标识符，模块4可以用它们做 switch/dispatch，也可以只看 name 字段生成代码 -->

| ID                          | Category  | Notes                                          |
|-----------------------------|-----------|------------------------------------------------|
| `classification_head`       | head      |                                                |
| `detection_head_anchor_free`| head      | YOLO-style                                     |
| `detection_head_transformer`| head      | DETR/RT-DETR only (fixed, non-swappable)       |
| `semantic_seg_head`         | head      |                                                |
| `panoptic_seg_head`         | head      | Mask2Former                                    |
| `feature_pooling_head`      | head      | GAP or CLS token, no trainable params          |
| `projection_head`           | head      | Contrastive learning                           |
| `cross_entropy_loss`        | loss      |                                                |
| `focal_loss`                | loss      | Class imbalance                                |
| `hungarian_matching_loss`   | loss      | DETR/RT-DETR only (fixed)                      |
| `dice_loss`                 | loss      | Segmentation                                   |
| `bce_dice_loss`             | loss      | Binary / medical segmentation                  |
| `infonce_loss`              | loss      | Contrastive / feature extraction               |
| `adamw`                     | optimizer | Standard for transformer finetuning            |
| `adam`                      | optimizer | CNN training and from-scratch                  |
| `sgd_momentum`              | optimizer | Large-scale CNN training                       |

---

## Using Both Formats Together

```python
results = retrieve_top3_hybrid(input_json, G, col)

# Structured: parse and dispatch to code generation templates
# structured 格式：按 action 类型分发到对应的代码生成模板
for tl in build_all_task_lists(results, G, fmt="structured"):
    for task in tl["tasks"]:
        dispatch(task["action"], task)

# NL: inject into LLM agent
# nl 格式：把 tasks 列表拼成 prompt，交给 LLM agent 生成代码
top_nl = build_all_task_lists(results, G, fmt="nl")[0]
prompt = "Implement a PyTorch training pipeline for the following tasks:\n"
prompt += "\n".join(f"- {t}" for t in top_nl["tasks"])
```

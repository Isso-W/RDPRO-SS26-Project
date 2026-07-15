# Module 3: API Reference for Module 4

<!-- Output of module 3: Receive the task description of module 1, return up to 3 model configurations, each configuration is packaged into a task list for use by module 4 -->

Module 3 takes a task description from Module 1 and returns a ranked list of model configurations, each packaged as an actionable task list.

---

## Module 2 → Module 3 interface alignment (to be confirmed)

> This section is used to align the input format with Module 2 leaders and has not yet been finalized.

### Module 3 currently expects input

```python
{
    "task_type": str, # Enumeration value, see table below
    "data_size": str, # "small" | "medium" | "large"
    "priority": str, # "speed" | "accuracy" | "balanced"
    "constraints": {
        "real_time": bool,
        "edge_deployment": bool,
        "class_imbalance": bool,
        "cross_modal": bool,
        "medical": bool,
    },
    "description": str, # Free text for semantic retrieval
}
```

Module 3 currently consumes **structured Boolean values**. If module 2 outputs keywords, a layer of mapping needs to be added between the two. This mapping can be placed on the output side of module 2 or on the input side of module 3 - you need to confirm who will do it.

---

### Keyword → structured field mapping reference

#### task_type (enumeration, must match exactly)

| User keywords (example) | Mapping value |
|------------------|--------|
| Detect, locate, frame, find, count | `object_detection` |
| Classify, identify, determine what it is, and label it | `classification` |
| Segmentation, masking, pixel-level annotation, and cutout | `image_segmentation` |
| Features, vectors, retrieval, similarity, embedding | `feature_extraction` |

#### data_size (recommended to support quantity range)

| User keywords (example) | Mapping value |
|------------------|--------|
| Little data, hundreds of images, expensive annotation, insufficient samples | `small` |
| Thousands to tens of thousands, self-collected data | `medium` |
| A large amount of data, millions of levels, public big data sets | `large` |

Digital conversion is implemented by `pipeline.derive_data_size()` (updated to dual signals on 2026-06-10):

**Total volume level** (cost side - total volume determines labeling/training cost):

| Task type | small | medium | large |
|---------|-------|--------|-------|
| classification / feature_extraction | ≤ 3,000 | 3,001-20,000 | > 20,000 |
| object_detection / image_segmentation | ≤ 1,500 | 1,501-10,000 | > 10,000 |

**Number of samples per category** (overfitting side, only classification tasks): ≤ 100 images/category → small, 101-1,000 → medium, > 1,000 → large.

Finally, the **more conservative** of the two signals is selected. Example: 25,000 sheets 200 categories = 125 sheets/category → total amount large but each category medium → final `medium`; 3,000 sheets 2 categories = 1,500 sheets/category → each category large but total amount small → final `small`.

#### priority

| User keywords (example) | Mapping value |
|------------------|--------|
| Fast, real-time, low-latency, lightweight, efficient, and fast | `speed` |
| High precision, good effect, no matter the speed, most accurate | `accuracy` |
| Others/not mentioned | `balanced` |

#### constraints (Boolean, multiple choices available)

| User keywords (example) | Field |
|------------------|------|
| Real-time, 30fps, video streaming, online inference | `real_time` |
| Mobile phone, mobile terminal, embedded, Raspberry Pi, Jetson, low power consumption | `edge_deployment` |
| Category imbalance, uneven samples, long tail, and less data of a certain category | `class_imbalance` |
| Medicine, medical, CT, X light, MRI, pathology, ultrasound, endoscopy | `medical` |
| Image and text, multi-modal, text image search, language alignment, CLIP | `cross_modal` |

---

### Keywords that are currently unmapped (needs discussion)

The following keywords actually exist among users, but are not covered by the existing fields in module 3. The processing method needs to be confirmed:

| Keyword type | Example | Current processing | suggestion |
|-----------|------|---------|------|
| Special perspective/domain | Drones, satellite imagery, industrial defects | Falling into the trap of `description` | Discuss whether a new constraint field is needed |
| Image conditions | Night, low light, small target, dense | Falling into the trap of `description` | Not processed yet, vector retrieval coverage |
| Data strategy preferences | Zero sample, few samples, transfer learning | No fields | Can be mapped to `data_size=small` + mandatory recommendation DINOv2/CLIP |
| Model preference | "I want to use YOLO", "Use Transformer" | No fields | Module 3 does not support the specified model. It is recommended that module 2 filter it out. |
| Training preferences | Training from scratch, no pre-training required | No fields | Discuss whether a new `prefer_scratch` field is needed |

---

### Questions that need to be aligned with module 2

1. **Who will do the keyword → structured mapping? ** The output side of module 2 or the input side of module 3?
2. **Will there be a specific figure for the amount of data? ** If so, who converts it into small/medium/large?
3. **task_type is the enumeration value determined by module 2, or is it also a keyword? ** Module 3 requires precise enumeration values ​​and cannot perform fuzzy matching.
4. **How ​​to deal with "zero samples/few samples"? ** Should `data_size=small` be directly mapped, or new fields should be added?
5. **How ​​to deal with the user-specified model ("I want to use YOLO")? ** It is recommended to filter it out in module 2 or upstream and not pass it to module 3.

---

## Quick Start

```python
from module3_kb_demo import (
    build_graph,
    build_vector_index,
    retrieve_top3_hybrid,
    build_all_task_lists,
)

G   = build_graph()       # Build component relationship diagram (backbone/head/loss/optimizer)
col = build_vector_index() # Build vector index (for semantic retrieval)

# Module 1 output (passed through to Module 3)
input_json = {
    "task_type": "object_detection", # classification | object_detection | image_segmentation | feature_extraction
    "data_size": "medium", # small | medium | large
    "priority": "speed", # speed | accuracy | balanced
    "constraints": {
        "real_time": True, # Do you need real-time reasoning?
        "edge_deployment": False, # Whether to deploy on edge/mobile terminal
        "class_imbalance": False, # Is there a class imbalance in the dataset?
        "cross_modal": False, # Whether cross-modal (image and text alignment) features are needed
        "medical": False, # Whether it is a medical imaging scene
    },
    "description": "Detect vehicles from traffic cameras at 30fps",
}

results    = retrieve_top3_hybrid(input_json, G, col)
task_lists = build_all_task_lists(results, G, fmt="structured")  # or fmt="nl"
```

<!-- task_lists is a list, with up to 3 items, arranged from high to low by recommendation score, [0] is the best recommendation --> `task_lists` is a list of up to 3 items, sorted by score descending. Use `task_lists[0]` for the top recommendation.

---

## Output Formats

### `fmt="structured"`: for deterministic code generation

<!-- Suitable for template filling code generation: each task has a fixed action type, which can be processed in order -->

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

<!-- head / loss / optimizer Three task are not guaranteed to appear, depending on whether there are compatible components in the picture --> `head` / `loss` / `optimizer` tasks may be absent if the graph has no compatible component for the task type.

---

### `fmt="nl"`: for LLM agent prompting

<!-- Suitable for LLM agent: tasks is a natural language list, directly spelled into prompt; model_config is structured metadata, used for internal references in prompt -->

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
| `score`        | `float`        | Combined retrieval score, 0-1                            |
| `backbone`     | `str`          | Backbone ID (e.g. `"yolov8"`, `"segformer"`)             |
| `backbone_name`| `str`          | Human-readable backbone name                             |
| `alternatives` | `list[str]`    | Other backbone IDs that are interchangeable              |

### `set_finetune_strategy` / `model_config` training fields

<!-- These three fields determine the training method and are the most important reference when generating training code in module 4 -->

| Field               | Values                              | Meaning                                                        |
|---------------------|-------------------------------------|----------------------------------------------------------------|
| `strategy`          | `"full"` / `"head_only"` / `"either"` | How to finetune the pretrained model                        |
| `freeze_backbone`   | `bool`                              | Whether to freeze backbone weights during training             |
| `scratch_viable`    | `bool`                              | Whether training from scratch is viable given the data size    |

`strategy` guide:
- `full`: update all weights (backbone + head). Required for task-specific models (YOLO, DETR, SegFormer).
- `head_only`: freeze backbone, only train the head. Typical for DINOv2, CLIP.
- `either`: both approaches work; choose based on data size and compute budget.

### Component type IDs

<!-- These ID are identifiers inside the knowledge base of module 3. Module 4 can use them to make switch/dispatch, or you can only look at the name field to generate code -->

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
# structured format: distributed to the corresponding code generation template according to the action type
for tl in build_all_task_lists(results, G, fmt="structured"):
    for task in tl["tasks"]:
        dispatch(task["action"], task)

# NL: inject into LLM agent
# nl format: spell the tasks list into prompt and give it to LLM agent to generate the code
top_nl = build_all_task_lists(results, G, fmt="nl")[0]
prompt = "Implement a PyTorch training pipeline for the following tasks:\n"
prompt += "\n".join(f"- {t}" for t in top_nl["tasks"])
```

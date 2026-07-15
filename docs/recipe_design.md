# recipe 层 — 超参推荐层设计记录

> 本文记录 recipe 层的实现设计；当前代码与测试位于 `recipe/`。
> 定位：**graph 决定"选什么组件"，recipe 层决定"选出来的怎么配置"。**
> 输出模型耦合的推荐级超参（image_size / learning_rate / epochs / augmentation），
> 并输出可选的 early-stopping metadata；Module 4 负责实际训练期早停行为。
> 执行级超参（batch_size / num_workers 等）仍由 Module 4 管理。

---

## 0. 现状（为什么需要这一层）

超参目前散在三处、无统一来源：

- `pipeline.py::derive_recommended_epochs` —— 唯一的"信号→HP"函数，孤儿，
  在 pipeline 层事后 `setdefault` 注入（run_kaggle_benchmark.py / pipeline.py）；
- `image_size` / `learning_rate` —— Module 4 模板里的固定默认值，不随数据/骨干变；
- `augmentation` —— 不存在。

recipe 层把这些收拢成**一个确定性模块**，Module 3 调用，输出带 provenance
（improvements §2 点名的"配置来源可追溯"问题）。**纯规则、无 LLM、可单测。**

---

## 1. 定位与调用点

**独立模块**（improvements §5 倾向 standalone——隔离、可测、可独立迭代）：

```
recipe/
  __init__.py
  layer.py       # 编排：build_recipe(config, input_json, backbone_facts, data_stats)
  tables.py      # 全部冻结查表（epochs / lr base / augment 档 / invariance veto / image 约束）
  augment.py     # 三维增广解析器（最复杂的一块，单独文件）
  tests/
```

**调用点**：`rag_retrieval.build_task_list`（rag_retrieval.py:1785，`model_config`
组装完 backbone/checkpoint/loss/finetune_strategy 之后）末尾调用一次，把
recipe 输出并进 `model_config`。理由：让 recipe 随推荐本身产出，而不是每个
调用方（pipeline / run_kaggle_benchmark）各自事后注入——同时**收编 epochs
孤儿**，消除重复注入。

**签名**：

```python
def build_recipe(
    config: dict,          # 已组装的 model_config（backbone/finetune_strategy/use_pretrained…）
    input_json: dict,      # Module 3 输入（data_size/priority/constraints/task_type/num_classes）
    backbone_facts: dict,  # 选中 backbone/checkpoint 的 graph 节点属性（image 约束、期望分辨率…）
    data_stats: dict | None = None,  # Module 2 统计（分辨率档、色彩模式）；缺省则降级
) -> tuple[dict, dict]:    # (recipe, provenance)
```

`build_task_list` 增一个可选参 `data_stats`，转手传入；缺省时 recipe 用保守
默认（graceful degradation，见各 §）。

---

## 2. 前置第 0 步 — 把 Module 2 统计接进来（image_size / 灰度 veto 的信号地基）

**现状**：`merge_modules`（pipeline.py:155）只保留 data_size / num_classes /
class_imbalance，**丢弃** Module 2 已算的分辨率（min/max/avg 宽高）、
`mode_distribution`（色彩）、`format_distribution`。recipe 的分辨率感知
image_size 和灰度 veto 因此**无信号**。

**改动**（沿 improvements §3 的 derive_* 模式）：

```python
def derive_resolution_tier(stats) -> str:   # "low"(<256) / "medium" / "high"(>=768) 按 avg 短边
def derive_color_mode(stats) -> str:        # "rgb" / "grayscale"（mode_distribution 主导判定）
```

`merge_modules` 把这两个字段透进 Module 3 输入（不进 constraints，单独放
`data_stats` 子字典，避免污染检索用的 constraints）。**不做此步，recipe 仍
可跑，但 image_size 退化为 backbone 默认、增广 veto 退化为全放行**（在
provenance 里标 `signal_missing`）。

---

## 3. 四个子决策（v0 范围：分类任务）

### 3.1 epochs（收编孤儿，逻辑不变）

`derive_recommended_epochs` + `_RECOMMENDED_EPOCHS` 表**整体迁入
`recipe/tables.py`**，pipeline.py 改为从 recipe 导入（保留一个 re-export
别名，避免破坏 run_kaggle_benchmark 的现有 import）。键 (data_size, mode)，
mode ∈ {head_only, finetune, scratch} 由 finetune_strategy + use_pretrained 推。
Provenance：`"epochs_table[{data_size},{mode}]"`。

### 3.2 image_size（improvements §4 的"收敛点"）

三个输入汇合，**按顺序**求解：

1. **基准**：优先取选中 **checkpoint 的期望输入分辨率**（backbone_facts；
   如 efficientnet_b0=224），无则取 family 默认（tables.py 的
   `_FAMILY_IMAGE_DEFAULT`）；
2. **分辨率 + 细粒度上调**：`data_stats.resolution_tier == "high"` **且**
   `constraints.fine_grained`（细粒度需要细节）→ 上调一档（224→384）；
   `priority == "speed"` 或 `data_size == "large"` → 不上调（高分辨率更慢）；
3. **硬约束吸附**：某些 backbone 要求整除（DINOv2 /14、Swin /32、ViT /16——
   实现时逐个核实除数，存 `_IMAGE_DIVISOR`）。把上两步结果**吸附到最近的
   合法值**。这是安全约束，最后执行、不可被跳过。

缺 data_stats → 停在第 1 步（backbone 默认）+ 第 3 步吸附。
Provenance：`"ckpt_default=224 | fine_grained+high_res bump→384 | snapped /14→392"`。

### 3.3 learning_rate（与 finetune_strategy 耦合）

二维查表 `_LR_BASE[(family_class, mode)]`，family_class ∈ {cnn, transformer}
（transformer 要更低 LR），mode 同 epochs：

| | head_only | finetune | scratch |
|---|---|---|---|
| cnn | 1e-3 | 1e-4 | 5e-4 |
| transformer | 1e-3 | 3e-5 | 3e-4 |

（数值为 v0 默认，待 recipes.json / A/B 校准。）warmup / schedule 类型**不在
此层**——它们需运行时反馈，归 Module 4（§7）。Provenance：`"lr_base[cnn,finetune]"`。

### 3.4 augmentation（三维：强度 ⊗ 不变性 ⊗ 日程，见 augment.py §4）

---

## 4. augment.py — 三维增广解析

输出结构：

```python
{"tier": "medium",
 "invariance": {"hflip": True, "vflip": False, "rot90": False, "color": True,
                "crop_scale_min": 0.8},
 "schedule": "taper_last_20pct"}
```

### 维度一：强度档（tier）——"加多少"

规则（按序，后者覆盖前者的档位调整）：

```
data_size=small  → heavy   ；medium → medium ；large → light
finetune_strategy=head_only → 压一档（冻结骨干适应不了强畸变）
constraints.few_shot=True   → heavy 且强制含 RandAugment
```

档内容（torchvision v2，不加新依赖）：none=resize+norm；light=RRC(0.8-1.0)+
HFlip；medium=+旋转/平移+ColorJitter+RandomErasing；heavy=+RandAugment+MixUp/CutMix。

### 维度二：不变性掩码（invariance）——"能加哪些"（安全裁决，硬规则，优先级最高）

veto 覆盖强度档给出的默认开关：

| 信号 | 来源 | veto |
|---|---|---|
| color_mode=grayscale | Module 2（data_stats） | `color=False`（灰度调色无意义） |
| domain ∈ {satellite, aerial, pathology, microscopy} | Module 1 语义（v0 缺，见下） | `vflip=True, rot90=True`（无固定方位） |
| domain ∈ {document, digit, ocr} | Module 1 语义 | `hflip=False, vflip=False, rot90=False`（翻转改标签） |
| domain=medical 且方位敏感（如胸片） | Module 1 语义 | `hflip=False` |
| constraints.fine_grained | 已有 | `crop_scale_min` 抬高到 0.5（激进裁剪会裁掉判别特征） |

**v0 诚实边界**：Module 1 目前**不抽 domain 语义**，所以方位/翻转类 veto 在
v0 **只有 grayscale 一条真实生效**（M2 色彩信号现成），其余 domain veto 是
**保守默认放行 + 代码骨架就位**，等 Module 1 加 `domain` 抽取字段后接上
（provenance 标 `domain_signal_missing`）。fine_grained 的 crop veto 用已有信号，v0 即生效。

### 维度三：日程（schedule）——"什么时候加"

v0 只出一条静态标签：`data_size in {small,medium}` → `"taper_last_20pct"`
（末段降一档，防末期畸变害收敛）；`large` → `"constant"`。**动态调度**（按
train/val gap 升降档）需运行时反馈，归 Module 4 v1（§7）。Module 4 模板按
标签实现具体 epoch 门限。

---

## 5. 输出与 provenance

并进 `model_config`：

```python
config["recipe"] = {"image_size":384, "learning_rate":1e-4, "epochs":20,
                    "augmentation": {...}}
config["recipe_provenance"] = {"image_size": "...", "learning_rate": "...",
                    "epochs": "...", "augmentation": "..."}
```

provenance 每字段一句话，记规则路径与是否有信号缺失——这是 improvements §2
点名的可追溯性，也是终期报告"每个超参可解释"的直接素材。

**校准 seam（v0 不自动化）**：tables.py 的默认值处留注释，标明哪些可被
kb_mining 的 recipes.json / A/B 结果覆盖。v0 用手工默认，覆盖是后续工作。

---

## 6. Module 4 消费

Module 4 模板读 `config["recipe"]`：image_size / learning_rate / epochs 直接用；
augmentation 三维在 dataloader 构造处翻译成 torchvision v2 transform 流水线
（tier→基础变换集，invariance→按开关增删，schedule→按标签设 epoch 门限）。
**向后兼容**：无 `recipe` 键时用模板现有默认（旧行为）。这块是 Module 4 改动，
与 recipe 层解耦——recipe 只产结构化配置，不产代码。

---

## 7. 范围纪律（improvements §2 的 HP 归属，明确不做）

- **归 Module 4、recipe 不碰**：batch_size（OOM 重试/梯度累积）、num_workers、
  mixed_precision、gradient_clip、早停、LR warmup 自动、OOM 重试、动态增广调度；
- **v0 不做**：检测/分割的 recipe（bbox/mask 需几何同步，Module 4 训练模板
  目前分类向）；LR scheduler 类型；per-checkpoint 精细 image_size 搜索；
  Module 1 的 domain 抽取（是 invariance 维度二的前置，单独立项）。

---

## 8. 测试计划（recipe/tests/，全离线纯函数）

- **子决策单测**：epochs 迁移后数值不变（回归）；image_size 吸附（DINOv2
  输入永远 /14）；lr 表命中；augment 三维各规则。
- **不变量测试**（对所有构造输入必成立）：①image_size 永远满足选中 backbone
  的整除约束；②head_only 永远拿不到 heavy 档；③grayscale 永远 `color=False`；
  ④fine_grained 永远 `crop_scale_min>=0.5`；⑤缺 data_stats 不崩、退化路径正确。
- **golden 式端到端**：几个代表输入（小数据细粒度 / 大数据速度优先 / 灰度医学 /
  few_shot）→ 断言完整 recipe。
- **集成回归**：`cd retrieval && python -m pytest test_golden.py test_rag_retrieval.py`
  全绿（build_task_list 加 recipe 后，现有断言不受影响——recipe 是新增字段）。

---

## 9. 实现顺序

0. §2 Module 2 统计接入 merge_modules + derive_* + 测试（半天；不做则 recipe
   降级运行，可作为并行项，但 image_size/灰度 veto 空转）
1. tables.py（迁 epochs + image/lr/augment 默认表）+ layer.py 编排 + epochs
   回归测试（半天）
2. image_size + learning_rate 子决策 + 不变量测试（半天）
3. augment.py 三维解析 + 测试（半天，最复杂）
4. 接入 build_task_list + 集成回归；Module 4 模板消费 recipe（§6）+ 向后兼容（半天）

预计净工作量 2–3 天。§2（Module 2 接入）和 Module 1 的 domain 抽取是两个
可独立推进的信号源前置——不阻塞 recipe 骨架，接上后 invariance 维度才完整。

---

## 10. 一句话总览

recipe 层 = 一个确定性模块，把 (已选骨干 + 数据信号) 映射成
image_size/lr/epochs/augmentation，每个值带来源。v0 覆盖分类；epochs 收编
现有孤儿，image_size 首次真正消费 Module 2 数据，augmentation 首次落地
（三维：查表强度 + 硬规则不变性 + 静态日程）。graph 供事实、recipe 做配置、
Module 4 变代码——三层各司其职。

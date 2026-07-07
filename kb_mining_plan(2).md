# kb_mining — Kaggle 优胜方案挖掘 → KB 增强：实现规格

> 本文档是可直接编码的实现计划。目标读者是执行编码的 agent/工程师。
> 设计讨论背景见对话记录；本文只保留结论和规格。
>
> **一次性项目**：不做 cron、不做增量、不做 notebook AST 解析、不建新 backbone
> 节点、不做留出集验证与短训 A/B（验证部分已明确暂缓）、**不改打分常数**
> （`w_vector` / bonus 值的调整在看到 consensus 表之后另开 PR）。

---

## 0. 目标与产出总览

从 2021 年以后已结束的 Kaggle CV 分类竞赛的 top 方案 write-up 中，挖掘
"数据集特征 → 组件选择"的统计共识，输出一份**人可五分钟读完的共识表**和
一份**按五档决策表分类的 KB 改动建议清单**。KB 数据本身（`EDGES` /
`EDGE_CONDITIONS` / 节点字段）的实际修改由人看完清单后另行提交，**不在本
流水线的自动化范围内**。

唯一随本项目落地的 KB 代码改动是 Phase B：让 `_select_components` 消费
loss 节点间的 `preferred_when` 边（现为死数据），用 KB 数据替换现有硬编码
loss 规则链的第一优先级。

```
kb_mining/
  __init__.py
  catalog.py        # 竞赛清单+特征卡、架构发布时间表、组件别名表（纯数据）
  harvest.py        # Meta Kaggle dump → data/posts.jsonl
  extract.py        # LLM 抽取 → data/facts.jsonl
  aggregate.py      # facts × 特征卡 → data/consensus.{json,md} + 三张侧表
  decide.py         # consensus × 现有 KB → data/proposals.md（五档决策）
  tests/
    fixtures/       # 迷你 CSV、canned LLM 响应、样例 facts
    test_harvest.py
    test_extract.py
    test_aggregate.py
    test_decide.py
kb_mining/data/     # 全部产物落这里（gitignore 原始 dump，产物 jsonl 入库）
```

管道各阶段幂等、可独立重跑；阶段间只通过 `kb_mining/data/` 下的文件通信。

---

## 0.5 前置第 0 步 — 数据源地基验证（半天，先于 catalog）

**整条链路的单点风险**：Competitions → 论坛 → 帖子的 JOIN 是否真实可行。
Meta Kaggle 的论坛关联字段历史上变动过，`Competitions.csv` 不保证有稳定的
`ForumId` 列。**在写任何其它代码之前**先证实/证伪：

1. 下载 `Competitions.csv`（小文件，可全量）+ `ForumTopics.csv` 表头与抽样行
   + `ForumMessages.csv` 仅表头与前几行（`head`，不全量下载）；
2. 确认 JOIN 键真实存在：Competitions 里 Slug 与论坛关联列（`ForumId` 或
   等价字段；若 ForumTopics 有直连竞赛的列则链路更短）；
3. **端到端验一条已知记录**：用 cassava 竞赛已知存在的 "1st place solution"
   帖，沿链路真实捞出其正文，肉眼确认是 write-up 全文而非摘要/截断；
4. 产出 `kb_mining/data/source_check.md`：记录实际列名、样例 JOIN 结果、
   ForumMessages 正文的实际格式（markdown/HTML）。**harvest.py 按此文件
   编码，不按本计划的"预期链路"编码。**

**链断了的备选方案**（按优先级）：① ForumTopics 若有其它竞赛关联列，改
链路；② 用社区维护的 solution 帖索引（如 farid.one/kaggle-solutions，按
竞赛聚合了方案帖 URL）拿到帖子清单，再逐帖取正文；③ 都不行则本项目数据
源方案需重议——**此时停下来向决策者汇报，不要硬写爬虫**。

---

## 1. catalog.py — 纯数据，无逻辑

### 1.1 竞赛清单 + 特征卡 `COMPETITIONS`

```python
COMPETITIONS: dict[str, dict] = {
    "<slug>": {
        "slug": str,            # kaggle 竞赛 slug
        "title": str,
        "start": "YYYY-MM",     # 开赛时间（共存性判断用）
        "end": "YYYY-MM",
        "task_type": "classification",
        "traits": {             # 特征卡：键 = 合法 condition 键去掉 "=True"
            "fine_grained": bool,
            "class_imbalance": bool,
            "medical": bool,
            "data_size": "small" | "medium" | "large",
        },
        "traits_verified": bool,  # 人工核对过置 True；aggregate 对未核对的打 warning
        "notes": str,             # 特殊性备注（多标签、metric-learning 味等）
    },
}
```

初始候选池（实现时逐个核实：①确实 2021-01 之后结束 ②讨论区有 ≥5 篇
带名次的 solution write-up ③特征卡取值；不满足的从清单删除，每个关键特征
`fine_grained` / `class_imbalance` / `medical` 至少保留 3 个竞赛）：

| slug | 年份 | 初判特征 |
|---|---|---|
| cassava-leaf-disease-classification | 2021 | fine_grained, 轻度 imbalance, medium |
| plant-pathology-2021-fgvc8 | 2021 | fine_grained（注意：多标签，notes 标记） |
| herbarium-2022-fgvc9 | 2022 | fine_grained, 长尾 imbalance, large |
| sorghum-id-fgvc-9 | 2022 | fine_grained |
| paddy-disease-classification | 2022 | fine_grained, small |
| happy-whale-and-dolphin | 2022 | fine_grained（metric-learning 味重，notes 标记） |
| mayo-clinic-strip-ai | 2022 | medical, small |
| rsna-breast-cancer-detection | 2023 | medical, 极端 imbalance |
| ubc-ocean | 2023 | medical, imbalance |
| isic-2024-challenge | 2024 | medical, 极端 imbalance, large |
| hms-harmful-brain-activity-classification | 2024 | medical, imbalance（输入为脑电频谱图渲染的图像，notes 标记非自然图像） |
| rsna-2024-lumbar-spine-degenerative-classification | 2024 | medical, imbalance（MRI 多部位分级，notes 标记多输出） |
| fathomnet-2025 | 2025 | fine_grained（海洋物种层级分类；社区规模较小，核实 write-up 数量是否 ≥5） |
| rsna-intracranial-aneurysm-detection | 2025 | medical, imbalance, large（CT/MR，1100+ 队，write-up 充足；notes 标记含定位子任务） |

**2026 年及未来竞赛的机械枚举**（弥补手工清单的时效盲区）：`catalog.py` 附带
一个辅助函数 `list_recent_cv_candidates(dump_dir) -> list[dict]`，扫描
Meta Kaggle `Competitions.csv`，过滤 ①截止时间在 2025-01 之后 ②标签/标题含
CV 分类信号（tag "Computer Vision"、标题含 classification/detection 等）
③参赛队数 ≥ 300（保证有足够 write-up），输出候选行（slug、title、起止时间、
队数）供人工挑选后补进 `COMPETITIONS`。此函数只做枚举、不自动入清单——
特征卡仍然人工核对。harvest 的 CLI 加 `--list-recent` 开关调用它。

特征卡填法：竞赛 Overview/Data 页描述 + write-up 交叉，可用 LLM 辅助初填，
但 `traits_verified=True` 必须人工过目（一共十几行，成本可忽略）。

### 1.2 架构发布时间表 `FAMILY_RELEASE`（共存性过滤用）

只覆盖 KB 的 14 个 backbone 家族（id 与 `retrieval/rag_retrieval.py`
`COMPONENTS` 完全一致）：

```python
FAMILY_RELEASE: dict[str, str] = {   # family_id -> "YYYY-MM"
    "resnet": "2015-12",  "efficientnet": "2019-05", "mobilenet_v3": "2019-05",
    "vit": "2020-10",     "swin_transformer": "2021-03", "convnext": "2022-01",
    "yolov8": "2023-01",  "detr": "2020-05",  "rt_detr": "2023-04",
    "segformer": "2021-05", "mask2former": "2021-12", "unet": "2015-05",
    "dinov2": "2023-04",  "clip_vit": "2021-01",
}
```

（实现时逐条核对论文/发布时间，以月为粒度即可。）

### 1.3 组件别名表 `MODEL_ALIASES` / `LOSS_ALIASES`

原始字符串 → KB id 的映射，按序匹配的 `(regex, family_id)` 列表，全部
不区分大小写；无一命中 → `"unknown"`：

```python
MODEL_ALIASES: list[tuple[str, str]] = [
    (r"(tf_)?efficientnet(v2)?", "efficientnet"),
    (r"convnext",                "convnext"),
    (r"swin",                    "swin_transformer"),
    (r"(deit|beit|^vit|_vit|vit_)", "vit"),
    (r"dinov2",                  "dinov2"),
    (r"clip",                    "clip_vit"),
    (r"(resnet|resnext|resnest|se_?resnext)", "resnet"),
    (r"mobilenet",               "mobilenet_v3"),
    # yolov8/detr/rt_detr/segformer/mask2former/unet 分类赛出现即映射，
    # 但 task_type 不符会在 decide 阶段被档 0 检查天然拦下
]
LOSS_ALIASES: list[tuple[str, str]] = [
    (r"focal",                        "focal_loss"),
    (r"(weighted|class.?weight).*(ce|cross.?entropy)", "focal_loss"),  # 加权CE并入focal证据，notes保留raw
    (r"cross.?entropy|\bce\b|label.?smooth", "cross_entropy_loss"),
    (r"bce.?dice",                    "bce_dice_loss"),
    (r"\bdice\b",                     "dice_loss"),
    (r"infonce|(?<!arc)contrastive",  "infonce_loss"),
    (r"arcface|cosface|triplet|metric.?learning", "unknown"),  # metric-learning 损失不归并，进侧表
    (r"hungarian|matching",           "hungarian_matching_loss"),
]
```

**loss 归并纪律**（loss 共识正是 Phase B 的消费对象，污染代价最高）：

1. **arcface/cosface/triplet 不归并**到 infonce——它们是 metric-learning
   的东西，归并会虚增 infonce 的 support。映射为 `unknown` 并进
   `unknown_components.json` 侧表（带 `metric_learning` 标签）；
2. **加权 CE→focal 保留归并**（KB 没有加权 CE 节点，而两者对应的 KB 行动
   是同一条边），但 consensus.md 里该行必须**拆分显示 raw 计数**；若加权
   CE 占合并票 > 50%，该行加 ⚠（证据主体不是字面上的 focal）；
3. **notes 标记为 metric-learning 味的竞赛（如 happy-whale）整场排除
   loss 投票**——其 loss 信号不适用于分类推荐；backbone 投票不受影响。

---

## 2. harvest.py — Meta Kaggle dump → posts.jsonl

**数据源**：Kaggle 官方数据集 `kaggle/meta-kaggle`（每日更新的全站 CSV
dump），只需三个文件：`Competitions.csv`、`ForumTopics.csv`、
`ForumMessages.csv`。用 Kaggle API 按文件单独下载
（`api.dataset_download_file("kaggle/meta-kaggle", <name>, path=...)`），
复用 `ingestion/kaggle_loader._authenticate()`。

**注意**：`ForumMessages.csv` 数 GB 级，**必须**用
`pandas.read_csv(chunksize=100_000)` 流式过滤，不可整表载入。
CSV 列名以实际 dump 表头为准（实现第一步先 `head -1` 核对），预期链路：

```
Competitions:  Slug → ForumId          （按 catalog.COMPETITIONS 的 slug 过滤）
ForumTopics:   ForumId → Topic Id/Title （标题过正则筛 solution 帖）
ForumMessages: ForumTopicId → Message   （取每帖楼主的首条消息即 write-up 正文）
```

**solution 帖判定正则**（对 topic 标题）：

```python
RANK_RE = re.compile(r"\b(\d{1,3})(st|nd|rd|th)\s+place\b|\bplace\s+(\d{1,3})\b", re.I)
SOLUTION_RE = re.compile(r"solution|write.?up|summary", re.I)
# 命中 RANK_RE 即收；仅命中 SOLUTION_RE 无名次的也收，rank 记 None
```

每竞赛收名次最靠前的至多 `MAX_POSTS_PER_COMP = 10` 篇（rank None 排最后）。

**召回不足的两级补救与明确政策**（solution 帖标题不保证含
place/solution——"My approach"、"Gold — timm ensemble" 这类会被正则漏掉）：

- **二级召回**：某竞赛正则命中 < 5 篇时，取该竞赛论坛按投票分（Score 列）
  最高的前 30 个 topic，用 LLM 判别"是否名次方案帖"（输入标题 + 正文前
  500 字符；走与 extract 相同的可注入 `llm_fn`，可测试）；
- **政策（写死，不留裁量）**：二级召回后 ≥5 篇 → 正常；3–4 篇 → 保留该
  竞赛，harvest 输出 warning；**< 3 篇 → 从本次挖掘中剔除该竞赛**，并在
  harvest 汇总里列明（特征卡保留，供未来补）。

**输出 `data/posts.jsonl`**，每行：

```json
{"competition": "<slug>", "topic_id": 123, "topic_title": "1st Place Solution",
 "rank": 1, "author_message_id": 456, "text": "<raw markdown/html 正文>",
 "post_date": "2021-02-20"}
```

CLI：`python -m kb_mining.harvest [--dump-dir kb_mining/data/meta_kaggle] [--force-download]`
（dump 目录已存在则跳过下载，同 `kaggle_loader` 的缓存习惯）。

---

## 3. extract.py — LLM 抽取 → facts.jsonl

**LLM 客户端**：复用 `features_extraction_api._provider()` +
`_client_for_provider(provider)`（OpenAI 兼容接口，默认 qwen）。
`temperature=0`。**可测试性要求**：核心函数签名为

```python
def extract_post(post: dict, llm_fn: Callable[[str, str], str]) -> dict | None
# llm_fn(system_prompt, user_content) -> raw completion text；
# 生产走真实 client，测试注入 canned 响应。
```

**输入截断**：正文超过 12_000 字符时保留前 9_000 + 后 3_000（write-up 的
模型配置多在开头，分数表常在结尾）。

**LLM 输出 schema**（prompt 要求纯 JSON，无 markdown 围栏；`raw` 字段抄
原文，映射由代码用别名表做，**不让 LLM 直接输出 KB id**，降低幻觉面）：

```json
{
  "kind": "single" | "ensemble" | "unclear",
  "members": [{"raw_model": "tf_efficientnet_b4_ns", "image_size": 512}],
  "loss_raw": "focal loss" | null,
  "best_single_model_raw": "..." | null,
  "best_single_score": 0.899 | null,
  "used_pseudo_labeling": bool,
  "used_tta": bool,
  "citations": ["our best single model was a B4 at 512px"]
}
```

**校验规则**（不过则整篇丢弃并记入 `data/extract_rejects.jsonl` 附原因）：

1. JSON 可解析、`kind` 合法、`members` 非空；
2. **引用校验**：`citations` 里至少一条在正文中能找到（做空白归一后的子串
   匹配即可）；防 LLM 幻觉的主要闸门；
3. `members` 数 > 12 视为抽飞，丢弃。

**代码侧后处理**（`extract_post` 内完成）：

- `raw_model` 过 `MODEL_ALIASES` → `family`；同一篇内**家族去重**（B4+B5
  只算一次 efficientnet），`image_size` 取该家族成员的众数；
- `loss_raw` 过 `LOSS_ALIASES` → `loss_kb`；
- `best_single_model_raw` 同样映射 → `best_single_family`。

**输出 `data/facts.jsonl`**，每行 = posts 行 + 上述解析结果（原始 raw 全
保留）。CLI：`python -m kb_mining.extract [--limit N]`（`--limit` 供试跑）。

---

## 4. aggregate.py — 共识计算（纯函数，无 IO 依赖 LLM）

### 4.1 投票规则（已定稿，勿改）

对每篇 fact、其中每个 family：

| 情形 | 票重 |
|---|---|
| `kind == "single"` | 1.0 |
| `kind == "ensemble"` 且该 family == `best_single_family` | 1.0 |
| `kind == "ensemble"` 其余成员 | 0.5 |
| `kind == "unclear"` | 0.5 |

loss 投票同理（loss 是篇级字段，票重取该篇的最高票重）。
`used_pseudo_labeling=True` 的篇，票重再 ×0.8（归因混淆折扣）。

### 4.2 资格过滤（共存性 + 2021）

fact 计入 (trait T, family A) 的支持度，须同时满足：

1. 所属竞赛 `end >= "2021-01"`（catalog 清单本身已保证，代码再断言一次）；
2. **共存规则**：`FAMILY_RELEASE[A] < 竞赛 start`。不满足的 fact 对该
   family 不计票，也不计入分母。

### 4.3 共识行

对每个 (T, A)，**T 仅取特征卡的 true 布尔特征**（`fine_grained` /
`class_imbalance` / `medical`）。`data_size` **不作为独立挖掘 trait**——
竞赛数据量与胜方 backbone 选择之间的因果远弱于布尔特征，且混杂严重。
`data_size` 只在两处使用：

1. **原型查询填充参数**（§5.1）：每个 trait 的原型查询，`data_size` 一律取
   该 trait 证据竞赛的 data_size 众数（fine_grained / medical /
   class_imbalance 全部同规则）；
2. **档 1 field-fix 的证据**（§5.2）：对每个 family A 单独统计其获票竞赛的
   data_size 分布，与 A 节点的 `data_size` 列表比对，矛盾则出档 1 建议——
   这是按 family 聚合，不经过 trait 投票。

```python
support = Σ votes(A) / Σ votes(all families)   # 仅在具有 T 的合格竞赛内
breadth = A 获票的不同竞赛数
```

**阈值常数**（模块顶层，允许 CLI 覆盖）：

```python
SUPPORT_MIN = 0.50
BREADTH_MIN = 2
```

### 4.4 输出

- `data/consensus.json` — 全部共识行（含未过阈值的），字段：
  `{trait, component_type, kb_id, support, breadth, votes, total_votes,
    n_competitions, passed: bool, evidence: [{competition, rank, raw, citation}]}`
- `data/consensus.md` — 人读的表：按 trait 分节、support 降序，归并痕迹
  （raw ≠ 规范名）以 `raw` 列显示；未核对特征卡的竞赛加 ⚠ 标注。
- 三张侧表（只存不耗）：
  - `data/unknown_components.json` — 映射失败的 raw 字符串计数（未来建节点的候选池）；
  - `data/recipes.json` — `(family, trait) → {image_size 众数与分布}`（等 recipe layer）；
  - `data/ensemble_cooccurrence.json` — 家族共现计数矩阵（等 §7 ensemble 阶段）。

CLI：`python -m kb_mining.aggregate [--support-min 0.5] [--breadth-min 2]`

---

## 5. decide.py — 五档决策清单

对 `consensus.json` 里 `passed=True` 的每行，对照**现有 KB** 分类，产出
`data/proposals.md`。**本模块只写建议，不改任何 KB 数据。**

### 5.1 原型查询

```python
ARCHETYPE_QUERY = {
    "fine_grained":    {"task_type": "classification", "data_size": <该trait证据竞赛的data_size众数>,
                        "priority": "balanced", "constraints": {"fine_grained": True},
                        "description": "fine-grained image classification"},
    "class_imbalance": {..., "constraints": {"class_imbalance": True}, ...},
    "medical":         {..., "constraints": {"medical": True}, ...},
    "data_size=small": {..., "data_size": "small", "constraints": {}, ...},
    # 以此类推
}
```

检索调用（从仓库根运行）：

```python
from retrieval.rag_retrieval import build_graph, build_vector_index, retrieve_top3_hybrid
col = build_vector_index(persist_path=str(REPO_ROOT / "retrieval" / "chroma_db_kb"))
```

（注意：`fine_grained` 现在不是合法 constraint 键，`_matches_condition` 对
未知键返回不匹配即可——档 0 检查里它自然等价于"无此信号的查询"，正确。）

### 5.2 五档判定（逐档尝试，取第一个适用的）

| 档 | 判定 | proposals.md 里写什么 |
|---|---|---|
| 0 confirmed | 原型查询 top-1（loss 则看 top-1 的 loss 字段）已是 A | "无需改动"；confirmed 行单独列一节（KB 正确性的正面证据） |
| 1 field-fix | A 的节点字段与证据矛盾（目前只检查一种：`data_size` 列表不含证据竞赛的 size 众数档） | 建议改哪个节点哪个字段、改成什么 |
| 2 edge-tune | `EDGES` 里已有源为 A 的 `preferred_when` 边，条件与 T 相关但不含 T | 建议对 `EDGE_CONDITIONS` 的具体修改（all→any / 加键） |
| 3 new-edge | 以上都不适用，且 T 是合法 condition 键 | 建议新边 `(A, <当前top-1>, preferred_when)` + 条件；**目标 = 原型查询当前 top-1**（打分不消费目标，纯文档语义） |
| 4 schema-ext | T 不是合法 condition 键（即 `fine_grained`） | 建议：constraints 加键 + Module 1 prompt 同步 + 档 3 的边；标注影响面 |

**冲突检查**（附加在每条建议上）：若建议与现有某条边**方向相反**（同一
condition 下 A、B 互换），或使任何 golden 用例的断言对象改变——标
`CONFLICT`，写明"需短训 A/B 仲裁，暂不应用"。检查 golden 的方式：直接在
决策输出里对每条档 1–4 建议注明"若应用，受影响的原型查询 top-3 变化"
（decide 内部对 graph 副本试应用、重跑原型查询、diff）。

**堆叠纪律检查**：试应用全部档 3/4 建议后，任何 backbone 由挖掘边获得的
条件 bonus 上限须 ≤ 1 条命中（多 trait 合并为一条 `any` 边）；违反则在
proposals.md 顶部告警。

CLI：`python -m kb_mining.decide`

---

## 6. Phase B — loss `preferred_when` 边接线（唯一的检索代码改动）

**文件**：`retrieval/rag_retrieval.py` `_select_components`（约 1283 行）。

**现状**：loss 选择是硬编码 if 链（`class_imbalance→focal`、
`segmentation→dice/bce_dice`、`detr→hungarian`）；loss 节点间的
`preferred_when` 边（`focal_loss → cross_entropy_loss`,
`condition={"all": ["class_imbalance=True"]}`）是死数据。

**改动**：注意现有代码结构是 `chosen = candidates[0]` 默认值 + 独立的
`if ctype == "loss":` 块，块内是 `if class_imbalance... / elif segmentation...
/ elif detr...` 整条链。边消费**不能**用 `elif` 拼进这条链——正确结构是把
现有整条链包进 `else:`（整体缩进一层），`chosen = candidates[0]` 默认值
保持在最前不动：

```python
chosen = candidates[0]  # default: 第一个兼容项（原样保留）

if ctype == "loss":
    # preferred_when 边消费：候选间两两偏好，条件匹配则胜者上位
    # （backbone 打分只用边的源+条件；此处是候选内选择，目标有意义）
    edge_pick = None
    for cand in candidates:
        for succ in graph.successors(cand):
            e = graph[cand][succ]
            if (e.get("relation") == "preferred_when"
                    and succ in candidates
                    and _matches_condition(e.get("condition", {}), input_json)):
                edge_pick = cand
                break
        if edge_pick:
            break
    if edge_pick is not None:
        chosen = edge_pick
    else:
        # ↓ 现有硬编码 if/elif 链整体原样移入此 else，仅缩进变化
        if c.get("class_imbalance") and "focal_loss" in candidates:
            ...
```

确定性要求：`candidates` 顺序即遍历顺序，首个命中者胜（与现有
`candidates[0]` 的确定性习惯一致）。**不删除**现有硬编码规则——它们覆盖
边尚未表达的情形（bce_dice、hungarian）；等挖掘产出的边补齐后再另行清理。

**测试**（加入 `retrieval/test_rag_retrieval.py` 风格的用例；第 1 条是
硬性要求，不给备选——否则"死边改活"这件事本身没有被验证过）：

1. **边路径必须被证明触发**：测试内构造一个图副本，添加一条合成 loss
   `preferred_when` 边，其选择结果与硬编码 fallback 链**不同**（例如给
   `cross_entropy_loss → focal_loss` 加条件 `medical=True`——语义是虚构
   的，仅用于测试），断言 medical 查询下选中 cross_entropy 而非 fallback
   结果。现实 KB 里 focal 边与硬编码规则结论一致，行为回归测不出边是否
   活着，必须靠合成边区分；
2. 类不平衡分类查询 → focal_loss；无不平衡 → cross_entropy_loss（行为
   回归，双路径都盖到）；
3. `cd retrieval && python -m pytest test_golden.py test_rag_retrieval.py -q`
   全绿（**验收硬条件**）。

---

## 7. 测试计划（kb_mining/tests/）

全部离线、无网络、无真实 LLM：

- `test_harvest.py`：fixtures 放三个迷你 CSV（2 竞赛、4 topic、6 message），
  验证 slug 过滤、标题正则（含 "1st place"、"Solution summary"、不相关帖）、
  rank 解析、每竞赛截断到 MAX_POSTS_PER_COMP、chunked 读取路径（fixture 也走
  chunksize=2 强制多 chunk）。
- `test_extract.py`：canned LLM 响应注入 `llm_fn`，覆盖：正常 single、
  ensemble+best_single、引用校验失败被拒、JSON 坏被拒、家族去重、别名表
  （含 `tf_efficientnetv2_m`→efficientnet、`seresnext50`→resnet、未知→unknown）。
- `test_aggregate.py`：手写 facts fixture 验证票重表（含伪标签折扣）、
  共存过滤（在 swin 发布前的竞赛里 swin 不计票不计分母）、support/breadth
  数值精确断言、阈值分界、三张侧表内容。
- `test_decide.py`：用真实 `build_graph()`（不需要 chroma——档 0 检查可
  注入假的检索函数 `retrieve_fn`），对构造的 consensus 行断言五档各命中
  一次、冲突标记、堆叠告警。

---

## 8. Runbook（全流程命令）

```bash
# 0. 前置：Kaggle 凭证（~/.kaggle/kaggle.json）、LLM 凭证（同 Module 1 的 env）
# 1. 收集（首跑下载 meta-kaggle 三个 CSV，ForumMessages 数 GB，耐心）
python -m kb_mining.harvest
# 2. 抽取（先 --limit 5 试跑核对 facts 质量，再全量）
python -m kb_mining.extract --limit 5
python -m kb_mining.extract
# 3. 聚合 + 决策
python -m kb_mining.aggregate
python -m kb_mining.decide
# 4. 人读 data/consensus.md + data/proposals.md，挑选要应用的改动 → 另开 PR
# 5. 回归（任何 KB 改动后）
cd retrieval && python -m pytest test_golden.py test_rag_retrieval.py -q
```

---

## 9. 验收标准

1. `kb_mining/tests/` 全绿，全程无网络依赖；
2. 真实跑通 harvest→extract→aggregate→decide：≥8 个竞赛、全局 ≥50 篇
   facts、每个关键 trait（fine_grained / class_imbalance / medical）有
   ≥3 个竞赛且各贡献 ≥3 篇 facts，`consensus.md` / `proposals.md` 生成
   且可读（单竞赛 ≥5 篇不再是硬指标，见 harvest 的召回政策）；
3. `extract_rejects.jsonl` 拒绝率 < 30%（更高说明 prompt 或截断策略要调）；
4. Phase B 合入后 `test_golden.py` + `test_rag_retrieval.py` 全绿；
5. 本 PR **不含**任何 `EDGES` / `EDGE_CONDITIONS` / 节点字段 / 打分常数的
   数据改动（Phase B 的代码改动除外）。

## 10. 实现顺序

0. **第 0 步数据源验证（§0.5）——先于一切**；证伪则停下重议数据源
1. `catalog.py`（数据核实是主要工作量：竞赛清单、发布时间、特征卡初填）
2. `harvest.py` + tests（先用 fixture 开发，最后真跑一次下载）
3. `extract.py` + tests（prompt 初稿见附录 A；预留 **0.5 天专项 prompt
   调试**——用 --limit 试跑、按 extract_rejects 的拒因迭代，这是全项目
   最可能反复的环节，不要挤占）
4. `aggregate.py` + tests（纯函数，最快）
5. `decide.py` + tests
6. Phase B 接线 + 回归（独立 commit）

预计净工作量 4–5 天（含第 0 步半天与 prompt 调试半天）；harvest 的真实
下载和 extract 的全量 LLM 调用是仅有的两处外部依赖，都放在各自阶段最后做。

---

## 附录 A — extract 系统 prompt 初稿

编码时以此为起点（write-up 语料是英文，prompt 用英文）；调试中的修改需
同步回本附录：

```text
You are extracting structured facts from a Kaggle competition solution write-up.
Return pure JSON only — no markdown fences, no commentary.

Output schema:
{
  "kind": "single" | "ensemble" | "unclear",
  "members": [{"raw_model": "<model name exactly as written>", "image_size": <int|null>}],
  "loss_raw": "<loss name exactly as written>" | null,
  "best_single_model_raw": "<model name>" | null,
  "best_single_score": <float|null>,
  "used_pseudo_labeling": true | false,
  "used_tta": true | false,
  "citations": ["<verbatim sentence copied from the post>"]
}

Rules:
1. members: every distinct model architecture in the FINAL submission only —
   ignore abandoned experiments. Copy names exactly as written
   (e.g. "tf_efficientnet_b4_ns"); do NOT normalize or expand them.
2. kind: "single" if the final submission is one model; "ensemble" if it
   averages/stacks several; "unclear" if you cannot tell.
3. loss_raw: the training loss of the main model(s); null if never stated.
4. best_single_model_raw / best_single_score: only if the post explicitly
   reports a best single-model score (e.g. "our best single model scored
   0.899"); otherwise null.
5. citations: 1–3 quotes copied character-for-character from the post that
   mention the models or the loss. They are used for automatic verification;
   paraphrased quotes will cause the whole extraction to be rejected.
6. If the post is not actually a solution write-up (e.g. a question or a
   congratulations thread), return {"kind": "unclear", "members": []}.
```

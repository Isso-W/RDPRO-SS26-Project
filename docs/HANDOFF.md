# Jiaozi Module 3 — 工作交接文档（kb_mining 挖掘 + A/B 仲裁）

> 面向接手者（用 Codex/AI 助手续做）。2026-07-06 快照。分支 `integration-recommender`。
> **本工作流所有改动均未 commit。** 先读本文件建立全局，再按各 plan 的细节动手。

---

## 0. 一句话背景

Jiaozi 是"给 CV 任务描述 → 推荐整套模型配置（backbone+head+loss+optimizer+checkpoint）"
的系统。**Module 3**（`retrieval/rag_retrieval.py`）是选型核心：一个 NetworkX 知识图
（KB）+ ChromaDB 向量检索的混合管道。本工作流做了两件事，都围绕"用 Kaggle 冠军方案
的真实证据来校验/改进 KB"：

1. **kb_mining/** —— 从 Kaggle 优胜 write-up 挖掘"数据集特征→组件选择"的统计共识，
   产出**建议清单**（人审后才改 KB），并接活了一条 KB 里的死边（Phase B）。
2. **experiments/ab_loss_imbalance/** —— 对挖掘产出的**唯一 CONFLICT**（class_imbalance
   下 loss 该默认 focal 还是 CE）做预注册 A/B 仲裁实验。**代码完成，未跑真训练。**

两条线的权威规格：仓库根 `kb_mining_plan(2).md`、`ab_loss_imbalance_plan.md`。

---

## 1. 线程 A：kb_mining 挖掘管道

### 1.1 数据流（各阶段幂等、可独立重跑，只经 `kb_mining/data/` 通信）

```
Meta Kaggle CSV dump ──harvest──▶ posts.jsonl ──extract(LLM)──▶ facts.jsonl
   ──aggregate──▶ consensus.{json,md} + 3 张侧表 ──decide──▶ proposals.md
```

- **harvest.py**：Meta Kaggle 官方 dump（`Competitions/ForumTopics/ForumMessages.csv`）
  → 每竞赛的优胜方案帖正文。链路：`Competitions.ForumId → ForumTopics.ForumId`，再用
  `ForumTopics.FirstForumMessageId → ForumMessages.Id`（直接外键取楼主首帖），
  ForumMessages（1.7GB）**单次 chunksize 流式**按 id 过滤。实证见 `data/source_check.md`。
- **extract.py**：复用 Module 1 的 LLM client（qwen），`extract_post(post, llm_fn)` 可注入。
  LLM 只吐 raw 字符串，KB id 映射由 `catalog.py` 别名表做（不让 LLM 输出 id，降幻觉）。
  有引用校验、组合损失→unknown、家族去重。`remap_facts()` 可在别名表变更后**免 LLM 重映射**。
- **aggregate.py**：加权投票算共识。核心概念（都在这实现）：
  - **按 task_type 分池**：分类/检测/分割 backbone 不混比。
  - **车架 vs 发动机角色分组**（`catalog.FAMILY_ROLE`）：检测/分割方案常是
    "U-Net(车架) + EfficientNet encoder(发动机)"，两组各自分母、各比各的，避免 encoder 稀释元架构。
  - **DOMINANCE 判据**：`passed = 多数决(support≥0.5,breadth≥2) OR dominance(ratio≥1.5,breadth≥3,margin≥0.10)`。
  - support 分母只算已知家族；`kb_coverage` 暴露 KB 外架构占比；unknown 进侧表（建新节点候选池）。
- **decide.py**：对每条 passed 共识，用**原型查询**打现有检索管道，按五档归类
  （0 confirmed / 1 field-fix / 2 edge-tune / 3 new-edge / 4 schema-ext）+ 档5 cross-role
  + 档6 finding。含**冲突检查**（反向边对，含跨条件）、堆叠告警。**只写建议，不改 KB。**
- **catalog.py**：纯数据 —— 31 个竞赛的特征卡（`traits_verified` 人工核对标志）、
  `FAMILY_RELEASE`（共存性过滤）、`MODEL_ALIASES/LOSS_ALIASES`、`FAMILY_ROLE`。

### 1.2 Phase B（唯一落地的 KB 代码改动，在 `retrieval/rag_retrieval.py`）

`_select_components` 的 loss 分支现在**消费 loss 节点间的 `preferred_when` 边**（原为死数据），
硬编码 if 链降为 fallback。图数据零改动，只是让检索开始读一条本就存在的边。测试见
`retrieval/test_rag_retrieval.py::TestPhaseBLossEdges`（含"边确实触发"的合成边证明）。

### 1.3 当前挖掘结论（`kb_mining/data/proposals.md`）

净结论：**KB 基本被验证是对的**。5 条 confirmed（efficientnet 是分类默认 backbone、
CE 是默认 loss、unet 是医疗分割首选，KB 都已对）；唯一实质改动是 class_imbalance 下
CE-vs-focal，但**方向与现有 focal 边冲突 → 交给线程 B 的 A/B 仲裁**。检测/分割的
backbone 共识因角色分组后 unet 在医疗分割冲到 0.59（干净信号）。

---

## 2. 线程 B：A/B 仲裁实验（`experiments/ab_loss_imbalance/`）

预注册实验：class_imbalance 下 loss 默认 focal（KB 现状）还是 CE（挖掘共识）？

- **paired 5-fold，两台跨域**：`siim_isic`（极端不平衡医疗二分类，roc_auc）+
  `cassava`（中度不平衡农业多类，**主判据 macro_f1**——accuracy 对不平衡盲）。
- **自适应噪声带**：平局带 `max(0.005, 2·SE)`，翻案必须跨过噪声（现状赢含糊局面）。
  三种 verdict（CE_WINS / FOCAL_WINS / TIE）都有对应 KB 动作，没有白跑。
- **文件**：`configs.py`（冻结矩阵+判据，唯一事实源）、`collect.py`（verdict，纯函数）、
  `run_ab.py`（算折/生成/训练/续跑）。
- **地基改动**（在 `module4_agent/code_generator.py`，向后兼容）：
  - 折注入：`model_config` 加 `fold_file`/`fold_index`，按样本 id 显式划分 val（paired 保证）。
  - 预测导出：`export_preds_path` → `val_preds.json`，run_ab 侧算 macro_f1/roc_auc/pr_auc bundle。

**状态：代码全完成，未跑真训练**（需 GPU + Kaggle 数据）。续接清单见
`experiments/ab_loss_imbalance/TODO.md`（权威）。

---

## 3. 跑 & 测（注意 cwd）

```bash
# 全部离线测试（无网络、无 LLM、无 GPU）
PYTHONPATH=. python -m pytest kb_mining/tests/ -q                 # 58 项
PYTHONPATH=. python -m pytest experiments/ab_loss_imbalance/tests/ -q   # 26 项
PYTHONPATH=. python -m pytest module4_agent/tests/ -q            # 63 项（含 6 折注入）
cd retrieval && python -m pytest test_golden.py test_rag_retrieval.py -q   # 53 项，必须 cwd=retrieval/

# kb_mining 全流程（harvest 需 Kaggle 凭证 + 下 dump；extract 需 LLM env）
PYTHONPATH=. python -m kb_mining.harvest
PYTHONPATH=. python -m kb_mining.extract        # 先 --limit 5 试跑
PYTHONPATH=. python -m kb_mining.aggregate
PYTHONPATH=. python -m kb_mining.decide

# A/B 真跑见 experiments/ab_loss_imbalance/TODO.md
```

**环境坑（务必知道）**：
- **测试 cwd**：`retrieval/` 的测试必须 `cd retrieval/` 跑；其余从仓库根 + `PYTHONPATH=.`。
- **Windows GBK 控制台**：print 里别用非 GBK 字符（emoji ⚠/✗ 会崩）；跑脚本加 `PYTHONIOENCODING=utf-8`。
- **后台任务 cwd**：后台 python 的 cwd 未必是仓库根，导入用 `sys.path.insert` 或绝对路径。
- **torch 现为 CPU 版**（`2.6.0+cpu`）——A/B 真训练前必须换 CUDA 版（见 TODO.md）。

---

## 4. 关键设计决定（别再推翻，已论证过）

1. **aggregate 分母只算已知家族**：否则冠军大量用 KB 外架构，support 全被压到 0，共识表全空没用。
2. **车架/发动机角色分组**：解决"U-Net+encoder 一方案两票互相稀释"；`FAMILY_ROLE` 是权威映射。
3. **DOMINANCE 判据**：Kaggle 选 backbone 是分布不是多数，纯"过半"永远出不了 backbone 提案。
4. **组合损失 → unknown**：检测/分割的加权组合损失压平成单票是伪象；`is_hybrid_loss` 判定。
5. **检测 loss / 分割网解检测 → findings 不进 proposals**：证据形态不支持改边。
6. **A/B：paired 5-fold + 噪声带**：原 2-seed±0.002 方向反了（噪声会驱动翻案）。cassava 主判据用
   macro_f1（accuracy 不平衡盲）。pretrained 冻结 `efficientnet_b0_imagenet`（不吃 Module 3 动态选）。
7. **traits_verified 工作流**：catalog 里 22/31 竞赛的特征卡仍是初判（`traits_verified=False`），
   采纳任何 proposal 前要先人工核对其证据竞赛。

---

## 5. 待办总览（细节见各文件）

| 优先 | 事项 | 权威文件 |
|---|---|---|
| A | A/B 真训练（换 CUDA torch → Kaggle 数据 → 跑 20 折 → collect verdict） | `experiments/ab_loss_imbalance/TODO.md` |
| B | verdict → §4 KB 动作 + golden 回归 + 小结 | `ab_loss_imbalance_plan.md` §4 |
| C | 4 条 confirmed 固化为 golden 断言 | `kb_mining/data/proposals.md` 档0 |
| C | 核对剩余竞赛 `traits_verified` 后重跑 aggregate/decide | `kb_mining/catalog.py` |
| D | 杂项：`kb_mining_plan(2).md` 改名归 docs/；**本工作流全部 commit** | — |

---

## 6. 权威详细文档（本文件之外）

- `kb_mining_plan(2).md` —— kb_mining 完整实现规格
- `ab_loss_imbalance_plan.md` —— A/B 实验完整规格（判据、§4 动作）
- `experiments/ab_loss_imbalance/TODO.md` —— A/B 续接清单
- `kb_mining/data/source_check.md` —— Meta Kaggle 数据源实证
- `CLAUDE.md` —— Module 3 架构总览（KB 结构、检索管道、输入 schema）
- `docs/MODULE3_API.md` —— Module 3→4 接口

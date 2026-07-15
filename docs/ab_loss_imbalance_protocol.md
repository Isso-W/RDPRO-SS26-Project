# A/B 仲裁实验 — class_imbalance 下 CE vs focal：实现规格

> kb_mining 产出的唯一 CONFLICT 的仲裁实验。可直接编码。
> 前置约定（不可事后更改）：**判据先于实验写死在本文档与 configs.py 里。**

---

## 0. 问题与裁决规则（预注册）

**问题**：分类任务 + `class_imbalance=True` 时，loss 默认该是 focal（KB 现状：
`focal_loss → cross_entropy_loss` 边 + `_select_components` 硬编码规则）还是
CE（Kaggle 共识：support 0.71，dominance ×2.66，breadth 10）？

**测试台**（v3：副台改用 catalog 现成条目，且两台跨域——医疗 + 农业）：

- **主**：`siim_isic`（极端不平衡医疗二分类，阳性 ~1.8%，metric=roc_auc）
- **副**：`cassava`（catalog 现成、管线已真实跑通：5 类叶病细粒度，中度
  不平衡（主导类 ~61%），21k 图）。**裁决指标用 macro-F1，不用官方
  accuracy**——本实验主题即"focal 在不平衡下有没有用"，而 accuracy 对不平衡
  盲（忽略少数类照样 ~61%），会让该台按构造倾向 TIE；换 imbalance 敏感的
  macro-F1 才能让 cassava 真正投票。此选择先于实验预注册，非事后 metric-
  shopping。accuracy 降为次级（仅记录）。
  不用 aptos（不在 vision_benchmark_catalog，新增是额外前置任务）；
  不用 diabetic_retinopathy（80GB+ 下载不值）。

**实验设计（v2：paired 5-fold，替换原 2-seed 单折）**：每个测试台上，
**分层 5 折，两臂共用同一套折**（paired），2 臂 × 5 折 = 10 次短训/台。
除 loss 外一切相同（同 checkpoint、同 image_size、同增广、同 LR/schedule/
epochs、同 sampler——**两臂都用普通 shuffle，不用加权采样**；loss×sampler
交互显式声明为范围外）。全局 seed 固定 42（折既提供重复，也消掉单折验证集
的抽样运气）。focal 超参用 Module 4 当前默认实现原样——仲裁的是"我们的
系统部署出的 focal vs 我们的系统部署出的 CE"，不是理想化 focal。

**统计量与裁决规则（写死；v2 修正——原 ±0.002 窄带方向反了，噪声会驱动
翻案）**：对每折 i 计算配对差 `Δ_i = metric(CE, fold_i) − metric(focal,
fold_i)`，取 `Δ̄ = mean(Δ)`，`SE = std(Δ)/√5`。**平局带自适应且不小于
噪声**：

- `Δ̄ ≥ max(0.005, 2×SE)` → 该测试台 **CE 胜**；
- `Δ̄ ≤ −max(0.005, 2×SE)` → 该测试台 **focal 胜**；
- 其间 → **TIE**。翻案必须跨过噪声；现状赢一切含糊局面。

**双台合并规则**：仅当无一台判 focal 胜、且至少一台判 CE 胜 → 总 verdict
CE_WINS（focal 对称同理）；其余 → TIE。结论文字中老实限定适用范围：
siim=极端不平衡·医疗·二分类，cassava=中度不平衡·农业细粒度·多类——
两台合计仍非全域 class_imbalance，KB 动作的注释里写明这一点。

**次级观测（仅记录、不进裁决，防 metric-shopping）——按台设，不进共享
BASE**（指标口径按台不同）：siim_isic 记 PR-AUC（极端不平衡下比 ROC-AUC
对 loss 更敏感）；cassava 记 accuracy（官方指标，作对照参考）。若主指标
TIE 而次级指标出现一致方向的大差，写进实验小结作为后续线索，但**不得**据
此改 KB。

**功率的诚实预期**：好预训练 + 微调下 CE/focal 的差常落在噪声内——TIE 是
一个**正当的、可写进报告的结局**（"在部署条件下未检出显著差异 → 维持现状"
即 CONFLICT 的 resolved-tie），不是实验失败。

**成本**：224px + efficientnet_b0，siim_isic 约 1–1.5h/次 ×10 ≈ 2–3 个
Kaggle GPU session；cassava（21k 图、8 epochs）约 0.5–1h/次 ×10 ≈ 1–2 个
session。均可挂机。

---

## 1. 文件结构

```
experiments/ab_loss_imbalance/
  configs.py     # 冻结的实验矩阵 + 裁决常数（唯一事实源）
  run_ab.py      # 驱动：算折→生成 Module 4 工程→按 (臂, 折) 顺序跑 10 次/台
  collect.py     # 汇总折级配对差 → 台级 + 总 verdict
  README.md      # 指向本文档
results/outcomes.jsonl   # 每次训练追加一条记录（schema 见 §3）
```

## 1.5 前置第 0 步 — Module 4 折注入机制（paired 的地基，必须先做）

**现状**：`code_generator.py` 只有内部 `val_split` 逻辑，没有任何外部折
注入能力——"两臂共用同一套折"目前无从实现，直接跑会退化成两臂各切各的
val，配对差 Δ_i 失去意义。**本步不完成，后续全部作废。**

**接口**：`model_config` 增加两个可选键——

- `"fold_file"`：折文件路径。格式：
  ```json
  {"seed": 42, "n_folds": 5, "stratified": true,
   "id_column": "image_id",
   "folds": [["id1","id7",...], ...]}   // 每折的 val 样本 id 列表
  ```
  **按样本 id 存，不按行号**——对 CSV 行序变化免疫；
- `"fold_index"`：0–4，本次训练用第几折做 val（该折 = val，其余 = train）。

**code_generator 改动**：`_build_local_dataloader` 相关模板中，若
`fold_file` 存在 → 按 id 显式划分并**旁路**内部 val_split；同时做完整性
校验（folds 并集 == CSV 全部 id 且两两不相交，否则报错拒跑）。两键缺省
→ 行为与现状完全一致（向后兼容）。

**测试**（进 `module4_agent/tests/`）：①fixture CSV + folds.json 生成工程
后，val 集与指定折的 id 精确相等；②同一 fold_file 的两次生成（不同 loss）
val 集完全一致——**paired 的机器可查证明**；③折文件不完整/有交集 → 拒跑；
④不带两键 → 旧行为回归。

**附带（同属地基）——多指标导出**：Module 4 benchmark eval 原生只算 catalog
里的单个 metric（cassava=accuracy、siim=roc_auc），但本实验 cassava 主判据要
macro_f1、siim 次级要 pr_auc（均非原生）。因此 run.py 训练结束需**导出该折
val 的预测 + 标签**（`val_preds.json`：`{"y_true": [...], "y_score": [...]}`），
run_ab 侧用 sklearn 算指标 bundle 写进 `val_metric`。eval 不吐预测则 collect
无从算 macro_f1/pr_auc——与折注入同为"现有 harness 缺的钩子"，一并在第 0 步补。

### configs.py

```python
MARGIN_FLOOR = 0.005          # 平局带下限；实际带宽 = max(0.005, 2*SE)
N_FOLDS = 5
GLOBAL_SEED = 42
ARMS = ("focal_loss", "cross_entropy_loss")
TESTBEDS = {
    "siim_isic": {"metric": "roc_auc",  "image_size": 224, "epochs": 8,
                  "secondary_metrics": ["pr_auc"]},
    "cassava":   {"metric": "macro_f1", "image_size": 224, "epochs": 8,
                  "secondary_metrics": ["accuracy"]},
}
BASE = {                      # 除 loss/fold 外全部冻结
    "backbone": "efficientnet",
    # 实现时：真实跑一次 Module 3 对两台查询的检索，把解析出的 checkpoint
    # id 硬编码在这里并注明日期——冻结，不做动态解析（否则 KB 后续变动会
    # 悄悄换掉实验的地基）。禁止留占位字面量。
    "pretrained": "efficientnet_b0_XXXX",
    "sampler": "shuffle",     # 显式声明：无加权采样
    "cv": {"n_folds": 5, "stratified": True, "shared_across_arms": True},
}
def build_matrix() -> list[dict]: ...   # 每台 10 个 run config；差异字段仅 loss, fold_index
```

**测试**：①断言每台矩阵 10 条、两两之间除 `loss`/`fold_index` 外所有键值
相等（防手滑引入第二个变量——本实验有效性的机器可查保证）；②断言两臂
引用同一个 `fold_file` 路径（配合 §1.5 测试②构成 paired 的完整证明链）；
③`pretrained` 不含占位符字样。

### run_ab.py

复用现有链路，不新造训练代码。每台流程：

1. `ingest_benchmark(<testbed>, data_root)` 下载数据；
2. **算折**：读 train CSV 标签，`StratifiedKFold(5, shuffle=True,
   random_state=42)`，按 §1.5 格式写 `folds_<testbed>.json`，算 sha256；
   folds 文件已存在则直接复用（幂等——重跑不得重切折）；
3. 生成 **一个** Module 4 工程（沿 `run_kaggle_benchmark.prepare_project`
   流程），对每个 run 覆写 `model_config` 的 `loss` / `fold_file` /
   `fold_index`，顺序训练 10 次；
4. 每次训练结束把记录（含 `fold_file_sha256`）追加进
   `results/outcomes.jsonl`（§3 schema），中断可续跑（按 jsonl 已有记录
   跳过已完成的 (臂, 折)）。

CLI：`python -m experiments.ab_loss_imbalance.run_ab [--testbed cassava]
[--data-root ...] [--only focal_loss:3]`（`--only 臂:折` 供单折试链路）

### collect.py

读 outcomes.jsonl → 每台输出折级配对表（5 行 Δ + Δ̄ ± SE）+ 台级 verdict
+ 按双台合并规则输出总 verdict。**verdict 逻辑必须有单元测试**：三种台级判
例、`Δ̄` 恰落 `max(0.005, 2×SE)` 边界、SE 极大时窄效应正确判 TIE、双台合并
的全部组合（含"一台 CE 胜一台 focal 胜 → TIE"）。无网络、纯函数。

## 3. outcomes.jsonl 记录 schema

```json
{"experiment": "ab_loss_imbalance", "benchmark": "siim_isic",
 "arm": "cross_entropy_loss", "fold": 3, "seed": 42,
 "config": {"backbone": "...", "pretrained": "...", "image_size": 224,
            "epochs": 8, "sampler": "shuffle"},
 "val_metric": {"roc_auc": 0.912, "pr_auc": 0.231}, "best_epoch": 6,
 "fold_file_sha256": "…", "kb_version": "<git sha>", "date": "2026-07-xx"}
```

## 4. verdict → KB 动作（三种结局都有事做，没有白跑的可能）

| verdict | KB 动作 |
|---|---|
| CE_WINS | ①翻转边：删 `focal_loss→cross_entropy_loss`，加 `cross_entropy_loss→focal_loss`（条件不变）；②同步改 `_select_components` 硬编码 imbalance 规则；③改 golden 中相应断言；④全部改动带溯源注释（`# ab_loss_imbalance 2026-07: paired 5-fold, siim Δ̄=+x.xxx±SE / cassava Δ̄=…；适用范围=医疗极端不平衡+农业中度不平衡`） |
| FOCAL_WINS | KB 不动；边上加注释"Kaggle 共识(0.71)与此相悖，经 siim_isic A/B 实证防御（focal +x.xxx）"；proposals 的 CONFLICT 标记 resolved-rejected |
| TIE | KB 不动；边注释记录"A/B 无显著差异，维持现状"；CONFLICT 标记 resolved-tie |

任一结局后：`cd retrieval && python -m pytest test_golden.py test_rag_retrieval.py -q`
全绿是收尾硬条件。

## 5. 验收标准

1. configs 冻结/paired/占位符三项测试 + collect verdict 测试全绿（离线）；
2. 两台共 20 次训练完成，outcomes.jsonl 有 20 条完整记录（折级）；
3. collect 输出台级 + 总 verdict，§4 对应动作已应用且 golden 全绿；
4. docs 里一段 5–10 行的实验小结（配对差表格 + verdict + 适用范围限定 +
   KB 动作），供终期报告直接引用。

## 6. 实现顺序

0. **§1.5 折注入机制 + 四项测试（半天）——不完成不得进入后续任何步骤**
1. configs.py + 三项冻结测试、collect.py + verdict 测试（半天，纯离线）
2. run_ab.py 接通算折/生成/覆写/续跑（半天，先在 cassava 上用
   `--only focal_loss:0` 单臂单折 1 epoch 验证链路）
3. 真跑：cassava 10 次（1–2 session）先出首个台级 verdict，再挂
   siim_isic 10 次（2–3 session）
4. 总 verdict → §4 动作 → golden 回归 → 实验小结（半天）

**范围外**（明确不做）：加权采样交互、focal 超参搜索、按次级指标改 KB。

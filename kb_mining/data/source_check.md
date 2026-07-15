# source_check.md — Meta Kaggle 数据源地基验证（§0.5）

**验证日期**：2026-07-03　**结论**：✅ 链路通，采用 Meta Kaggle 官方 dump，不走备选方案。

harvest.py **按本文件的实际列名编码**，不按 kb_mining_plan 的"预期链路"编码。

## 下载的文件（`kaggle/meta-kaggle`，逐文件 `dataset_download_file`）

| 文件 | 大小 | 载入方式 |
|---|---|---|
| `Competitions.csv` | 149.6 MB | 可 `usecols` 全量载入 |
| `ForumTopics.csv` | 70.4 MB | 可 `usecols` 全量载入 |
| `ForumMessages.csv` | **1.70 GB** | **必须 `chunksize` 流式**，按 Id 集合过滤 |

下载不产生 `.zip`——kaggle 2.2.3 的 `dataset_download_file` 直接落 `.csv`。

## 实际列名（以本次 dump 为准）

- **Competitions.csv**：`Id, Slug, Title, ForumId, EnabledDate, DeadlineDate,
  TotalTeams, HostSegmentTitle, Overview, DatasetDescription`（后两列内嵌竞赛
  描述正文——特征卡初填**不必爬竞赛网页**）。
- **ForumTopics.csv**：`Id, ForumId, FirstForumMessageId, Title, Score,
  TotalMessages, CreationDate`。
- **ForumMessages.csv**：`Id, ForumTopicId, PostUserId, PostDate,
  ReplyToForumMessageId, Message, RawMarkdown, Medal, MedalAwardDate`。

## 确认可行的 JOIN 链（比计划设想的更短）

```
Competitions.ForumId  ==  ForumTopics.ForumId          # 按 catalog slug 定位竞赛论坛
ForumTopics.FirstForumMessageId  ==  ForumMessages.Id  # 直接外键指向楼主首帖！
```

**关键优化（harvest 据此实现）**：`FirstForumMessageId` 是指向楼主首帖的直接
外键，**不需要按 `ForumTopicId` 分组扫 ForumMessages**。正确做法：

1. Competitions 过滤出 catalog 里的竞赛 → 得 `ForumId` 集合；
2. ForumTopics 过滤这些 ForumId + 标题正则筛 solution 帖 → 得
   `FirstForumMessageId` 集合（连同 rank/score/title）；
3. ForumMessages **单次 chunksize 流式**，`Id ∈ 该集合` 即取正文——1.70 GB 只
   扫一遍，命中集合很小（每竞赛 ≤10）。

## 正文格式

- 正文优先用 **`RawMarkdown`**（markdown，喂 LLM 更干净），空则回退 `Message`（HTML）。
- 注意：RawMarkdown 里仍可能内嵌 HTML 标签（如 `<b>`）和图片链接，extract 的
  截断/引用校验需容忍。
- `PostDate` 格式 `MM/DD/YYYY HH:MM:SS`（→ posts.jsonl 的 `post_date` 取 `YYYY-MM-DD`）。

## 端到端实证（cassava-leaf-disease-classification）

- ForumId **1000771**，论坛 833 帖，标题正则命中 **46 篇** solution 帖（远超 ≥5）。
- 1st place（topic 221957 → FirstForumMessageId **1216990**）正文 **8222 字符**，
  PostDate 02/24/2021，内容含 EfficientNet/ResNet/ResNext/ViT/DeiT/MobileNet、
  "ensemble of four models"、best single "B4: 89.4%"——正是 extract 目标素材。
- 假阳性观察：正则会误收 "Can't wait to see 1st team's solution"（许愿帖）、
  "Top solutions in ..."（汇总帖）——交给 extract 的引用校验 / "非方案帖→unclear"
  过滤，噪声可接受。

## 对 catalog 的连带修正（已回填真实值）

- 全部 14 竞赛的 `start/end` 用 dump 的 `EnabledDate/DeadlineDate` 真实值填入。
- **`ubc-ocean` slug 错误 → 实际为 `UBC-OCEAN`（全大写）**。harvest 的 slug
  过滤须大小写敏感或对齐 dump 原值。
- `fathomnet-2025` 仅 **79 队**、`herbarium-2022-fgvc9` **134 队**、
  `sorghum-id-fgvc-9` **252 队**——write-up 可能不足 5，由 harvest 召回政策定去留。

## 备选方案

未触发。若未来 dump 结构变化导致链断，见 kb_mining_plan §0.5 的三级备选。

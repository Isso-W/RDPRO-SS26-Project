# proposals.md — KB 改动建议（五档决策）

> 本文件仅为建议；KB 数据由人看完后另行提交。

## 档 0 — confirmed（5 条）

- **[classification · backbone] efficientnet** ×「class_imbalance」 **[DOMINANCE]**
  - 'efficientnet' 已是 class_imbalance 原型查询的 top-1，无需改动。
- **[classification · backbone] efficientnet** ×「fine_grained」 **[DOMINANCE]**
  - 'efficientnet' 已是 fine_grained 原型查询的 top-1，无需改动。
- **[classification · loss] cross_entropy_loss** ×「fine_grained」 **[DOMINANCE]**
  - 'cross_entropy_loss' 已是 fine_grained 原型查询的 top-1，无需改动。
- **[classification · loss] cross_entropy_loss** ×「medical」 **[DOMINANCE]**
  - 'cross_entropy_loss' 已是 medical 原型查询的 top-1，无需改动。
- **[image_segmentation · backbone] unet** ×「medical」 **[DOMINANCE]**
  - 'unet' 已是 medical 原型查询的 top-1，无需改动。

## 档 3 — new-edge（1 条）

- **[classification · loss] cross_entropy_loss** ×「class_imbalance」 **[CONFLICT]** **[DOMINANCE]**
  - 新增边 ('cross_entropy_loss', 'focal_loss', preferred_when)，条件 {'any': ['class_imbalance=True']}；目标=当前 top-1（打分不消费目标，纯语义）。
  - CONFLICT：已存在反向边 'focal_loss'→'cross_entropy_loss' 且条件相交（['class_imbalance=True']），需短训 A/B 仲裁，暂不应用。

## 档 5 — cross-role（无对应 RAG 槽位）（1 条）

- **[object_detection · backbone] efficientnet** ×「medical」
  - 跨角色：'efficientnet'（engine）与当前 top-1 'yolov8'（frame）角色不同。RAG 对 object_detection 只选 frame，未建模 engine 槽位——本条共识暂无对应 RAG 决策，不提边。

## Findings（3 条，仅记录，不改 KB）

- **[object_detection · backbone] unet** ×「class_imbalance」
  - 分割架构 'unet' 出现在检测任务共识（support=0.6056，breadth=2）——'用分割网解检测赛'的路径，KB 无法表达此跨任务用法；记为 finding，不动分割边。
- **[object_detection · loss] cross_entropy_loss** ×「class_imbalance」 **[DOMINANCE]**
  - 检测任务的 loss 共识（'cross_entropy_loss' support=0.7105）来自组合损失，压平成单票是伪象，不代表 CE/focal 之间的选择——记为 finding，不提边。
- **[object_detection · loss] cross_entropy_loss** ×「medical」
  - 检测任务的 loss 共识（'cross_entropy_loss' support=0.7838）来自组合损失，压平成单票是伪象，不代表 CE/focal 之间的选择——记为 finding，不提边。

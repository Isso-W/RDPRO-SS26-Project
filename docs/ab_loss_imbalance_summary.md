# A/B 仲裁实验小结 — class_imbalance 下 CE vs focal

*2026-07-07。规格见 `docs/ab_loss_imbalance_protocol.md`，代码见 `experiments/ab_loss_imbalance/`。*

## 问题

kb_mining 挖掘出的唯一 CONFLICT：分类任务 + `class_imbalance=True` 时，loss 默认
该用 **focal**（KB 现状：`focal_loss→cross_entropy_loss` 边 + `_select_components`
硬编码规则）还是 **CE**（Kaggle 优胜方案挖掘共识：support 0.71 / dominance ×2.66 /
breadth 10）？

## 方法（预注册）

- **paired 5-fold**：两臂（focal / cross_entropy）共用同一套分层折，除 loss 外一切
  冻结（efficientnet_b0 @ 224px、5 epochs、AdamW、普通 shuffle 采样、seed 42）。
- **判据**：每折配对差 `Δ = metric(CE) − metric(focal)`；平局带 `max(0.005, 2·SE)`
  —— 翻案须越过噪声，现状赢含糊局面。
- 计划两台跨域：`siim_isic`（极端不平衡·医疗·二分类）+ `cassava`（中度不平衡·农业·
  多类）。**siim 因算力成本（~23GB 下载 + ~3h 训练 + Colab 计算单元）本轮略过，
  仅完成 cassava。**

## 结果 — cassava（主指标 macro-F1，5 折）

| 折 | Δ(CE − focal) |
|---|---|
| 0 | +0.0145 |
| 1 | +0.0041 |
| 2 | +0.0031 |
| 3 | +0.0041 |
| 4 | +0.0117 |

**Δ̄ = +0.0075，SE = 0.0023，平局带 = ±0.005 → CE_WINS**（5/5 折方向一致偏 CE）。

原始记录：`experiments/ab_loss_imbalance/results/outcomes.jsonl`（10 条 = 2 臂 × 5 折）。10 条的 `fold_file_sha256` 完全
相同 —— **paired 的机器可查证明**：两臂确实在同一套折上比较。

## 结论与对 KB 的处置

- **cassava 台**：CE 稳定优于 focal —— 5 折全为正、方向一致、Δ̄ 越过噪声带；且与
  挖掘共识（0.71）**方向一致**。是"挖掘信号 + 实验验证"两条独立证据，不是孤证。
- **适用范围限定**：中度不平衡的自然图像多类分类。**极端不平衡 / 医疗 regime 未做
  A/B 验证**（siim 略过）—— 而 focal 理论上恰在极端不平衡最有用，此处未证否。
- **KB 处置（保守，与证据强度匹配）**：单台 + 薄边际（Δ̄ 0.0075 仅略高于带 0.005）
  + 极端 regime 未测 → **不足以支撑把 `focal→CE` 边全局翻向**。CONFLICT 标为
  **resolved-lean-CE**：证据倾向 CE，但保留现状（focal 默认）不动 KB，待极端不平衡
  台补测后再定是否翻边。这也是预注册平局带哲学的延续 —— 含糊局面现状赢。

## 复现

```bash
python -m pytest experiments/ab_loss_imbalance/tests -q
python -m experiments.ab_loss_imbalance.run_ab --testbed cassava
python -m experiments.ab_loss_imbalance.collect
```

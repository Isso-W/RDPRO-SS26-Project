# ab_loss_imbalance — CE vs focal 的 A/B 仲裁

kb_mining 产出的唯一 CONFLICT（class_imbalance 下 loss 该默认 focal 还是 CE）的
预注册仲裁实验。**规格与判据见仓库根 `ab_loss_imbalance_plan.md`（唯一权威）。**

## 组件

- `configs.py` — 冻结的实验矩阵 + 裁决常数（唯一事实源，预注册）
- `collect.py` — 折级配对差 → 台级 + 双台 verdict（自适应噪声带，纯函数）
- `run_ab.py` — 算折 → 生成工程 → 按 (臂, 折) 顺序训练 → 落 outcomes（需 GPU + 数据）
- `tests/` — configs 冻结 / collect 裁决 / run_ab 纯逻辑（全离线）
- `results/outcomes.jsonl` — 每折一条记录（训练时追加，续跑可断点）

## 跑法

```bash
# 离线自检（判据 + 纯逻辑）
python -m pytest experiments/ab_loss_imbalance/tests/ -q
# 真跑（Kaggle 凭证 + GPU；先单折试链路）
python -m experiments.ab_loss_imbalance.run_ab --testbed cassava --only focal_loss:0
python -m experiments.ab_loss_imbalance.run_ab --testbed cassava
python -m experiments.ab_loss_imbalance.run_ab --testbed siim_isic
# 汇总裁决
python -m experiments.ab_loss_imbalance.collect
```

verdict → KB 动作见 plan §4；任一结局后 `cd retrieval && pytest test_golden.py
test_rag_retrieval.py -q` 全绿是收尾硬条件。

# A/B 实验 — 续接清单（2026-07-06 暂停于此）

规格见仓库根 `ab_loss_imbalance_plan.md`。下面记录**已完成**与**待做**，续接直接照做。

## ✅ 已完成（离线，全部有测试）

- **Step 0 折注入地基**（`module4_agent/code_generator.py`）
  - `_fold_split_indices`：按样本 id 的 paired 折划分 + 完整性校验（并集==全部、不相交）
  - CSV 分支：有 `fold_file`+`fold_index` 时旁路内部 val_split；缺省 = 旧行为（向后兼容）
  - `_eval_on_dataloader`：有 `export_preds_path` 时导出 `val_preds.json`（y_true / y_prob）
  - 测试 `module4_agent/tests/test_fold_injection.py`（6 项）；Module4 全套 63 绿、无回归
- **Step 1 预注册冻结**：`configs.py`（矩阵 + 判据常数，pretrained=efficientnet_b0_imagenet 无占位符）、`collect.py`（自适应噪声带 verdict）；`tests/`（20 项）
- **Step 2 驱动**：`run_ab.py`（算折 / 生成 / 覆写 / subprocess / 续跑 + 指标 bundle）；`tests/`（6 项）
- 实验测试合计 **26 绿**；plan 已补"多指标导出"note

## ⏸ 待做 A：本地 4060 跑真训练（Step 3，需 GPU + Kaggle）

1. **[拦路] 换 CUDA 版 torch** —— 当前是 `2.6.0+cpu`（`cuda available: False`）。
   `pip uninstall -y torch && pip install torch --index-url https://download.pytorch.org/whl/cu124`
   装完确认 `torch.cuda.is_available() == True`。
2. **接受 Kaggle 规则**（网页各一次）：`siim-isic-melanoma-classification`、`cassava-leaf-disease-classification`。
3. **留磁盘** ~50GB（siim JPEG ~23GB 下载+解压）。
4. **[可选，Windows 稳妥] 给 `run_ab._frozen_config` 加 `"num_workers": 0`** —— 防 Windows DataLoader 多进程卡顿（224px 小图，num_workers=0 不太拖速度）。*我提议过，用户暂停，未加。*
5. 跑（可续跑，`completed_pairs` 跳过已完成臂/折；预估 20 次共 6–8h）：
   ```
   python -m experiments.ab_loss_imbalance.run_ab --testbed cassava --only focal_loss:0  # 先单折试链路
   python -m experiments.ab_loss_imbalance.run_ab --testbed cassava
   python -m experiments.ab_loss_imbalance.run_ab --testbed siim_isic
   python -m experiments.ab_loss_imbalance.collect
   ```

## ⏸ 待做 B：verdict → KB 动作（Step 4，离线，跑完 A 再做）

按 `collect` 的总 verdict，照 plan §4：
- **CE_WINS**：①翻边 `focal_loss→cross_entropy_loss` 改为 `cross_entropy_loss→focal_loss`（条件不变）；②同步改 `retrieval/rag_retrieval.py` `_select_components` 硬编码 imbalance 规则；③改 golden 相应断言；④全部带溯源注释。
- **FOCAL_WINS**：KB 不动；边加实证防御注释；proposals CONFLICT 标 resolved-rejected。
- **TIE**：KB 不动；边注释"A/B 无显著差异，维持现状"；CONFLICT 标 resolved-tie。
- 任一结局后 `cd retrieval && python -m pytest test_golden.py test_rag_retrieval.py -q` 全绿。
- docs 补 5–10 行实验小结（配对差表 + verdict + 适用范围限定：均为医疗/农业不平衡，非全域）。

## ⏸ 待做 C：与 A/B 无关的遗留（kb_mining 侧，随时可做）

- **4 条 confirmed 固化为 golden 断言**：efficientnet×{fine_grained, class_imbalance}、cross_entropy×{fine_grained, medical}（分类）。unet×医疗分割那条 `test_medical_seg_small` 已覆盖。
- **CE-vs-focal CONFLICT** 就是本 A/B 要仲裁的对象——待 verdict 出来才 resolved。
- 杂项：`kb_mining_plan(2).md` 文件名带括号（可 `git mv` 到 `docs/`）；本轮所有工作**均未 commit**。

# 在 Colab 运行 MLE-STAR

最直接的实验入口是
[`mlestar_kaggle_experiments.ipynb`](../mlestar_kaggle_experiments.ipynb)：修改
`BENCHMARK_KEY` 后运行所有单元格即可。它默认只生成 DataOps 实验计划、数据清单和
审计记录；只有生成的 pipeline 已通过 OOF 与 submission schema 检查后，才应请求 Kaggle
评分。

开始前需要由你完成两项账户操作：在 Kaggle 接受目标竞赛规则，并在 Colab 的 Secrets 中保存 API token。不要把 token、`kaggle.json`、数据、checkpoint、OOF 预测或 submission 提交到 Git。

在 Colab 中选择 GPU Runtime，挂载 Google Drive，然后执行：

```bash
git clone <your-jiaozi-repository-url>
cd Jiaozi
pip install -r requirements.txt
```

在 Secrets 中只设置 `KAGGLE_API_TOKEN`，以及需要代码生成时的 `OPENAI_API_KEY`
或 `JIAOZI_DASHSCOPE_API_KEY`。将 Secret 映射到当前 runtime 环境变量后，先验证
Kaggle 身份和竞赛访问权限：

```bash
kaggle --version
kaggle competitions files plant-pathology-2020-fgvc7
```

把数据和输出放在 Drive，而非仓库目录：

```bash
export RUN_ROOT=/content/drive/MyDrive/jiaozi-runs/plant-pathology-2020
mkdir -p "$RUN_ROOT/data" "$RUN_ROOT/runs"
kaggle competitions download -c plant-pathology-2020-fgvc7 -p "$RUN_ROOT/data"
```

先用 `--plan-only` 生成并检查 DataOps DAG、数据清单和审计结果；确认任务 JSON
中的 metric、label columns、fold strategy 和 submission schema 后，再开始 GPU 训练。
正式选择必须使用固定的 `folds.parquet` 和 OOF 指标；只跑一个 fold 的结果只能作为
smoke test，不能用于最终 ensemble。

截图中的十个竞赛已经在 `benchmarks.catalog` 中有各自的 modality、metric 和
submission contract。特别注意：APTOS 必须按 QWK 的序数问题处理，Global Wheat 是检测，
Ultrasound 是 mask/RLE 分割，Denoising Dirty Documents 是 image-to-image RMSE；不能用
通用分类 accuracy 代替它们。对于已关闭的历史竞赛，保留本地 OOF 结果和 submission
schema 验证；只有 Kaggle API 实际接受时才记录新的 public score。

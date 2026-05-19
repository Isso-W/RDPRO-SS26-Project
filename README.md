# Jiaozi
<img width="1355" height="794" alt="图片" src="https://github.com/user-attachments/assets/2fb43697-3493-402c-849b-50d41078cd9d" />

饺子（暂定）
平台地址：https://jiaozi-automl-891059684610.us-west1.run.app/

Readme先用来写一些文档或者注意事项，也可以在下面留言

## 计划
目前计划将项目分成4个大部分
1. 处理用户自然语言输入 @codetraveller66
    接受用户自然语言输入，输出为字符串的list，包含模型输入，输出，模型大小，类型 等等
    1. 准备测试，测试输入  x5
    2. 选择并调用大语言模型agent
    3. 测试不同提示词效果
    4. （可选）生成用于RAG Agent的模型提示词
关键词qwen api, system message, user message, api key

2. 处理用户数据集输入
   接受用户数据集输入，输出为数据集信息，数据类型的json，（可选）数据集理解的字符串list。包含数据集大小，离群值 等等 @haoyue-chen
    1. 找找数据集，包括adult在内的3个小数据集
    2. 优先完成csv类型数据，准备json输出，包含数据列名以及数据类型，包含label列相关信息
    3. 询问可能的数据attribute，作为另一个输出，包含比如数据集大小等相关信息
    4. （可选）数据清洗
    5. （可选）结合agent理解数据集相关信息，并输出类似模块一的字符串list
    6. （可选）准备应付不同数据类型，如图像数据，表格数据等

  
3. 选择正确模型
   1. 收集资料，准备模型数据库
   2. 选择RAG或者词向量匹配，寻找最适配的模型


4. 生成模型代码
   1. 讨论一下这一块的输入
   2. 用ReAct框架构建agent
   3. 看看有没有测试代码能否运行的工具
   4. 输出training code和inference code
   5. 考虑evaluation metrics



# 周会时间 周五 德国时间/中国时间 1 p.m./ 8 p.m.
## 留言板


急急急，真得先来个项目名字了！
别来了就饺子了
饺子🥟这个名字非常好！ 非常赞成！
https://huggingface.co/models

## CV Auto-DL Codegen Submodule

This repo now includes a workflow-first implementation for the CV Auto-DL code generation submodule under `cv_autodl_agent/`.

Current scope:

- input contract: `DatasetManifest + RetrievedModelCandidate[]`
- workflow: candidate selection -> training spec -> baseline codegen -> baseline run -> ablation -> targeted refinement -> review -> notebook export
- task families: `classification`, `segmentation`, `detection`
- local fallback execution mode: `simulate`
- Colab demo execution mode: `real`
- Colab handoff: generated `train.py`, `dataset.py`, `inference.py`, `requirements.txt`, `notebook.ipynb`

Quickstart:

```bash
cd /Users/wang/Desktop/TUB2025sose/ML_project/Jiaozi
python3 -m cv_autodl_agent \
  --manifest examples/classification_manifest.json \
  --candidates examples/classification_candidates.json \
  --output-dir demo_run
```

Colab demo:

- open `examples/colab_demo.ipynb` from the repository root
- run the cells in order
- the output prints the selected model, baseline metric, refined metric, ablation winner, review status, generated notebook path, real CIFAR-10 metric, and checkpoint path
- the demo uses a small CIFAR-10 subset by default so Colab can finish quickly; increase `max_train_samples`, `max_val_samples`, and `max_epochs` in the manifest for a stronger model

Example inputs:

- `examples/classification_manifest.json`
- `examples/classification_candidates.json`
- `examples/cifar10_manifest.json`
- `examples/cifar10_candidates.json`
- `examples/food101_manifest.json`
- `examples/food101_candidates.json`
- `examples/segmentation_manifest.json`
- `examples/segmentation_candidates.json`
- `examples/detection_manifest.json`
- `examples/detection_candidates.json`

Expected demo output summary:

- `examples/expected_classification_summary.json`

Run tests:

```bash
cd /Users/wang/Desktop/TUB2025sose/ML_project/Jiaozi
python3 -m unittest discover -s tests -v
```

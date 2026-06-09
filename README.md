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

## Current Pipeline

当前主线是 `pipeline.py`，它串联 Module 1、Module 2、Module 3 和
`module4_agent`：

```text
用户自然语言 + HuggingFace 数据集 ID
→ Module 1: 解析任务类型、优先级和约束
→ Module 2: 分析数据集规模和类别分布
→ Module 3: 推荐 Top 3 CV 模型配置
→ Module 4: 生成训练、评估和推理代码
```

Integrated pipeline example:

```bash
python pipeline.py \
  --query "classify images on a small dataset" \
  --dataset uoft-cs/cifar10 \
  --fmt nl \
  --module4-output generated_pipeline \
  --module4-no-smoke
```

`pipeline.py` writes the Module 3 candidates to
`generated_pipeline/module3_candidates.json`, then asks Module 4 to generate the
training/evaluation/inference project.

## Module 4 Agent

`module4_agent/` is the active Module 4 implementation. It consumes Module 3
candidate outputs, treats structured `model_config` as the source of truth, and
generates a local runnable project:

```text
generated/
  configs.json
  generation_info.json
  utils.py
  model_utils.py
  smoke_data.py
  model.py
  train.py
  evaluate.py
  infer.py
  run.py
  run_experiments.py
  requirements.txt
  README_generated.md
  module4_summary.json
```

Standalone Module 4 usage:

```bash
python3 -m module4_agent \
  --input module4_agent/examples/sample_m3_output.json \
  --output generated/
```

Run without local smoke tests:

```bash
python3 -m module4_agent \
  --input module4_agent/examples/sample_m3_output.json \
  --output generated/ \
  --no-smoke
```

Local API keys can be stored in `.env`. Copy `.env.example` to `.env`, then fill
in your local key:

```bash
cp .env.example .env
```

The pipeline loads `.env` automatically. To use Qwen/DashScope:

```bash
JIAOZI_LLM_PROVIDER=qwen
M4_LLM_PROVIDER=qwen
JIAOZI_DASHSCOPE_API_KEY=...
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
M1_QWEN_MODEL=qwen-plus
M4_QWEN_MODEL=qwen-plus
```

For fully offline Module 4 generation, use the template fallback:

```bash
M4_LLM_PROVIDER=none
```

You can also override Module 4 from the command line:

```bash
--module4-llm-provider qwen
```

Generated projects include `generation_info.json`, which records whether
`model.py` came from Qwen or from the template fallback.

Run tests:

```bash
python3 -m unittest test_pipeline.py -v
```

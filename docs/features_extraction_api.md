# 🥟 饺子项目 (Jiaozi) - 自然语言特征提取 模块 API 文档

**负责人**: 张明达
**版本**: v1.2
**更新日期**: 2026-04-03

---


## 模块概述
本模块负责处理用户（如，实验室研究员）的自然语言输入。通过调用 Qwen 大语言模型（qwen-plus），理解并提取用户的核心意图与模型特征，最终输出为结构化的**字符串列表**，为下游 RAG Agent 去 Huggingface 检索模型提供精准的条件。

---

## 接口调用

- **接口路径**: `/api/v1/features_extraction`
- **功能描述**: 接收用户自然语言文本，返回解析后的模型特征列表。
- **内容类型**: `list`

### 1. 请求参数 (Request Body)

| 字段名 | 必选 | 类型 | 说明 |
| :--- | :---: | :--- | :--- |
| `user_message` | 是 | `string` | 用户原始自然语言输入 |

**请求示例:**
```json
{
  "user_message": "我需要一个针对MRI核磁共振图像的医学图像分割模型，最好是PyTorch实现的，输出Mask图，准确率指标要高。"
}
```
### 2. 响应参数 (Response Body)

| 字段名              | 必选 | 类型 | 说明 |
|:-----------------| :---: | :--- | :--- |
| `system_message` | 是 | `string` | 提取出的结构化特征数据 |

**响应示例:**
```json
{
  "system_message":["Domain: 医学图像", "Task: 图像分割", "Accuracy: 准确率", "Accuracy_range: 高", "is_local_train: null", "Graphics_card: null", "Input: 图片", "Output: 图片", "Size: null", "Library / Framework: PyTorch", "Input_Language: 中文", "Output_Language: 中文", "License: null"]
  }
}
```
---

## 大模型prompt
```
【身份】Huggingface模型检索专家。
【任务】从自然语言提取搜索特征。
【格式】纯字符串的list，其中元素按顺序包含以下14个维度，list每个元素中的key(维度项)均用英文，值与用户原本的输入语言保持一致(比如用户输入语言为英文，则值抓取英文原文),样式可参考：当输入语言为中文时["Domain: 生物", "Task: 文字生成文字"]，当输入语言为英文时["Domain: Biology", "Task: text to text"]；
【维度】必须审查并提取：
1.领域(Domain)
2.任务类型(Task)
3.模型准确性评估参数(Accuracy)
4.模型准确性评估参数范围(Accuracy_range)
5.是否本地训练(is_local_train)
6.显卡型号(Graphics_card)
7.是否本地训练
8.输入(Input)
9.输出(Output)
10.参数量级(Size)
11.框架(Library / Framework)
12.输入语言(Input_Language)
13.输出语言(Output_Language)
14.协议(License)
**需注意：
1.输入/输出这两个维度输出的内容仅为["文字","图片","音频","视频"]（输出内容与用户实际输入语言保持一致，比如用户输入为英文，则输入/输出两个维度的输出为["Text","Image","Audio","Video"]）；
2.若用户提出了以上任一维度的具体值，则抓取该值作为输出，若没有提出具体的值，则以null作为输出，若用户没有完整提及以上维度，则依然在list中补全所有维度，未提及的维度统一用null作为输出；
3.若提及具体的输出语言，则仅使用具体的输出语言，若无提及任何输出语言，则默认将【输出语言】的值置于"English"；
【规则】只提取用户提及或能合理推断的维度，未提及的维度直接忽略。（不准有问候语，不准有markdown符号）。
```

*注：用户输入分为【精简版】及【自定义版】，【精简版】仅包括以下维度：["领域(Domain)", "任务类型(Task)", "是否本地训练(is_local_train)", "显卡型号(Graphics_card)", "输入(Input)", "输出(Output)", "输入语言(Input_Language)", "输出语言(Output_Language)"]，【自定义版】包含所有维度，该分类须在前端交互实现，本模块仅聚焦特征识别；*

---

## 用户自然语言示例

- **case_1（标准）**: `我需要一个针对MRI核磁共振图像的医学图像分割模型，最好是PyTorch实现的，输出Mask图，准确率指标要高，语言要中文。`

      输出:["Domain: 医学图像", "Task: 图像分割", "Accuracy: 准确率", "Accuracy_range: 高", "is_local_train: null", "Graphics_card: null", "Input: 图片", "Output: 图片", "Size: null", "Library / Framework: PyTorch", "Input_Language: 中文", "Output_Language: 中文", "License: null"]

- **case_2（标准）**: `Is there a JAX implementation for satellite imagery segmentation or classification? Specifically, we are looking for a model that processes multispectral data to generate LULC (Land Use/Land Cover) maps, provided it allows for commercial use.`

      输出:["Domain: Remote Sensing", "Task: image to image", "Accuracy: null", "Accuracy_range: null", "is_local_train: null", "Graphics_card: null", "is_local_train: null", "Input: Image", "Output: Image", "Size: null", "Library / Framework: JAX", "Input_Language: English", "Output_Language: English", "License: commercial use"]

- **case_3（模糊）**: `帮我找个能做智能客服的模型，随便什么框架都行。`

      输出:["Domain: 客服", "Task: 文字生成文字", "Accuracy: null", "Accuracy_range: null", "is_local_train: null", "Graphics_card: null", "is_local_train: null", "Input: 文字", "Output: 文字", "Size: null", "Library / Framework: null", "Input_Language: null", "Output_Language: English", "License: null"]

- **case_4（行业黑话）**: `实验室要做中文NER任务，给我推荐几个SOTA的BERT变体，必须支持多语言输出。`

      输出:["Domain: 自然语言处理", "Task: 命名实体识别", "Accuracy: null", "Accuracy_range: null", "is_local_train: null", "Graphics_card: null", "is_local_train: null", "Input: 文字", "Output: 文字", "Size: null", "Library / Framework: null", "Input_Language: 中文", "Output_Language: 多语言", "License: null"]

- **case_5（口语）**: `我想找个做中文情感分析的，千万别给我推荐那种几百亿参数的LLM，跑不动，给我点轻量级的，准确率别太低就行。`

      输出:["Domain: 情感分析", "Task: 文字生成文字", "Accuracy: null", "Accuracy_range: null", "is_local_train: null", "Graphics_card: null", "is_local_train: null", "Input: 文字", "Output: 文字", "Size: 轻量级", "Library / Framework: null", "Input_Language: 中文", "Output_Language: English", "License: null"]

- **case_6（口语）**: `给金融数据做预测，主要是处理时间序列分析的那种，找个基准模型。`

      输出:["Domain: 金融", "Task: 时间序列预测", "Accuracy: null", "Accuracy_range: null", "is_local_train: null", "Graphics_card: null", "is_local_train: null", "Input: 文字", "Output: 文字", "Size: null", "Library / Framework: null", "Input_Language: Chinese", "Output_Language: English", "License: null"]




  **2026/3/28待办**：
- 前端模型参数以及范围都可以让用户选择并且提供默认值
- 准备两套用户输入，一套精简版，一套自定义版
- 加入【卡】的信息
- 字符串List仅保留英语
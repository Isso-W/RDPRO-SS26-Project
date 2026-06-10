from __future__ import annotations

import json
import os
import re
import textwrap

from env_loader import load_env_file


def _provider() -> str:
    load_env_file()
    return os.getenv("JIAOZI_LLM_PROVIDER", os.getenv("M1_LLM_PROVIDER", "qwen")).strip().lower()


def _client_for_provider(provider: str):
    from openai import OpenAI

    if provider == "openai":
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY")), os.getenv("M1_OPENAI_MODEL", "gpt-4o")
    return (
        OpenAI(
            api_key=os.getenv("JIAOZI_DASHSCOPE_API_KEY"),
            base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        ),
        os.getenv("M1_QWEN_MODEL", "qwen-plus"),
    )


def extract_model_features_api(user_message: str):
    try:
        provider = _provider()
        client, model = _client_for_provider(provider)
        system_message = textwrap.dedent('''\
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
                    【规则】只提取用户提及或能合理推断的维度，未提及的维度直接忽略。（不准有问候语，不准有markdown符号）。''').strip()
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"错误信息：{e}")

# ═══════════════════════════════════════════════════════════════════════════════
# Module 3 对接：CV 任务特征提取 + 校验
# ═══════════════════════════════════════════════════════════════════════════════

_CV_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a computer vision task analysis expert. The user will describe a CV task
    in natural language (in any language). Extract structured fields and return pure
    JSON only — no extra text, no markdown fences.

    Required fields:

    1. task_type — exactly one of:
       "classification"  (image classification, recognition, labeling)
       "object_detection" (object detection, localization, bounding boxes, counting)
       "image_segmentation" (segmentation, masks, pixel-level annotation)
       "feature_extraction" (feature extraction, embeddings, similarity retrieval)

    2. priority — exactly one of:
       "speed"    (user emphasizes fast, real-time, lightweight, low latency, efficient)
       "accuracy" (user emphasizes high accuracy, best performance, state-of-the-art)
       "balanced" (no clear preference, or user wants both)

    3. constraints — an object with the following boolean fields:
       "real_time": real-time inference needed (30fps, video stream, online inference)
       "edge_deployment": deploy on edge/mobile (phone, embedded, Raspberry Pi, Jetson)
       "class_imbalance": dataset has class imbalance (rare classes, long-tail distribution)
       "cross_modal": cross-modal capability needed (image-text alignment, text-to-image search, multimodal)
       "medical": medical imaging scenario (CT, X-ray, MRI, pathology, ultrasound)
       "zero_shot": zero-shot capability needed (no labeled data, zero-shot classification)
       "few_shot": few-shot capability needed (very few labeled samples)

    Rules:
    - Only extract what the user explicitly mentions or what can be reasonably inferred.
    - Set any unmentioned constraint to false.
    - If priority cannot be determined, set it to "balanced".
    - Output pure JSON only — no greetings, no explanations, no markdown.
""").strip()

_VALID_TASK_TYPES = {
    "classification", "object_detection", "image_segmentation", "feature_extraction",
}
_VALID_PRIORITIES = {"speed", "accuracy", "balanced"}
_CONSTRAINT_KEYS = [
    "real_time", "edge_deployment", "class_imbalance",
    "cross_modal", "medical", "zero_shot", "few_shot",
]

_TASK_TYPE_ALIASES = {
    "detection":            "object_detection",
    "det":                  "object_detection",
    "segmentation":         "image_segmentation",
    "semantic_segmentation":"image_segmentation",
    "seg":                  "image_segmentation",
    "cls":                  "classification",
    "extraction":           "feature_extraction",
    "embedding":            "feature_extraction",
    "retrieval":            "feature_extraction",
}


def _extract_cv_features(user_message: str) -> str | None:
    """调用配置的模型提取 CV 任务结构化字段，返回原始输出字符串。"""
    try:
        provider = _provider()
        client, model = _client_for_provider(provider)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CV_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"Module 1 LLM 调用失败：{e}")
        return None


def parse_module1_output(raw: str, user_message: str) -> dict:
    """
    解析 LLM 返回的 JSON 字符串，校验并补全为 Module 3 可消费的 dict。

    容错处理：
    - 去除 markdown 代码块包裹
    - enum 值不合法时回退默认值
    - 缺失字段自动补全
    - data_size 不在此处提取（由 Module 2 提供），默认留 "medium"
    """
    # 去除 markdown 代码块
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # LLM 返回了非法 JSON，用全默认值兜底——但要让用户知道结果已退化
        print(
            "[Module 1] 警告：LLM 输出不是合法 JSON，已回退到默认值 "
            "(task_type=classification, priority=balanced)。原始输出片段："
            f"{cleaned[:200]!r}"
        )
        parsed = {}

    # task_type 校验 + 别名映射（LLM 可能返回 null 或非字符串）
    task_type = str(parsed.get("task_type") or "").lower().strip()
    if task_type not in _VALID_TASK_TYPES:
        task_type = _TASK_TYPE_ALIASES.get(task_type, "classification")

    # priority 校验
    priority = str(parsed.get("priority") or "").lower().strip()
    if priority not in _VALID_PRIORITIES:
        priority = "balanced"

    # constraints 校验
    raw_constraints = parsed.get("constraints", {})
    if not isinstance(raw_constraints, dict):
        raw_constraints = {}
    constraints = {k: bool(raw_constraints.get(k, False)) for k in _CONSTRAINT_KEYS}

    return {
        "task_type":   task_type,
        "data_size":   "medium",
        "priority":    priority,
        "constraints": constraints,
        "description": user_message,
    }


def module1_pipeline(user_message: str) -> dict | None:
    """
    Module 1 入口：用户自然语言 → Module 3 可消费的结构化 dict。

    返回格式与 Module 3 的 retrieve_top3_hybrid() 输入完全对齐。
    data_size 字段预填 "medium"，待 Module 2 覆盖。
    """
    raw = _extract_cv_features(user_message)
    if raw is None:
        return None
    return parse_module1_output(raw, user_message)


if __name__ == "__main__":
    user_message = input("请输入您对于模型的要求：")

    print("\n--- 原始 14 维提取 ---")
    result = extract_model_features_api(user_message)
    print(result)

    print("\n--- Module 3 对接输出 ---")
    m3_input = module1_pipeline(user_message)
    print(json.dumps(m3_input, indent=2, ensure_ascii=False))

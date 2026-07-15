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
                    Role: Hugging Face model search assistant.
                    Task: extract search features from a natural-language request.
                    Format: return a plain Python-style list of strings. The keys must be
                    English. Values should keep the user's original language when possible.
                    Example for English input: ["Domain: Biology", "Task: text to text"].

                    Review and extract these 14 fields, in order:
                        1. Domain
                        2. Task
                        3. Accuracy
                        4. Accuracy_range
                        5. is_local_train
                        6. Graphics_card
                        7. local_training
                        8. Input
                        9. Output
                        10. Size
                        11. Library / Framework
                        12. Input_Language
                        13. Output_Language
                        14. License

                    Notes:
                    1. Input and Output should use only Text, Image, Audio, or Video,
                       translated to the user's language when the request is not in English.
                    2. If the user gives a value for a field, extract it. If a field is not
                       mentioned, include it with null.
                    3. If no output language is mentioned, set Output_Language to English.
                    4. Return only the list. Do not add greetings, markdown, or commentary.''').strip()
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message}
            ]
        )
        try:
            import cost_meter
            cost_meter.record_llm_call(cost_meter.tokens_from_response(completion))
        except Exception:
            pass
        return completion.choices[0].message.content
    except Exception as e:
        print(f"[Error] {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# Module 3 adapter: CV task extraction and validation.
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

    4. evaluation_metric — exactly one of (how the model will be scored). Set it only when
       the user mentions or clearly implies a metric; otherwise use "accuracy":
       "accuracy"  (plain correctness)
       "macro_f1"  (macro / per-class F1; good for imbalanced classification)
       "roc_auc"   (ROC AUC; binary scoring, "AUC")
       "qwk"       (quadratic weighted kappa; ordinal grading / severity levels)
       "log_loss"  (cross-entropy / log loss; probability-calibrated scoring)

    Rules:
    - Only extract what the user explicitly mentions or what can be reasonably inferred.
    - Set any unmentioned constraint to false.
    - If priority cannot be determined, set it to "balanced".
    - If no metric is mentioned, set evaluation_metric to "accuracy".
    - Output pure JSON only — no greetings, no explanations, no markdown.
""").strip()

_VALID_TASK_TYPES = {
    "classification", "object_detection", "image_segmentation", "feature_extraction",
}
_VALID_PRIORITIES = {"speed", "accuracy", "balanced"}
_VALID_METRICS = {"accuracy", "macro_f1", "roc_auc", "qwk", "log_loss"}
_METRIC_ALIASES = {
    "auc": "roc_auc", "roc": "roc_auc", "auroc": "roc_auc",
    "f1": "macro_f1", "macro-f1": "macro_f1", "macro f1": "macro_f1", "f1_score": "macro_f1",
    "kappa": "qwk", "cohen_kappa": "qwk", "quadratic_weighted_kappa": "qwk",
    "logloss": "log_loss", "cross_entropy": "log_loss", "multiclass_log_loss": "log_loss",
}
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
    """Call the configured model and return the raw structured-field output."""
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
        try:
            import cost_meter
            cost_meter.record_llm_call(cost_meter.tokens_from_response(completion))
        except Exception:
            pass
        return completion.choices[0].message.content
    except Exception as e:
        print(f"[Module 1] LLM call failed: {e}")
        return None


def parse_module1_output(raw: str, user_message: str) -> dict:
    """
    Parse the LLM JSON response into the dictionary consumed by Module 3.

    Tolerates common response issues:
    - strips markdown code fences
    - falls back when enum values are invalid
    - fills missing fields
    - leaves data_size at "medium"; Module 2 overwrites it later
    """
    # Strip common markdown code fences.
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Keep the pipeline runnable, but make the degraded parse visible.
        print(
            "[Module 1] Warning: LLM output is not valid JSON, falling back to defaults "
            f"(task_type=classification, priority=balanced). Raw output snippet: {cleaned[:200]!r}"
        )
        parsed = {}

    # Validate task_type and handle common aliases.
    task_type = str(parsed.get("task_type") or "").lower().strip()
    if task_type not in _VALID_TASK_TYPES:
        task_type = _TASK_TYPE_ALIASES.get(task_type, "classification")

    # Validate priority.
    priority = str(parsed.get("priority") or "").lower().strip()
    if priority not in _VALID_PRIORITIES:
        priority = "balanced"

    # Validate constraints.
    raw_constraints = parsed.get("constraints", {})
    if not isinstance(raw_constraints, dict):
        raw_constraints = {}
    constraints = {k: bool(raw_constraints.get(k, False)) for k in _CONSTRAINT_KEYS}

    # Validate evaluation_metric and handle aliases.
    metric = str(parsed.get("evaluation_metric") or "").lower().strip()
    metric = _METRIC_ALIASES.get(metric, metric)
    if metric not in _VALID_METRICS:
        metric = "accuracy"

    return {
        "task_type":         task_type,
        "data_size":         "medium",
        "priority":          priority,
        "constraints":       constraints,
        "evaluation_metric": metric,
        "description":       user_message,
    }


def module1_pipeline(user_message: str) -> dict | None:
    """
    Module 1 entry point: user text to Module 3 structured input.

    The returned format matches retrieve_top3_hybrid(). data_size starts as
    "medium" and is overwritten by Module 2.
    """
    raw = _extract_cv_features(user_message)
    if raw is None:
        return None
    return parse_module1_output(raw, user_message)


if __name__ == "__main__":
    user_message = input("Enter your model requirements: ")

    print("\n--- Raw 14-dimension extraction ---")
    result = extract_model_features_api(user_message)
    print(result)

    print("\n--- Module 3 compatible output ---")
    m3_input = module1_pipeline(user_message)
    print(json.dumps(m3_input, indent=2, ensure_ascii=False))

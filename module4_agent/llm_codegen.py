"""
LLM 代码生成层 — 通过环境变量切换 provider。

支持的 provider（通过 M4_LLM_PROVIDER 环境变量）：
  - "none"     — 跳过 LLM，回退到模板生成（默认）
  - "qwen"     — 阿里 DashScope（需要 JIAOZI_DASHSCOPE_API_KEY）
  - "openai"   — OpenAI（需要 OPENAI_API_KEY）
  - "vertex"   — Google Vertex AI Gemini（需要 GOOGLE_APPLICATION_CREDENTIALS）
"""

from __future__ import annotations

import ast
import json
import os
import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from env_loader import load_env_file

from .schemas import TrainingSpec


_LAST_GENERATION_ERROR = ""


def _set_generation_error(message: str) -> None:
    global _LAST_GENERATION_ERROR
    _LAST_GENERATION_ERROR = message.strip()


def get_last_generation_error() -> str:
    """Return the latest provider or generated-code validation failure."""

    return _LAST_GENERATION_ERROR


def _record_llm_cost(response) -> None:
    """Best-effort: record this LLM call + its tokens in the process cost meter."""
    try:
        import cost_meter
        cost_meter.record_llm_call(cost_meter.tokens_from_response(response))
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# Provider 抽象
# ═══════════════════════════════════════════════════════════════════════════════

def _call_llm(system_prompt: str, user_prompt: str, provider: str | None = None) -> str | None:
    """调用 LLM，返回原始文本。失败返回 None。

    provider 为 None 时回退到 M4_LLM_PROVIDER 环境变量。"""
    provider = (provider or get_provider()).strip().lower()

    if provider == "none":
        return None

    if provider == "openai":
        return _call_openai(system_prompt, user_prompt)
    if provider == "vertex":
        return _call_vertex(system_prompt, user_prompt)
    if provider == "qwen":
        return _call_qwen(system_prompt, user_prompt)
    return None


def get_provider() -> str:
    """Return the configured model.py generation provider."""

    load_env_file()
    return os.environ.get("M4_LLM_PROVIDER", "none").strip().lower()


def _response_text(response: Any) -> str | None:
    """Extract text from OpenAI-compatible SDK, dict, or direct-string responses."""

    if response is None:
        return None
    if isinstance(response, bytes):
        response = response.decode("utf-8", errors="replace")
    if isinstance(response, str):
        text = response.strip()
        if not text:
            return None
        try:
            decoded = json.loads(text)
        except (json.JSONDecodeError, TypeError):
            return text
        if decoded == response:
            return text
        return _response_text(decoded)
    if isinstance(response, list):
        parts = [_response_text(item) for item in response]
        joined = "".join(part for part in parts if part)
        return joined or None
    if isinstance(response, dict):
        for key in ("output_text", "text", "content"):
            extracted = _response_text(response.get(key))
            if extracted:
                return extracted
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            extracted = _response_text(choices[0])
            if extracted:
                return extracted
        output = response.get("output")
        extracted = _response_text(output)
        if extracted:
            return extracted
        message = response.get("message")
        extracted = _response_text(message)
        if extracted:
            return extracted
        return None

    output_text = getattr(response, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    choices = getattr(response, "choices", None)
    if choices:
        extracted = _response_text(choices[0])
        if extracted:
            return extracted

    message = getattr(response, "message", None)
    extracted = _response_text(message)
    if extracted:
        return extracted

    content = getattr(response, "content", None)
    extracted = _response_text(content)
    if extracted:
        return extracted

    output = getattr(response, "output", None)
    extracted = _response_text(output)
    if extracted:
        return extracted

    model_dump = getattr(response, "model_dump", None)
    if callable(model_dump):
        try:
            return _response_text(model_dump())
        except Exception:
            return None
    return None


def _call_qwen(system_prompt: str, user_prompt: str) -> str | None:
    try:
        load_env_file()
        from openai import OpenAI
        client = OpenAI(
            api_key=os.environ["JIAOZI_DASHSCOPE_API_KEY"],
            base_url=os.environ.get("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        )
        resp = client.chat.completions.create(
            model=os.environ.get("M4_QWEN_MODEL", "qwen-plus"),
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        _record_llm_cost(resp)
        return _response_text(resp)
    except Exception as e:
        print(f"[LLM] Qwen call failed: {e}")
        return None


def _call_openai(system_prompt: str, user_prompt: str) -> str | None:
    try:
        load_env_file()
        from openai import OpenAI
        client_kwargs = {"api_key": os.environ["OPENAI_API_KEY"]}
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
        wire_api = os.environ.get("M4_OPENAI_WIRE_API", "chat_completions").strip().lower()
        if base_url:
            client_kwargs["base_url"] = _normalize_openai_base_url(base_url, wire_api)
        client = OpenAI(**client_kwargs)
        model = os.environ.get("M4_OPENAI_MODEL", "gpt-4o")
        if wire_api in {"responses", "response"}:
            resp = client.responses.create(
                model=model,
                instructions=system_prompt,
                input=user_prompt,
            )
        else:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        _record_llm_cost(resp)
        text = _response_text(resp)
        if not text:
            raise ValueError(f"Unsupported empty response type: {type(resp).__name__}")
        return text
    except Exception as e:
        _set_generation_error(f"OpenAI call failed: {e}")
        print(f"[LLM] {_LAST_GENERATION_ERROR}")
        return None


def _normalize_openai_base_url(base_url: str, wire_api: str) -> str:
    """Normalize custom endpoints for the OpenAI SDK.

    Responses-compatible gateways such as Codex providers commonly expose
    ``/responses`` at their configured root. Chat Completions gateways usually
    expose ``/v1/chat/completions``, so a bare origin receives ``/v1``.
    """

    cleaned = base_url.strip().rstrip("/")
    parsed = urlsplit(cleaned)
    if (
        wire_api not in {"responses", "response"}
        and parsed.scheme in {"http", "https"}
        and parsed.netloc
        and parsed.path in {"", "/"}
    ):
        parsed = parsed._replace(path="/v1")
        return urlunsplit(parsed)
    return cleaned


def _call_vertex(system_prompt: str, user_prompt: str) -> str | None:
    try:
        load_env_file()
        import vertexai
        from vertexai.generative_models import GenerationConfig, GenerativeModel

        project = os.environ.get("M4_VERTEX_PROJECT", "my-agent-498201")
        location = os.environ.get("M4_VERTEX_LOCATION", "us-central1")
        model_name = os.environ.get("M4_VERTEX_MODEL", "gemini-2.0-flash")

        vertexai.init(project=project, location=location)
        model = GenerativeModel(model_name, system_instruction=system_prompt)
        resp = model.generate_content(user_prompt, generation_config=GenerationConfig(temperature=0.0))
        return resp.text
    except Exception as e:
        print(f"[LLM] Vertex AI call failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 输出清洗
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_python(raw: str) -> str:
    """从 LLM 输出中提取纯 Python 代码，去除 markdown 包裹。"""
    match = re.search(r"```python\s*\n(.*?)```", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    # 没有 markdown 包裹，去除可能的 ``` 行
    cleaned = re.sub(r"```\w*\s*", "", raw).strip().rstrip("`")
    return cleaned


def _validate_model_python(source: str) -> str | None:
    """Return a reason when provider output is not a usable model.py."""

    stripped = source.lstrip()
    lowered = stripped[:500].lower()
    if any(
        marker in lowered
        for marker in (
            "<!doctype html",
            "<html",
            "<head",
            "<body",
            "access denied",
            "cloudflare",
        )
    ):
        return "provider returned an HTML or gateway error page instead of Python"
    try:
        tree = ast.parse(source, filename="model.py")
    except SyntaxError as exc:
        return f"provider returned invalid Python: {exc}"
    if not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "build_model"
        for node in ast.walk(tree)
    ):
        return "provider output does not define required build_model(config)"
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt 模板
# ═══════════════════════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = """\
You are a PyTorch code generator. You produce clean, runnable Python files.
Rules:
- Output ONLY pure Python code, no explanations, no markdown fences.
- Use standard PyTorch, torchvision, and transformers libraries.
- All functions must have type hints.
- Do not include if __name__ == "__main__" blocks.
- Do not add comments explaining what the code does.
- task_type values are EXACTLY: "classification", "object_detection", "image_segmentation", "feature_extraction". Never use abbreviations like "detection" or "segmentation".
"""

_MODEL_PY_PROMPT = """\
Generate `model.py` for the following spec:

- task_type: {task_type}
- backbone: {backbone}
- head: {head}
- num_classes: 3 (configurable via config dict)

IMPORTANT — you MUST use these imports and helpers. Do NOT load models yourself:

```
from model_utils import load_backbone, apply_freeze
from utils import get_value, as_int, as_bool, task_type
```

Helper signatures (always pass `default`):
- `load_backbone(config) -> tuple[nn.Module, int]` — returns (backbone, out_channels). Backbone outputs spatial features [B, C, H', W']. Never load torchvision/HF models yourself.
- `apply_freeze(model, config)` — freezes backbone params per finetune_strategy. Call this on the final model before returning.
- `get_value(config, key, default)`, `as_int(value, default)`, `as_bool(value, default)`, `task_type(config) -> str`

Required export: `build_model(config: dict) -> nn.Module`

Steps:
1. Call `backbone, ch = load_backbone(config)` to get the backbone.
2. Build an nn.Module subclass that wraps backbone + a task-specific head.
3. The backbone outputs [B, ch, H', W']. Build a head that maps this to the task output.
   If backbone output is already 2D [B, D] (transformer backbones), skip pooling/flatten.
4. Call `apply_freeze(model, config)` before returning.

CRITICAL — forward() return types (the template train.py and evaluate.py depend on these exact types):
- classification: forward(x) returns a BARE TENSOR [B, num_classes]. NO dict, NO targets parameter. Just `return logits`.
- object_detection: forward(x, targets=None) returns a dict with "pred_boxes" [B,1,4], "pred_logits" [B,1,num_classes]. If targets is not None, also include "loss" (L1 boxes + CrossEntropy classes).
- image_segmentation: forward(x) returns a BARE TENSOR [B, num_classes, H, W]. Use Conv2d(ch, num_classes, 1) + F.interpolate to input spatial size. NO dict.
- feature_extraction: forward(x) returns a BARE TENSOR [B, embedding_dim], L2-normalized. NO dict.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# 公开接口
# ═══════════════════════════════════════════════════════════════════════════════

def generate_model_py(spec: TrainingSpec, feedback: str = "", provider: str | None = None) -> str | None:
    """用 LLM 生成 model.py（使用 model_utils helper）。失败返回 None。"""
    _set_generation_error("")
    prompt = _MODEL_PY_PROMPT.format(
        task_type=spec.task_type,
        backbone=spec.backbone,
        head=spec.head,
    )
    if feedback:
        prompt += f"\n\nPrevious attempt failed with this feedback:\n{feedback}\nFix the issues."
    raw = _call_llm(_SYSTEM_PROMPT, prompt, provider=provider)
    if not raw:
        if not get_last_generation_error():
            _set_generation_error("provider returned no content")
        return None
    source = _extract_python(raw)
    invalid_reason = _validate_model_python(source)
    if invalid_reason:
        _set_generation_error(invalid_reason)
        print(f"[LLM] Rejected provider output: {invalid_reason}. Using template fallback.")
        return None
    return source

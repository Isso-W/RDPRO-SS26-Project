import sys
from types import ModuleType
from types import SimpleNamespace

from module4_agent.llm_codegen import (
    _call_openai,
    _normalize_openai_base_url,
    _response_text,
    generate_model_py,
    get_last_generation_error,
)
from module4_agent.schemas import TrainingSpec


def test_response_text_accepts_direct_string():
    assert _response_text("print('generated')") == "print('generated')"


def test_response_text_accepts_json_encoded_string():
    payload = '{"choices":[{"message":{"content":"print(\\"generated\\")"}}]}'

    assert _response_text(payload) == 'print("generated")'


def test_response_text_accepts_chat_completion_object():
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content="print('chat')"),
            )
        ]
    )

    assert _response_text(response) == "print('chat')"


def test_response_text_accepts_responses_api_object():
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                content=[
                    SimpleNamespace(
                        type="output_text",
                        text="print('responses')",
                    )
                ]
            )
        ]
    )

    assert _response_text(response) == "print('responses')"


def test_call_openai_accepts_direct_string_from_compatible_endpoint(monkeypatch):
    calls = {}

    class FakeCompletions:
        def create(self, **kwargs):
            calls["request"] = kwargs
            return "print('direct')"

    class FakeOpenAI:
        def __init__(self, **kwargs):
            calls["client"] = kwargs
            self.chat = SimpleNamespace(completions=FakeCompletions())

    fake_module = ModuleType("openai")
    fake_module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.test")
    monkeypatch.setenv("M4_OPENAI_MODEL", "gpt-5.5")
    monkeypatch.delenv("M4_OPENAI_WIRE_API", raising=False)

    assert _call_openai("system", "user") == "print('direct')"
    assert calls["client"]["base_url"] == "https://example.test/v1"
    assert calls["request"]["model"] == "gpt-5.5"


def test_call_openai_supports_responses_wire_api(monkeypatch):
    calls = {}

    class FakeResponses:
        def create(self, **kwargs):
            calls["request"] = kwargs
            return SimpleNamespace(output_text="print('responses')")

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.responses = FakeResponses()

    fake_module = ModuleType("openai")
    fake_module.OpenAI = FakeOpenAI
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("M4_OPENAI_MODEL", "gpt-5.5")
    monkeypatch.setenv("M4_OPENAI_WIRE_API", "responses")

    assert _call_openai("system", "user") == "print('responses')"
    assert calls["request"] == {
        "model": "gpt-5.5",
        "instructions": "system",
        "input": "user",
    }


def test_responses_wire_keeps_gateway_root():
    assert (
        _normalize_openai_base_url("https://yybb.codes", "responses")
        == "https://yybb.codes"
    )


def test_generate_model_rejects_html(monkeypatch):
    monkeypatch.setattr(
        "module4_agent.llm_codegen._call_llm",
        lambda *_args, **_kwargs: "<!doctype html><html><body>Login</body></html>",
    )

    assert generate_model_py(TrainingSpec(), provider="openai") is None
    assert "HTML" in get_last_generation_error()


def test_generate_model_rejects_invalid_python(monkeypatch):
    monkeypatch.setattr(
        "module4_agent.llm_codegen._call_llm",
        lambda *_args, **_kwargs: "this is not valid python",
    )

    assert generate_model_py(TrainingSpec(), provider="openai") is None
    assert "invalid Python" in get_last_generation_error()


def test_generate_model_accepts_valid_build_model(monkeypatch):
    monkeypatch.setattr(
        "module4_agent.llm_codegen._call_llm",
        lambda *_args, **_kwargs: (
            "from torch import nn\n"
            "def build_model(config: dict) -> nn.Module:\n"
            "    return nn.Identity()\n"
        ),
    )

    generated = generate_model_py(TrainingSpec(), provider="openai")

    assert generated is not None
    assert "def build_model" in generated
    assert get_last_generation_error() == ""


def test_chat_completion_retries_without_temperature():
    """Models that reject temperature=0 (e.g. gpt-5.x) should be retried without it."""
    from module4_agent.llm_codegen import _chat_completion

    class _Completions:
        def __init__(self):
            self.calls = []

        def create(self, **kwargs):
            self.calls.append(kwargs)
            if "temperature" in kwargs:
                raise RuntimeError("Error code: 400 - 'temperature' does not support 0 with this model")
            return "ok"

    class _Client:
        def __init__(self):
            self.chat = type("C", (), {"completions": _Completions()})()

    client = _Client()
    assert _chat_completion(client, "gpt-5.5", "sys", "usr") == "ok"
    assert len(client.chat.completions.calls) == 2  # with temperature, then without


def test_chat_completion_propagates_other_errors():
    from module4_agent.llm_codegen import _chat_completion

    class _Completions:
        def create(self, **kwargs):
            raise RuntimeError("401 authentication error")

    class _Client:
        def __init__(self):
            self.chat = type("C", (), {"completions": _Completions()})()

    import pytest
    with pytest.raises(RuntimeError, match="authentication"):
        _chat_completion(_Client(), "m", "s", "u")


def test_generate_model_py_self_corrects(monkeypatch):
    """Invalid output is fed back and corrected before giving up."""
    import module4_agent.llm_codegen as llm

    valid = "from model_utils import load_backbone, apply_freeze\ndef build_model(config):\n    return None\n"
    invalid = "def not_build_model():\n    pass\n"
    seq = [invalid, valid]
    monkeypatch.setattr(llm, "_call_llm", lambda *a, **k: seq.pop(0))
    out = generate_model_py(TrainingSpec(), provider="openai", max_attempts=2)
    assert out is not None and "build_model" in out
    assert seq == []  # both attempts used (1 reject + 1 success)


def test_generate_model_py_transport_failure_no_retry(monkeypatch):
    """A transport failure (no content) falls back immediately without retrying."""
    import module4_agent.llm_codegen as llm

    calls = []

    def _fake(*args, **kwargs):
        calls.append(1)
        return None

    monkeypatch.setattr(llm, "_call_llm", _fake)
    assert generate_model_py(TrainingSpec(), provider="openai", max_attempts=3) is None
    assert len(calls) == 1


def test_generate_model_py_falls_back_after_exhausting_attempts(monkeypatch):
    import module4_agent.llm_codegen as llm

    # valid Python but no build_model -> rejected every attempt -> template fallback
    monkeypatch.setattr(llm, "_call_llm", lambda *a, **k: "def other():\n    pass\n")
    assert generate_model_py(TrainingSpec(), provider="openai", max_attempts=2) is None
    assert "build_model" in get_last_generation_error()

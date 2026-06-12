import sys
from types import ModuleType
from types import SimpleNamespace

from module4_agent.llm_codegen import _call_openai, _response_text


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
    assert calls["client"]["base_url"] == "https://example.test"
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

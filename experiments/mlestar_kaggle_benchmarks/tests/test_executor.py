from __future__ import annotations

import json
import sys

import pytest

from mlestar.executor import execute_adapter, sanitized_child_environment


def test_sanitized_environment_removes_kaggle_openai_and_provider_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KAGGLE_API_TOKEN", "kaggle-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "provider-secret")
    monkeypatch.setenv("MLESTAR_SAFE_SETTING", "keep-me")

    environment = sanitized_child_environment({"CUSTOM_PROVIDER_TOKEN": "also-secret"})

    assert "KAGGLE_API_TOKEN" not in environment
    assert "OPENAI_API_KEY" not in environment
    assert "ANTHROPIC_API_KEY" not in environment
    assert "CUSTOM_PROVIDER_TOKEN" not in environment
    assert environment["MLESTAR_SAFE_SETTING"] == "keep-me"


def test_execute_adapter_child_observes_no_credentials(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KAGGLE_API_TOKEN", "kaggle-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-secret")
    script = (
        "import json, os; "
        "print(json.dumps({name: os.getenv(name) for name in "
        "['KAGGLE_API_TOKEN', 'OPENAI_API_KEY', 'CUSTOM_PROVIDER_API_KEY']}))"
    )

    result = execute_adapter(
        [sys.executable, "-c", script],
        tmp_path,
        environment={"CUSTOM_PROVIDER_API_KEY": "provider-secret"},
        check=True,
    )

    assert json.loads(result.stdout) == {
        "KAGGLE_API_TOKEN": None,
        "OPENAI_API_KEY": None,
        "CUSTOM_PROVIDER_API_KEY": None,
    }


def test_execute_adapter_rejects_a_shell_command(tmp_path) -> None:
    with pytest.raises(TypeError, match="sequence"):
        execute_adapter("echo unsafe", tmp_path)  # type: ignore[arg-type]

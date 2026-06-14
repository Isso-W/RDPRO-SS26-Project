import asyncio
import io
from types import SimpleNamespace
from typing import get_type_hints

from agents import mcp_client
from agents.mcp_client import call_tool, mcp_session, result_value
from mcp_server.server import extract_strategy_cards


EXPECTED_TOOLS = {
    "ingest_external_source",
    "summarize_source",
    "extract_strategy_cards",
    "upsert_strategy_card",
    "search_strategy_cards",
    "get_past_experiments",
    "generate_experiment_configs",
    "run_experiment",
    "read_metrics",
    "compare_results",
    "write_experiment_result",
    "generate_experiment_report",
}


def test_stdio_error_stream_uses_real_file_descriptor_in_notebook(monkeypatch):
    class NotebookStream:
        def fileno(self):
            raise io.UnsupportedOperation("fileno")

    monkeypatch.setattr(mcp_client.sys, "stderr", NotebookStream())

    with mcp_client._stdio_error_stream() as stream:
        assert isinstance(stream.fileno(), int)


def test_result_value_decodes_string_structured_content():
    result = SimpleNamespace(
        structuredContent='{"status": "success"}',
        content=[],
    )

    assert result_value(result) == {"status": "success"}


def test_result_value_unwraps_string_result_envelope():
    result = SimpleNamespace(
        structuredContent={"result": '{"status": "success"}'},
        content=[],
    )

    assert result_value(result) == {"status": "success"}


def test_result_value_raises_for_tool_errors():
    result = SimpleNamespace(
        isError=True,
        structuredContent=None,
        content=[SimpleNamespace(text="invalid arguments")],
    )

    try:
        result_value(result)
    except RuntimeError as exc:
        assert str(exc) == "invalid arguments"
    else:
        raise AssertionError("Expected MCP tool error to be raised")


def test_extract_strategy_cards_output_contract_is_object():
    assert get_type_hints(extract_strategy_cards)["return"] is dict


def test_stdio_server_exposes_all_tools_and_calls_compare(tmp_path, monkeypatch):
    monkeypatch.setenv("JIAOZI_KB_ROOT", str(tmp_path / "kb"))
    monkeypatch.setenv("JIAOZI_WORKSPACE_ROOT", str(tmp_path / "workspace"))

    async def exercise():
        async with mcp_session() as session:
            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            assert names == EXPECTED_TOOLS
            result = await call_tool(
                session,
                "compare_results",
                {
                    "baseline_metrics": {"metric_value": 1.0},
                    "experiment_metrics": [
                        {"experiment_name": "better", "metric_value": 0.8}
                    ],
                    "target_metric": "log_loss",
                },
            )
            assert result["best_experiment"] == "better"

    asyncio.run(exercise())

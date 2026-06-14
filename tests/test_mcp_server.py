import asyncio

from agents.mcp_client import call_tool, mcp_session


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

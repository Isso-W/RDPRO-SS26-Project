"""Official FastMCP stdio server exposing the Jiaozi low-token loop."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from .context import build_context
from .tools.experiment_tools import (
    compare_results_service,
    generate_experiment_configs_service,
    get_past_experiments_service,
    read_metrics_service,
    run_experiment_service,
    write_experiment_result_service,
)
from .tools.knowledge_tools import (
    extract_strategy_cards_service,
    ingest_external_source_service,
    search_strategy_cards_service,
    summarize_source_service,
    upsert_strategy_card_service,
)
from .tools.report_tools import generate_experiment_report_service


app_context = build_context()
mcp = FastMCP("jiaozi-mle")


@mcp.tool()
def ingest_external_source(source_name: str, url: str, source_type: str = "html") -> dict:
    return ingest_external_source_service(app_context, source_name, url, source_type)


@mcp.tool()
def summarize_source(source_id: str) -> dict:
    return summarize_source_service(app_context, source_id)


@mcp.tool()
def extract_strategy_cards(
    source_summary_id: str,
    domain: str = "fine_grained_classification",
) -> list[dict]:
    return extract_strategy_cards_service(app_context, source_summary_id, domain=domain)


@mcp.tool()
def upsert_strategy_card(card: dict) -> dict:
    return upsert_strategy_card_service(app_context, card)


@mcp.tool()
def search_strategy_cards(
    query: str,
    task_type: str = "classification",
    domain: str = "fine_grained_classification",
    target_metric: str = "log_loss",
    top_k: int = 5,
) -> list[dict]:
    return search_strategy_cards_service(
        app_context, query, task_type, domain, target_metric, top_k
    )


@mcp.tool()
def get_past_experiments(dataset_id: str, top_k: int = 10) -> list[dict]:
    return get_past_experiments_service(app_context, dataset_id, top_k)


@mcp.tool()
def generate_experiment_configs(
    baseline_config: dict,
    strategy_cards: list[dict],
    past_experiments: list[dict] | None = None,
    max_experiments: int = 3,
    max_changed_variables: int = 2,
) -> list[dict]:
    return generate_experiment_configs_service(
        app_context,
        baseline_config,
        strategy_cards,
        past_experiments,
        max_experiments,
        max_changed_variables,
    )


@mcp.tool()
def run_experiment(project_dir: str, experiment_name: str, config: dict) -> dict:
    return run_experiment_service(app_context, project_dir, experiment_name, config)


@mcp.tool()
def read_metrics(run_result: dict) -> dict:
    return read_metrics_service(run_result)


@mcp.tool()
def compare_results(
    baseline_metrics: dict,
    experiment_metrics: list[dict],
    target_metric: str = "log_loss",
) -> dict:
    return compare_results_service(baseline_metrics, experiment_metrics, target_metric)


@mcp.tool()
def write_experiment_result(
    dataset_id: str,
    fingerprint: dict,
    proposal: dict,
    metrics: dict,
    baseline_metric: float | None = None,
) -> dict:
    return write_experiment_result_service(
        app_context, dataset_id, fingerprint, proposal, metrics, baseline_metric
    )


@mcp.tool()
def generate_experiment_report(report: dict, output_path: str) -> dict:
    return generate_experiment_report_service(app_context, report, output_path)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

"""Stage 1: learn compact strategy cards from the fixed Dog Breed source package."""

from __future__ import annotations

import argparse
import asyncio
import json

from knowledge.sources import DOG_BREED_SOURCES

from .mcp_client import call_tool, mcp_session


async def learn_fixed_sources(min_successful_sources: int = 3) -> dict:
    source_results = []
    cards = []
    llm_tokens = 0
    successful = 0
    async with mcp_session() as session:
        for source in DOG_BREED_SOURCES:
            ingested = await call_tool(session, "ingest_external_source", source)
            source_results.append(ingested)
            if not ingested or ingested.get("status") != "success":
                continue
            try:
                summary = await call_tool(
                    session,
                    "summarize_source",
                    {"source_id": ingested["source"]["id"]},
                )
                llm_tokens += int(summary.pop("_llm_tokens", 0) or 0)
                extracted = await call_tool(
                    session,
                    "extract_strategy_cards",
                    {"source_summary_id": summary["id"]},
                )
                llm_tokens += int((extracted or {}).get("_llm_tokens", 0) or 0)
                for card in (extracted or {}).get("cards", []):
                    cards.append(await call_tool(session, "upsert_strategy_card", {"card": card}))
                successful += 1
            except Exception as exc:
                ingested["status"] = "failed"
                ingested["processing_error"] = str(exc)
    if successful < min_successful_sources:
        raise RuntimeError(
            f"Only {successful} fixed sources succeeded; at least {min_successful_sources} are required."
        )
    return {
        "successful_sources": successful,
        "failed_sources": len(source_results) - successful,
        "strategy_cards": cards,
        "source_results": source_results,
        "llm_calls": successful * 2,
        "llm_tokens": llm_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Learn Dog Breed strategy cards through MCP.")
    parser.add_argument("--min-sources", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(asyncio.run(learn_fixed_sources(args.min_sources)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

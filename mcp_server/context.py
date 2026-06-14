"""Runtime dependencies shared by MCP tools."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from knowledge import KnowledgeStore
from recommender import OutcomeMemory


@dataclass
class AppContext:
    store: KnowledgeStore
    memory: OutcomeMemory
    workspace_root: Path


def build_context() -> AppContext:
    store = KnowledgeStore()
    store.bootstrap_packaged_cards()
    workspace = Path(os.getenv("JIAOZI_WORKSPACE_ROOT", "workspace")).expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    memory_path = os.getenv(
        "JIAOZI_OUTCOME_MEMORY",
        str(store.experiments / "outcomes.jsonl"),
    )
    return AppContext(
        store=store,
        memory=OutcomeMemory(memory_path),
        workspace_root=workspace,
    )

"""Canonical JSON storage plus a rebuildable SQLite FTS strategy index."""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import shutil
from pathlib import Path
from typing import Any

from .schemas import SourceRecord, SourceSummary, StrategyCard


class KnowledgeStore:
    def __init__(self, root: str | Path | None = None):
        self.root = Path(root or os.getenv("JIAOZI_KB_ROOT", "knowledge_base")).expanduser()
        self.raw_sources = self.root / "raw_sources"
        self.source_summaries = self.root / "source_summaries"
        self.strategy_cards = self.root / "strategy_cards"
        self.experiments = self.root / "experiments"
        self.index_dir = self.root / "index"
        self.index_path = self.index_dir / "strategy_cards.sqlite3"
        for path in (
            self.raw_sources,
            self.source_summaries,
            self.strategy_cards,
            self.experiments,
            self.index_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def bootstrap_packaged_cards(self) -> int:
        """Copy repository seed cards into an empty external/Drive knowledge root."""
        packaged = Path(__file__).resolve().parent.parent / "knowledge_base" / "strategy_cards"
        if not packaged.is_dir() or packaged.resolve() == self.strategy_cards.resolve():
            return 0
        copied = 0
        for source in packaged.glob("*.json"):
            destination = self.strategy_cards / source.name
            if not destination.exists():
                shutil.copy2(source, destination)
                copied += 1
        if copied:
            self.rebuild_index()
        return copied

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def save_source(self, source: SourceRecord) -> Path:
        path = self.raw_sources / f"{source.id}.json"
        self._atomic_json(path, source.to_dict())
        return path

    def save_summary(self, summary: SourceSummary) -> Path:
        path = self.source_summaries / f"{summary.id}.json"
        self._atomic_json(path, summary.to_dict())
        return path

    def get_card(self, card_id: str) -> StrategyCard | None:
        path = self.strategy_cards / f"{card_id}.json"
        if not path.exists():
            return None
        return StrategyCard.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def all_cards(self) -> list[StrategyCard]:
        cards = []
        for path in sorted(self.strategy_cards.glob("*.json")):
            cards.append(StrategyCard.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        return cards

    def upsert_card(self, card: StrategyCard) -> StrategyCard:
        existing = self.get_card(card.id)
        if existing is None:
            for candidate in self.all_cards():
                if (
                    candidate.task_type == card.task_type
                    and candidate.domain == card.domain
                    and candidate.strategy_name.casefold() == card.strategy_name.casefold()
                ):
                    existing = candidate
                    card.id = candidate.id
                    break
        if existing is not None:
            seen = {(item.source_id, item.note, item.url) for item in existing.evidence}
            for evidence in card.evidence:
                key = (evidence.source_id, evidence.note, evidence.url)
                if key not in seen:
                    existing.evidence.append(evidence)
                    seen.add(key)
            existing.summary = card.summary or existing.summary
            existing.use_when = list(dict.fromkeys(existing.use_when + card.use_when))
            existing.avoid_when = list(dict.fromkeys(existing.avoid_when + card.avoid_when))
            existing.compatible_with = list(
                dict.fromkeys(existing.compatible_with + card.compatible_with)
            )
            existing.target_metrics = list(dict.fromkeys(existing.target_metrics + card.target_metrics))
            existing.experiment_template.update(card.experiment_template)
            existing.risk = card.risk or existing.risk
            existing.risk_level = card.risk_level or existing.risk_level
            existing.priority = max(existing.priority, card.priority)
            card = existing
        card.validate()
        self._atomic_json(self.strategy_cards / f"{card.id}.json", card.to_dict())
        self.rebuild_index()
        return card

    def rebuild_index(self) -> None:
        connection = sqlite3.connect(self.index_path)
        try:
            connection.executescript(
                """
                DROP TABLE IF EXISTS strategy_cards_fts;
                DROP TABLE IF EXISTS strategy_cards;
                CREATE TABLE strategy_cards (
                    id TEXT PRIMARY KEY,
                    task_type TEXT NOT NULL,
                    domain TEXT NOT NULL,
                    component TEXT NOT NULL,
                    priority REAL NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE VIRTUAL TABLE strategy_cards_fts USING fts5(
                    id UNINDEXED,
                    strategy_name,
                    summary,
                    use_when,
                    compatible_with,
                    target_metrics
                );
                """
            )
            for card in self.all_cards():
                payload = json.dumps(card.to_dict(), ensure_ascii=False)
                connection.execute(
                    "INSERT INTO strategy_cards VALUES (?, ?, ?, ?, ?, ?)",
                    (card.id, card.task_type, card.domain, card.component, card.priority, payload),
                )
                connection.execute(
                    "INSERT INTO strategy_cards_fts VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        card.id,
                        card.strategy_name,
                        card.summary,
                        " ".join(card.use_when),
                        " ".join(card.compatible_with),
                        " ".join(card.target_metrics),
                    ),
                )
            connection.commit()
        finally:
            connection.close()

    def search_cards(
        self,
        query: str,
        *,
        task_type: str = "classification",
        domain: str | None = None,
        target_metric: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if not self.index_path.exists():
            self.rebuild_index()
        terms = [term.replace('"', "") for term in query.split() if term.strip()]
        match = " OR ".join(f'"{term}"' for term in terms) or '"classification"'
        sql = (
            "SELECT c.payload, bm25(strategy_cards_fts) AS text_score "
            "FROM strategy_cards_fts JOIN strategy_cards c USING(id) "
            "WHERE strategy_cards_fts MATCH ? AND c.task_type = ?"
        )
        params: list[Any] = [match, task_type]
        if domain:
            sql += " AND c.domain IN (?, 'general', 'fine_grained_classification')"
            params.append(domain)
        sql += " ORDER BY text_score ASC, c.priority DESC LIMIT ?"
        params.append(max(1, min(int(top_k), 20)))
        connection = sqlite3.connect(self.index_path)
        try:
            rows = connection.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            rows = []
        finally:
            connection.close()
        if not rows:
            cards = [
                card for card in self.all_cards()
                if card.task_type == task_type
                and (not domain or card.domain in {domain, "general", "fine_grained_classification"})
            ]
            cards.sort(key=lambda item: item.priority, reverse=True)
            rows = [(json.dumps(card.to_dict()), 0.0) for card in cards[:top_k]]
        results = []
        for payload, text_score in rows:
            card = json.loads(payload)
            if target_metric and card.get("target_metrics"):
                if target_metric not in card["target_metrics"]:
                    continue
            results.append(
                {
                    "id": card["id"],
                    "strategy_name": card["strategy_name"],
                    "component": card["component"],
                    "summary": card["summary"],
                    "experiment_template": card["experiment_template"],
                    "risk": card.get("risk", ""),
                    "priority": card.get("priority", 0.0),
                    "score": round(float(card.get("priority", 0.0)) - float(text_score), 4),
                }
            )
        return results[:top_k]

    def record_observation(
        self,
        card_id: str,
        result: dict[str, Any],
        *,
        improved: bool,
    ) -> StrategyCard | None:
        card = self.get_card(card_id)
        if card is None:
            return None
        card.observed_results.append(result)
        adjustment = 0.05 if improved else -0.03
        card.priority = round(min(1.0, max(0.0, card.priority + adjustment)), 4)
        return self.upsert_card(card)

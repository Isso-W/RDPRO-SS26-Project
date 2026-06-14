"""Outcome memory — a growing log of (fingerprint, config, result) from real runs.

This is the substrate for "the recommender gets better as it's used": each run
appends a record; the ranker queries similar past records to predict how a
candidate will do on a new dataset. A plain JSONL file — inspectable, portable,
and doubles as training data for a learned predictor later.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .fingerprint import fingerprint_distance


class OutcomeMemory:
    def __init__(self, path: str | Path = "recommender/outcomes.jsonl"):
        self.path = Path(path)

    def log(
        self,
        fingerprint: dict,
        config: dict,
        result: dict,
        dataset_id: str | None = None,
    ) -> None:
        """Append one run outcome. `result` should hold at least metric_name/metric_value."""
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "dataset_id": dataset_id,
            "fingerprint": fingerprint,
            "config": config,
            "result": result,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def all(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records

    def query_similar(
        self,
        fingerprint: dict,
        k: int = 5,
        backbone: str | None = None,
    ) -> list[dict]:
        """Return the k most similar past records (optionally only for one backbone).

        Each returned record gains a 'distance' field (lower = more similar).
        Records with a different task_type (distance = inf) are excluded.
        """
        scored = []
        for record in self.all():
            cfg = record.get("config", {})
            if backbone is not None and (cfg.get("backbone") != backbone):
                continue
            dist = fingerprint_distance(fingerprint, record.get("fingerprint", {}))
            if dist == float("inf"):
                continue
            scored.append({**record, "distance": dist})
        scored.sort(key=lambda r: r["distance"])
        return scored[:k]

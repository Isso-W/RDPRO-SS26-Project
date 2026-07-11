"""Citation-preserving model retrieval boundary for MLE-STAR initialization."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, Sequence
from urllib.parse import urlsplit, urlunsplit

from .contracts import TaskSpec


@dataclass(frozen=True)
class SearchEvidence:
    title: str
    url: str
    excerpt: str
    model_hint: str = ""
    license_note: str = ""
    retrieved_at: str = ""

    def __post_init__(self) -> None:
        if not self.title or not self.url or not self.excerpt:
            raise ValueError("Search evidence needs title, URL and excerpt.")
        parsed = urlsplit(self.url)
        if parsed.scheme not in {"https", "http"} or not parsed.netloc:
            raise ValueError("Search evidence needs an http(s) URL.")
        if not self.retrieved_at:
            object.__setattr__(self, "retrieved_at", datetime.now(timezone.utc).isoformat())
        object.__setattr__(
            self,
            "url",
            urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", "")),
        )


class SearchProvider(Protocol):
    def search(self, query: str, *, limit: int) -> Sequence[SearchEvidence]: ...


@dataclass(frozen=True)
class StaticSearchProvider:
    """Offline fixture provider used for repeatable tests and dry runs."""

    evidence: tuple[SearchEvidence, ...]

    def search(self, query: str, *, limit: int) -> Sequence[SearchEvidence]:
        del query
        return self.evidence[:limit]


def model_query(task: TaskSpec) -> str:
    """Build the model-centric search query used by the paper's first stage."""

    return " ".join(
        part
        for part in (
            task.modality,
            "machine learning model",
            f"metric {task.metric.name}",
            task.description,
            "official documentation or paper example Python",
        )
        if part
    )


def retrieve_evidence(task: TaskSpec, provider: SearchProvider, *, limit: int = 4) -> tuple[SearchEvidence, ...]:
    """Retrieve unique, citation-ready model evidence without fabricating hits."""

    if limit < 1:
        raise ValueError("limit must be positive")
    selected: list[SearchEvidence] = []
    seen: set[str] = set()
    for item in provider.search(model_query(task), limit=limit * 3):
        if item.url in seen:
            continue
        selected.append(item)
        seen.add(item.url)
        if len(selected) == limit:
            break
    return tuple(selected)

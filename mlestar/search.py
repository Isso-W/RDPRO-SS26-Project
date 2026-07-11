"""Evidence-backed model retrieval for the initial MLE-STAR candidates."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit, urlunsplit

from .contracts import SearchEvidence, TaskContract


class SearchUnavailable(RuntimeError):
    """A configured provider cannot perform web search in this environment."""


class SearchProvider(Protocol):
    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Return compact, citation-ready search hits."""


@dataclass
class StaticSearchProvider:
    """Deterministic provider for tests and manually curated evidence."""

    results: list[dict[str, Any]]

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        del query
        return [dict(item) for item in self.results[:limit]]


@dataclass
class OpenAIWebSearchProvider:
    """Thin adapter for an OpenAI Responses client with a web-search tool."""

    client: Any
    model: str

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        try:
            response = self.client.responses.create(
                model=self.model,
                input=query,
                tools=[{"type": "web_search"}],
            )
        except Exception as exc:  # provider-specific errors must not become invented evidence
            raise SearchUnavailable(f"OpenAI web search failed: {type(exc).__name__}: {exc}") from exc
        hits = _extract_response_hits(response)
        if not hits:
            raise SearchUnavailable("OpenAI web search returned no citation-ready results.")
        return hits[:limit]


def retrieve_model_evidence(
    task: TaskContract,
    provider: SearchProvider,
    *,
    limit: int = 4,
    output_path: str | Path | None = None,
) -> list[SearchEvidence]:
    """Retrieve, normalize, de-duplicate, and persist model evidence."""

    if limit < 1:
        raise ValueError("limit must be positive.")
    query = build_model_query(task)
    evidence: list[SearchEvidence] = []
    seen_urls: set[str] = set()
    for raw in provider.search(query, limit=max(limit * 3, limit)):
        normalized = _normalize_hit(raw)
        if normalized is None or normalized.url in seen_urls:
            continue
        seen_urls.add(normalized.url)
        evidence.append(normalized)
        if len(evidence) == limit:
            break
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([item.to_dict() for item in evidence], indent=2, sort_keys=True), encoding="utf-8")
    return evidence


def build_model_query(task: TaskContract) -> str:
    constraints = "; ".join(task.constraints)
    return " ".join(
        part
        for part in (
            f"{task.modality} machine learning model",
            f"metric {task.metric.name}",
            task.description,
            constraints,
            "official paper or documentation example Python code",
        )
        if part
    )


def _normalize_hit(raw: dict[str, Any]) -> SearchEvidence | None:
    title = str(raw.get("title") or "").strip()
    url = _canonical_url(str(raw.get("url") or ""))
    summary = str(raw.get("summary") or raw.get("snippet") or "").strip()
    if not title or not url or not summary:
        return None
    code = str(raw.get("example_code") or raw.get("code") or "").strip()
    model_hint = str(raw.get("model_hint") or "").strip() or _model_hint(summary, code)
    return SearchEvidence(
        title=title,
        url=url,
        summary=summary,
        model_hint=model_hint,
        example_code=code,
        license_note=str(raw.get("license_note") or raw.get("license") or "").strip(),
    )


def _canonical_url(value: str) -> str:
    parsed = urlsplit(value.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/"), "", ""))


def _model_hint(summary: str, code: str) -> str:
    for text in (code, summary):
        match = re.search(r"(?:use|using|model\s*[=:]|import)\s+([A-Za-z][A-Za-z0-9_.-]+)", text, re.IGNORECASE)
        if match:
            return match.group(1).rstrip(".,;:()")
    return ""


def _extract_response_hits(response: Any) -> list[dict[str, Any]]:
    """Extract URL annotations from common OpenAI Responses output shapes."""

    output = getattr(response, "output", None) or (response.get("output") if isinstance(response, dict) else []) or []
    hits: list[dict[str, Any]] = []
    for item in output:
        content = getattr(item, "content", None) or (item.get("content") if isinstance(item, dict) else []) or []
        for block in content:
            annotations = getattr(block, "annotations", None) or (block.get("annotations") if isinstance(block, dict) else []) or []
            text = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else "") or ""
            for annotation in annotations:
                url = getattr(annotation, "url", None) or (annotation.get("url") if isinstance(annotation, dict) else None)
                title = getattr(annotation, "title", None) or (annotation.get("title") if isinstance(annotation, dict) else None)
                if url and title:
                    hits.append({"title": title, "url": url, "snippet": text})
    return hits

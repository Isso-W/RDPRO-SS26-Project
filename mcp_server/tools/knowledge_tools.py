"""Knowledge ingestion, compression, card extraction, and retrieval services."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

from knowledge import Evidence, SourceRecord, SourceSummary, StrategyCard
from knowledge.schemas import utc_now


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._ignored = 0

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style", "nav", "footer", "svg"}:
            self._ignored += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "nav", "footer", "svg"} and self._ignored:
            self._ignored -= 1

    def handle_data(self, data: str) -> None:
        if not self._ignored:
            self.parts.append(data)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "source"


def _priority_value(value: Any) -> float:
    if isinstance(value, str):
        normalized = value.strip().lower()
        named = {"low": 0.25, "medium": 0.5, "moderate": 0.5, "high": 0.8}
        if normalized in named:
            return named[normalized]
        value = normalized
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.5


def _risk_level(value: Any) -> str:
    normalized = str(value or "medium").strip().lower()
    if normalized == "moderate":
        return "medium"
    return normalized if normalized in {"low", "medium", "high"} else "medium"


def _component(value: Any) -> str:
    normalized = _slug(str(value or "augmentation"))
    aliases = {
        "data_augmentation": "augmentation",
        "loss_function": "loss",
        "learning_rate_scheduler": "scheduler",
        "fine_tuning": "finetune",
        "finetuning": "finetune",
        "transfer_learning": "finetune",
        "model": "backbone",
    }
    return aliases.get(normalized, normalized)


def _clean_text(raw: str, source_type: str) -> str:
    if source_type == "html" or "<html" in raw[:1000].lower():
        parser = _TextExtractor()
        parser.feed(raw)
        raw = "\n".join(parser.parts)
    raw = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", raw)
    raw = re.sub(r"<[^>]+>", " ", raw)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip()


def ingest_external_source_service(context, source_name: str, url: str, source_type: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "Jiaozi-Knowledge-Learner/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            raw = response.read().decode("utf-8", errors="replace")
        cleaned = _clean_text(raw, source_type)
        if len(cleaned) < 200:
            raise ValueError("Source did not contain enough readable text.")
        digest = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()
        source_id = f"source_{_slug(source_name)}_{digest[:8]}"
        content_path = context.store.raw_sources / f"{source_id}.txt"
        content_path.write_text(cleaned, encoding="utf-8")
        record = SourceRecord(
            id=source_id,
            source_name=source_name,
            source_type=source_type,
            content_path=str(content_path),
            url=url,
            content_sha256=digest,
            created_at=utc_now(),
        )
        context.store.save_source(record)
        return {"status": "success", "source": record.to_dict(), "characters": len(cleaned)}
    except Exception as exc:
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        record = SourceRecord(
            id=f"source_{_slug(source_name)}_{digest[:8]}",
            source_name=source_name,
            source_type=source_type,
            content_path="",
            url=url,
            content_sha256="",
            error=str(exc),
        )
        context.store.save_source(record)
        return {"status": "failed", "source": record.to_dict(), "error": str(exc)}


def _llm_json(system_prompt: str, user_text: str) -> tuple[dict[str, Any], int]:
    from openai import OpenAI

    provider = os.getenv("KNOWLEDGE_LLM_PROVIDER", os.getenv("JIAOZI_LLM_PROVIDER", "qwen"))
    if provider == "openai":
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        model = os.getenv("KNOWLEDGE_OPENAI_MODEL", "gpt-4.1-mini")
    else:
        client = OpenAI(
            api_key=os.getenv("JIAOZI_DASHSCOPE_API_KEY"),
            base_url=os.getenv(
                "DASHSCOPE_BASE_URL",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
        )
        model = os.getenv("KNOWLEDGE_QWEN_MODEL", "qwen-plus")
    request = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
    }
    try:
        response = client.chat.completions.create(
            **request,
            response_format={"type": "json_object"},
        )
    except Exception as first_error:
        if "response_format" not in str(first_error).lower():
            raise
        response = client.chat.completions.create(**request)
    tokens = 0
    try:
        import cost_meter

        tokens = cost_meter.tokens_from_response(response)
        cost_meter.record_llm_call(tokens)
    except Exception:
        pass
    content = response.choices[0].message.content or "{}"
    return json.loads(re.sub(r"```(?:json)?|```", "", content).strip()), tokens


def _source_record(context, source_id: str) -> SourceRecord:
    path = context.store.raw_sources / f"{source_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Unknown source_id: {source_id}")
    return SourceRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))


def summarize_source_service(context, source_id: str) -> dict:
    source = _source_record(context, source_id)
    if not source.content_path:
        raise ValueError(f"Source {source_id} has no successful content.")
    text = Path(source.content_path).read_text(encoding="utf-8")
    payload, tokens = _llm_json(
        (
            "Compress the supplied machine-learning source into JSON. Return keys: "
            "summary, models, augmentations, losses, optimizers, schedulers, inference, ensemble. "
            "Each category is a short list. Preserve only actionable evidence for image classification."
        ),
        text[:60000],
    )
    summary = SourceSummary(
        id=f"summary_{source.id}",
        source_id=source.id,
        summary=str(payload.get("summary", "")),
        models=list(payload.get("models") or []),
        augmentations=list(payload.get("augmentations") or []),
        losses=list(payload.get("losses") or []),
        optimizers=list(payload.get("optimizers") or []),
        schedulers=list(payload.get("schedulers") or []),
        inference=list(payload.get("inference") or []),
        ensemble=list(payload.get("ensemble") or []),
    )
    context.store.save_summary(summary)
    return {**summary.to_dict(), "_llm_tokens": tokens}


def extract_strategy_cards_service(
    context,
    source_summary_id: str,
    *,
    domain: str = "fine_grained_classification",
) -> list[dict]:
    path = context.store.source_summaries / f"{source_summary_id}.json"
    if not path.exists():
        raise FileNotFoundError(f"Unknown source_summary_id: {source_summary_id}")
    summary = SourceSummary.from_dict(json.loads(path.read_text(encoding="utf-8")))
    source = _source_record(context, summary.source_id)
    payload, tokens = _llm_json(
        (
            "Create compact experiment strategy cards from this source summary. Return JSON "
            "with key cards, a list. Each card must contain strategy_name, component, summary, "
            "use_when, avoid_when, compatible_with, target_metrics, experiment_template, risk, "
            "risk_level, priority. Allowed executable template fields: augmentation, "
            "randaugment_num_ops, randaugment_magnitude, mixup_alpha, cutmix_alpha, "
            "label_smoothing, optimizer, scheduler, learning_rate, finetune_strategy, "
            "freeze_backbone, unfreeze_last_n_blocks, backbone_learning_rate, "
            "head_learning_rate, backbone, pretrained_hf_id, use_pretrained, image_size, "
            "batch_size, tta_horizontal_flip. "
            "Use at most two fields per card and never enable MixUp and CutMix together."
        ),
        json.dumps(summary.to_dict(), ensure_ascii=False),
    )
    cards = []
    for index, raw in enumerate(payload.get("cards") or [], start=1):
        name = str(raw.get("strategy_name", "")).strip()
        if not name:
            continue
        try:
            component = _component(raw.get("component"))
            card = StrategyCard(
                id=f"strategy_{_slug(component)}_{_slug(name)}_001",
                task_type="classification",
                domain=domain,
                strategy_name=name,
                component=component,
                summary=str(raw.get("summary", "")),
                use_when=list(raw.get("use_when") or []),
                avoid_when=list(raw.get("avoid_when") or []),
                compatible_with=list(raw.get("compatible_with") or []),
                target_metrics=list(raw.get("target_metrics") or ["log_loss"]),
                experiment_template=dict(raw.get("experiment_template") or {}),
                evidence=[
                    Evidence(source_id=source.id, note=summary.summary[:400], url=source.url)
                ],
                risk=str(raw.get("risk", "")),
                risk_level=_risk_level(raw.get("risk_level")),
                priority=_priority_value(raw.get("priority", 0.5)),
            )
            cards.append(card.to_dict())
        except (TypeError, ValueError):
            continue
    return {"cards": cards, "_llm_tokens": tokens}


def upsert_strategy_card_service(context, card: dict) -> dict:
    return context.store.upsert_card(StrategyCard.from_dict(card)).to_dict()


def search_strategy_cards_service(
    context,
    query: str,
    task_type: str = "classification",
    domain: str = "fine_grained_classification",
    target_metric: str = "log_loss",
    top_k: int = 5,
) -> list[dict]:
    return context.store.search_cards(
        query,
        task_type=task_type,
        domain=domain,
        target_metric=target_metric,
        top_k=top_k,
    )

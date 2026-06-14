"""Process-global cost meter for one pipeline+train run.

Tracks the things that make Jiaozi's efficiency the selling point — LLM calls,
LLM tokens, training runs/epochs, and wall-clock — so a run's *cost* can be reported
and compared head-to-head against an agentic pipeline (AIDE / MLE-STAR).

Instrumentation is best-effort and must never break a call site: record_* swallows
nothing of value but is wrapped in try/except by callers.
"""

from __future__ import annotations

import threading
import time

_lock = threading.Lock()
_state: dict = {}


def reset() -> None:
    with _lock:
        _state.clear()
        _state.update(llm_calls=0, llm_tokens=0, train_runs=0, epochs=0, _start=time.time())


def _ensure() -> None:
    if not _state:
        _state.update(llm_calls=0, llm_tokens=0, train_runs=0, epochs=0, _start=time.time())


def record_llm_call(tokens: int = 0) -> None:
    with _lock:
        _ensure()
        _state["llm_calls"] += 1
        _state["llm_tokens"] += int(tokens or 0)


def record_training(epochs: int = 0, runs: int = 1) -> None:
    with _lock:
        _ensure()
        _state["train_runs"] += int(runs)
        _state["epochs"] += int(epochs or 0)


def report() -> dict:
    with _lock:
        if not _state:
            return {"llm_calls": 0, "llm_tokens": 0, "train_runs": 0, "epochs": 0, "wall_clock_sec": 0.0}
        out = {k: v for k, v in _state.items() if not k.startswith("_")}
        out["wall_clock_sec"] = round(time.time() - _state["_start"], 2)
        return out


def tokens_from_response(response) -> int:
    """Best-effort total-token count from an OpenAI-style response (chat or responses API)."""
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    for attr in ("total_tokens", "total_token_count"):
        value = getattr(usage, attr, None)
        if value is None and isinstance(usage, dict):
            value = usage.get(attr)
        if value:
            return int(value)
    return 0

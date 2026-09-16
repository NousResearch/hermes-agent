"""Small, optional audit trace for iterative web research.

The trace is intentionally callback-based and context-local. When no sink is
installed, emitters are no-ops so generic web tools retain existing behavior.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator
from urllib.parse import urlsplit, urlunsplit

_MAX_TEXT = 512
_MAX_ITEMS = 20
_sink: ContextVar[Callable[[dict[str, Any]], None] | None] = ContextVar("research_trace_sink", default=None)


def _text(value: Any) -> str:
    return str(value or "")[:_MAX_TEXT]


def _url(value: Any) -> str:
    """Keep source identity while dropping query/fragment values that may carry tokens."""
    value = _text(value)
    try:
        parts = urlsplit(value)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))[:_MAX_TEXT]
    except ValueError:
        return ""


def _metadata(source: Any, position: int) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {"url": "", "title": "", "description": "", "position": position}
    return {"title": _text(source.get("title")), "url": _url(source.get("url")),
            "description": _text(source.get("description")), "position": position}


def emit(event_type: str, **fields: Any) -> None:
    """Send one already-shaped trace event to the current optional sink."""
    sink = _sink.get()
    if sink is None:
        return
    try:
        sink({"type": event_type, **fields})
    except Exception:
        return  # instrumentation must never change tool behavior


def emit_query(query: Any, provider: Any, limit: int) -> None:
    emit("research.query", query=_text(query), provider=_text(provider), limit=int(limit))


def emit_sources(sources: list[Any]) -> None:
    emit("research.sources", sources=[_metadata(item, i) for i, item in enumerate(sources[:_MAX_ITEMS], 1)])


def emit_extraction(provider: Any, results: list[Any]) -> None:
    entries = []
    for item in results[:_MAX_ITEMS]:
        if not isinstance(item, dict):
            continue
        entries.append({"url": _url(item.get("url")), "status": "error" if item.get("error") else "ok",
                        "error": _text(item.get("error")) if item.get("error") else None})
    emit("research.extraction", provider=_text(provider), results=entries)


def emit_decision(decision: Any, details: Any = None) -> None:
    if isinstance(details, str):
        details = _text(details)
    elif isinstance(details, dict):
        details = {str(k)[:64]: _text(v) for k, v in list(details.items())[:10]}
    elif not isinstance(details, (int, float, bool)):
        details = None
    emit("research.decision", decision=_text(decision), details=details)


def emit_completion(status: Any, *, source_count: int = 0, extraction_count: int = 0) -> None:
    emit("research.completed", status=_text(status), source_count=int(source_count), extraction_count=int(extraction_count))


@contextmanager
def trace_context(sink: Callable[[dict[str, Any]], None] | None) -> Iterator[None]:
    """Install *sink* for the current execution context."""
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)

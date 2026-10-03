"""Bounded, opt-in audit events for iterative web research.

The context-local sink is installed only for a run that explicitly requests
research tracing. Events retain routing metadata, not prompts, result text,
request bodies, headers, tokens, or exception messages.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator
from urllib.parse import urlsplit, urlunsplit

_MAX_ITEMS = 20
_ALLOWED_EVENTS = frozenset({
    "research.query", "research.sources", "research.extraction",
    "research.decision", "research.completed",
})
_ALLOWED_FIELDS = {
    "research.query": frozenset({"provider", "limit"}),
    "research.sources": frozenset({"sources"}),
    "research.extraction": frozenset({"provider", "results"}),
    "research.decision": frozenset({"decision"}),
    "research.completed": frozenset({"status", "source_count", "extraction_count"}),
}
_sink: ContextVar[Callable[[dict[str, Any]], None] | None] = ContextVar("research_trace_sink", default=None)


def _bounded_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, min(int(value), _MAX_ITEMS))
    except (TypeError, ValueError, OverflowError):
        return default


def _label(value: Any) -> str:
    """Keep a bounded identifier, never arbitrary text."""
    value = str(value or "")
    return value if value.isascii() and value.replace("-", "").replace("_", "").isalnum() and len(value) <= 64 else ""


def _url(value: Any) -> str:
    """Canonical HTTP(S) URL without credentials, query, or fragment."""
    try:
        parts = urlsplit(str(value or ""))
        if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
            return ""
        host = parts.hostname.lower()
        port = parts.port
        netloc = host if not port or (parts.scheme.lower(), port) in {("http", 80), ("https", 443)} else f"{host}:{port}"
        return urlunsplit((parts.scheme.lower(), netloc, parts.path, "", ""))[:512]
    except (TypeError, ValueError):
        return ""


def emit(event_type: str, **fields: Any) -> None:
    """Send one pre-shaped, allowlisted event to this context's optional sink."""
    sink = _sink.get()
    if sink is None or event_type not in _ALLOWED_EVENTS:
        return
    try:
        allowed = _ALLOWED_FIELDS[event_type]
        sink({"type": event_type, **{key: value for key, value in fields.items() if key in allowed}})
    except Exception:
        return


def emit_query(_query: Any, provider: Any, limit: int) -> None:
    emit("research.query", provider=_label(provider), limit=_bounded_int(limit))


def emit_sources(sources: list[Any]) -> None:
    items = sources if isinstance(sources, list) else []
    emit("research.sources", sources=[
        {"url": _url(item.get("url")) if isinstance(item, dict) else "", "position": position}
        for position, item in enumerate(items[:_MAX_ITEMS], 1)
    ])


def emit_extraction(provider: Any, results: list[Any]) -> None:
    items = results if isinstance(results, list) else []
    entries = []
    for item in items[:_MAX_ITEMS]:
        if isinstance(item, dict):
            entries.append({"url": _url(item.get("url")), "status": "error" if item.get("error") else "ok"})
    emit("research.extraction", provider=_label(provider), results=entries)


def emit_decision(decision: Any, _details: Any = None) -> None:
    emit("research.decision", decision=_label(decision))


def emit_completion(status: Any, *, source_count: int = 0, extraction_count: int = 0) -> None:
    emit("research.completed", status=_label(status), source_count=_bounded_int(source_count),
         extraction_count=_bounded_int(extraction_count))


def enabled_for_request(body: Any) -> bool:
    """Return true only for the explicit, per-run boolean opt-in."""
    return isinstance(body, dict) and body.get("research_trace") is True


@contextmanager
def trace_context(sink: Callable[[dict[str, Any]], None] | None) -> Iterator[None]:
    """Install *sink* for this run only; ContextVar prevents session/profile leaks."""
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)

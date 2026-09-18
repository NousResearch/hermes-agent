"""Bounded, opt-in audit events for iterative web research.

The context-local sink is installed only for a run that explicitly requests
research tracing. Every value crossing this boundary is minimized and passed
through Hermes' canonical secret redactor; raw web content never enters it.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator
from urllib.parse import unquote, urlsplit

_MAX_TEXT = 512
_MAX_ITEMS = 20
_MAX_DETAIL_KEYS = 10
_ALLOWED_EVENTS = frozenset({
    "research.query", "research.sources", "research.extraction",
    "research.decision", "research.completed",
})
_ALLOWED_FIELDS = {
    "research.query": frozenset({"query", "provider", "limit"}),
    "research.sources": frozenset({"sources"}),
    "research.extraction": frozenset({"provider", "results"}),
    "research.decision": frozenset({"decision", "details"}),
    "research.completed": frozenset({"status", "source_count", "extraction_count"}),
}
_sink: ContextVar[Callable[[dict[str, Any]], None] | None] = ContextVar("research_trace_sink", default=None)
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)\b(password|passphrase|token|secret|api[_ -]?key|authorization)\s*[:=]\s*[^\s&,;]+"
)
_BEARER_VALUE = re.compile(r"(?i)\b(bearer)\s+[^\s&,;]+")


def _redact(value: Any) -> str:
    """Redact before applying the size limit; fail closed if policy code fails."""
    if value is None:
        return ""
    try:
        from agent.redact import redact_sensitive_text
        text = redact_sensitive_text(str(value), force=True)
        text = _SENSITIVE_ASSIGNMENT.sub(lambda match: f"{match.group(1)}=<redacted>", text)
        text = _BEARER_VALUE.sub(r"\1 <redacted>", text)
        return text[:_MAX_TEXT]
    except Exception:
        return "<redacted>"


def _url(value: Any) -> str:
    """Keep only an HTTP(S) origin and conservative, non-sensitive path segments."""
    raw = _redact(value)
    try:
        parts = urlsplit(raw)
        if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
            return ""
        host = parts.hostname
        try:
            port = parts.port
        except ValueError:
            return ""
        origin = f"{parts.scheme.lower()}://{host}"
        if port and not ((parts.scheme.lower() == "http" and port == 80) or
                         (parts.scheme.lower() == "https" and port == 443)):
            origin += f":{port}"
        segments = []
        for segment in parts.path.split("/"):
            segment = _redact(unquote(segment))
            if not segment or len(segment) > 64 or not re.fullmatch(r"[A-Za-z0-9._~-]+", segment):
                if segment:
                    break
                continue
            segments.append(segment)
            if len(segments) == 8:
                break
        result = origin + ("/" + "/".join(segments) if segments else "")
        return result[:_MAX_TEXT]
    except (TypeError, ValueError):
        return ""


def _bounded_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, min(int(value), _MAX_ITEMS))
    except (TypeError, ValueError, OverflowError):
        return default


def _metadata(source: Any, position: int) -> dict[str, Any]:
    if not isinstance(source, dict):
        return {"url": "", "title": "", "description": "", "position": position}
    return {
        "title": _redact(source.get("title")),
        "url": _url(source.get("url")),
        "description": _redact(source.get("description")),
        "position": position,
    }


def emit(event_type: str, **fields: Any) -> None:
    """Send a bounded event; unknown event types and sink failures are ignored."""
    sink = _sink.get()
    if sink is None or event_type not in _ALLOWED_EVENTS:
        return
    try:
        allowed = _ALLOWED_FIELDS[event_type]
        sink({"type": event_type, **{key: value for key, value in fields.items() if key in allowed}})
    except Exception:
        return  # instrumentation must never change tool behavior


def emit_query(query: Any, provider: Any, limit: int) -> None:
    emit("research.query", query=_redact(query), provider=_redact(provider), limit=_bounded_int(limit))


def emit_sources(sources: list[Any]) -> None:
    items = sources if isinstance(sources, list) else []
    emit("research.sources", sources=[_metadata(item, i) for i, item in enumerate(items[:_MAX_ITEMS], 1)])


def emit_extraction(provider: Any, results: list[Any]) -> None:
    items = results if isinstance(results, list) else []
    entries = []
    for item in items[:_MAX_ITEMS]:
        if not isinstance(item, dict):
            continue
        error = item.get("error")
        entries.append({"url": _url(item.get("url")), "status": "error" if error else "ok",
                        "error": _redact(error) if error else None})
    emit("research.extraction", provider=_redact(provider), results=entries)


def emit_decision(decision: Any, details: Any = None) -> None:
    if isinstance(details, str):
        details = _redact(details)
    elif isinstance(details, dict):
        details = {_redact(k)[:64]: _redact(v) for k, v in list(details.items())[:_MAX_DETAIL_KEYS]}
    elif not isinstance(details, (int, float, bool)):
        details = None
    emit("research.decision", decision=_redact(decision), details=details)


def emit_completion(status: Any, *, source_count: int = 0, extraction_count: int = 0) -> None:
    emit("research.completed", status=_redact(status), source_count=_bounded_int(source_count),
         extraction_count=_bounded_int(extraction_count))


def enabled_for_request(body: Any) -> bool:
    """Return true only for the versioned, explicit per-run opt-in."""
    return isinstance(body, dict) and body.get("research_trace") is True


@contextmanager
def trace_context(sink: Callable[[dict[str, Any]], None] | None) -> Iterator[None]:
    """Install *sink* for this run only; never leak it across sessions/profiles."""
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)

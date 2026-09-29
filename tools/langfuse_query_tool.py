"""Langfuse query tool — read back traces/observations for self-observation.

The bundled ``plugins/observability/langfuse`` plugin is write-only: it
opens traces and ends them, but nothing in this codebase reads them back.
This tool is the first reader, built for the Dream cron job's weekly
self-observation pass but generically usable from any agent turn.

Query design verified live against a self-hosted Langfuse v3 instance:

- ``GET /api/public/traces`` accepts ``fields=core,io`` for a compact
  payload (name/timestamps/input/output, no scores/metrics/observations).
  On this deployment, ``fields`` values that include ``metrics``,
  ``scores``, or ``observations`` return HTTP 500 (a ClickHouse
  column-resolution bug) — never request those field groups here.
- The ``filter`` query param takes a JSON array of filter objects, ANDed
  together. A ``level`` column filter must use
  ``{"type": "stringOptions", "column": "level", "operator": "any of",
  "value": [...]}`` — ``categoryOptions`` 400s on this column (it expects
  a ``key`` field that doesn't apply here).
- Server-side ``level`` filtering only catches genuine SDK/OTEL-flagged
  errors (``ERROR``/``WARNING``). The narration-only tool-call failure
  this tool was built to help catch (a local model emitting
  ``{"content": "[Calling tool", "tool_calls": []}`` instead of a real
  tool call) is logged at ``level: DEFAULT`` — it is invisible to a level
  filter and requires a client-side heuristic pass over trace bodies.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from plugins.observability.langfuse import _validate_langfuse_key

logger = logging.getLogger(__name__)

_BASE_URL_DEFAULT = "https://cloud.langfuse.com"
_MAX_FIELD_CHARS = 4000
_MAX_PAGES = 3
_PAGE_LIMIT = 100

# Heuristic markers for the narration-only tool-call failure: the model
# describes an intended tool call in plain text instead of emitting a real
# structured tool_call. Expect to tune this list as real Dream runs surface
# false positives/negatives -- not a documented Langfuse feature, just a
# pattern match over trace output bodies.
_NARRATION_MARKERS = (
    re.compile(r"^\[?calling\b", re.IGNORECASE),
    re.compile(r"\bi(?:'ll| will)\s+(?:now\s+)?call\b.*\btool\b", re.IGNORECASE),
    re.compile(r"\binvoking\b.*\btool\b", re.IGNORECASE),
)


def _env(name: str, default: str = "") -> str:
    import os

    return os.environ.get(name, default).strip()


def _lf_creds() -> Optional[tuple]:
    """Return (public_key, secret_key, base_url) from env, or None if unset/placeholder.

    Server-side only -- never exposed to the LLM/prompt as a tool argument
    or echoed back in a tool result.
    """
    public_key = _env("HERMES_LANGFUSE_PUBLIC_KEY") or _env("LANGFUSE_PUBLIC_KEY")
    secret_key = _env("HERMES_LANGFUSE_SECRET_KEY") or _env("LANGFUSE_SECRET_KEY")
    if not (public_key and secret_key):
        return None
    issues = [
        msg
        for msg in (
            _validate_langfuse_key("HERMES_LANGFUSE_PUBLIC_KEY", public_key),
            _validate_langfuse_key("HERMES_LANGFUSE_SECRET_KEY", secret_key),
        )
        if msg
    ]
    if issues:
        logger.warning("langfuse_query: credentials look like placeholders (%s)", "; ".join(issues))
        return None
    base_url = _env("HERMES_LANGFUSE_BASE_URL") or _env("LANGFUSE_BASE_URL") or _BASE_URL_DEFAULT
    return public_key, secret_key, base_url.rstrip("/")


def check_langfuse_query_requirements() -> bool:
    return _lf_creds() is not None


def _truncate(value: Any, limit: int = _MAX_FIELD_CHARS) -> Any:
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + f"...[truncated, {len(value)} chars total]"
    return value


def _looks_like_narration_failure(output: Any) -> Optional[str]:
    """Return a one-line symptom string if `output` matches the narration-only pattern, else None."""
    if not isinstance(output, dict):
        return None
    tool_calls = output.get("tool_calls")
    content = output.get("content")
    if tool_calls not in (None, []):
        return None
    if not isinstance(content, str):
        return None
    stripped = content.strip()
    if not stripped:
        return "empty response, no tool_calls (dead turn)"
    for pattern in _NARRATION_MARKERS:
        if pattern.search(stripped):
            return f"narrated instead of calling: {stripped[:120]!r}"
    return None


def _get(base_url: str, auth: tuple, path: str, params: Dict[str, Any]) -> Dict[str, Any]:
    resp = httpx.get(f"{base_url}{path}", auth=auth, params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _list_errors(since: Optional[str], until: Optional[str], limit: int) -> str:
    creds = _lf_creds()
    if creds is None:
        return json.dumps(
            {"error": "Langfuse credentials not configured or look like placeholders"}, ensure_ascii=False
        )
    public_key, secret_key, base_url = creds
    auth = (public_key, secret_key)
    limit = max(1, min(int(limit or 20), _PAGE_LIMIT))

    time_filter: List[Dict[str, Any]] = []
    if since:
        time_filter.append({"type": "datetime", "column": "timestamp", "operator": ">=", "value": since})
    if until:
        time_filter.append({"type": "datetime", "column": "timestamp", "operator": "<=", "value": until})

    findings_by_id: Dict[str, Dict[str, Any]] = {}

    # Pass 1: server-side level filter -- catches genuine SDK/OTEL errors cheaply.
    level_filter = time_filter + [
        {"type": "stringOptions", "column": "level", "operator": "any of", "value": ["ERROR", "WARNING"]}
    ]
    try:
        for page in range(1, _MAX_PAGES + 1):
            data = _get(
                base_url,
                auth,
                "/api/public/traces",
                {
                    "fields": "core,io",
                    "filter": json.dumps(level_filter),
                    "limit": limit,
                    "page": page,
                },
            )
            items = data.get("data") or []
            for t in items:
                findings_by_id[t["id"]] = {
                    "trace_id": t.get("id"),
                    "timestamp": t.get("timestamp"),
                    "name": t.get("name"),
                    "symptom": "flagged ERROR/WARNING level by Langfuse",
                }
            if len(items) < limit:
                break
    except httpx.HTTPError as e:
        logger.warning("langfuse_query list_errors (level pass) failed: %s", e)

    # Pass 2: client-side heuristic over the same time window (no level
    # filter) for the narration-only-tool-call pattern the level filter
    # cannot see (logged at level=DEFAULT).
    try:
        for page in range(1, _MAX_PAGES + 1):
            data = _get(
                base_url,
                auth,
                "/api/public/traces",
                {
                    "fields": "core,io",
                    "filter": json.dumps(time_filter) if time_filter else None,
                    "limit": limit,
                    "page": page,
                },
            )
            items = data.get("data") or []
            for t in items:
                symptom = _looks_like_narration_failure(t.get("output"))
                if symptom:
                    tid = t.get("id")
                    if tid not in findings_by_id:
                        findings_by_id[tid] = {
                            "trace_id": tid,
                            "timestamp": t.get("timestamp"),
                            "name": t.get("name"),
                            "symptom": symptom,
                        }
            if len(items) < limit:
                break
    except httpx.HTTPError as e:
        logger.warning("langfuse_query list_errors (heuristic pass) failed: %s", e)

    return json.dumps({"count": len(findings_by_id), "findings": list(findings_by_id.values())}, ensure_ascii=False)


def _get_trace(trace_id: str) -> str:
    if not trace_id:
        return json.dumps({"error": "trace_id is required"}, ensure_ascii=False)
    creds = _lf_creds()
    if creds is None:
        return json.dumps(
            {"error": "Langfuse credentials not configured or look like placeholders"}, ensure_ascii=False
        )
    public_key, secret_key, base_url = creds
    auth = (public_key, secret_key)
    try:
        data = _get(base_url, auth, f"/api/public/traces/{trace_id}", {})
    except httpx.HTTPStatusError as e:
        return json.dumps({"error": f"Langfuse returned {e.response.status_code}"}, ensure_ascii=False)
    except httpx.HTTPError as e:
        return json.dumps({"error": f"Langfuse request failed: {e}"}, ensure_ascii=False)

    truncated = {k: _truncate(v) for k, v in data.items()}
    return json.dumps(truncated, ensure_ascii=False)


def langfuse_query(
    action: str,
    since: Optional[str] = None,
    until: Optional[str] = None,
    trace_id: Optional[str] = None,
    limit: int = 20,
) -> str:
    action = (action or "").strip().lower()
    if action == "list_errors":
        return _list_errors(since, until, limit)
    if action == "get_trace":
        return _get_trace(trace_id)
    return json.dumps({"error": f"unknown action {action!r}, expected 'list_errors' or 'get_trace'"}, ensure_ascii=False)


LANGFUSE_QUERY_SCHEMA = {
    "name": "langfuse_query",
    "description": (
        "Read back traces from the self-hosted Langfuse observability instance for "
        "self-observation -- find recent tool-call failures, narration-only responses, "
        "and other anomalies Hermes had no other way to notice. Read-only."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["list_errors", "get_trace"],
                "description": (
                    "'list_errors': time-boxed digest of anomalous traces (not full bodies). "
                    "'get_trace': full detail for one trace_id."
                ),
            },
            "since": {
                "type": "string",
                "description": "ISO8601 start of the time window. list_errors only.",
            },
            "until": {
                "type": "string",
                "description": "ISO8601 end of the time window, default now. list_errors only.",
            },
            "trace_id": {
                "type": "string",
                "description": "Trace id to fetch full detail for. get_trace only.",
            },
            "limit": {
                "type": "integer",
                "description": "Max results per page for list_errors, default 20, capped at 100.",
            },
        },
        "required": ["action"],
    },
}


# --- Registry ---
from tools.registry import registry

registry.register(
    name="langfuse_query",
    toolset="langfuse",
    schema=LANGFUSE_QUERY_SCHEMA,
    handler=lambda args, **kw: langfuse_query(
        action=args.get("action", ""),
        since=args.get("since"),
        until=args.get("until"),
        trace_id=args.get("trace_id"),
        limit=args.get("limit", 20),
    ),
    check_fn=check_langfuse_query_requirements,
    emoji="🔭",
)

"""Shared Agent Memory tools.

Thin Hermes tool wrappers over a local/shared FastAPI service. The service owns
policy, secret scanning, audit logging, Markdown/Basic Memory persistence, and
status transitions.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from tools.registry import registry


DEFAULT_BASE_URL = "http://127.0.0.1:8001"


def _base_url() -> str:
    return os.getenv("SAM_API_BASE_URL", DEFAULT_BASE_URL).rstrip("/")


def _api_key() -> str | None:
    return os.getenv("SAM_API_KEY") or os.getenv("SHARED_MEMORY_API_KEY")


def check_shared_memory_requirements() -> bool:
    """Return True when the API key exists and the service is healthy."""
    if not _api_key():
        return False
    try:
        req = urllib.request.Request(f"{_base_url()}/health", method="GET")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return 200 <= resp.status < 300
    except Exception:
        return False


def _request(method: str, path: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
    key = _api_key()
    if not key:
        return {"error": "SAM_API_KEY is not configured"}

    data = None
    headers = {"X-API-Key": key, "Accept": "application/json"}
    if body is not None:
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib.request.Request(f"{_base_url()}{path}", data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return json.loads(raw) if raw else {"status": resp.status}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            detail = json.loads(raw)
        except Exception:
            detail = raw[:1000]
        return {"error": "shared_memory_http_error", "status": exc.code, "detail": detail}
    except Exception as exc:
        return {"error": "shared_memory_request_failed", "type": type(exc).__name__, "message": str(exc)}


def _redact_actor_fields(value: Any) -> Any:
    """Avoid leaking configured API-key labels/values into model context."""
    if isinstance(value, dict):
        return {
            key: ("[REDACTED]" if key in {"created_by", "reviewed_by", "actor"} and val else _redact_actor_fields(val))
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [_redact_actor_fields(item) for item in value]
    return value


def _json_result(payload: dict[str, Any]) -> str:
    return json.dumps(_redact_actor_fields(payload), ensure_ascii=False, indent=2)


def shared_memory_search(query: str | None = None, statuses: list[str] | None = None, limit: int = 10) -> str:
    """Search the Shared Agent Memory service."""
    body: dict[str, Any] = {"query": query, "limit": limit}
    if statuses is not None:
        body["statuses"] = statuses
    return _json_result(_request("POST", "/search", body))


def shared_memory_create_draft(
    title: str,
    content: str,
    directory: str = "",
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Create a draft memory. The service performs secret scanning and audit logging."""
    return _json_result(
        _request(
            "POST",
            "/memory/draft",
            {
                "title": title,
                "content": content,
                "directory": directory,
                "tags": tags or [],
                "metadata": metadata or {},
            },
        )
    )


def shared_memory_get(memory_id: str) -> str:
    """Read one memory record by id/permalink."""
    encoded = urllib.parse.quote(memory_id, safe="/")
    return _json_result(_request("GET", f"/memory/{encoded}"))


def shared_memory_update(
    memory_id: str,
    title: str | None = None,
    content: str | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    expected_sha256: str | None = None,
) -> str:
    """Update a memory with optional optimistic locking via expected_sha256."""
    body: dict[str, Any] = {}
    if title is not None:
        body["title"] = title
    if content is not None:
        body["content"] = content
    if tags is not None:
        body["tags"] = tags
    if metadata is not None:
        body["metadata"] = metadata
    if expected_sha256 is not None:
        body["expected_sha256"] = expected_sha256
    if not body:
        return _json_result({"error": "no_update_fields"})
    encoded = urllib.parse.quote(memory_id, safe="/")
    return _json_result(_request("PATCH", f"/memory/{encoded}", body))


def shared_memory_update_status(memory_id: str, action: str) -> str:
    """Move a memory through service policy: review, accept, supersede, archive."""
    if action not in {"review", "accept", "supersede", "archive"}:
        return _json_result({"error": "invalid_action", "allowed": ["review", "accept", "supersede", "archive"]})
    encoded = urllib.parse.quote(memory_id, safe="/")
    return _json_result(_request("POST", f"/memory/{encoded}/{action}"))


_SHARED_MEMORY_SEARCH_SCHEMA = {
    "name": "shared_memory_search",
    "description": "Search the Shared Agent Memory service. Prefer accepted/reviewed memories for durable project/user facts. Returns matching Markdown-backed memory records.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query. Omit or empty to list recent/visible memory if supported."},
            "statuses": {
                "type": "array",
                "items": {"type": "string", "enum": ["draft", "reviewed", "accepted", "superseded", "archived", "quarantined"]},
                "description": "Optional status filter. Default service behavior should prefer accepted/reviewed.",
            },
            "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
        },
    },
}

_SHARED_MEMORY_CREATE_DRAFT_SCHEMA = {
    "name": "shared_memory_create_draft",
    "description": "Create a draft in Shared Agent Memory. Use for durable facts, project state, decisions, or lessons that should be reviewable. Never store secrets, tokens, passwords, or temporary task progress.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Human-readable memory title."},
            "content": {"type": "string", "description": "Memory content in clear, factual Markdown."},
            "directory": {"type": "string", "description": "Optional vault directory, e.g. projects, systems, decisions, playbooks, people."},
            "tags": {"type": "array", "items": {"type": "string"}},
            "metadata": {"type": "object", "additionalProperties": True},
        },
        "required": ["title", "content"],
    },
}

_SHARED_MEMORY_GET_SCHEMA = {
    "name": "shared_memory_get",
    "description": "Read a single Shared Agent Memory record by id/permalink.",
    "parameters": {
        "type": "object",
        "properties": {"memory_id": {"type": "string", "description": "Memory id/permalink returned by search or create."}},
        "required": ["memory_id"],
    },
}

_SHARED_MEMORY_UPDATE_SCHEMA = {
    "name": "shared_memory_update",
    "description": "Update an existing Shared Agent Memory record. Use expected_sha256 from get/search to avoid overwriting concurrent edits.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string"},
            "title": {"type": "string"},
            "content": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "metadata": {"type": "object", "additionalProperties": True},
            "expected_sha256": {"type": "string", "description": "Optional sha256 from current record for conflict detection."},
        },
        "required": ["memory_id"],
    },
}

_SHARED_MEMORY_UPDATE_STATUS_SCHEMA = {
    "name": "shared_memory_update_status",
    "description": "Move a Shared Agent Memory record through policy-controlled states: review, accept, supersede, archive. Do not accept important memories without human approval.",
    "parameters": {
        "type": "object",
        "properties": {
            "memory_id": {"type": "string"},
            "action": {"type": "string", "enum": ["review", "accept", "supersede", "archive"]},
        },
        "required": ["memory_id", "action"],
    },
}


registry.register(
    name="shared_memory_search",
    toolset="shared_memory",
    schema=_SHARED_MEMORY_SEARCH_SCHEMA,
    handler=lambda args, **kw: shared_memory_search(
        query=args.get("query"),
        statuses=args.get("statuses"),
        limit=args.get("limit", 10),
    ),
    check_fn=check_shared_memory_requirements,
    requires_env=["SAM_API_KEY"],
    emoji="🧠",
)

registry.register(
    name="shared_memory_create_draft",
    toolset="shared_memory",
    schema=_SHARED_MEMORY_CREATE_DRAFT_SCHEMA,
    handler=lambda args, **kw: shared_memory_create_draft(
        title=args.get("title", ""),
        content=args.get("content", ""),
        directory=args.get("directory", ""),
        tags=args.get("tags"),
        metadata=args.get("metadata"),
    ),
    check_fn=check_shared_memory_requirements,
    requires_env=["SAM_API_KEY"],
    emoji="🧠",
)

registry.register(
    name="shared_memory_get",
    toolset="shared_memory",
    schema=_SHARED_MEMORY_GET_SCHEMA,
    handler=lambda args, **kw: shared_memory_get(memory_id=args.get("memory_id", "")),
    check_fn=check_shared_memory_requirements,
    requires_env=["SAM_API_KEY"],
    emoji="🧠",
)

registry.register(
    name="shared_memory_update",
    toolset="shared_memory",
    schema=_SHARED_MEMORY_UPDATE_SCHEMA,
    handler=lambda args, **kw: shared_memory_update(
        memory_id=args.get("memory_id", ""),
        title=args.get("title"),
        content=args.get("content"),
        tags=args.get("tags"),
        metadata=args.get("metadata"),
        expected_sha256=args.get("expected_sha256"),
    ),
    check_fn=check_shared_memory_requirements,
    requires_env=["SAM_API_KEY"],
    emoji="🧠",
)

registry.register(
    name="shared_memory_update_status",
    toolset="shared_memory",
    schema=_SHARED_MEMORY_UPDATE_STATUS_SCHEMA,
    handler=lambda args, **kw: shared_memory_update_status(
        memory_id=args.get("memory_id", ""),
        action=args.get("action", ""),
    ),
    check_fn=check_shared_memory_requirements,
    requires_env=["SAM_API_KEY"],
    emoji="🧠",
)

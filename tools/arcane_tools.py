"""Arcane workspace tools.

These tools let a Hermes agent inspect and update the Arcane chat+artifact
workspace through Arcane's HTTP API. They never read Arcane's filesystem
directly; file paths are sent to Arcane as encoded API path parameters so
Arcane remains the authority for traversal checks and artifact policy.
"""

from __future__ import annotations

import json
import os
from typing import Any
from urllib.parse import quote

import requests

from agent.events import redact_string
from tools.registry import registry

DEFAULT_ARCANE_BASE_URL = "http://127.0.0.1:8787"
HTTP_TIMEOUT_SECONDS = 20
MAX_FILE_CONTENT_CHARS = 20000
MAX_MESSAGE_CONTENT_CHARS = 4000
MAX_ERROR_CHARS = 1000


def _json(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def _success(**payload: Any) -> str:
    return _json({"success": True, **payload})


def _error(message: Any, **payload: Any) -> str:
    return _json({"success": False, "error": _redact(message), **payload})


def _redact(value: Any) -> str:
    text = redact_string(str(value or ""))
    token = os.getenv("ARCANE_ACCESS_TOKEN", "")
    if token:
        text = text.replace(token, "[redacted]")
    return text[:MAX_ERROR_CHARS]


def _truncate(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + "\n[truncated]", True


def _base_url() -> str:
    return (os.getenv("ARCANE_BASE_URL", "").strip() or DEFAULT_ARCANE_BASE_URL).rstrip("/")


def _session_id(session_id: Any = None) -> str:
    value = str(session_id or os.getenv("ARCANE_SESSION_ID", "")).strip()
    if not value:
        raise ValueError("session_id is required or ARCANE_SESSION_ID must be set")
    return value


def _artifact_path(path: Any) -> str:
    value = str(path or "").strip()
    if not value:
        raise ValueError("path is required")
    return value


def _encoded(value: str) -> str:
    return quote(value, safe="")


def _headers(*, accept_text: bool = False, content_type: str | None = None) -> dict[str, str]:
    headers = {"Accept": "text/plain" if accept_text else "application/json"}
    token = os.getenv("ARCANE_ACCESS_TOKEN", "")
    if token:
        headers["x-arcane-token"] = token
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def _request_json(method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    try:
        response = requests.request(method, url, timeout=HTTP_TIMEOUT_SECONDS, **kwargs)
    except requests.RequestException as exc:
        raise RuntimeError(f"Arcane HTTP request failed: {_redact(exc)}") from exc

    if response.status_code >= 400:
        detail = _redact(getattr(response, "text", "") or getattr(response, "reason", ""))
        raise RuntimeError(f"Arcane API returned HTTP {response.status_code}: {detail}")
    try:
        data = response.json()
    except ValueError as exc:
        raise RuntimeError("Arcane API returned non-JSON response") from exc
    if not isinstance(data, dict):
        raise RuntimeError("Arcane API returned unexpected JSON payload")
    return data


def _request_text(method: str, url: str, **kwargs: Any) -> str:
    try:
        response = requests.request(method, url, timeout=HTTP_TIMEOUT_SECONDS, **kwargs)
    except requests.RequestException as exc:
        raise RuntimeError(f"Arcane HTTP request failed: {_redact(exc)}") from exc

    if response.status_code >= 400:
        detail = _redact(getattr(response, "text", "") or getattr(response, "reason", ""))
        raise RuntimeError(f"Arcane API returned HTTP {response.status_code}: {detail}")
    return response.text


def _session_url(session_id: str) -> str:
    return f"{_base_url()}/api/sessions/{_encoded(session_id)}"


def _file_url(session_id: str, path: str) -> str:
    return f"{_session_url(session_id)}/files/{_encoded(path)}"


def _artifact_url(session_id: str, path: str) -> str:
    return f"{_base_url()}/artifact/{_encoded(session_id)}/{_encoded(path)}"


def _coerce_positive_int(value: Any, default: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, min(parsed, maximum))


def _compact_message(message: Any) -> Any:
    if not isinstance(message, dict):
        return message
    compact = dict(message)
    content = compact.get("content")
    if isinstance(content, str):
        compact["content"], compact["contentTruncated"] = _truncate(content, MAX_MESSAGE_CONTENT_CHARS)
    return compact


def _compact_session_payload(
    payload: dict[str, Any],
    *,
    message_limit: int = 20,
    run_event_limit: int = 50,
) -> dict[str, Any]:
    out = {
        "session": payload.get("session"),
        "files": payload.get("files") or [],
        "artifactFiles": payload.get("artifactFiles") or [],
        "run": payload.get("run"),
        "runs": payload.get("runs") or [],
    }
    messages = payload.get("messages")
    if isinstance(messages, list):
        out["messages"] = [_compact_message(item) for item in messages[-message_limit:]]
        out["messagesTruncated"] = len(messages) > message_limit
    run_events = payload.get("runEvents")
    if isinstance(run_events, list):
        out["runEvents"] = run_events[-run_event_limit:]
        out["runEventsTruncated"] = len(run_events) > run_event_limit
    return out


def arcane_get_session(
    session_id: str | None = None,
    message_limit: int = 20,
    run_event_limit: int = 50,
    task_id: str | None = None,
) -> str:
    """Get Arcane session metadata, recent messages, artifact files, and runs."""
    try:
        sid = _session_id(session_id)
        payload = _request_json("GET", _session_url(sid), headers=_headers())
        return _success(
            session_id=sid,
            **_compact_session_payload(
                payload,
                message_limit=_coerce_positive_int(message_limit, 20, 100),
                run_event_limit=_coerce_positive_int(run_event_limit, 50, 200),
            ),
        )
    except Exception as exc:
        return _error(exc)


def arcane_list_files(session_id: str | None = None, task_id: str | None = None) -> str:
    """List artifact files in an Arcane session."""
    try:
        sid = _session_id(session_id)
        payload = _request_json("GET", _session_url(sid), headers=_headers())
        return _success(
            session_id=sid,
            files=payload.get("files") or [],
            artifactFiles=payload.get("artifactFiles") or [],
        )
    except Exception as exc:
        return _error(exc)


def arcane_read_file(
    path: str,
    session_id: str | None = None,
    task_id: str | None = None,
) -> str:
    """Read one artifact file from an Arcane session."""
    try:
        sid = _session_id(session_id)
        rel_path = _artifact_path(path)
        content = _request_text("GET", _artifact_url(sid, rel_path), headers=_headers(accept_text=True))
        preview, truncated = _truncate(content, MAX_FILE_CONTENT_CHARS)
        return _success(
            session_id=sid,
            path=rel_path,
            content=preview,
            size=len(content),
            truncated=truncated,
        )
    except Exception as exc:
        return _error(exc)


def arcane_write_file(
    path: str,
    content: str,
    session_id: str | None = None,
    task_id: str | None = None,
) -> str:
    """Write one artifact file in an Arcane session."""
    try:
        sid = _session_id(session_id)
        rel_path = _artifact_path(path)
        text = str(content or "")
        payload = _request_json(
            "PUT",
            _file_url(sid, rel_path),
            headers=_headers(content_type="text/plain; charset=utf-8"),
            data=text.encode("utf-8"),
        )
        return _success(
            session_id=sid,
            path=rel_path,
            file=payload.get("file"),
            files=payload.get("files") or [],
            artifactFiles=payload.get("artifactFiles") or [],
            bytes=len(text.encode("utf-8")),
        )
    except Exception as exc:
        return _error(exc)


def arcane_create_snapshot(
    summary: str | None = None,
    session_id: str | None = None,
    task_id: str | None = None,
) -> str:
    """Create an Arcane artifact snapshot for rollback/review."""
    try:
        sid = _session_id(session_id)
        payload = _request_json(
            "POST",
            f"{_session_url(sid)}/snapshots",
            headers=_headers(content_type="application/json"),
            json={"summary": str(summary or "")},
        )
        return _success(
            session_id=sid,
            snapshot=payload.get("snapshot") or {
                key: payload.get(key)
                for key in ("id", "summary", "createdAt")
                if key in payload
            },
            files=payload.get("files") or [],
            artifactFiles=payload.get("artifactFiles") or [],
        )
    except Exception as exc:
        return _error(exc)


def _arg(args: dict[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in args:
            return args.get(name)
    return default


_SESSION_ID_SCHEMA = {
    "type": "string",
    "description": "Arcane session id. Defaults to ARCANE_SESSION_ID for the current Arcane run.",
}

registry.register(
    name="arcane_get_session",
    toolset="arcane",
    schema={
        "name": "arcane_get_session",
        "description": "Get Arcane session metadata, recent messages, files, runs, and run events.",
        "parameters": {
            "type": "object",
            "properties": {
                "session_id": _SESSION_ID_SCHEMA,
                "message_limit": {
                    "type": "integer",
                    "description": "Maximum recent messages to include. Default 20, max 100.",
                    "default": 20,
                },
                "run_event_limit": {
                    "type": "integer",
                    "description": "Maximum recent run events to include. Default 50, max 200.",
                    "default": 50,
                },
            },
            "additionalProperties": False,
        },
    },
    handler=lambda args, **kw: arcane_get_session(
        session_id=_arg(args, "session_id", "sessionId"),
        message_limit=_arg(args, "message_limit", "messageLimit", default=20),
        run_event_limit=_arg(args, "run_event_limit", "runEventLimit", default=50),
        task_id=kw.get("task_id"),
    ),
    description="Get Arcane session metadata.",
    emoji="🜁",
)

registry.register(
    name="arcane_list_files",
    toolset="arcane",
    schema={
        "name": "arcane_list_files",
        "description": "List artifact files in the current Arcane workspace session.",
        "parameters": {
            "type": "object",
            "properties": {"session_id": _SESSION_ID_SCHEMA},
            "additionalProperties": False,
        },
    },
    handler=lambda args, **kw: arcane_list_files(
        session_id=_arg(args, "session_id", "sessionId"),
        task_id=kw.get("task_id"),
    ),
    description="List Arcane artifact files.",
    emoji="🜁",
)

registry.register(
    name="arcane_read_file",
    toolset="arcane",
    schema={
        "name": "arcane_read_file",
        "description": "Read one artifact file from the current Arcane workspace session.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative artifact path such as index.html, styles.css, or nested/file.js.",
                },
                "session_id": _SESSION_ID_SCHEMA,
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    handler=lambda args, **kw: arcane_read_file(
        path=_arg(args, "path", default=""),
        session_id=_arg(args, "session_id", "sessionId"),
        task_id=kw.get("task_id"),
    ),
    description="Read an Arcane artifact file.",
    emoji="🜁",
)

registry.register(
    name="arcane_write_file",
    toolset="arcane",
    schema={
        "name": "arcane_write_file",
        "description": "Write one artifact file in the current Arcane workspace session.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Relative artifact path such as index.html, styles.css, or nested/file.js.",
                },
                "content": {
                    "type": "string",
                    "description": "Full file content to write.",
                },
                "session_id": _SESSION_ID_SCHEMA,
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        },
    },
    handler=lambda args, **kw: arcane_write_file(
        path=_arg(args, "path", default=""),
        content=_arg(args, "content", default=""),
        session_id=_arg(args, "session_id", "sessionId"),
        task_id=kw.get("task_id"),
    ),
    description="Write an Arcane artifact file.",
    emoji="🜁",
)

registry.register(
    name="arcane_create_snapshot",
    toolset="arcane",
    schema={
        "name": "arcane_create_snapshot",
        "description": "Create a snapshot of the current Arcane artifact files.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Short summary of why the snapshot was created.",
                },
                "session_id": _SESSION_ID_SCHEMA,
            },
            "additionalProperties": False,
        },
    },
    handler=lambda args, **kw: arcane_create_snapshot(
        summary=_arg(args, "summary", default=""),
        session_id=_arg(args, "session_id", "sessionId"),
        task_id=kw.get("task_id"),
    ),
    description="Create an Arcane artifact snapshot.",
    emoji="🜁",
)

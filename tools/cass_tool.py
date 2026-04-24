"""CASS CLI tools for cross-agent session intelligence.

CASS (Coding Agent Session Search) indexes coding-agent histories across
providers such as Claude Code, Codex, and Gemini.  Hermes uses these wrappers as
an explicit continuity backend instead of shelling out through the generic
terminal tool for every recall query.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from collections.abc import Iterable
from typing import Any

from tools.registry import registry, tool_error, tool_result

_CASS_INSTALL_HINT = (
    "Install cass and ensure it is on PATH. Expected binary name: cass. "
    "See https://github.com/Dicklesworthstone/coding_agent_session_search"
)
_ALLOWED_SEARCH_MODES = {"lexical", "semantic", "hybrid"}
_ALLOWED_EXPORT_FORMATS = {"markdown", "text", "json", "html"}
_ALLOWED_ANALYTICS_KINDS = {"status", "tokens", "tools", "models"}
_ALLOWED_GROUP_BY = {"hour", "day", "week", "month", "none"}
_SEARCH_DEFAULT_LIMIT = 5
_SEARCH_MAX_LIMIT = 25
_DEFAULT_SEARCH_CONTENT_LENGTH = 2000


def check_cass_requirements() -> bool:
    """Return True when the CASS CLI is available on PATH."""
    return shutil.which("cass") is not None


def _parse_json_maybe(text: str) -> Any | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _normalize_cli_error(
    *,
    command: str,
    returncode: int,
    stdout: str,
    stderr: str,
) -> str:
    parsed = _parse_json_maybe(stdout) or _parse_json_maybe(stderr)
    if isinstance(parsed, dict) and isinstance(parsed.get("error"), dict):
        error = parsed["error"]
        kind = error.get("kind")
        message = error.get("message") or stderr or stdout or f"cass {command} failed"
        hint = error.get("hint") or ""
        if kind == "semantic-unavailable" and "--mode lexical" not in hint:
            hint = (hint + " Use --mode lexical as a fallback.").strip()
        return tool_result(
            success=False,
            command=command,
            error=message,
            code=error.get("code", returncode),
            kind=kind,
            hint=hint,
            retryable=error.get("retryable"),
        )

    message = (stderr or stdout or f"cass {command} exited with status {returncode}").strip()
    return tool_error(message, success=False, command=command, code=returncode)


def _run_cass(args: list[str], *, command: str, timeout: int = 120) -> str:
    if not check_cass_requirements():
        return tool_error(
            "cass CLI not found on PATH",
            success=False,
            command=command,
            hint=_CASS_INSTALL_HINT,
        )

    argv = ["cass", *args]
    try:
        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return tool_error(
            f"cass {command} timed out after {timeout}s",
            success=False,
            command=command,
            code="timeout",
        )
    except OSError as exc:
        return tool_error(str(exc), success=False, command=command)

    if result.returncode != 0:
        return _normalize_cli_error(
            command=command,
            returncode=result.returncode,
            stdout=result.stdout,
            stderr=result.stderr,
        )

    parsed = _parse_json_maybe(result.stdout)
    return tool_result({
        "success": True,
        "command": command,
        "data": parsed if parsed is not None else result.stdout,
    })


def _as_list(value: str | Iterable[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    return [str(item) for item in value if str(item).strip()]


def _append_repeated(args: list[str], flag: str, values: str | Iterable[str] | None) -> None:
    for value in _as_list(values):
        args.extend([flag, value])


def _bounded_int(value: int | str | None, *, default: int, minimum: int, maximum: int) -> int:
    try:
        number = int(value) if value is not None else default
    except (TypeError, ValueError):
        number = default
    return min(max(number, minimum), maximum)


def cass_status() -> str:
    """Return CASS index health and database status."""
    return _run_cass(["status", "--json"], command="status", timeout=30)


def cass_search(
    query: str,
    limit: int = 5,
    mode: str = "lexical",
    workspace: str | list[str] | None = None,
    agent: str | list[str] | None = None,
    since: str | None = None,
    until: str | None = None,
    days: int | None = None,
    offset: int = 0,
    fields: str | None = None,
    max_content_length: int | None = None,
    source: str | None = None,
) -> str:
    """Search indexed coding-agent sessions with CASS."""
    if not (query or "").strip():
        return tool_error("query is required", success=False, command="search")
    if mode not in _ALLOWED_SEARCH_MODES:
        return tool_error(
            f"mode must be one of: {', '.join(sorted(_ALLOWED_SEARCH_MODES))}",
            success=False,
            command="search",
        )

    bounded_limit = _bounded_int(
        limit,
        default=_SEARCH_DEFAULT_LIMIT,
        minimum=1,
        maximum=_SEARCH_MAX_LIMIT,
    )
    bounded_content_length = _bounded_int(
        max_content_length,
        default=_DEFAULT_SEARCH_CONTENT_LENGTH,
        minimum=200,
        maximum=10_000,
    )
    args = [
        "search",
        "--json",
        "--limit",
        str(bounded_limit),
        "--mode",
        mode,
    ]
    if offset:
        args.extend(["--offset", str(max(0, int(offset)))])
    _append_repeated(args, "--workspace", workspace)
    _append_repeated(args, "--agent", agent)
    if since:
        args.extend(["--since", since])
    if until:
        args.extend(["--until", until])
    if days is not None:
        args.extend(["--days", str(max(0, int(days)))])
    if fields:
        args.extend(["--fields", fields])
    if max_content_length is not None:
        args.extend(["--max-content-length", str(bounded_content_length)])
    else:
        args.extend(["--max-content-length", str(_DEFAULT_SEARCH_CONTENT_LENGTH)])
    if source:
        args.extend(["--source", source])
    args.extend(["--", query])

    return _run_cass(args, command="search")


def cass_context(path: str, limit: int = 5) -> str:
    """Find sessions related to a CASS source/session path."""
    if not (path or "").strip():
        return tool_error("path is required", success=False, command="context")
    args = ["context", "--json", "--limit", str(_bounded_int(limit, default=5, minimum=1, maximum=25)), "--", path]
    return _run_cass(args, command="context")


def cass_timeline(
    since: str | None = None,
    until: str | None = None,
    today: bool = False,
    agent: str | list[str] | None = None,
    group_by: str = "day",
    source: str | None = None,
) -> str:
    """Return activity timeline data from indexed CASS sessions."""
    if group_by not in {"hour", "day", "none"}:
        return tool_error("group_by must be one of: hour, day, none", success=False, command="timeline")
    args = ["timeline", "--json", "--group-by", group_by]
    if today:
        args.append("--today")
    if since:
        args.extend(["--since", since])
    if until:
        args.extend(["--until", until])
    _append_repeated(args, "--agent", agent)
    if source:
        args.extend(["--source", source])
    return _run_cass(args, command="timeline")


def cass_export(
    session_path: str,
    format: str = "markdown",
    output_path: str | None = None,
    include_tools: bool = False,
) -> str:
    """Export a CASS session transcript."""
    if not (session_path or "").strip():
        return tool_error("session_path is required", success=False, command="export")
    if format not in _ALLOWED_EXPORT_FORMATS:
        return tool_error(
            f"format must be one of: {', '.join(sorted(_ALLOWED_EXPORT_FORMATS))}",
            success=False,
            command="export",
        )
    args = ["export", "--format", format]
    if output_path:
        args.extend(["--output", output_path])
    if include_tools:
        args.append("--include-tools")
    args.extend(["--", session_path])
    return _run_cass(args, command="export")


def cass_analytics(
    kind: str = "tokens",
    days: int | None = 7,
    workspace: str | list[str] | None = None,
    agent: str | list[str] | None = None,
    source: str | None = None,
    group_by: str | None = None,
    since: str | None = None,
    until: str | None = None,
) -> str:
    """Return token/tool/model analytics from CASS rollups."""
    if kind not in _ALLOWED_ANALYTICS_KINDS:
        return tool_error(
            f"kind must be one of: {', '.join(sorted(_ALLOWED_ANALYTICS_KINDS))}",
            success=False,
            command="analytics",
        )
    args = ["analytics", kind, "--json"]
    if kind != "status":
        if days is not None:
            args.extend(["--days", str(max(0, int(days)))])
        if since:
            args.extend(["--since", since])
        if until:
            args.extend(["--until", until])
        _append_repeated(args, "--workspace", workspace)
        _append_repeated(args, "--agent", agent)
        if source:
            args.extend(["--source", source])
        if group_by:
            if group_by not in _ALLOWED_GROUP_BY - {"none"}:
                return tool_error(
                    "group_by must be one of: hour, day, week, month",
                    success=False,
                    command="analytics",
                )
            args.extend(["--group-by", group_by])
    return _run_cass(args, command="analytics")


_CASS_STATUS_SCHEMA = {
    "name": "cass_status",
    "description": "Check CASS index health and database status for coding-agent session history.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

_CASS_SEARCH_SCHEMA = {
    "name": "cass_search",
    "description": (
        "Search coding-agent session history indexed by CASS across Claude Code, Codex, Gemini, and other agents. "
        "Use lexical mode for reliable keyword recall; semantic/hybrid require a CASS vector index."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "limit": {"type": "integer", "description": "Maximum number of hits to return (bounded 1-25).", "default": 5, "minimum": 1, "maximum": 25},
            "mode": {"type": "string", "enum": sorted(_ALLOWED_SEARCH_MODES), "default": "lexical"},
            "workspace": {"type": "string", "description": "Workspace path filter."},
            "agent": {"type": "string", "description": "Agent slug filter, e.g. claude_code."},
            "since": {"type": "string", "description": "Start date/time filter."},
            "until": {"type": "string", "description": "End date/time filter."},
            "days": {"type": "integer", "description": "Filter to last N days."},
            "offset": {"type": "integer", "default": 0},
            "fields": {"type": "string", "description": "Optional CASS fields selector."},
            "max_content_length": {"type": "integer", "description": "Content/snippet truncation length (bounded 200-10000).", "default": 2000, "minimum": 200, "maximum": 10000},
            "source": {"type": "string", "description": "CASS source filter."},
        },
        "required": ["query"],
    },
}

_CASS_CONTEXT_SCHEMA = {
    "name": "cass_context",
    "description": "Find CASS sessions related to a source/session file path.",
    "parameters": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "CASS source/session path."},
            "limit": {"type": "integer", "default": 5},
        },
        "required": ["path"],
    },
}

_CASS_TIMELINE_SCHEMA = {
    "name": "cass_timeline",
    "description": "Show activity timeline for indexed CASS sessions.",
    "parameters": {
        "type": "object",
        "properties": {
            "since": {"type": "string"},
            "until": {"type": "string"},
            "today": {"type": "boolean", "default": False},
            "agent": {"type": "string"},
            "group_by": {"type": "string", "enum": ["hour", "day", "none"], "default": "day"},
            "source": {"type": "string"},
        },
        "required": [],
    },
}

_CASS_EXPORT_SCHEMA = {
    "name": "cass_export",
    "description": "Export a CASS session transcript as markdown, text, JSON, or HTML.",
    "parameters": {
        "type": "object",
        "properties": {
            "session_path": {"type": "string", "description": "Path to a CASS source/session file."},
            "format": {"type": "string", "enum": sorted(_ALLOWED_EXPORT_FORMATS), "default": "markdown"},
            "output_path": {"type": "string", "description": "Optional output file path. If omitted, returns stdout content."},
            "include_tools": {"type": "boolean", "default": False},
        },
        "required": ["session_path"],
    },
}

_CASS_ANALYTICS_SCHEMA = {
    "name": "cass_analytics",
    "description": "Query CASS token, tool, model, or analytics status rollups.",
    "parameters": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": sorted(_ALLOWED_ANALYTICS_KINDS), "default": "tokens"},
            "days": {"type": "integer", "default": 7},
            "workspace": {"type": "string"},
            "agent": {"type": "string"},
            "source": {"type": "string"},
            "group_by": {"type": "string", "enum": ["hour", "day", "week", "month"]},
            "since": {"type": "string"},
            "until": {"type": "string"},
        },
        "required": [],
    },
}

registry.register(
    name="cass_status",
    toolset="cass",
    schema=_CASS_STATUS_SCHEMA,
    handler=lambda args, **kw: cass_status(),
    check_fn=check_cass_requirements,
    description="Check CASS health and indexing status",
    emoji="🗂️",
)

registry.register(
    name="cass_search",
    toolset="cass",
    schema=_CASS_SEARCH_SCHEMA,
    handler=lambda args, **kw: cass_search(
        query=args.get("query", ""),
        limit=args.get("limit", 5),
        mode=args.get("mode", "lexical"),
        workspace=args.get("workspace"),
        agent=args.get("agent"),
        since=args.get("since"),
        until=args.get("until"),
        days=args.get("days"),
        offset=args.get("offset", 0),
        fields=args.get("fields"),
        max_content_length=args.get("max_content_length"),
        source=args.get("source"),
    ),
    check_fn=check_cass_requirements,
    description="Search CASS coding-agent session history",
    emoji="🔎",
    max_result_size_chars=60_000,
)

registry.register(
    name="cass_context",
    toolset="cass",
    schema=_CASS_CONTEXT_SCHEMA,
    handler=lambda args, **kw: cass_context(
        path=args.get("path", ""),
        limit=args.get("limit", 5),
    ),
    check_fn=check_cass_requirements,
    description="Find related CASS sessions for a path",
    emoji="🧭",
    max_result_size_chars=40_000,
)

registry.register(
    name="cass_timeline",
    toolset="cass",
    schema=_CASS_TIMELINE_SCHEMA,
    handler=lambda args, **kw: cass_timeline(
        since=args.get("since"),
        until=args.get("until"),
        today=args.get("today", False),
        agent=args.get("agent"),
        group_by=args.get("group_by", "day"),
        source=args.get("source"),
    ),
    check_fn=check_cass_requirements,
    description="Show CASS activity timeline",
    emoji="🕰️",
    max_result_size_chars=40_000,
)

registry.register(
    name="cass_export",
    toolset="cass",
    schema=_CASS_EXPORT_SCHEMA,
    handler=lambda args, **kw: cass_export(
        session_path=args.get("session_path", ""),
        format=args.get("format", "markdown"),
        output_path=args.get("output_path"),
        include_tools=args.get("include_tools", False),
    ),
    check_fn=check_cass_requirements,
    description="Export CASS session transcripts",
    emoji="📤",
    max_result_size_chars=80_000,
)

registry.register(
    name="cass_analytics",
    toolset="cass",
    schema=_CASS_ANALYTICS_SCHEMA,
    handler=lambda args, **kw: cass_analytics(
        kind=args.get("kind", "tokens"),
        days=args.get("days", 7),
        workspace=args.get("workspace"),
        agent=args.get("agent"),
        source=args.get("source"),
        group_by=args.get("group_by"),
        since=args.get("since"),
        until=args.get("until"),
    ),
    check_fn=check_cass_requirements,
    description="Query CASS analytics rollups",
    emoji="📊",
    max_result_size_chars=80_000,
)

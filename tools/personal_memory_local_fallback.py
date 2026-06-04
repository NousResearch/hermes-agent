"""Read-only local fallback for Răzvan's Personal Memory MCP import.

This module is intentionally not auto-registered as a normal built-in tool.
``tools.mcp_tool`` calls :func:`register_personal_memory_local_fallback`
only when the configured ``personal-memory`` MCP server cannot connect.  The
fallback keeps memory reads usable from the local macOS import archive while
OAuth is expired/revoked, but write-capable operations still fail loudly so we
never pretend the authoritative external MCP was updated.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from hermes_constants import get_hermes_home
from tools.registry import registry

logger = logging.getLogger(__name__)

_SERVER_NAME = "personal-memory"
_TOOLSET_NAME = "mcp-personal-memory"


@dataclass
class LocalMemory:
    name: str
    kind: str = "memory"
    type: str = "note"
    scopes: list[str] | None = None
    description: str = ""
    updated_at: str = ""
    content: str = ""


def _memory_roots() -> Iterable[Path]:
    home = Path(get_hermes_home())
    yield home / "memory"
    for path in sorted(
        (home / "migration" / "openclaw").glob("*/archive/workspace/memory"),
        reverse=True,
    ):
        yield path


def _find_import_dir() -> Path | None:
    for root in _memory_roots():
        if (root / "personal-memory-import-index.md").exists():
            return root
    return None


def has_local_personal_memory_import() -> bool:
    return _find_import_dir() is not None


def _chunk_paths(root: Path, include_daily: bool = True) -> list[Path]:
    paths = sorted(root.glob("personal-memory-import-*.md"))
    return [
        p
        for p in paths
        if p.name != "personal-memory-import-index.md"
        and (include_daily or "-daily-" not in p.name)
    ]


def _parse_backtick_meta(block: str, key: str) -> str:
    m = re.search(rf"^- {re.escape(key)}: `([^`]*)`\s*$", block, re.MULTILINE)
    return m.group(1).strip() if m else ""


def _parse_plain_meta(block: str, key: str) -> str:
    m = re.search(rf"^- {re.escape(key)}:\s*(.*?)\s*$", block, re.MULTILINE)
    return m.group(1).strip() if m else ""


def _parse_memories(include_daily: bool = True) -> list[LocalMemory]:
    root = _find_import_dir()
    if root is None:
        return []

    memories: list[LocalMemory] = []
    for path in _chunk_paths(root, include_daily=include_daily):
        text = path.read_text(encoding="utf-8")
        # Top-level memory entries are rendered as "## slug". Content inside a
        # memory may contain deeper headings (###/####), but not another "##"
        # until the next entry in the import format.
        matches = list(re.finditer(r"(?m)^## ([A-Za-z0-9_.-]+)\s*$", text))
        for idx, match in enumerate(matches):
            start = match.start()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            block = text[start:end].strip()
            name = match.group(1)
            content_match = re.search(r"(?m)^### Content\s*$", block)
            content = block[content_match.end():].strip() if content_match else ""
            scopes_raw = _parse_backtick_meta(block, "scopes")
            scopes = [s.strip() for s in scopes_raw.split(",") if s.strip()] if scopes_raw else []
            memories.append(
                LocalMemory(
                    name=name,
                    kind=_parse_backtick_meta(block, "kind") or "memory",
                    type=_parse_backtick_meta(block, "type") or "note",
                    scopes=scopes,
                    updated_at=_parse_backtick_meta(block, "updated_at"),
                    description=_parse_plain_meta(block, "description"),
                    content=content,
                )
            )
    return memories


def _scope_matches(memory: LocalMemory, scope: str | None) -> bool:
    if not scope or scope == "*":
        return True
    scopes = memory.scopes or []
    return "*" in scopes or scope in scopes


def _filtered_memories(
    *,
    kind: str | None = None,
    type: str | None = None,
    scope: str | None = None,
    include_daily: bool = True,
) -> list[LocalMemory]:
    items = _parse_memories(include_daily=include_daily)
    if kind:
        items = [m for m in items if m.kind == kind]
    if type:
        items = [m for m in items if m.type == type]
    if scope:
        items = [m for m in items if _scope_matches(m, scope)]
    return items


def _metadata(memory: LocalMemory) -> dict[str, Any]:
    return {
        "name": memory.name,
        "kind": memory.kind,
        "type": memory.type,
        "scopes": memory.scopes or [],
        "description": memory.description,
        "updated_at": memory.updated_at,
        "source": "local_personal_memory_import_fallback",
    }


def _json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _read_only_error(operation: str) -> str:
    return _json({
        "error": (
            f"Personal Memory MCP is offline and the local macOS import is read-only; "
            f"cannot perform {operation}. Re-authenticate with `hermes mcp login personal-memory` "
            f"to write to the authoritative MCP."
        ),
        "fallback": "local_read_only",
    })


def _bootstrap_handler(args: dict, **_: Any) -> str:
    scope = args.get("scope") or "assistant"
    recent_limit = int(args.get("recent_limit") or 5)
    include_daily = bool(args.get("include_daily", False))
    souls = _filtered_memories(kind="soul", scope=scope, include_daily=include_daily)
    recent = _filtered_memories(scope=scope, include_daily=include_daily)
    recent = [m for m in recent if m.kind != "soul"][: max(1, min(recent_limit, 20))]

    lines = [
        "[personal-memory local fallback] MCP is offline; using read-only macOS import.",
        f"Scope: {scope}",
        "",
        "## SOUL",
    ]
    for memory in souls:
        lines.append(f"### {memory.name}\n{memory.content or memory.description}".strip())
    lines.append("\n## Recent memories")
    for memory in recent:
        lines.append(f"### {memory.name}\n{memory.content or memory.description}".strip())
    lines.append("\n## Index")
    for memory in _filtered_memories(scope=scope, include_daily=include_daily):
        lines.append(f"- `{memory.name}` ({memory.kind} · {memory.type}) - {memory.description}")
    return "\n\n".join(lines)


def _list_handler(args: dict, **_: Any) -> str:
    items = _filtered_memories(
        kind=args.get("kind"),
        type=args.get("type"),
        scope=args.get("scope"),
    )
    return _json([_metadata(m) for m in items])


def _read_handler(args: dict, **_: Any) -> str:
    name = args.get("name")
    if not name:
        return _json({"error": "Missing required parameter 'name'"})
    for memory in _parse_memories(include_daily=True):
        if memory.name == name:
            return memory.content or memory.description
    return _json({"error": f"Memory '{name}' not found in local Personal Memory import"})


def _search_handler(args: dict, **_: Any) -> str:
    query = (args.get("query") or "").strip().lower()
    limit = int(args.get("limit") or 20)
    if not query:
        return _json({"error": "Missing required parameter 'query'"})
    results = []
    for memory in _parse_memories(include_daily=True):
        haystack = "\n".join([memory.name, memory.description, memory.content]).lower()
        if query in haystack:
            idx = max(haystack.find(query), 0)
            raw = "\n".join([memory.description, memory.content]).replace("\n", " ")
            snippet = raw[max(0, idx - 120): idx + 240]
            results.append({**_metadata(memory), "snippet": snippet})
        if len(results) >= max(1, min(limit, 50)):
            break
    return _json(results)


def _list_prompts_handler(args: dict, **_: Any) -> str:
    return _json({"prompts": [], "fallback": "local_read_only"})


def _get_prompt_handler(args: dict, **_: Any) -> str:
    return _json({"error": "No prompts are available in the local Personal Memory import", "fallback": "local_read_only"})


def _schema(name: str, description: str, properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "description": description + " (local read-only fallback when Personal Memory MCP is offline)",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required or [],
        },
    }


def register_personal_memory_local_fallback(reason: str = "") -> list[str]:
    """Register read-only fallback tools for the configured personal-memory MCP."""
    if not has_local_personal_memory_import():
        return []

    tools: list[tuple[str, dict[str, Any], Any]] = [
        (
            "mcp_personal_memory_bootstrap_context",
            _schema("mcp_personal_memory_bootstrap_context", "Load SOUL, recent memories, and index", {
                "recent_limit": {"type": "integer", "minimum": 1, "maximum": 20},
                "include_daily": {"type": "boolean"},
                "scope": {"type": "string"},
            }),
            _bootstrap_handler,
        ),
        (
            "mcp_personal_memory_list_memories",
            _schema("mcp_personal_memory_list_memories", "List memory metadata", {
                "kind": {"type": "string", "enum": ["soul", "memory", "daily"]},
                "type": {"type": "string", "enum": ["user", "feedback", "project", "reference", "note"]},
                "scope": {"type": "string"},
            }),
            _list_handler,
        ),
        (
            "mcp_personal_memory_read_memory",
            _schema("mcp_personal_memory_read_memory", "Read one memory by slug", {"name": {"type": "string"}}, ["name"]),
            _read_handler,
        ),
        (
            "mcp_personal_memory_search_memories",
            _schema("mcp_personal_memory_search_memories", "Search local imported memories", {
                "query": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            }, ["query"]),
            _search_handler,
        ),
        (
            "mcp_personal_memory_save_memory",
            _schema("mcp_personal_memory_save_memory", "Save memory to Personal Memory MCP", {
                "name": {"type": "string"},
                "kind": {"type": "string"},
                "type": {"type": "string"},
                "scopes": {"type": "array", "items": {"type": "string"}},
                "description": {"type": "string"},
                "content": {"type": "string"},
            }, ["name", "content"]),
            lambda args, **kwargs: _read_only_error("save_memory"),
        ),
        (
            "mcp_personal_memory_delete_memory",
            _schema("mcp_personal_memory_delete_memory", "Delete memory from Personal Memory MCP", {"name": {"type": "string"}}, ["name"]),
            lambda args, **kwargs: _read_only_error("delete_memory"),
        ),
        (
            "mcp_personal_memory_export_conversation",
            _schema("mcp_personal_memory_export_conversation", "Export conversation to Personal Memory MCP", {
                "title": {"type": "string"},
                "content": {"type": "string"},
                "source": {"type": "string"},
            }, ["title", "content"]),
            lambda args, **kwargs: _read_only_error("export_conversation"),
        ),
        (
            "mcp_personal_memory_list_prompts",
            _schema("mcp_personal_memory_list_prompts", "List Personal Memory MCP prompts", {}),
            _list_prompts_handler,
        ),
        (
            "mcp_personal_memory_get_prompt",
            _schema("mcp_personal_memory_get_prompt", "Get a Personal Memory MCP prompt", {"name": {"type": "string"}, "arguments": {"type": "object"}}, ["name"]),
            _get_prompt_handler,
        ),
    ]

    registered = []
    for tool_name, schema, handler in tools:
        registry.register(
            name=tool_name,
            toolset=_TOOLSET_NAME,
            schema=schema,
            handler=handler,
            check_fn=has_local_personal_memory_import,
            is_async=False,
            description=schema["description"],
        )
        registered.append(tool_name)
    registry.register_toolset_alias(_SERVER_NAME, _TOOLSET_NAME)
    logger.warning(
        "Personal Memory MCP unavailable%s; registered %d local read-only fallback tool(s)",
        f" ({reason})" if reason else "",
        len(registered),
    )
    return registered

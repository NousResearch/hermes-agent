"""Vault tools — model-tool surface for Hermes Vault (Obsidian-compatible PKM).

Provides tools for the agent to search, read, write, append, and explore
bidirectional links in the user's local-first Markdown vault.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

from tools.registry import registry, tool_error
from workstation.vault import VaultManager

logger = logging.getLogger(__name__)

_vault_manager: Optional[VaultManager] = None


def get_vault_manager() -> VaultManager:
    global _vault_manager
    if _vault_manager is None:
        _vault_manager = VaultManager()
    return _vault_manager


def _ok(**data: Any) -> str:
    return json.dumps({"status": "ok", **data}, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_vault_search(args: dict, **kw) -> str:
    query = str(args.get("query") or "").strip()
    if not query:
        return tool_error("query cannot be empty")
    limit = int(args.get("limit") or 10)
    try:
        mgr = get_vault_manager()
        results = mgr.search(query, limit=limit)
        return _ok(results=results, count=len(results))
    except Exception as e:
        logger.exception("vault_search failed")
        return tool_error(f"vault_search failed: {e}")


def _handle_vault_read(args: dict, **kw) -> str:
    title = str(args.get("title") or args.get("note") or "").strip()
    if not title:
        return tool_error("title or note name is required")
    try:
        mgr = get_vault_manager()
        note = mgr.get_note(title)
        if not note:
            return tool_error(f"note '{title}' not found in vault")
        return _ok(note=note)
    except Exception as e:
        logger.exception("vault_read failed")
        return tool_error(f"vault_read failed: {e}")


def _handle_vault_write(args: dict, **kw) -> str:
    title = str(args.get("title") or "").strip()
    content = str(args.get("content") or "").strip()
    if not title:
        return tool_error("title is required")
    if not content:
        return tool_error("content is required")
    tags = args.get("tags")
    frontmatter = args.get("frontmatter")
    subfolder = args.get("subfolder")
    try:
        mgr = get_vault_manager()
        note = mgr.write_note(
            title=title,
            content=content,
            tags=tags,
            frontmatter=frontmatter,
            subfolder=subfolder,
        )
        return _ok(message=f"Note '{title}' saved successfully", note=note)
    except Exception as e:
        logger.exception("vault_write failed")
        return tool_error(f"vault_write failed: {e}")


def _handle_vault_append(args: dict, **kw) -> str:
    title = str(args.get("title") or "").strip()
    content = str(args.get("content") or "").strip()
    if not title:
        return tool_error("title is required")
    if not content:
        return tool_error("content is required")
    try:
        mgr = get_vault_manager()
        note = mgr.append_note(title=title, append_content=content)
        return _ok(message=f"Appended to '{title}' successfully", note=note)
    except Exception as e:
        logger.exception("vault_append failed")
        return tool_error(f"vault_append failed: {e}")


def _handle_vault_backlinks(args: dict, **kw) -> str:
    title = str(args.get("title") or "").strip()
    if not title:
        return tool_error("title is required")
    try:
        mgr = get_vault_manager()
        note = mgr.get_note(title)
        if not note:
            return tool_error(f"note '{title}' not found in vault")
        return _ok(
            title=note["title"],
            backlinks=note["backlinks"],
            forward_links=note["forward_links"],
        )
    except Exception as e:
        logger.exception("vault_backlinks failed")
        return tool_error(f"vault_backlinks failed: {e}")


def _handle_vault_graph(args: dict, **kw) -> str:
    try:
        mgr = get_vault_manager()
        graph = mgr.get_graph()
        return _ok(
            nodes_count=len(graph["nodes"]),
            edges_count=len(graph["edges"]),
            graph=graph,
        )
    except Exception as e:
        logger.exception("vault_graph failed")
        return tool_error(f"vault_graph failed: {e}")


# ---------------------------------------------------------------------------
# Tool Schemas
# ---------------------------------------------------------------------------

VAULT_SEARCH_SCHEMA = {
    "name": "vault_search",
    "description": "Search Markdown notes in the local-first Hermes Vault by keyword, topic, or tag.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search terms to match in titles, tags, or content."},
            "limit": {"type": "integer", "description": "Maximum number of results to return (default 10)."},
        },
        "required": ["query"],
    },
}

VAULT_READ_SCHEMA = {
    "name": "vault_read",
    "description": "Read a Markdown note from the Vault including frontmatter, tags, outgoing wikilinks, and incoming backlinks.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Title or relative path of the note to read."},
        },
        "required": ["title"],
    },
}

VAULT_WRITE_SCHEMA = {
    "name": "vault_write",
    "description": "Create or overwrite a Markdown note in the Vault. Use [[Wikilinks]] to connect to other notes.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Note title (becomes filename: <title>.md)."},
            "content": {"type": "string", "description": "Markdown body of the note. May contain [[Wikilinks]] and #tags."},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional list of tags."},
            "frontmatter": {"type": "object", "description": "Optional YAML frontmatter properties dictionary."},
            "subfolder": {"type": "string", "description": "Optional subfolder within vault."},
        },
        "required": ["title", "content"],
    },
}

VAULT_APPEND_SCHEMA = {
    "name": "vault_append",
    "description": "Append Markdown text or log entries to an existing note in the Vault (creates note if missing).",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Title of the note to append to."},
            "content": {"type": "string", "description": "Text or markdown to append."},
        },
        "required": ["title", "content"],
    },
}

VAULT_BACKLINKS_SCHEMA = {
    "name": "vault_backlinks",
    "description": "Query incoming references (backlinks) and outgoing links for a specific note in the Vault.",
    "parameters": {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Title of the note."},
        },
        "required": ["title"],
    },
}

VAULT_GRAPH_SCHEMA = {
    "name": "vault_graph",
    "description": "Get the interconnected knowledge graph of the Vault (nodes and wikilink edges).",
    "parameters": {
        "type": "object",
        "properties": {},
    },
}


# ---------------------------------------------------------------------------
# Registrations
# ---------------------------------------------------------------------------

registry.register(
    name="vault_search",
    toolset="vault",
    schema=VAULT_SEARCH_SCHEMA,
    handler=_handle_vault_search,
    emoji="🔍",
)

registry.register(
    name="vault_read",
    toolset="vault",
    schema=VAULT_READ_SCHEMA,
    handler=_handle_vault_read,
    emoji="📖",
)

registry.register(
    name="vault_write",
    toolset="vault",
    schema=VAULT_WRITE_SCHEMA,
    handler=_handle_vault_write,
    emoji="✍️",
)

registry.register(
    name="vault_append",
    toolset="vault",
    schema=VAULT_APPEND_SCHEMA,
    handler=_handle_vault_append,
    emoji="📝",
)

registry.register(
    name="vault_backlinks",
    toolset="vault",
    schema=VAULT_BACKLINKS_SCHEMA,
    handler=_handle_vault_backlinks,
    emoji="🔗",
)

registry.register(
    name="vault_graph",
    toolset="vault",
    schema=VAULT_GRAPH_SCHEMA,
    handler=_handle_vault_graph,
    emoji="🕸️",
)

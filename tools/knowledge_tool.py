#!/usr/bin/env python3
"""
Knowledge Store Tool — thin subprocess wrapper around knowledge binary.

Calls `knowledge` (Rust) or `knowledge-py` (Python fallback).
No imports of knowledge_fallback. Knowledge store is external.
"""

import json
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

VALID_CATEGORIES = ["pitfalls", "fixes", "workflows", "facts"]


def _find_binary() -> str | None:
    """Find the knowledge binary (Rust preferred, Python fallback)."""
    candidates = [
        Path.home() / ".hermes" / "bin" / "knowledge",  # Rust
        Path("/usr/local/bin/knowledge"),
    ]
    for p in candidates:
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    # Python fallback
    py_fallback = Path.home() / ".hermes" / "bin" / "knowledge-py"
    if py_fallback.is_file():
        return str(py_fallback)
    return None


def _run(*args: str) -> str:
    """Run knowledge binary, return stdout. Raises on error."""
    bin_path = _find_binary()
    if not bin_path:
        raise FileNotFoundError(
            "knowledge binary not found. "
            "Install: cargo install knowledge-db"
        )

    result = subprocess.run(
        [bin_path, *args],
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            result.stderr.strip() or f"exit code {result.returncode}"
        )
    return result.stdout


def knowledge_write(category: str, content: str) -> str:
    if category not in VALID_CATEGORIES:
        return _tool_error(
            f"Unknown category '{category}'. Available: {', '.join(VALID_CATEGORIES)}"
        )
    try:
        path = _run("append", category, content)
        return json.dumps(
            {"stored_at": path.strip(), "category": category, "success": True},
            ensure_ascii=False,
        )
    except Exception as e:
        return _tool_error(str(e))


def knowledge_read(category: str) -> str:
    if category not in VALID_CATEGORIES:
        return _tool_error(
            f"Unknown category '{category}'. Available: {', '.join(VALID_CATEGORIES)}"
        )
    try:
        content = _run("read", category)
        if not content.strip():
            return json.dumps(
                {"category": category, "entries": 0, "content": ""},
                ensure_ascii=False,
            )
        return content
    except Exception as e:
        return _tool_error(str(e))


def knowledge_search(category: str, query: str) -> str:
    if category not in VALID_CATEGORIES:
        return _tool_error(
            f"Unknown category '{category}'. Available: {', '.join(VALID_CATEGORIES)}"
        )
    try:
        return _run("search", category, query)
    except Exception as e:
        return _tool_error(str(e))


def _tool_error(message: str) -> str:
    return json.dumps({"error": message, "success": False}, ensure_ascii=False)


def knowledge_tool(
    action: str, category: str = "", content: str = "", query: str = ""
) -> str:
    action = action.lower().strip()
    if action == "write":
        return knowledge_write(category, content)
    elif action == "read":
        return knowledge_read(category)
    elif action == "search":
        return knowledge_search(category, query)
    return _tool_error(f"Unknown action '{action}'. Use: write, read, search")


def check_knowledge_requirements() -> bool:
    if _find_binary():
        logger.info("knowledge store: binary found")
        return True
    logger.warning("knowledge store: no binary found")
    return False  # Can't work without at least knowledge-py


KNOWLEDGE_SCHEMA = {
    "name": "knowledge",
    "description": (
        "Write, read, and search durable knowledge across sessions. "
        "Four categories: pitfalls (bugs, gotchas), fixes (solutions), "
        "workflows (procedures), facts (observations).\n\n"
        "Write entries using ## field headers:\n"
        "  ## tool\\nfastapi\\n\\n## severity\\nhigh\\n\\n## source\\nWhat happened\\n\\n## fix\\nHow to fix\n\n"
        "Search with tokens and field:value filters: "
        "\"fastapi timeout\", \"tool:fastapi\", \"tool:fastapi timeout\"."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["write", "read", "search"],
            },
            "category": {
                "type": "string",
                "enum": ["pitfalls", "fixes", "workflows", "facts"],
            },
            "content": {
                "type": "string",
                "description": "Markdown with ## headers. For 'write'.",
            },
            "query": {
                "type": "string",
                "description": "Tokens or field:value filters. For 'search'.",
            },
        },
        "required": ["action", "category"],
    },
}

from tools.registry import registry

registry.register(
    name="knowledge",
    toolset="memory",
    schema=KNOWLEDGE_SCHEMA,
    handler=lambda args, **kw: knowledge_tool(
        action=args.get("action", ""),
        category=args.get("category", ""),
        content=args.get("content", ""),
        query=args.get("query", ""),
    ),
    check_fn=check_knowledge_requirements,
    emoji="📚",
)

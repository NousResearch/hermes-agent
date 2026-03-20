#!/usr/bin/env python3
"""
QMD Tool — Native Hermes Anticipatory Memory Integration

Provides semantic memory search via the local QMD server (FastAPI + FAISS).
No external MCP — Hermes talks directly to the QMD server over HTTP.

Schema (tool use by agent):
    qmd_memory { action, query, content, role, tags, top_k }

Actions:
    add       — store a new memory (content required)
    query     — semantic search (query required)
    recent    — get N most recent memories
    delete    — delete a memory by id
    status    — server health + index stats
    clear     — wipe all memories
"""

import httpx
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

QMD_URL = "http://127.0.0.1:8181"
TIMEOUT = 10.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _client() -> httpx.Client:
    return httpx.Client(base_url=QMD_URL, timeout=TIMEOUT)


def _get(path: str, **kw) -> dict:
    try:
        r = _client().get(path, **kw)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        raise RuntimeError(
            "QMD server is not running. Start it with: "
            "python3 ~/.hermes/qmd_server/daemon.py start"
        )
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"QMD HTTP {e.response.status_code}: {e.response.text}")


def _post(path: str, json: dict = None, **kw) -> dict:
    try:
        r = _client().post(path, json=json, **kw)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        raise RuntimeError(
            "QMD server is not running. Start it with: "
            "python3 ~/.hermes/qmd_server/daemon.py start"
        )
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"QMD HTTP {e.response.status_code}: {e.response.text}")


def _delete(path: str) -> dict:
    try:
        r = _client().delete(path)
        r.raise_for_status()
        return r.json()
    except httpx.ConnectError:
        raise RuntimeError(
            "QMD server is not running. Start it with: "
            "python3 ~/.hermes/qmd_server/daemon.py start"
        )
    except httpx.HTTPStatusError as e:
        raise RuntimeError(f"QMD HTTP {e.response.status_code}: {e.response.text}")


# ---------------------------------------------------------------------------
# Tool schema
# ---------------------------------------------------------------------------

QMD_SCHEMA = {
    "name": "qmd_memory",
    "description": "Persistent semantic memory for Hermes. Store and retrieve knowledge "
                  "that persists across sessions. Use 'query' before tasks to check if "
                  "relevant context already exists. Use 'add' after discoveries to save them.",
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "query", "recent", "delete", "status", "clear"],
                "description": "Operation to perform",
            },
            "content": {
                "type": "string",
                "description": "Memory text (for action=add). Store key facts, "
                              "preferences, conventions, and discoveries here.",
            },
            "query": {
                "type": "string",
                "description": "Search query (for action=query). Ask naturally — "
                              "e.g. 'what does user prefer' not keywords.",
            },
            "role": {
                "type": "string",
                "description": "Who this memory belongs to: 'user', 'agent', or 'system'. "
                              "Defaults to 'agent'.",
                "enum": ["user", "agent", "system"],
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional topic tags for filtering.",
            },
            "top_k": {
                "type": "integer",
                "description": "Max results to return (for query/recent). Default: 5.",
            },
            "memory_id": {
                "type": "string",
                "description": "Memory ID (for action=delete).",
            },
            "limit": {
                "type": "integer",
                "description": "Number of recent memories to fetch (for action=recent). Default: 10.",
            },
        },
        "required": ["action"],
    },
}


# ---------------------------------------------------------------------------
# Tool handler
# ---------------------------------------------------------------------------

def qmd_memory(
    action: str,
    content: Optional[str] = None,
    query: Optional[str] = None,
    role: Optional[str] = "agent",
    tags: Optional[list] = None,
    top_k: Optional[int] = 5,
    memory_id: Optional[str] = None,
    limit: Optional[int] = 10,
    **kwargs,
) -> Dict[str, Any]:
    tags = tags or []

    if action == "add":
        if not content:
            return {"error": "content is required for action=add"}
        result = _post("/memories", json={"content": content, "role": role, "tags": tags})
        return {
            "ok": True,
            "message": f"Memory stored (id={result['id']})",
            "id": result["id"],
        }

    elif action == "query":
        if not query:
            return {"error": "query is required for action=query"}
        results = _get("/memories", params={"q": query, "top_k": top_k or 5})
        if not results:
            return {"ok": True, "results": [], "message": "No matching memories found"}
        formatted = [
            {
                "id": r["id"],
                "content": r["content"],
                "role": r["role"],
                "tags": r.get("tags", []),
                "created_at": r["created_at"],
                "score": round(r["score"], 3),
            }
            for r in results
        ]
        return {"ok": True, "results": formatted}

    elif action == "recent":
        results = _get("/memories/recent", params={"limit": limit or 10})
        formatted = [
            {
                "id": r["id"],
                "content": r["content"],
                "role": r["role"],
                "tags": r.get("tags", []),
                "created_at": r["created_at"],
            }
            for r in results
        ]
        return {"ok": True, "results": formatted}

    elif action == "delete":
        if not memory_id:
            return {"error": "memory_id is required for action=delete"}
        result = _delete(f"/memories/{memory_id}")
        return {"ok": True, "message": f"Memory {memory_id} deleted"}

    elif action == "status":
        result = _get("/status")
        return {"ok": True, **result}

    elif action == "clear":
        result = _delete("/memories")
        return {"ok": True, "message": "All memories cleared"}

    else:
        return {"error": f"Unknown action: {action}"}


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def check_qmd_requirements() -> bool:
    """QMD tool is available when the server is reachable."""
    try:
        _get("/status")
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from tools.registry import registry

registry.register(
    name="qmd_memory",
    toolset="memory",
    schema=QMD_SCHEMA,
    handler=lambda args, **kw: qmd_memory(
        action=args.get("action", ""),
        content=args.get("content"),
        query=args.get("query"),
        role=args.get("role", "agent"),
        tags=args.get("tags"),
        top_k=args.get("top_k"),
        memory_id=args.get("memory_id"),
        limit=args.get("limit"),
    ),
    check_fn=check_qmd_requirements,
    emoji="🧠",
)

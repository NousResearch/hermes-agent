"""OpenViking tool — semantic knowledge base via OpenViking REST API.

Registers ten LLM-callable tools:

Core tools (data access):
- ``viking_search``       -- semantic search over the knowledge base
- ``viking_read``         -- read content at viking:// URIs (abstract/overview/full)
- ``viking_browse``       -- filesystem-style listing/tree/stat
- ``viking_remember``     -- store explicit memories
- ``viking_add_resource`` -- ingest URLs/docs into the knowledge base

Higher-level tools (knowledge management):
- ``ov_context``          -- get assembled session context (LLM-synthesized)
- ``ov_stats``            -- memory health stats by category
- ``ov_grep``             -- pattern search across knowledge content
- ``ov_organize``         -- filesystem management (mkdir, mv, rm)
- ``ov_relate``           -- link/unlink related resources

These tools work as a standalone toolset, available whenever
``OPENVIKING_ENDPOINT`` is set.  They can coexist with the memory
provider plugin (which adds lifecycle hooks like session sync,
prefetch, and automatic memory extraction).  When both are active,
the memory provider's registry entries take precedence due to
``run_agent.py``'s re-registration on init.

Authentication uses ``OPENVIKING_API_KEY`` (optional) and trusted-mode
headers ``X-OpenViking-Account`` / ``X-OpenViking-User`` (optional,
default: "default").

API validated against live OpenViking server's /openapi.json spec.
Endpoint mapping:
  Filesystem:  GET /api/v1/fs/{ls,tree,stat}?uri=  POST mkdir, mv  DELETE rm
  Content:     GET /api/v1/content/{read,abstract,overview}?uri=
  Search:      POST /api/v1/search/{find,grep}
  Resources:   POST /api/v1/resources
  Sessions:    GET /api/v1/sessions/{id}/context
  Stats:       GET /api/v1/stats/memories
  Relations:   GET /api/v1/relations  POST /api/v1/relations/link  DELETE unlink
"""

import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

_DEFAULT_ENDPOINT = "http://127.0.0.1:1933"
_TIMEOUT = 30.0


# ---------------------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------------------

def _get_httpx():
    """Lazy import httpx to avoid hard dependency."""
    try:
        import httpx
        return httpx
    except ImportError:
        return None


def _get_config():
    """Return (endpoint, api_key, account, user) from env vars at call time."""
    return (
        os.getenv("OPENVIKING_ENDPOINT", _DEFAULT_ENDPOINT).rstrip("/"),
        os.getenv("OPENVIKING_API_KEY", ""),
        os.getenv("OPENVIKING_ACCOUNT", "default"),
        os.getenv("OPENVIKING_USER", "default"),
    )


def _headers() -> dict:
    """Build request headers with auth."""
    _, api_key, account, user = _get_config()
    h = {"Content-Type": "application/json"}
    if api_key:
        h["X-API-Key"] = api_key
    h["X-OpenViking-Account"] = account or "default"
    h["X-OpenViking-User"] = user or "default"
    return h


def _url(path: str) -> str:
    endpoint, _, _, _ = _get_config()
    return f"{endpoint}{path}"


def _get(path: str, **kwargs) -> dict:
    httpx = _get_httpx()
    resp = httpx.get(_url(path), headers=_headers(), timeout=_TIMEOUT, **kwargs)
    resp.raise_for_status()
    return resp.json()


def _post(path: str, payload: dict = None, **kwargs) -> dict:
    httpx = _get_httpx()
    resp = httpx.post(
        _url(path), json=payload or {}, headers=_headers(),
        timeout=_TIMEOUT, **kwargs
    )
    resp.raise_for_status()
    return resp.json()


def _delete(path: str, payload: dict = None, **kwargs) -> dict:
    httpx = _get_httpx()
    kw = {"headers": _headers(), "timeout": _TIMEOUT, **kwargs}
    if payload:
        kw["json"] = payload
    resp = httpx.request("DELETE", _url(path), **kw)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def _check_openviking_available() -> bool:
    """Tool is only available when OPENVIKING_ENDPOINT is set and httpx importable."""
    if not os.getenv("OPENVIKING_ENDPOINT"):
        return False
    return _get_httpx() is not None


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _handle_search(args: dict, **kw) -> str:
    """Handler for viking_search tool."""
    query = args.get("query", "")
    if not query:
        return json.dumps({"error": "query is required"})

    payload: Dict[str, Any] = {"query": query}
    mode = args.get("mode", "auto")
    if mode != "auto":
        payload["mode"] = mode
    if args.get("scope"):
        payload["target_uri"] = args["scope"]
    if args.get("limit"):
        payload["limit"] = args["limit"]

    try:
        resp = _post("/api/v1/search/find", payload)
    except Exception as e:
        logger.error("viking_search error: %s", e)
        return json.dumps({"error": f"Search failed: {e}"})

    result = resp.get("result", {})

    formatted = []
    for ctx_type in ("memories", "resources", "skills"):
        items = result.get(ctx_type, [])
        for item in items:
            entry = {
                "uri": item.get("uri", ""),
                "type": ctx_type.rstrip("s"),
                "score": round(item.get("score", 0), 3),
                "abstract": item.get("abstract", ""),
            }
            if item.get("relations"):
                entry["related"] = [r.get("uri") for r in item["relations"][:3]]
            formatted.append(entry)

    return json.dumps({
        "results": formatted,
        "total": result.get("total", len(formatted)),
    }, ensure_ascii=False)


def _handle_read(args: dict, **kw) -> str:
    """Handler for viking_read tool."""
    uri = args.get("uri", "")
    if not uri:
        return json.dumps({"error": "uri is required"})

    level = args.get("level", "overview")
    endpoint_map = {
        "abstract": "/api/v1/content/abstract",
        "overview": "/api/v1/content/overview",
        "full": "/api/v1/content/read",
    }
    endpoint = endpoint_map.get(level, "/api/v1/content/overview")

    try:
        resp = _get(endpoint, params={"uri": uri})
    except Exception as e:
        logger.error("viking_read error: %s", e)
        return json.dumps({"error": f"Read failed: {e}"})

    result = resp.get("result", "")
    content = result if isinstance(result, str) else result.get("content", "")

    if len(content) > 8000:
        content = content[:8000] + "\n\n[... truncated, use a more specific URI or abstract level]"

    return json.dumps({
        "uri": uri,
        "level": level,
        "content": content,
    }, ensure_ascii=False)


def _handle_browse(args: dict, **kw) -> str:
    """Handler for viking_browse tool."""
    action = args.get("action", "list")
    path = args.get("path", "viking://")

    endpoint_map = {
        "tree": "/api/v1/fs/tree",
        "list": "/api/v1/fs/ls",
        "stat": "/api/v1/fs/stat",
    }
    endpoint = endpoint_map.get(action, "/api/v1/fs/ls")

    try:
        resp = _get(endpoint, params={"uri": path})
    except Exception as e:
        logger.error("viking_browse error: %s", e)
        return json.dumps({"error": f"Browse failed: {e}"})

    result = resp.get("result", {})

    if action in ("list", "tree") and isinstance(result, list):
        entries = []
        for e in result[:50]:
            entries.append({
                "name": e.get("rel_path", e.get("name", "")),
                "uri": e.get("uri", ""),
                "type": "dir" if e.get("isDir") else "file",
                "abstract": e.get("abstract", ""),
            })
        return json.dumps({"path": path, "entries": entries}, ensure_ascii=False)

    return json.dumps(result, ensure_ascii=False)


def _handle_remember(args: dict, **kw) -> str:
    """Handler for viking_remember tool."""
    content = args.get("content", "")
    if not content:
        return json.dumps({"error": "content is required"})

    category = args.get("category", "")
    text = f"[Remember] {content}"
    if category:
        text = f"[Remember — {category}] {content}"

    # Store via the sessions API — uses a tool-scoped session ID
    session_id = os.getenv("HERMES_SESSION_ID", "default")

    try:
        _post(f"/api/v1/sessions/{session_id}/messages", {
            "role": "user",
            "content": text,
        })
    except Exception as e:
        logger.error("viking_remember error: %s", e)
        return json.dumps({"error": f"Remember failed: {e}"})

    return json.dumps({
        "status": "stored",
        "message": "Memory recorded. Will be extracted and indexed on session commit.",
    })


def _handle_add_resource(args: dict, **kw) -> str:
    """Handler for viking_add_resource tool."""
    url = args.get("url", "")
    if not url:
        return json.dumps({"error": "url is required"})

    payload: Dict[str, Any] = {"path": url}
    if args.get("reason"):
        payload["reason"] = args["reason"]

    try:
        resp = _post("/api/v1/resources", payload)
    except Exception as e:
        logger.error("viking_add_resource error: %s", e)
        return json.dumps({"error": f"Add resource failed: {e}"})

    result = resp.get("result", {})

    return json.dumps({
        "status": "added",
        "root_uri": result.get("root_uri", ""),
        "message": "Resource queued for processing. Use viking_search after a moment to find it.",
    }, ensure_ascii=False)


def _handle_context(args: dict, **kw) -> str:
    """Handler for ov_context — get assembled session context."""
    session_id = args.get("session_id", "")
    if not session_id:
        session_id = os.getenv("HERMES_SESSION_ID", "default")

    params = {}
    if args.get("token_budget"):
        params["token_budget"] = args["token_budget"]

    try:
        resp = _get(f"/api/v1/sessions/{session_id}/context", params=params)
    except Exception as e:
        logger.error("ov_context error: %s", e)
        return json.dumps({"error": f"Context retrieval failed: {e}"})

    result = resp.get("result", {})
    # The context endpoint returns assembled context — could be a string or structured
    if isinstance(result, str):
        content = result
    else:
        content = result.get("context", result.get("content", json.dumps(result)))

    if len(content) > 10000:
        content = content[:10000] + "\n\n[... truncated at 10k chars]"

    return json.dumps({
        "session_id": session_id,
        "context": content,
    }, ensure_ascii=False)


def _handle_stats(args: dict, **kw) -> str:
    """Handler for ov_stats — memory health stats."""
    params = {}
    if args.get("category"):
        params["category"] = args["category"]

    try:
        resp = _get("/api/v1/stats/memories", params=params)
    except Exception as e:
        logger.error("ov_stats error: %s", e)
        return json.dumps({"error": f"Stats retrieval failed: {e}"})

    result = resp.get("result", resp)
    return json.dumps(result, ensure_ascii=False)


def _handle_grep(args: dict, **kw) -> str:
    """Handler for ov_grep — pattern search across knowledge content."""
    pattern = args.get("pattern", "")
    if not pattern:
        return json.dumps({"error": "pattern is required"})

    payload: Dict[str, Any] = {"pattern": pattern}
    payload["uri"] = args.get("uri", "viking://")
    if args.get("case_insensitive") is not None:
        payload["case_insensitive"] = args["case_insensitive"]
    if args.get("limit"):
        payload["node_limit"] = args["limit"]

    try:
        resp = _post("/api/v1/search/grep", payload)
    except Exception as e:
        logger.error("ov_grep error: %s", e)
        return json.dumps({"error": f"Grep failed: {e}"})

    result = resp.get("matches", resp.get("result", []))

    # Format results for the model
    if isinstance(result, list):
        formatted = []
        for item in result[:30]:
            if isinstance(item, str):
                formatted.append({"match": item})
            else:
                entry = {
                    "uri": item.get("uri", ""),
                    "match": item.get("match", item.get("content", "")),
                }
                if item.get("line"):
                    entry["line"] = item["line"]
                formatted.append(entry)
        return json.dumps({"results": formatted, "total": len(result)}, ensure_ascii=False)

    return json.dumps(result, ensure_ascii=False)


def _handle_organize(args: dict, **kw) -> str:
    """Handler for ov_organize — filesystem management (mkdir, mv, rm)."""
    action = args.get("action", "")
    if action not in ("mkdir", "mv", "rm"):
        return json.dumps({"error": "action must be 'mkdir', 'mv', or 'rm'"})

    try:
        if action == "mkdir":
            uri = args.get("uri", "")
            if not uri:
                return json.dumps({"error": "uri is required for mkdir"})
            resp = _post("/api/v1/fs/mkdir", {"uri": uri})
            return json.dumps({
                "action": "mkdir",
                "uri": uri,
                "status": "created",
            })

        elif action == "mv":
            from_uri = args.get("from_uri", "")
            to_uri = args.get("to_uri", "")
            if not from_uri or not to_uri:
                return json.dumps({"error": "from_uri and to_uri are required for mv"})
            resp = _post("/api/v1/fs/mv", {"from_uri": from_uri, "to_uri": to_uri})
            return json.dumps({
                "action": "mv",
                "from": from_uri,
                "to": to_uri,
                "status": "moved",
            })

        elif action == "rm":
            uri = args.get("uri", "")
            if not uri:
                return json.dumps({"error": "uri is required for rm"})
            resp = _delete("/api/v1/fs", params={"uri": uri})
            return json.dumps({
                "action": "rm",
                "uri": uri,
                "status": "deleted",
            })

    except Exception as e:
        logger.error("ov_organize error: %s", e)
        return json.dumps({"error": f"Organize failed: {e}"})


def _handle_relate(args: dict, **kw) -> str:
    """Handler for ov_relate — link/unlink related resources."""
    action = args.get("action", "get")

    try:
        if action == "get":
            uri = args.get("uri", "")
            if not uri:
                return json.dumps({"error": "uri is required for get"})
            resp = _get("/api/v1/relations", params={"uri": uri})
            result = resp.get("result", [])
            return json.dumps({"uri": uri, "relations": result}, ensure_ascii=False)

        elif action == "link":
            from_uri = args.get("from_uri", "")
            to_uris = args.get("to_uris", [])
            if not from_uri or not to_uris:
                return json.dumps({"error": "from_uri and to_uris are required for link"})
            payload = {"from_uri": from_uri, "to_uris": to_uris}
            if args.get("reason"):
                payload["reason"] = args["reason"]
            resp = _post("/api/v1/relations/link", payload)
            return json.dumps({
                "action": "link",
                "from": from_uri,
                "to": to_uris,
                "status": "linked",
            })

        elif action == "unlink":
            from_uri = args.get("from_uri", "")
            to_uri = args.get("to_uri", "")
            if not from_uri or not to_uri:
                return json.dumps({"error": "from_uri and to_uri are required for unlink"})
            resp = _delete("/api/v1/relations/link", payload={
                "from_uri": from_uri, "to_uri": to_uri,
            })
            return json.dumps({
                "action": "unlink",
                "from": from_uri,
                "to": to_uri,
                "status": "unlinked",
            })

        else:
            return json.dumps({"error": "action must be 'get', 'link', or 'unlink'"})

    except Exception as e:
        logger.error("ov_relate error: %s", e)
        return json.dumps({"error": f"Relate failed: {e}"})


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

VIKING_SEARCH_SCHEMA = {
    "name": "viking_search",
    "description": (
        "Semantic search over the OpenViking knowledge base. "
        "Returns ranked results with viking:// URIs for deeper reading. "
        "Use mode='deep' for complex queries that need reasoning across "
        "multiple sources, 'fast' for simple lookups."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
            "mode": {
                "type": "string", "enum": ["auto", "fast", "deep"],
                "description": "Search depth (default: auto).",
            },
            "scope": {
                "type": "string",
                "description": (
                    "Viking URI prefix to scope search "
                    "(e.g. 'viking://resources/docs/')."
                ),
            },
            "limit": {"type": "integer", "description": "Max results (default: 10)."},
        },
        "required": ["query"],
    },
}

VIKING_READ_SCHEMA = {
    "name": "viking_read",
    "description": (
        "Read content at a viking:// URI. Three detail levels:\n"
        "  abstract - ~100 token summary (L0)\n"
        "  overview - ~2k token key points (L1)\n"
        "  full - complete content (L2)\n"
        "Start with abstract/overview, only use full when you need details."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "uri": {"type": "string", "description": "viking:// URI to read."},
            "level": {
                "type": "string", "enum": ["abstract", "overview", "full"],
                "description": "Detail level (default: overview).",
            },
        },
        "required": ["uri"],
    },
}

VIKING_BROWSE_SCHEMA = {
    "name": "viking_browse",
    "description": (
        "Browse the OpenViking knowledge store like a filesystem.\n"
        "  list - show directory contents\n"
        "  tree - show hierarchy\n"
        "  stat - show metadata for a URI"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string", "enum": ["tree", "list", "stat"],
                "description": "Browse action.",
            },
            "path": {
                "type": "string",
                "description": (
                    "Viking URI path (default: viking://). "
                    "Examples: 'viking://resources/', 'viking://user/memories/'."
                ),
            },
        },
        "required": ["action"],
    },
}

VIKING_REMEMBER_SCHEMA = {
    "name": "viking_remember",
    "description": (
        "Explicitly store a fact or memory in the OpenViking knowledge base. "
        "Use for important information the agent should remember long-term. "
        "The system automatically categorizes and indexes the memory."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The information to remember."},
            "category": {
                "type": "string",
                "enum": ["preference", "entity", "event", "case", "pattern"],
                "description": "Memory category (default: auto-detected).",
            },
        },
        "required": ["content"],
    },
}

VIKING_ADD_RESOURCE_SCHEMA = {
    "name": "viking_add_resource",
    "description": (
        "Add a URL or document to the OpenViking knowledge base. "
        "Supports web pages, GitHub repos, PDFs, markdown, code files. "
        "The system automatically parses, indexes, and generates summaries."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL or path of the resource to add."},
            "reason": {
                "type": "string",
                "description": "Why this resource is relevant (improves search).",
            },
        },
        "required": ["url"],
    },
}


# --- Higher-level tools ---

OV_CONTEXT_SCHEMA = {
    "name": "ov_context",
    "description": (
        "Get assembled session context from OpenViking — a synthesized view of "
        "relevant memories, resources, and session history for the current conversation. "
        "Similar to honcho_context but uses OpenViking's retrieval engine. "
        "Optionally specify a token budget to control context size."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "session_id": {
                "type": "string",
                "description": "Session ID to get context for (default: current session).",
            },
            "token_budget": {
                "type": "integer",
                "description": "Max tokens for the returned context (default: server decides).",
            },
        },
        "required": [],
    },
}

OV_STATS_SCHEMA = {
    "name": "ov_stats",
    "description": (
        "Get memory health statistics from the OpenViking knowledge base. "
        "Shows counts by category (preferences, entities, events, cases, patterns), "
        "hotness distribution, and staleness metrics. "
        "Use to understand what the knowledge base contains and its quality."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "description": (
                    "Filter stats by a single category "
                    "(e.g. 'cases', 'patterns', 'preferences', 'entities', 'events'). "
                    "Omit for all categories."
                ),
            },
        },
        "required": [],
    },
}

OV_GREP_SCHEMA = {
    "name": "ov_grep",
    "description": (
        "Pattern search across OpenViking knowledge content — like grep but for "
        "the knowledge base. Searches inside document content, not just metadata. "
        "Use when you need exact string/regex matching rather than semantic search."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Search pattern (supports regex)."},
            "uri": {
                "type": "string",
                "description": "Scope to a URI prefix (e.g. 'viking://resources/'). Default: all.",
            },
            "case_insensitive": {
                "type": "boolean",
                "description": "Case-insensitive matching (default: false).",
            },
            "limit": {
                "type": "integer",
                "description": "Max results (default: server decides).",
            },
        },
        "required": ["pattern"],
    },
}

OV_ORGANIZE_SCHEMA = {
    "name": "ov_organize",
    "description": (
        "Manage the OpenViking knowledge base filesystem — create directories, "
        "move resources, or delete entries. Like honcho_workspace but more granular.\n"
        "  mkdir — create a new directory at a viking:// URI\n"
        "  mv — move/rename a resource from one URI to another\n"
        "  rm — delete a resource or directory"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string", "enum": ["mkdir", "mv", "rm"],
                "description": "Filesystem action.",
            },
            "uri": {
                "type": "string",
                "description": "Target URI for mkdir/rm (e.g. 'viking://resources/new-project/').",
            },
            "from_uri": {
                "type": "string",
                "description": "Source URI for mv.",
            },
            "to_uri": {
                "type": "string",
                "description": "Destination URI for mv.",
            },
        },
        "required": ["action"],
    },
}

OV_RELATE_SCHEMA = {
    "name": "ov_relate",
    "description": (
        "Manage relationships between resources in the OpenViking knowledge base.\n"
        "  get — show what a resource is linked to\n"
        "  link — create a relationship between resources (with optional reason)\n"
        "  unlink — remove a relationship\n"
        "Relationships help OpenViking surface related content during search."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string", "enum": ["get", "link", "unlink"],
                "description": "Relation action (default: get).",
            },
            "uri": {
                "type": "string",
                "description": "URI to get relations for (get action).",
            },
            "from_uri": {
                "type": "string",
                "description": "Source URI for link/unlink.",
            },
            "to_uris": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Target URIs to link to (link action).",
            },
            "to_uri": {
                "type": "string",
                "description": "Target URI to unlink from (unlink action).",
            },
            "reason": {
                "type": "string",
                "description": "Why these resources are related (improves search, link action only).",
            },
        },
        "required": ["action"],
    },
}


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

from tools.registry import registry

registry.register(
    name="viking_search",
    toolset="openviking",
    schema=VIKING_SEARCH_SCHEMA,
    handler=_handle_search,
    check_fn=_check_openviking_available,
    requires_env=["OPENVIKING_ENDPOINT"],
    emoji="V",
)

registry.register(
    name="viking_read",
    toolset="openviking",
    schema=VIKING_READ_SCHEMA,
    handler=_handle_read,
    check_fn=_check_openviking_available,
    requires_env=["OPENVIKING_ENDPOINT"],
    emoji="V",
)

registry.register(
    name="viking_browse",
    toolset="openviking",
    schema=VIKING_BROWSE_SCHEMA,
    handler=_handle_browse,
    check_fn=_check_openviking_available,
    requires_env=["OPENVIKING_ENDPOINT"],
    emoji="V",
)

registry.register(
    name="viking_remember",
    toolset="openviking",
    schema=VIKING_REMEMBER_SCHEMA,
    handler=_handle_remember,
    check_fn=_check_openviking_available,
    requires_env=["OPENVIKING_ENDPOINT"],
    emoji="V",
)

registry.register(
    name="viking_add_resource",
    toolset="openviking",
    schema=VIKING_ADD_RESOURCE_SCHEMA,
    handler=_handle_add_resource,
    check_fn=_check_openviking_available,
    requires_env=["OPENVIKING_ENDPOINT"],
    emoji="V",
)

registry.register(
    name="ov_context",
    toolset="openviking",
    schema=OV_CONTEXT_SCHEMA,
    handler=_handle_context,
    check_fn=_check_openviking_available,
    requires_env=["OPENVIKING_ENDPOINT"],
    emoji="V",
)

registry.register(
    name="ov_stats",
    toolset="openviking",
    schema=OV_STATS_SCHEMA,
    handler=_handle_stats,
    check_fn=_check_openviking_available,
    requires_env=["OPENVIKING_ENDPOINT"],
    emoji="V",
)

registry.register(
    name="ov_grep",
    toolset="openviking",
    schema=OV_GREP_SCHEMA,
    handler=_handle_grep,
    check_fn=_check_openviking_available,
    requires_env=["OPENVIKING_ENDPOINT"],
    emoji="V",
)

registry.register(
    name="ov_organize",
    toolset="openviking",
    schema=OV_ORGANIZE_SCHEMA,
    handler=_handle_organize,
    check_fn=_check_openviking_available,
    requires_env=["OPENVIKING_ENDPOINT"],
    emoji="V",
)

registry.register(
    name="ov_relate",
    toolset="openviking",
    schema=OV_RELATE_SCHEMA,
    handler=_handle_relate,
    check_fn=_check_openviking_available,
    requires_env=["OPENVIKING_ENDPOINT"],
    emoji="V",
)

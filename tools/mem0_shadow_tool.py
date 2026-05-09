"""mem0 shadow-memory tools.

This is intentionally a tool lane, not a MemoryProvider cutover. It lets Hermes
compare Cortex/cmem recall with self-hosted mem0 shadow recall on demand while
keeping the live memory provider unchanged.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from hermes_constants import get_hermes_home
from plugins.memory.mem0 import _Mem0RestClient, _as_bool, _as_int
from tools.registry import registry, tool_error


CONFIG_FILE = "mem0_shadow.json"


MEM0_SHADOW_SEARCH_SCHEMA = {
    "name": "mem0_shadow_search",
    "description": (
        "Search the self-hosted mem0 shadow shared-memory lane. Use this only to "
        "compare shadow recall against Cortex/cmem or source artifacts; it is not "
        "the authoritative memory store. Requires MEM0_SHADOW_* config."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Memory query."},
            "top_k": {"type": "integer", "description": "Max returned results (default: 5, max: 25)."},
            "project": {"type": "string", "description": "Optional project metadata filter override."},
            "user_id": {"type": "string", "description": "Optional shadow namespace override; defaults to config."},
            "agent_id": {"type": "string", "description": "Optional shadow agent namespace override; defaults to config."},
            "include_raw": {"type": "boolean", "description": "Include the raw mem0 result objects (default: false)."},
        },
        "required": ["query"],
    },
}


def _load_shadow_config() -> Dict[str, Any]:
    """Load mem0 shadow tool config from env and $HERMES_HOME/mem0_shadow.json.

    Env vars are preferred for secrets; the JSON file is useful for non-secret
    namespace defaults. File values override env only when non-empty, matching
    Hermes memory-provider config behavior.
    """
    cfg: Dict[str, Any] = {
        "base_url": os.environ.get("MEM0_SHADOW_BASE_URL") or os.environ.get("MEM0_BASE_URL", ""),
        "api_key": os.environ.get("MEM0_SHADOW_API_KEY", ""),
        "user_id": os.environ.get("MEM0_SHADOW_USER_ID", "joohyun-memory-shadow-r3"),
        "agent_id": os.environ.get("MEM0_SHADOW_AGENT_ID", "cortex-shadow-structured-r3"),
        "project": os.environ.get("MEM0_SHADOW_PROJECT", "mina-operating-system"),
        "candidate_k": os.environ.get("MEM0_SHADOW_CANDIDATE_K", "50"),
        "local_rerank": os.environ.get("MEM0_SHADOW_LOCAL_RERANK", "true"),
        "strict_search": os.environ.get("MEM0_SHADOW_STRICT_SEARCH", "true"),
    }

    path = get_hermes_home() / CONFIG_FILE
    if path.exists():
        try:
            file_cfg = json.loads(path.read_text(encoding="utf-8"))
            cfg.update({k: v for k, v in file_cfg.items() if v is not None and v != ""})
        except Exception:
            pass
    return cfg


def _requirements_met() -> bool:
    cfg = _load_shadow_config()
    return bool(cfg.get("base_url") and cfg.get("api_key"))


def _search(args: Dict[str, Any], **_kwargs) -> str:
    cfg = _load_shadow_config()
    base_url = str(cfg.get("base_url") or "").rstrip("/")
    api_key = str(cfg.get("api_key") or "")
    if not base_url or not api_key:
        return tool_error(
            "mem0 shadow search is not configured. Set MEM0_SHADOW_BASE_URL and "
            "MEM0_SHADOW_API_KEY, or create $HERMES_HOME/mem0_shadow.json."
        )

    query = str(args.get("query") or "").strip()
    if not query:
        return tool_error("Missing required parameter: query")

    top_k = max(1, min(_as_int(args.get("top_k"), default=5), 25))
    user_id = str(args.get("user_id") or cfg.get("user_id") or "").strip()
    agent_id = str(args.get("agent_id") or cfg.get("agent_id") or "").strip()
    project = str(args.get("project") or cfg.get("project") or "").strip()
    include_raw = _as_bool(args.get("include_raw"), default=False)
    strict_search = _as_bool(cfg.get("strict_search"), default=True)

    filters: Dict[str, Any] = {"user_id": user_id}
    if strict_search:
        if agent_id:
            filters["agent_id"] = agent_id
        if project:
            filters["project"] = project

    client = _Mem0RestClient(
        base_url=base_url,
        api_key=api_key,
        candidate_k=max(1, _as_int(cfg.get("candidate_k"), default=50)),
        local_rerank=_as_bool(cfg.get("local_rerank"), default=True),
    )

    try:
        response = client.search(query=query, filters=filters, top_k=top_k, rerank=False)
    except Exception as exc:
        return tool_error(f"mem0 shadow search failed: {exc}")

    raw_results = response.get("results", []) if isinstance(response, dict) else (response or [])
    items = []
    for item in raw_results[:top_k]:
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") or {}
        items.append({
            "memory": item.get("memory") or item.get("text") or "",
            "score": item.get("score"),
            "source_id": metadata.get("source_id"),
            "project": metadata.get("project") or project,
            "record_type": metadata.get("record_type") or metadata.get("type"),
            "metadata": metadata,
        })

    payload: Dict[str, Any] = {
        "query": query,
        "count": len(items),
        "filters": filters,
        "shadow_only": True,
        "authoritative": False,
        "results": items,
    }
    if include_raw:
        payload["raw"] = raw_results[:top_k]
    return json.dumps(payload, ensure_ascii=False)


registry.register(
    name="mem0_shadow_search",
    toolset="memory",
    schema=MEM0_SHADOW_SEARCH_SCHEMA,
    handler=_search,
    check_fn=_requirements_met,
    requires_env=["MEM0_SHADOW_BASE_URL", "MEM0_SHADOW_API_KEY"],
    description="Search self-hosted mem0 shadow shared-memory lane",
    emoji="🧠",
)

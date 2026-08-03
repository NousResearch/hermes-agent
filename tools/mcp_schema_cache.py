"""Persistent MCP tool-schema cache for lazy server startup.

Stores per-server tool manifests on disk so Hermes can register MCP tools
into the agent snapshot without spawning the stdio child process at idle
dashboard startup. Cache entries are keyed by server name + a fingerprint
of the connection config (command/args/url/tools filters).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CACHE_FILENAME = "mcp_schema_cache.json"
_cache_lock = threading.Lock()


def _cache_path() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / _CACHE_FILENAME


def config_fingerprint(config: dict) -> str:
    """Stable hash of the connection-defining parts of an MCP server config."""
    tools_filter = config.get("tools") or {}
    payload = {
        "command": config.get("command"),
        "args": config.get("args") or [],
        "url": config.get("url"),
        "transport": config.get("transport"),
        "tools_include": sorted(tools_filter.get("include") or []),
        "tools_exclude": sorted(tools_filter.get("exclude") or []),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _load_all() -> Dict[str, Any]:
    path = _cache_path()
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception as exc:
        logger.debug("Could not read MCP schema cache %s: %s", path, exc)
        return {}


def _save_all(data: Dict[str, Any]) -> None:
    path = _cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def get_cached_entry(server_name: str, fingerprint: str) -> Optional[dict]:
    """Return cached entry when fingerprint matches, else None."""
    with _cache_lock:
        entry = _load_all().get(server_name)
    if not isinstance(entry, dict):
        return None
    if entry.get("fingerprint") != fingerprint:
        return None
    return entry


def has_cached_entry(server_name: str, fingerprint: str) -> bool:
    return get_cached_entry(server_name, fingerprint) is not None


def write_cache_entry(
    server_name: str,
    fingerprint: str,
    *,
    tools: List[dict],
    utility_tools: Optional[List[dict]] = None,
) -> None:
    """Persist tool schemas after a successful live connect."""
    with _cache_lock:
        data = _load_all()
        data[server_name] = {
            "fingerprint": fingerprint,
            "tools": tools,
            "utility_tools": utility_tools or [],
        }
        _save_all(data)


def clear_cache_entry(server_name: str) -> None:
    with _cache_lock:
        data = _load_all()
        if server_name in data:
            del data[server_name]
            _save_all(data)


def tools_from_cache_entry(entry: dict) -> List[dict]:
    """Return cached MCP tool dicts (name, description, inputSchema)."""
    tools = entry.get("tools")
    return list(tools) if isinstance(tools, list) else []


def utility_tools_from_cache_entry(entry: dict) -> List[dict]:
    util = entry.get("utility_tools")
    return list(util) if isinstance(util, list) else []

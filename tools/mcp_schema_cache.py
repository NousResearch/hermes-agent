"""Persistent MCP tool-schema cache for lazy server startup: per-server tool manifests on
disk so Hermes can register MCP tools into the agent snapshot without spawning the stdio
child at idle dashboard startup. Entries are keyed by server name + a fingerprint of the
connection config (command/args/url/tools filters)."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_CACHE_FILENAME = "mcp_schema_cache.json"
_cache_lock = threading.Lock()


def _cache_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cache" / _CACHE_FILENAME


def config_fingerprint(config: dict) -> str:
    """Stable hash of the connection-defining parts of an MCP server config."""
    tools_filter = config.get("tools") or {}
    payload = {
        "command": config.get("command"),
        "args": config.get("args") or [],
        "url": config.get("url"),
        "transport": config.get("transport"),
        "tools_include": sorted(tools_filter.get("include") or []),
        "tools_exclude": sorted(tools_filter.get("exclude") or [])}
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
    from utils import atomic_json_write
    # 0o600: the cache file is trusted input on the lazy registration path, keep it user-only.
    atomic_json_write(_cache_path(), data, mode=0o600)


def get_cached_entry(server_name: str, fingerprint: str) -> Optional[dict]:
    """Return cached entry when fingerprint matches (and TTL holds), else None. ``tools/list``
    results may carry ``ttlMs`` (SEP-2549); an entry older than a recorded TTL is a miss so the
    next startup re-probes instead of serving a stale manifest forever. Entries without a TTL
    never expire. ``cacheScope`` is irrelevant: this cache is per-user local disk."""
    with _cache_lock:
        entry = _load_all().get(server_name)
    if not isinstance(entry, dict) or entry.get("fingerprint") != fingerprint:
        return None
    ttl_ms = entry.get("ttl_ms")
    written_at = entry.get("written_at")
    expired = (isinstance(ttl_ms, (int, float)) and isinstance(written_at, (int, float))
               and (time.time() - written_at) * 1000.0 >= float(ttl_ms))
    return None if expired else entry


def write_cache_entry(server_name: str, fingerprint: str, *, tools: List[dict],
                      utility_tools: Optional[List[dict]] = None, ttl_ms: Optional[float] = None,
                      cache_scope: Optional[str] = None) -> None:
    """Persist tool schemas after a successful live connect. ``ttl_ms`` / ``cache_scope`` are
    the server's ``tools/list`` SEP-2549 hints; ``written_at`` anchors TTL expiry."""
    entry = {"fingerprint": fingerprint, "tools": tools, "utility_tools": utility_tools or []}
    if isinstance(ttl_ms, (int, float)):
        entry["ttl_ms"] = ttl_ms
        entry["written_at"] = time.time()
    if cache_scope:
        entry["cache_scope"] = cache_scope
    with _cache_lock:
        data = _load_all()
        # Write-through fires on every registration (reconnects, list_changed); skip the
        # rewrite when the entry is byte-identical on disk. TTL'd entries always rewrite:
        # written_at must advance or the entry would expire at its ORIGINAL write time.
        if "written_at" not in entry and data.get(server_name) == entry:
            return
        data[server_name] = entry
        _save_all(data)


def _list_field(entry: dict, key: str) -> List[dict]:
    value = entry.get(key)
    return list(value) if isinstance(value, list) else []


def tools_from_cache_entry(entry: dict) -> List[dict]:
    """Return cached MCP tool dicts (name, description, inputSchema)."""
    return _list_field(entry, "tools")


def utility_tools_from_cache_entry(entry: dict) -> List[dict]:
    """Return cached ``{schema, handler_key}`` utility rows."""
    return _list_field(entry, "utility_tools")


# ---- BEGIN PLUGIN-COMPAT (revert-scheduled; see COMPAT_MANIFEST.md) ----
# Names external plugins imported from this module before the Sep 2026 decomposition.
# Internal code MUST NOT use these (scripts/check_compat_pointers.py fails CI if it does).
# The whole block is removed by reverting the commit that added it.

def clear_cache_entry(server_name: str) -> None:
    with _cache_lock:
        data = _load_all()
        if server_name in data:
            del data[server_name]
            _save_all(data)

def has_cached_entry(server_name: str, fingerprint: str) -> bool:
    return get_cached_entry(server_name, fingerprint) is not None
# ---- END PLUGIN-COMPAT ----

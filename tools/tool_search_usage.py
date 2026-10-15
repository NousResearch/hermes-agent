"""Usage-learned eager tools for tool search.

A deferred tool the user's sessions reach for constantly (a workflow tool, smart-home
controls) costs a ``tool_describe`` round and a ``tool_call`` wrapper every session, and small
models drop the wrapper's ``name`` once the arguments grow long. Tools used in many recent
sessions are therefore shown directly. The counts come from the active profile's ``state.db``
(read-only) and are cached in its ``cache/tool_usage_counts.json`` for ``_CACHE_TTL_SECONDS``, so
the visible tool set changes at most that often and the prompt cache stays warm in between.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Iterable

logger = logging.getLogger(__name__)

_WINDOW_DAYS = 14
_CACHE_TTL_SECONDS = 6 * 3600
# A learned tool whose schema is larger than this stays deferred: the point is saving
# rounds, not shipping a huge schema with every request.
_MAX_SCHEMA_CHARS = 6000

_lock = threading.Lock()
# Keyed by state.db path: one process serves several profiles, each with its own history.
_memo: dict[Path, tuple[float, dict[str, int]]] = {}


def _paths() -> tuple[Path, Path]:
    from hermes_constants import get_hermes_home
    home = get_hermes_home()
    return home / "state.db", home / "cache" / "tool_usage_counts.json"


def _query_counts(db_path: Path) -> dict[str, int]:
    """Distinct sessions per tool name over the window (role='tool' rows carry the real name,
    also for calls made through the tool_call bridge)."""
    if not db_path.exists():
        return {}
    since = time.time() - _WINDOW_DAYS * 86400
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=2)
    try:
        rows = conn.execute(
            "SELECT tool_name, COUNT(DISTINCT session_id) FROM messages "
            "WHERE role = 'tool' AND tool_name IS NOT NULL AND timestamp > ? GROUP BY tool_name",
            (since,)).fetchall()
    finally:
        conn.close()
    return {str(name): int(n) for name, n in rows if name}


def recent_tool_session_counts() -> dict[str, int]:
    """Cached ``{tool_name: sessions}`` for the active profile; ``{}`` when the history cannot be
    read."""
    now = time.time()
    db_path, cache_path = _paths()
    with _lock:
        memo = _memo.get(db_path)
        if memo and now - memo[0] < _CACHE_TTL_SECONDS:
            return memo[1]
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8-sig"))
            if now - float(cached["computed_at"]) < _CACHE_TTL_SECONDS:
                _memo[db_path] = (float(cached["computed_at"]), dict(cached["counts"]))
                return _memo[db_path][1]
        except (OSError, ValueError, KeyError, TypeError):
            pass
        try:
            counts = _query_counts(db_path)
        except sqlite3.Error as e:
            logger.debug("tool usage counts unavailable: %s", e)
            counts = {}
        _memo[db_path] = (now, counts)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = cache_path.with_suffix(".tmp")
            tmp.write_text(json.dumps({"computed_at": now, "counts": counts}), encoding="utf-8")
            tmp.replace(cache_path)
        except OSError as e:
            logger.debug("tool usage cache not written: %s", e)
        return counts


def learned_eager_tools(deferrable_defs: Iterable[dict[str, Any]], *, min_sessions: int,
                        max_tools: int, counts: dict[str, int] | None = None) -> frozenset[str]:
    """The most-used deferrable tools (at least ``min_sessions`` recent sessions, at most
    ``max_tools`` of them, small schemas only)."""
    if min_sessions <= 0 or max_tools <= 0:
        return frozenset()
    counts = recent_tool_session_counts() if counts is None else counts
    candidates = []
    for td in deferrable_defs:
        fn = td.get("function", td) if isinstance(td, dict) else {}
        name = fn.get("name", "")
        used = counts.get(name, 0)
        if used >= min_sessions and len(json.dumps(fn, ensure_ascii=False)) <= _MAX_SCHEMA_CHARS:
            candidates.append((-used, name))
    return frozenset(name for _, name in sorted(candidates)[:max_tools])

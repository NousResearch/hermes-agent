#!/usr/bin/env python3
"""Opt-in local per-tool usage tracker and analytics.

Records every tool call (name, timestamp, success, estimated token cost) in a
sidecar SQLite store under ``<HERMES_HOME>/tool_usage.db``.  All data stays
local — no outbound telemetry.

Enable via ``tools.analytics: true`` in config.yaml (default off).
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_CONFIG_KEY = "tools.analytics"
_DB_NAME = "tool_usage.db"
_PRUNE_THRESHOLD = 10  # sessions with zero calls before suggesting disable

# ---------------------------------------------------------------------------
# Schema & DB management
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tool_calls (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tool_name   TEXT NOT NULL,
    ts          REAL NOT NULL,
    success     INTEGER NOT NULL DEFAULT 1,
    token_est   INTEGER NOT NULL DEFAULT 0,
    session_id  TEXT,
    turn_id     TEXT
);
CREATE INDEX IF NOT EXISTS idx_tool_calls_tool ON tool_calls(tool_name);
CREATE INDEX IF NOT EXISTS idx_tool_calls_ts   ON tool_calls(ts);
CREATE TABLE IF NOT EXISTS call_counts (
    tool_name   TEXT PRIMARY KEY,
    total       INTEGER NOT NULL DEFAULT 0,
    successes   INTEGER NOT NULL DEFAULT 0,
    last_used   REAL
);
"""

_local = threading.local()


def _db_path() -> Path:
    return get_hermes_home() / _DB_NAME


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    path = _db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    conn.executescript(_SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def record_call(
    tool_name: str,
    success: bool | None = None,
    *,
    token_est: int = 0,
    result: str | None = None,
    session_id: str | None = None,
    turn_id: str | None = None,
) -> None:
    """Record one tool call in the local store.

    Args:
        tool_name: Name of the tool called.
        success: Whether the call succeeded. If None, derived from result.
        token_est: Estimated token cost. If 0, estimated from result length.
        result: The raw result string from the tool dispatch. Used to derive
            success when not explicitly provided and to estimate token cost.
    """
    if not is_enabled():
        return
    if success is None and result is not None:
        success = _is_successful(result)
    if token_est == 0 and result is not None:
        token_est = _estimate_tokens(result)
    try:
        with _connect() as conn:
            now = time.time()
            conn.execute(
                "INSERT INTO tool_calls (tool_name, ts, success, token_est, session_id, turn_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (tool_name, now, 1 if success else 0, token_est, session_id, turn_id),
            )
            conn.execute(
                "INSERT INTO call_counts (tool_name, total, successes, last_used) "
                "VALUES (?, 1, ?, ?) "
                "ON CONFLICT(tool_name) DO UPDATE SET "
                "total = total + 1, "
                "successes = successes + ?, "
                "last_used = MAX(last_used, ?)",
                (tool_name, 1 if success else 0, now, 1 if success else 0, now),
            )
    except Exception:
        logger.warning("tool_usage: failed to record call to %s", tool_name, exc_info=True)


def _is_successful(result: str) -> bool:
    """Determine if a tool call succeeded based on its result string."""
    if result.startswith("[TOOL_ERROR]"):
        return False
    try:
        parsed = json.loads(result)
        if isinstance(parsed, dict):
            if parsed.get("error"):
                return False
            if parsed.get("success") is False:
                return False
    except (json.JSONDecodeError, TypeError):
        pass
    return True


def _estimate_tokens(result: str) -> int:
    """Rough token estimate from result length (~4 chars per token)."""
    return max(1, len(result) // 4)


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def _dict_row(row: sqlite3.Row) -> Dict[str, Any]:
    return dict(zip(row.keys(), row))


def tool_summary() -> Dict[str, Any]:
    """Return per-tool aggregate stats."""
    if not _db_path().exists():
        return {"tools": [], "total_calls": 0}
    with _connect() as conn:
        rows = conn.execute(
            "SELECT tool_name, total, successes, "
            "ROUND(CAST(successes AS REAL) / MAX(total, 1) * 100, 1) AS success_pct, "
            "last_used "
            "FROM call_counts ORDER BY total DESC"
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) AS c FROM tool_calls").fetchone()["c"]
    return {"tools": [_dict_row(r) for r in rows], "total_calls": total}


def cost_summary() -> Dict[str, Any]:
    """Return per-tool estimated token cost breakdown."""
    if not _db_path().exists():
        return {"tools": [], "total_tokens": 0}
    with _connect() as conn:
        rows = conn.execute(
            "SELECT tool_name, COUNT(*) AS calls, "
            "SUM(token_est) AS total_tokens, "
            "ROUND(AVG(token_est), 1) AS avg_tokens "
            "FROM tool_calls GROUP BY tool_name ORDER BY total_tokens DESC"
        ).fetchall()
        total = conn.execute("SELECT COALESCE(SUM(token_est), 0) AS t FROM tool_calls").fetchone()["t"]
    return {"tools": [_dict_row(r) for r in rows], "total_tokens": total}


def session_tool_counts(min_sessions: int = 1) -> Dict[str, int]:
    """Return how many unique sessions each tool was called in."""
    if not _db_path().exists():
        return {}
    with _connect() as conn:
        rows = conn.execute(
            "SELECT tool_name, COUNT(DISTINCT session_id) AS sessions FROM tool_calls "
            "WHERE session_id IS NOT NULL AND session_id != '' "
            "GROUP BY tool_name"
        ).fetchall()
    return {r["tool_name"]: r["sessions"] for r in rows}


# ---------------------------------------------------------------------------
# Prune suggestions
# ---------------------------------------------------------------------------

def suggest_prune() -> List[Dict[str, Any]]:
    """Return tools/toolsets used so rarely they could be disabled.

    A tool is a prune candidate when it has been called in fewer than
    ``_PRUNE_THRESHOLD`` distinct sessions.  Returns a list sorted by
    increasing session count (coldest first).
    """
    per_tool = session_tool_counts(min_sessions=1)
    if not per_tool:
        return []
    candidates = [
        {"tool_name": name, "sessions": count}
        for name, count in sorted(per_tool.items(), key=lambda kv: kv[1])
        if count < _PRUNE_THRESHOLD
    ]
    return candidates


def generate_prune_diff() -> Dict[str, Any]:
    """Generate a suggested ``tools.<platform>.disabled`` diff."""
    candidates = suggest_prune()
    if not candidates:
        return {"suggestions": [], "message": "All tools are actively used across sessions."}
    disabled = [c["tool_name"] for c in candidates]
    return {
        "suggestions": candidates,
        "diff_hint": (
            "Consider adding to your config.yaml under the relevant platform:\n"
            "  platform_toolsets.cli:\n"
            "    - hermes-cli\n"
            "  platform_toolsets.cli.disabled:\n"
            + "\n".join(f"    - {name}" for name in disabled)
        ),
    }


# ---------------------------------------------------------------------------
# Analytics report (plain text for CLI / gateway)
# ---------------------------------------------------------------------------

def analytics_report(verbose: bool = False) -> str:
    """Build a human-readable analytics report string."""
    summary = tool_summary()
    costs = cost_summary()
    prune = generate_prune_diff()

    if summary["total_calls"] == 0:
        return "No tool usage data recorded yet. Enable `tools.analytics: true` in config.yaml."

    lines = [f"Tool Usage Analytics ({summary['total_calls']} total calls)", "=" * 50]

    lines.append("")
    lines.append("Top tools by call count:")
    for t in summary["tools"][:15]:
        lines.append(
            f"  {t['tool_name']:<30s}  {t['total']:>5d} calls  "
            f"{t['success_pct']:.0f}% success"
        )
    if len(summary["tools"]) > 15:
        lines.append(f"  ... and {len(summary['tools']) - 15} more tools")

    lines.append("")
    lines.append("Estimated token cost by tool:")
    for t in costs["tools"][:10]:
        lines.append(
            f"  {t['tool_name']:<30s}  {t['total_tokens']:>8,d} tokens  "
            f"avg {t['avg_tokens']:.0f}/call"
        )
    if costs["total_tokens"]:
        lines.append(f"  {'TOTAL':<30s}  {costs['total_tokens']:>8,d} tokens")

    if prune["suggestions"]:
        lines.append("")
        lines.append("Prune suggestions (consider disabling):")
        for c in prune["suggestions"]:
            pct = max(1, int(c["sessions"] / max(summary["total_calls"], 1) * 100))
            lines.append(f"  {c['tool_name']:<30s}  used in {c['sessions']} session(s)")
    else:
        lines.append("")
        lines.append("No prune candidates — all tools are actively used.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Config gate
# ---------------------------------------------------------------------------

def is_enabled() -> bool:
    """Return whether the tool usage tracker is enabled in config."""
    try:
        from hermes_cli.config import load_config, cfg_get
        cfg = load_config()
        raw = cfg_get(cfg, "tools", "analytics", default=False)
    except Exception:
        return False
    return _normalize(raw)


def _normalize(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"on", "true", "yes", "1", "enabled"}
    return False


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def cmd_analytics(args: Any) -> None:
    """``hermes tools analytics`` handler."""
    if not is_enabled():
        print(
            "Tool analytics is disabled.\n"
            "Enable it with:  hermes config set tools.analytics true"
        )
        return
    print(analytics_report(verbose=getattr(args, "verbose", False)))


def cmd_prune(args: Any) -> None:
    """``hermes tools prune`` handler."""
    if not is_enabled():
        print(
            "Tool analytics is disabled.\n"
            "Enable it with:  hermes config set tools.analytics true"
        )
        return
    result = generate_prune_diff()
    if result["suggestions"]:
        print(f"Prune candidates ({len(result['suggestions'])} found):\n")
        for c in result["suggestions"]:
            print(f"  {c['tool_name']}  (used in {c['sessions']} session(s))")
        print("")
        print(result["diff_hint"])
    else:
        print(result["message"])


def cmd_record_stats() -> str:
    """Return a brief stats line for status displays."""
    summary = tool_summary()
    if summary["total_calls"] == 0:
        return "No tool usage data."
    return f"Tool usage: {summary['total_calls']} calls across {len(summary['tools'])} tools."
"""Feedback collector for Hermes Agent Sloop loop system.

Automatically collects execution feedback from cron job runs (success/failure,
duration, error type) and stores it in ~/.hermes/sloop_feedback.db (SQLite).

The collector can also aggregate feedback data for the sloop dashboard.
"""

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

# ── DB location ──────────────────────────────────────────────────────────────

def get_feedback_db_path() -> Path:
    """Return path to the sloop feedback SQLite database."""
    return get_hermes_home() / "sloop_feedback.db"


# ── Schema ───────────────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id      TEXT NOT NULL,
    job_name    TEXT,
    timestamp   TEXT NOT NULL,
    success     INTEGER NOT NULL,  -- 1 = ok, 0 = fail
    duration_ms REAL,
    error_type  TEXT,
    error_msg   TEXT,
    delivery_ok INTEGER DEFAULT 1, -- 1 = delivered, 0 = delivery failed
    metadata    TEXT               -- JSON blob for extensibility
);

CREATE INDEX IF NOT EXISTS idx_feedback_job_id ON feedback(job_id);
CREATE INDEX IF NOT EXISTS idx_feedback_timestamp ON feedback(timestamp);
CREATE INDEX IF NOT EXISTS idx_feedback_success ON feedback(success);
"""


@contextmanager
def _get_conn():
    """Context manager yielding a SQLite connection with WAL mode."""
    db_path = get_feedback_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _ensure_schema():
    """Create tables if they don't exist."""
    with _get_conn() as conn:
        conn.executescript(_SCHEMA_SQL)


# ── Recording ────────────────────────────────────────────────────────────────

def record_feedback(
    job_id: str,
    job_name: Optional[str] = None,
    success: bool = False,
    duration_ms: Optional[float] = None,
    error_type: Optional[str] = None,
    error_msg: Optional[str] = None,
    delivery_ok: bool = True,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Record a single job execution feedback entry."""
    _ensure_schema()
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO feedback
               (job_id, job_name, timestamp, success, duration_ms,
                error_type, error_msg, delivery_ok, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job_id,
                job_name,
                datetime.now(timezone.utc).isoformat(),
                1 if success else 0,
                duration_ms,
                error_type,
                error_msg,
                1 if delivery_ok else 0,
                json.dumps(metadata) if metadata else None,
            ),
        )


# ── Query helpers ────────────────────────────────────────────────────────────

def get_recent_feedback(
    job_id: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Get recent feedback entries, optionally filtered by job_id."""
    _ensure_schema()
    with _get_conn() as conn:
        if job_id:
            rows = conn.execute(
                "SELECT * FROM feedback WHERE job_id = ? ORDER BY timestamp DESC LIMIT ?",
                (job_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM feedback ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]


def get_success_rate(
    job_id: Optional[str] = None,
    hours: int = 24,
) -> Dict[str, Any]:
    """Compute success rate over the last N hours."""
    _ensure_schema()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with _get_conn() as conn:
        if job_id:
            rows = conn.execute(
                "SELECT success FROM feedback WHERE job_id = ? AND timestamp >= ?",
                (job_id, cutoff),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT success FROM feedback WHERE timestamp >= ?",
                (cutoff,),
            ).fetchall()
    total = len(rows)
    successes = sum(1 for r in rows if r["success"])
    failures = total - successes
    rate = (successes / total * 100) if total > 0 else 0.0
    return {
        "total": total,
        "successes": successes,
        "failures": failures,
        "rate_percent": round(rate, 1),
        "hours": hours,
    }


def get_avg_duration(
    job_id: Optional[str] = None,
    hours: int = 24,
) -> Optional[float]:
    """Compute average execution duration in ms over the last N hours."""
    _ensure_schema()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with _get_conn() as conn:
        if job_id:
            row = conn.execute(
                "SELECT AVG(duration_ms) as avg_ms FROM feedback "
                "WHERE job_id = ? AND timestamp >= ? AND duration_ms IS NOT NULL",
                (job_id, cutoff),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT AVG(duration_ms) as avg_ms FROM feedback "
                "WHERE timestamp >= ? AND duration_ms IS NOT NULL",
                (cutoff,),
            ).fetchone()
    return round(row["avg_ms"], 1) if row and row["avg_ms"] is not None else None


def get_failure_trend(
    job_id: Optional[str] = None,
    buckets: int = 12,
    bucket_minutes: int = 60,
) -> List[Dict[str, Any]]:
    """Get failure count per time bucket for trend visualization.

    Returns a list of ``buckets`` dicts, each covering ``bucket_minutes`` minutes,
    ordered oldest-first (left-to-right for ASCII charts).
    """
    _ensure_schema()
    now = datetime.now(timezone.utc)
    results = []
    with _get_conn() as conn:
        for i in range(buckets - 1, -1, -1):
            start = (now - timedelta(minutes=(i + 1) * bucket_minutes)).isoformat()
            end = (now - timedelta(minutes=i * bucket_minutes)).isoformat()
            if job_id:
                row = conn.execute(
                    "SELECT COUNT(*) as total, SUM(success) as ok_count "
                    "FROM feedback WHERE job_id = ? AND timestamp >= ? AND timestamp < ?",
                    (job_id, start, end),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT COUNT(*) as total, SUM(success) as ok_count "
                    "FROM feedback WHERE timestamp >= ? AND timestamp < ?",
                    (start, end),
                ).fetchone()
            total = row["total"] or 0
            ok = row["ok_count"] or 0
            results.append({
                "time_start": start,
                "total": total,
                "failures": total - ok,
                "successes": ok,
            })
    return results


def get_error_types(
    job_id: Optional[str] = None,
    hours: int = 24,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """Get top error types from recent failures."""
    _ensure_schema()
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    with _get_conn() as conn:
        if job_id:
            rows = conn.execute(
                "SELECT error_type, COUNT(*) as cnt FROM feedback "
                "WHERE job_id = ? AND timestamp >= ? AND success = 0 "
                "AND error_type IS NOT NULL "
                "GROUP BY error_type ORDER BY cnt DESC LIMIT ?",
                (job_id, cutoff, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT error_type, COUNT(*) as cnt FROM feedback "
                "WHERE timestamp >= ? AND success = 0 "
                "AND error_type IS NOT NULL "
                "GROUP BY error_type ORDER BY cnt DESC LIMIT ?",
                (cutoff, limit),
            ).fetchall()
    return [{"error_type": r["error_type"], "count": r["cnt"]} for r in rows]


def get_consecutive_failures(job_id: str) -> int:
    """Count consecutive failures for a job (most recent runs)."""
    _ensure_schema()
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT success FROM feedback WHERE job_id = ? "
            "ORDER BY timestamp DESC LIMIT 50",
            (job_id,),
        ).fetchall()
    count = 0
    for row in rows:
        if not row["success"]:
            count += 1
        else:
            break
    return count


def cleanup_old_feedback(days: int = 90) -> int:
    """Delete feedback entries older than N days. Returns count deleted."""
    _ensure_schema()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    with _get_conn() as conn:
        cursor = conn.execute(
            "DELETE FROM feedback WHERE timestamp < ?", (cutoff,)
        )
        return cursor.rowcount


# ── Tool registration ────────────────────────────────────────────────────────

def sloop_feedback(action: str, job_id: Optional[str] = None, **kwargs) -> str:
    """Tool entry point for feedback operations."""
    try:
        if action == "record":
            record_feedback(
                job_id=job_id or kwargs.get("job_id", ""),
                job_name=kwargs.get("job_name"),
                success=kwargs.get("success", False),
                duration_ms=kwargs.get("duration_ms"),
                error_type=kwargs.get("error_type"),
                error_msg=kwargs.get("error_msg"),
                delivery_ok=kwargs.get("delivery_ok", True),
                metadata=kwargs.get("metadata"),
            )
            return json.dumps({"success": True, "message": "Feedback recorded"})

        elif action == "recent":
            entries = get_recent_feedback(
                job_id=job_id, limit=kwargs.get("limit", 50)
            )
            return json.dumps({"success": True, "entries": entries}, indent=2)

        elif action == "stats":
            stats = get_success_rate(job_id=job_id, hours=kwargs.get("hours", 24))
            avg_dur = get_avg_duration(job_id=job_id, hours=kwargs.get("hours", 24))
            errors = get_error_types(job_id=job_id, hours=kwargs.get("hours", 24))
            return json.dumps(
                {
                    "success": True,
                    "success_rate": stats,
                    "avg_duration_ms": avg_dur,
                    "top_errors": errors,
                },
                indent=2,
            )

        elif action == "trend":
            trend = get_failure_trend(
                job_id=job_id,
                buckets=kwargs.get("buckets", 12),
                bucket_minutes=kwargs.get("bucket_minutes", 60),
            )
            return json.dumps({"success": True, "trend": trend}, indent=2)

        elif action == "cleanup":
            deleted = cleanup_old_feedback(days=kwargs.get("days", 90))
            return json.dumps(
                {"success": True, "deleted": deleted}
            )

        return json.dumps(
            {"success": False, "error": f"Unknown action: {action}"}
        )

    except Exception as e:
        logger.exception("sloop_feedback error")
        return json.dumps({"success": False, "error": str(e)})


# ── Registry ─────────────────────────────────────────────────────────────────

from tools.registry import registry

SLOOP_FEEDBACK_SCHEMA = {
    "name": "sloop_feedback",
    "description": (
        "Collect and query execution feedback for Hermes Sloop cron jobs. "
        "Use action='record' to log a job execution, 'recent' to view entries, "
        "'stats' for success rate and error analysis, 'trend' for failure trends, "
        "'cleanup' to purge old entries."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "description": "One of: record, recent, stats, trend, cleanup",
            },
            "job_id": {
                "type": "string",
                "description": "Filter by job ID (for recent/stats/trend) or record for a specific job",
            },
            "job_name": {
                "type": "string",
                "description": "Human-readable job name (for record)",
            },
            "success": {
                "type": "boolean",
                "description": "Whether the execution succeeded (for record)",
            },
            "duration_ms": {
                "type": "number",
                "description": "Execution duration in milliseconds (for record)",
            },
            "error_type": {
                "type": "string",
                "description": "Category of error (for record): e.g. timeout, api_error, agent_error",
            },
            "error_msg": {
                "type": "string",
                "description": "Error message detail (for record)",
            },
            "delivery_ok": {
                "type": "boolean",
                "description": "Whether delivery succeeded (for record, default true)",
            },
            "hours": {
                "type": "integer",
                "description": "Time window in hours for stats/trend (default 24)",
            },
            "limit": {
                "type": "integer",
                "description": "Max entries to return (for recent, default 50)",
            },
            "buckets": {
                "type": "integer",
                "description": "Number of time buckets for trend (default 12)",
            },
            "bucket_minutes": {
                "type": "integer",
                "description": "Minutes per bucket for trend (default 60)",
            },
            "days": {
                "type": "integer",
                "description": "Days to keep for cleanup (default 90)",
            },
        },
        "required": ["action"],
    },
}


registry.register(
    name="sloop_feedback",
    toolset="sloop",
    schema=SLOOP_FEEDBACK_SCHEMA,
    handler=lambda args, **kw: sloop_feedback(
        action=args.get("action", ""),
        job_id=args.get("job_id"),
        job_name=args.get("job_name"),
        success=args.get("success", False),
        duration_ms=args.get("duration_ms"),
        error_type=args.get("error_type"),
        error_msg=args.get("error_msg"),
        delivery_ok=args.get("delivery_ok", True),
        hours=args.get("hours", 24),
        limit=args.get("limit", 50),
        buckets=args.get("buckets", 12),
        bucket_minutes=args.get("bucket_minutes", 60),
        days=args.get("days", 90),
    ),
    check_fn=lambda: True,  # Always available — uses local SQLite
    emoji="📊",
)

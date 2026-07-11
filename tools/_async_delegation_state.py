"""Best-effort filesystem writers for async-delegation observability.

The dashboard, /agents slash command, and 'delegations N' vital pill all read:

  ~/.hermes/state/active-delegations.json   — point-in-time roster
  ~/.hermes/state/events.jsonl              — append-only lifecycle log

Both live OUTSIDE the agent process (separate dashboards / MCP-only clients),
so the in-memory ``_records`` dict in ``async_delegation.py`` is invisible to
them. This module bridges that gap: every dispatch/finalize calls in here and
we rewrite/append the on-disk files.

Best-effort by design — every public call swallows IO errors and logs at WARN.
A failed state write must NEVER crash the worker thread (Bug 4 root cause:
prior code had no writers at all, so the surface was permanently stale).
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

logger = logging.getLogger(__name__)

_STATE_DIR = Path(os.path.expanduser("~/.hermes/state"))
_ACTIVE_FILE = _STATE_DIR / "active-delegations.json"
_EVENTS_FILE = _STATE_DIR / "events.jsonl"

# Fields safe to expose on disk (drop interrupt_fn — not JSON-serialisable;
# truncate goal/context to keep the file lightweight for /api/delegations).
_GOAL_TRUNC = 400
_CONTEXT_TRUNC = 600
_SUMMARY_TRUNC = 800


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _truncate(s: Optional[str], n: int) -> Optional[str]:
    if not s:
        return s
    if len(s) <= n:
        return s
    return s[: n - 1] + "…"


def _record_to_public(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Map an internal record to the disk schema."""
    dispatched_at = rec.get("dispatched_at")
    completed_at = rec.get("completed_at")
    return {
        "delegation_id": rec.get("delegation_id"),
        "status": rec.get("status"),
        "goal": _truncate(rec.get("goal"), _GOAL_TRUNC),
        "context": _truncate(rec.get("context"), _CONTEXT_TRUNC),
        "toolsets": rec.get("toolsets"),
        "role": rec.get("role"),
        "model": rec.get("model"),
        "session_key": rec.get("session_key"),
        "dispatched_at": dispatched_at,
        "dispatched_at_iso": (
            datetime.fromtimestamp(dispatched_at, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            if isinstance(dispatched_at, (int, float))
            else None
        ),
        "completed_at": completed_at,
        "completed_at_iso": (
            datetime.fromtimestamp(completed_at, tz=timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            )
            if isinstance(completed_at, (int, float))
            else None
        ),
        "duration_seconds": (
            round(completed_at - dispatched_at, 2)
            if isinstance(dispatched_at, (int, float))
            and isinstance(completed_at, (int, float))
            else None
        ),
    }


def write_state_snapshot(records: Iterable[Dict[str, Any]]) -> None:
    """Rewrite active-delegations.json with a fresh snapshot.

    Atomic via tempfile + rename. Silent on IO error.
    """
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": _now_iso(),
            "delegations": [_record_to_public(r) for r in records],
        }
        # atomic write: tempfile in same dir, then rename
        fd, tmp_path = tempfile.mkstemp(
            prefix=".active-delegations.", suffix=".json", dir=str(_STATE_DIR)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2, default=str)
                fh.write("\n")
            os.replace(tmp_path, _ACTIVE_FILE)
        except Exception:
            # cleanup tmp on failure
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("write_state_snapshot failed: %s", exc)


def append_event(kind: str, record: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
    """Append one lifecycle event line to events.jsonl.

    Schema:
      {
        "ts": iso,
        "kind": "delegate.task_spawned" | "delegate.task_completed" | "delegate.task_failed",
        "delegation_id": str,
        "session_key": str,
        "goal_preview": str (≤200 chars),
        "summary": str (only on completed/failed; ≤800 chars),
        "status": str,
        "duration_seconds": float (only on completed/failed),
      }
    """
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        dispatched_at = record.get("dispatched_at")
        completed_at = record.get("completed_at")
        entry: Dict[str, Any] = {
            "ts": _now_iso(),
            "kind": kind,
            "delegation_id": record.get("delegation_id"),
            "session_key": record.get("session_key"),
            "goal_preview": _truncate(record.get("goal"), 200),
            "status": record.get("status"),
        }
        if isinstance(dispatched_at, (int, float)) and isinstance(
            completed_at, (int, float)
        ):
            entry["duration_seconds"] = round(completed_at - dispatched_at, 2)
        if extra:
            entry.update(extra)
        with open(_EVENTS_FILE, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")
    except Exception as exc:  # noqa: BLE001
        logger.warning("append_event(%s) failed: %s", kind, exc)


# Convenience event-emit shortcuts used by async_delegation.py callsites.

def emit_spawned(record: Dict[str, Any]) -> None:
    append_event("delegate.task_spawned", record)


def emit_finalized(record: Dict[str, Any], result: Dict[str, Any], status: str) -> None:
    kind = (
        "delegate.task_completed"
        if status == "completed"
        else "delegate.task_failed"
    )
    extra: Dict[str, Any] = {
        "summary": _truncate(result.get("summary"), _SUMMARY_TRUNC),
        "api_calls": result.get("api_calls"),
    }
    if status != "completed":
        extra["error"] = _truncate(result.get("error"), 400)
    append_event(kind, record, extra=extra)

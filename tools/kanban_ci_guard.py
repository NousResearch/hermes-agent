"""Core Kanban completion gate for governed local-CI tasks.

This module intentionally has no plugin imports: dispatcher workers use their
assignee profile, so the feedback plugin may be disabled even though the
worker still owns a governed local-CI task.
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any


_BLOCK_MESSAGE = (
    "CI completion rejected: no typed passing durable CI receipt matches this task's exact PR head/base "
    "and dispatch. Run the governed github-pr-feedback audit-pr command; its deterministic receipt and "
    "handoff own completion. If audit cannot run, use kanban_block with the exact blocker. A summary or "
    "claimed command is not CI evidence."
)
_UNAVAILABLE_MESSAGE = (
    "CI completion rejected: the durable CI task binding or receipt could not be verified. "
    "Run the governed github-pr-feedback audit-pr command; its deterministic receipt and handoff own "
    "completion. If audit cannot run, use kanban_block with the exact blocker."
)


def completion_block(task_id: str | None = None) -> str | None:
    """Return a blocking message for an unproven governed CI completion."""
    worker_task = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    target = str(task_id or worker_task or "").strip()
    if not target:
        return None
    try:
        root = Path(os.environ.get("HERMES_CONTROL_HOME", "").strip() or _default_hermes_root())
        path = root / "github-pr-feedback" / "ledger.sqlite3"
        if not path.exists():
            return None
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1)) as connection:
            connection.execute("BEGIN")
            bindings = connection.execute(
                "SELECT repository, pr_number, feedback_id, head_sha, claimed_at "
                "FROM feedback_receipts WHERE task_id = ? AND feedback_kind = 'pr_local_ci'",
                (target,),
            ).fetchall()
            if not bindings or all(_has_receipt(connection, binding) for binding in bindings):
                return None
        return _BLOCK_MESSAGE
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return _UNAVAILABLE_MESSAGE


def _default_hermes_root() -> str:
    try:
        from hermes_constants import get_default_hermes_root
    except ImportError:
        return str(Path.home() / ".hermes")
    return str(get_default_hermes_root())


def _has_receipt(connection: sqlite3.Connection, binding: tuple[Any, ...]) -> bool:
    repository, number, feedback_id, head, claimed_at = binding
    claimed = datetime.fromisoformat(claimed_at)
    if claimed.tzinfo is None:
        return False
    rows = connection.execute(
        "SELECT evidence_json FROM ci_audit_receipts WHERE repository = ? AND pr_number = ? "
        "AND head_sha = ? ORDER BY completed_at DESC",
        (repository, number, head),
    )
    for (payload_text,) in rows:
        payload = json.loads(payload_text)
        identity = payload.get("identity")
        if not isinstance(identity, dict):
            continue
        started_at = datetime.fromisoformat(str(payload["started_at"]))
        if (
            payload.get("status") == "passed"
            and identity.get("repository") == repository
            and identity.get("pr_number") == number
            and identity.get("head_sha") == head
            and str(feedback_id) == f"local-ci-audit-v2:{str(identity.get('base_sha', '')).casefold()}"
            and started_at.tzinfo is not None
            and started_at >= claimed
        ):
            return True
    return False

"""Core Kanban completion gate for governed feedback tasks.

Dispatched workers use their assignee profile, so the feedback plugin may not be
enabled there even though the control plane owns the durable feedback ledger.
This boundary check keeps the feedback acknowledgement contract active without
requiring every worker profile to duplicate plugin configuration.
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any


_BLOCK_MESSAGE = (
    "Feedback completion rejected: this task still has an unacknowledged feedback dispatch. "
    "Finish the authorized push and factual reply, then run the exact governed complete-feedback "
    "command; use retire-feedback only for its verified closed-PR case. If the contract cannot be "
    "completed, use kanban_block with the actual blocker. A summary or local commit is not an acknowledgement."
)
_UNAVAILABLE_MESSAGE = (
    "Feedback completion rejected: the task's durable feedback completion contract could not be verified. "
    "Finish the authorized push and factual reply, then run the exact governed complete-feedback command; "
    "use retire-feedback only for its verified closed-PR case. If the contract cannot be completed, use "
    "kanban_block with the actual blocker."
)


def completion_block(task_id: str | None = None) -> str | None:
    """Return a blocking message for an unacknowledged worker feedback task."""
    worker_task = os.environ.get("HERMES_KANBAN_TASK", "").strip()
    if not worker_task:
        return None
    target = str(task_id or worker_task).strip()
    if not target:
        return _UNAVAILABLE_MESSAGE
    try:
        root = Path(os.environ.get("HERMES_CONTROL_HOME", "").strip() or _default_hermes_root())
        path = root / "github-pr-feedback" / "ledger.sqlite3"
        # A normal Kanban worker can exist without the optional feedback ledger.
        if not path.exists():
            return None
        with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1)) as connection:
            rows = connection.execute(
                "SELECT status, action_status FROM feedback_receipts WHERE task_id = ? "
                "AND feedback_kind IN ('review_comment', 'issue_comment', 'review', 'pr_repair') "
                "AND NOT (feedback_kind = 'pr_repair' AND feedback_id LIKE 'report:%')",
                (target,),
            ).fetchall()
            if not rows or all(
                status == "completed" and action in {"completed", "superseded"}
                for status, action in rows
            ):
                return None
        return _BLOCK_MESSAGE
    except (OSError, sqlite3.Error, ValueError, TypeError):
        return _UNAVAILABLE_MESSAGE


def _default_hermes_root() -> str:
    try:
        from hermes_constants import get_default_hermes_root
    except ImportError:
        return str(Path.home() / ".hermes")
    return str(get_default_hermes_root())

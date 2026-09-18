"""Agent Inbox contracts (``tui_gateway/methods_inbox.py``): read-only cross-session aggregation
of persisted automation state and live pending server→client requests."""

from __future__ import annotations

from .base import JsonValue, Params, Result
from .common import ProfileParams
from .registry import method


class InboxParams(ProfileParams):
    limit: int = 200


class InboxPendingApproval(Result):
    count: int
    description: str = ""
    command_redacted: bool = True


class PendingClarify(Result):
    count: int


class InboxItem(Result):
    session_key: str
    title: str = ""
    source: str = ""
    cwd: str = ""
    lanes: list[str] = []
    goal: JsonValue = None
    loop: JsonValue = None
    heartbeat: JsonValue = None
    pending_approval: InboxPendingApproval | None = None
    pending_clarify: PendingClarify | None = None
    # Requests that ended without an answer (timed out / withdrawn). They stay listed so
    # the operator can still see what died and redo it; dismissed only explicitly.
    expired_request_count: int = 0
    categories: list[str] = []
    subagent_count: int = 0
    subagent_count_unavailable: bool = False
    background_task_count: int = 0
    background_task_count_unavailable: bool = False


class InboxCounts(Result):
    needs_you: int = 0
    running: int = 0
    waiting: int = 0
    scheduled: int = 0
    total: int = 0


class InboxCategoryCounts(Result):
    goals: int = 0
    loops: int = 0
    heartbeats: int = 0
    subagents: int = 0
    background_tasks: int = 0
    other: int = 0


class InboxCoverage(Result):
    profile: str = ""
    connection_scope: str = "active connection and profile only"
    scanned_sessions: int = 0
    partial: bool = False
    approval_scope: str = "live gateway approval queue"
    clarify_scope: str = "live open sessions only"
    errors: list[str] = []


class InboxResult(Result):
    items: list[InboxItem] = []
    counts: InboxCounts = InboxCounts()
    categories: InboxCategoryCounts = InboxCategoryCounts()
    badge: str = "none"
    coverage: InboxCoverage = InboxCoverage()


class InboxListResult(Result):
    """Wrapper matching the handler's actual return shape: ``{"inbox": {...}}``."""

    inbox: InboxResult


method(
    "inbox.list",
    params=InboxParams,
    result=InboxListResult,
    doc="Read-only cross-session inbox aggregation for the active profile.",
)

"""Contracts: composer-scheduled messages (#111873).

A draft the user deferred: the exact text arrives later as an ordinary user turn in the SAME
session. One-shot, cancellable, at most once — and deliberately free of any provider, model,
delivery-target or recurrence field, because the composer flow must not expose them (advanced
scheduling stays in the scheduled-jobs UI).

Handlers: ``tui_gateway/methods_schedule.py``. Store + firing policies:
``hermes_cli/scheduled_messages.py``.
"""

from __future__ import annotations

from .base import Result
from .common import SessionParams
from .registry import method


class ScheduledMessageView(Result):
    """One pending item as the composer panel renders it.

    ``due_at`` / ``created_at`` are epoch SECONDS (the backend's unit); the client converts to
    milliseconds and formats in the user's own timezone, so nothing stored can go stale across a
    DST shift or a machine move. ``display_text`` is the server-truncated preview.
    """

    id: str
    text: str
    display_text: str
    due_at: float
    created_at: float = 0.0
    status: str = "pending"
    # Past its due time and still waiting: the session was busy, or not open in a running backend.
    overdue: bool = False


class ScheduledMessageCreateParams(SessionParams):
    """``due_at`` is epoch seconds, a millisecond epoch (above 1e11, i.e. JS ``Date.now()``), or an
    ISO 8601 timestamp; a naive timestamp is read as the user's local wall clock, which is what a
    ``datetime-local`` input means. No other field exists: scheduling a message configures nothing.
    """

    text: str
    due_at: float | str


class ScheduledMessageListParams(SessionParams):
    """Every pending message bound to ``session_id``, soonest first."""


class ScheduledMessageCancelParams(SessionParams):
    message_id: str


class ScheduledMessageListResult(Result):
    messages: list[ScheduledMessageView] = []


class ScheduledMessageCreateResult(Result):
    """The list the client should render, plus the item that was just persisted."""

    message: ScheduledMessageView
    messages: list[ScheduledMessageView] = []


class ScheduledMessageCancelResult(Result):
    cancelled: bool
    messages: list[ScheduledMessageView] = []


method("schedule.message.create", params=ScheduledMessageCreateParams, result=ScheduledMessageCreateResult,
       doc="Persist a pending message for this session. Never sends it: it fires when it comes due.")
method("schedule.message.list", params=ScheduledMessageListParams, result=ScheduledMessageListResult,
       doc="Pending scheduled messages for this session (soonest due first).")
method("schedule.message.cancel", params=ScheduledMessageCancelParams, result=ScheduledMessageCancelResult,
       doc="Cancel a still-pending scheduled message; a fired one is history and is refused.")

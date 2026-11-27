"""Delivery status (``delivery.status``): the read-only state of deferred deliveries a session
has not consumed yet, for the status bar and the dashboard.

Sources covered (and only these):

- the transcript: detached API delegation deliveries, recorded as ``user`` rows with
  ``display_kind IN ('async_delegation_complete', 'hidden')`` and
  ``display_metadata.delegation_id`` (``hermes_state_detached_results.py``);
- the bot relay mailbox: per-delivery receipts under the profile home's
  ``runtime/bot_live_delivery/`` (``gateway/session_bot_mailbox.py``).

NOT covered: the plugin outbox (``delivery_store.py`` / ``deliveries.db`` in the plugin's own
home). The core never reads it, so this verb cannot promise its numbers; a client that needs
the plugin outbox's pending count reads the plugin's own surface.
"""

from __future__ import annotations

from typing import Literal

from .base import Result
from .common import ProfileParams
from .registry import canonical_method


class DeliveryStatusParams(ProfileParams):
    """One session (``session_id``) or every visible session (``scope="all"``); exactly one of
    the two. Neither is a default to "everything": without one the call is refused
    (fail-closed), so a bare poll never leaks another session's deliveries."""

    session_id: str | None = None
    scope: Literal["all"] | None = None
    # ``scope="all"`` only: at most this many visible sessions, same 1..200 bound as
    # ``session.list``.
    limit: int | None = None
    # Window ``delivered_recent`` counts over, in seconds (default 86400: the last day).
    window_seconds: int | None = None


class DeliverySourceMap(Result):
    """Which source feeds each number. A count is fed by its source even when it is zero."""

    pending: str
    delivered_recent: str
    dead_letter: str
    last_delivery_at: str


class DeliveryStatusRow(Result):
    """One session's delivery state.

    ``pending`` counts detached deliveries recorded after the session's last assistant row —
    the documented approximation of "recorded and not yet consumed by a turn": the core keeps
    no "seen" column on ``messages``, so a delivery the user already saw through another
    channel still counts until the next turn runs. ``oldest_pending_age_seconds`` is null when
    nothing is pending. ``delivered_recent`` counts detached deliveries plus settled relay
    receipts inside ``window_seconds``. ``dead_letter`` counts failed relay receipts (the
    transcript has no dead-letter state). ``last_delivery_at`` is the newest delivery
    timestamp across both sources, or null when the session has none.
    """

    session_id: str
    pending: int
    oldest_pending_age_seconds: float | None
    delivered_recent: int
    dead_letter: int
    last_delivery_at: float | None
    source: DeliverySourceMap


class DeliveryStatusResult(Result):
    sessions: list[DeliveryStatusRow]
    scope: Literal["session", "all"]
    window_seconds: int


canonical_method(
    "delivery.status",
    params=DeliveryStatusParams,
    result=DeliveryStatusResult,
    doc="Read-only delivery state for one session or every visible session: pending detached "
    "deliveries (recorded after the last assistant row — a documented approximation, the "
    "core keeps no 'seen' column), deliveries in the window, failed relay receipts "
    "(dead-letter) and the newest delivery time. Sources: the transcript's detached-result "
    "rows plus the bot relay receipts; the plugin outbox is NOT covered. Requires "
    "``session:read``; a session of another profile answers ``permission_denied`` / "
    "``profile_mismatch`` with no data. Writes nothing, touches no turn, lease or cache, "
    "and answers during the restart drain.",
)

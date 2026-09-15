"""Composer-scheduled messages: one one-shot future user turn in the CURRENT chat (#111873).

Product scope is the messenger gesture: the user writes a normal draft, picks "Schedule", picks a
date and time, and the exact text arrives later as an ordinary user turn in the same conversation.
So this is deliberately NOT ``hermes cron``: a cron job runs in its own session with its own
provider/model/delivery config, which is the configuration the issue asks the default flow to hide.

Storage and firing reuse the session-scoped scheduling base already behind ``/goal``, ``/loop`` and
``/heartbeat`` — ``hermes_cli/scheduled_messages.py`` (SessionDB ``state_meta`` keyed by the durable
session key) plus the per-session notification poller that already fires those from the
session-owner process. Nothing new polls; nothing new persists.

Wire contract (no provider, model, delivery-target or recurrence fields — the UI must not grow
them):
  schedule.message.create {session_id, text, due_at}   -> {message, messages}
  schedule.message.list   {session_id}                  -> {messages}
  schedule.message.cancel {session_id, message_id}      -> {cancelled, messages}

``due_at`` is epoch seconds, or a number above 1e11 read as a millisecond epoch (JS ``Date.now()``),
or an ISO 8601 string; a naive timestamp is the user's own wall clock (that is what a
``datetime-local`` input means). Firing: see ``_maybe_fire_tui_scheduled_message``.
"""

from __future__ import annotations

import contextlib
import logging
import time
from datetime import datetime

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped

logger = logging.getLogger(__name__)

# A millisecond epoch is ~1.7e12; a second epoch will not pass 1e11 until the year 5138.
_MS_EPOCH_FLOOR = 1e11

# Reached only for an item the poller could not hand to a seat it claimed. Bounded and logged:
# the item stays pending, so nothing is lost, and the next poll retries.
_FIRE_STATUS_KIND = "schedule"


def _coerce_due_at(raw) -> float:
    """``due_at`` from the client in any of the three accepted shapes. ValueError on garbage."""
    if isinstance(raw, bool) or raw is None:
        raise ValueError("due_at is required")
    if isinstance(raw, (int, float)):
        value = float(raw)
        return value / 1000.0 if value > _MS_EPOCH_FLOOR else value
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            raise ValueError("due_at is required")
        try:
            parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text)
        except ValueError as exc:
            raise ValueError("due_at must be an ISO 8601 timestamp or epoch seconds") from exc
        # Naive input IS the user's local wall clock: `datetime-local` carries no zone, so
        # interpreting it as UTC would fire hours off.
        return parsed.timestamp() if parsed.tzinfo else parsed.astimezone().timestamp()
    raise ValueError("due_at must be an ISO 8601 timestamp or epoch seconds")


def _scheduled_snapshot(session: dict, *, message: dict | None = None) -> dict:
    """Reply body after any mutation: the list the client should render, plus the new item."""
    from hermes_cli.scheduled_messages import list_messages

    key = str(session.get("session_key") or "")
    home = _session_home(session)
    payload = {"messages": list_messages(key, home=home)}
    if message is not None:
        payload["message"] = message
    return payload


def _scheduled_session(params: dict, rid):
    """(session, session_key, home) for a schedule RPC, or (None, err)."""
    session, err = _sess_nowait(params, rid)
    if err:
        return None, None, None, err
    key = str(session.get("session_key") or "")
    if not key:
        # No durable identity yet (a draft that was never persisted): a message bound to it could
        # not be found again after a restart, which is the one thing this feature promises.
        return None, None, None, _err(rid, 4125, "session has no durable id yet; send one message first")
    return session, key, _session_home(session), None


@method("schedule.message.create")
def _(rid, params: dict) -> dict:
    """Persist a pending message. Never sends it: the poller fires it when it comes due."""
    from hermes_cli.scheduled_messages import schedule_message

    session, key, home, err = _scheduled_session(params, rid)
    if err is not None:
        return err
    text = params.get("text")
    if not isinstance(text, str) or not text.strip():
        return _err(rid, 4002, "text is required")
    try:
        due_at = _coerce_due_at(params.get("due_at"))
        message = schedule_message(key, text, due_at, home=home)
    except ValueError as exc:
        return _err(rid, 4004, str(exc))
    logger.info("scheduled message accepted: session_key=%s due_in=%.0fs chars=%d",
                key, message.due_at - time.time(), len(message.text))
    return _ok(rid, _scheduled_snapshot(session, message=message.public(now=time.time())))


@method("schedule.message.list")
def _(rid, params: dict) -> dict:
    session, _key, _home, err = _scheduled_session(params, rid)
    if err is not None:
        return err
    return _ok(rid, _scheduled_snapshot(session))


@method("schedule.message.cancel")
def _(rid, params: dict) -> dict:
    """Drop a still-pending item. A fired item is history, not cancellable — hence the bool."""
    from hermes_cli.scheduled_messages import cancel_message

    session, key, home, err = _scheduled_session(params, rid)
    if err is not None:
        return err
    message_id = str(params.get("message_id") or "")
    if not message_id:
        return _err(rid, 4002, "message_id is required")
    cancelled = cancel_message(key, message_id, home=home)
    if not cancelled:
        return _err(rid, 4044, "no pending scheduled message with that id")
    return _ok(rid, {"cancelled": True, **_scheduled_snapshot(session)})


def _maybe_fire_tui_scheduled_message(sid: str, session: dict) -> None:
    """Fire one due scheduled message for an idle session (#111873).

    Same shape as ``_maybe_fire_tui_heartbeat_tick``: claim the idle session first (a racing user
    prompt wins, and a busy session is never preempted — the item simply stays due and the next
    poll retries, which is how this rides the per-session queueing rules instead of opening a
    concurrent turn), then re-enter through ``_run_prompt_submit`` as a plain user turn.

    At most once: ``claim_due`` persists pending -> fired BEFORE the dispatch, so a crash in that
    window cannot re-fire on the next start. A dispatch that never started a turn rewinds the claim
    so the item is not silently consumed.
    """
    try:
        from hermes_cli.scheduled_messages import abandon_fire, claim_due, due_items, format_due_display
    except Exception:
        return
    if not (key := str(session.get("session_key") or "")):
        return
    try:
        home = _session_home(session)
        due = due_items(key, home=home)
    except Exception as exc:
        logger.debug("scheduled message scan failed for %s: %s", key, exc)
        return
    if not due:
        return
    for item in due:
        if not _notif_claim_turn(session):
            return  # busy — stays due, retried on the next poll (never a concurrent turn)
        claimed = claim_due(key, item.id, home=home)
        if claimed is None:
            _notif_release_turn(session)  # a peer fired or cancelled it between scan and claim
            continue
        started = False
        try:
            _emit("status.update", sid, {
                "kind": _FIRE_STATUS_KIND,
                "text": f"⏰ Scheduled message ({format_due_display(claimed.due_at)}) is due — sending…"})
            logger.info("scheduled message firing: session_key=%s message_id=%s", key, claimed.id)
            started = bool(_run_prompt_submit(f"__schedule__{claimed.id}", sid, session, claimed.text))
        except Exception as exc:
            _notif_log_failure("scheduled message dispatch failed", exc)
        if not started:
            # ``_run_prompt_submit`` releases ``running`` itself when it refuses the turn; make it
            # unconditional, then put the item back so a later poll retries it.
            _notif_release_turn(session)
            with contextlib.suppress(Exception):
                abandon_fire(key, claimed.id, home=home)
        return  # one item per poll: they are user turns, and they stay in due order


def register(server) -> None:
    """Rebind this module's handlers and helpers onto the server namespace."""
    bind_module(globals(), server, skip=("_",))

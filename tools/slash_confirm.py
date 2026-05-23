"""Generic slash-command confirmation primitive (gateway-side).

Slash commands that have a non-destructive but expensive side effect worth
surfacing to the user (currently only ``/reload-mcp``, which invalidates
the provider prompt cache) route through this module.

Two delivery paths:

  1. Button UI — adapters that override ``send_slash_confirm`` render
     three inline buttons (Approve Once / Always Approve / Cancel).  The
     button callback calls ``resolve(session_key, confirm_id, choice)``.

  2. Text fallback — adapters without button UIs get a plain text prompt.
     Users reply with ``/approve``, ``/always``, or ``/cancel``; the
     gateway's ``_handle_message`` intercepts those replies and calls
     ``resolve()`` directly.

State is stored module-level (like ``tools.approval``) so platform
adapters can resolve callbacks without needing a backreference to the
``GatewayRunner`` instance.  The CLI path (``cli.py``) uses a local
synchronous variant — see ``_prompt_slash_confirm`` there.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Pending confirmations keyed by gateway session_key.  Each entry:
#   {
#       "confirm_id": str,
#       "command":    str,                       # e.g. "reload-mcp"
#       "handler":    Callable[[str], Awaitable[Optional[str]]],
#       "created_at": float,                     # time.time()
#   }
_pending: Dict[str, Dict[str, Any]] = {}
_lock = threading.RLock()

# Default timeout — a pending confirm older than this is discarded when
# the next message arrives for the same session.  Buttons work up until
# the adapter drops the callback_data (Telegram: ~48h; Discord: ephemeral;
# Slack: 3s ack + long-lived actions).
DEFAULT_TIMEOUT_SECONDS = 300
_VALID_CHOICES = frozenset({"once", "always", "cancel"})


def _record_slash_confirm_audit(**kwargs) -> None:
    try:
        from hermes_cli.audit_log import record_approval_audit_event
        record_approval_audit_event(
            event_type="slash_confirmation.decision",
            **kwargs,
        )
    except Exception as exc:
        logger.debug("Slash confirmation audit write failed: %s", exc, exc_info=True)


def normalize_confirmation_choice(choice: str) -> str | None:
    """Return an exact slash-confirm choice, or None for near-misses."""
    normalized = str(choice or "").strip().lower()
    if normalized in _VALID_CHOICES:
        return normalized
    return None


def register(
    session_key: str,
    confirm_id: str,
    command: str,
    handler: Callable[[str], Awaitable[Optional[str]]],
) -> None:
    """Register a pending slash-command confirmation.

    Overwrites any prior pending confirm for the same ``session_key`` — the
    user invoking a new confirmable command supersedes the stale one.
    """
    with _lock:
        _pending[session_key] = {
            "confirm_id": confirm_id,
            "command": command,
            "handler": handler,
            "created_at": time.time(),
        }


def get_pending(session_key: str) -> Optional[Dict[str, Any]]:
    """Return the pending confirm dict for a session, or None."""
    with _lock:
        entry = _pending.get(session_key)
        return dict(entry) if entry else None


def clear(session_key: str) -> None:
    """Drop the pending confirm for ``session_key`` without running it."""
    with _lock:
        _pending.pop(session_key, None)


def clear_if_stale(session_key: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> bool:
    """Drop the pending confirm if older than ``timeout`` seconds.

    Returns True if an entry was dropped.
    """
    with _lock:
        entry = _pending.get(session_key)
        if not entry:
            return False
        if time.time() - float(entry.get("created_at", 0) or 0) > timeout:
            _pending.pop(session_key, None)
            return True
        return False


async def resolve(
    session_key: str,
    confirm_id: str,
    choice: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Optional[str]:
    """Resolve a pending confirm.

    ``choice`` must be one of ``"once"``, ``"always"``, or ``"cancel"``.
    Returns the handler's output string (to be sent as a follow-up
    message), or ``None`` if the confirm was stale, already resolved, or
    the confirm_id doesn't match.

    Safe to call from an asyncio callback (button click) or from the
    gateway's message intercept path.
    """
    normalized_choice = normalize_confirmation_choice(choice)
    if normalized_choice is None:
        _record_slash_confirm_audit(
            decision="blocked",
            status="blocked_invalid_choice",
            reason="invalid_confirmation_choice",
            choice=str(choice or ""),
            command=None,
            session_key=session_key,
            surface="gateway",
            risk_tier="R3",
            extra={"confirm_id": confirm_id},
        )
        return None
    choice = normalized_choice

    with _lock:
        entry = _pending.get(session_key)
        if not entry:
            _record_slash_confirm_audit(
                decision="skipped",
                status="skipped_no_pending_confirmation",
                reason="no_pending_confirmation",
                choice=choice,
                command=None,
                session_key=session_key,
                surface="gateway",
                risk_tier="R3",
                extra={"confirm_id": confirm_id},
            )
            return None
        if entry.get("confirm_id") != confirm_id:
            # Stale confirm_id — superseded by a newer prompt on the same session.
            _record_slash_confirm_audit(
                decision="blocked",
                status="blocked_stale_confirmation_id",
                reason="stale_confirmation_id",
                choice=choice,
                command=entry.get("command", "?"),
                session_key=session_key,
                surface="gateway",
                risk_tier="R3",
                extra={"confirm_id": confirm_id},
            )
            return None
        # Pop before we run the handler to prevent duplicate callbacks
        # (e.g. button double-click) from running it twice.
        _pending.pop(session_key, None)
        if time.time() - float(entry.get("created_at", 0) or 0) > timeout:
            _record_slash_confirm_audit(
                decision="blocked",
                status="blocked_confirmation_timeout",
                reason="confirmation_timeout",
                choice=choice,
                command=entry.get("command", "?"),
                session_key=session_key,
                surface="gateway",
                risk_tier="R3",
                extra={"confirm_id": confirm_id},
            )
            return None
        handler = entry.get("handler")
        command = entry.get("command", "?")

    if not handler:
        _record_slash_confirm_audit(
            decision="blocked",
            status="blocked_missing_handler",
            reason="missing_confirmation_handler",
            choice=choice,
            command=command,
            session_key=session_key,
            surface="gateway",
            risk_tier="R3",
            extra={"confirm_id": confirm_id},
        )
        return None
    try:
        result = await handler(choice)
    except Exception as exc:
        logger.error(
            "Slash-confirm handler for /%s raised: %s",
            command, exc, exc_info=True,
        )
        _record_slash_confirm_audit(
            decision="blocked",
            status="blocked_handler_error",
            reason="confirmation_handler_error",
            choice=choice,
            command=command,
            session_key=session_key,
            surface="gateway",
            risk_tier="R3",
            extra={"confirm_id": confirm_id},
        )
        return f"❌ Error handling confirmation: {exc}"
    _record_slash_confirm_audit(
        decision="denied" if choice == "cancel" else "approved",
        status="cancelled_by_user" if choice == "cancel" else "allowed_user_approved",
        reason="user_cancelled" if choice == "cancel" else "user_approved",
        approval_scope=choice,
        choice=choice,
        command=command,
        session_key=session_key,
        surface="gateway",
        risk_tier="R3",
        extra={"confirm_id": confirm_id},
    )
    return result if isinstance(result, str) else None


def resolve_sync_compat(
    loop: asyncio.AbstractEventLoop,
    session_key: str,
    confirm_id: str,
    choice: str,
) -> Optional[str]:
    """Synchronous helper: schedule resolve() on a loop and wait for the result.

    Used by platform callback paths that run on a different thread than the
    event loop (e.g. Discord's button click handler in some configurations).
    Prefer the async ``resolve()`` from an async context.
    """
    try:
        from agent.async_utils import safe_schedule_threadsafe
        fut = safe_schedule_threadsafe(
            resolve(session_key, confirm_id, choice), loop,
            logger=logger,
            log_message="resolve_sync_compat scheduling failed",
        )
        if fut is None:
            return None
        return fut.result(timeout=30)
    except Exception as exc:
        logger.error("resolve_sync_compat failed: %s", exc)
        return None

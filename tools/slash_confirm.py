"""Generic slash-command confirmation primitive (gateway-side).

Slash commands with an expensive side effect (currently only ``/reload-mcp``, which invalidates
the provider prompt cache) route through here. Button-UI adapters render Approve Once / Always
Approve / Cancel and call ``resolve()``; text-only adapters get a prompt and the gateway
intercepts ``/approve``, ``/always``, ``/cancel``. State is module-level (like ``tools.approval``)
so adapters can resolve without a ``GatewayRunner`` backreference. The CLI has its own
synchronous variant (``_prompt_slash_confirm`` in ``cli.py``).
"""

from __future__ import annotations

import functools
import logging
import threading
import time
from typing import Any, Awaitable, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# session_key -> {"confirm_id", "command", "handler", "created_at", "authority"}
_pending: dict[str, dict[str, Any]] = {}
_lock = threading.RLock()

# Older pending confirms are discarded when the session's next message arrives (buttons live
# as long as the adapter keeps callback_data).
DEFAULT_TIMEOUT_SECONDS = 300


def register(session_key: str, confirm_id: str, command: str,
             handler: Callable[[str], Awaitable[Optional[str]]], authority: Optional[dict] = None) -> None:
    """Register a pending confirm, superseding any prior one for the session. *authority* says who
    may answer it (``gateway.approval_actor.slash_confirm_refusal``); default: the current turn's."""
    if authority is None:
        from tools.approval_context import prompt_authority
        authority = prompt_authority()
    with _lock:
        _pending[session_key] = {"confirm_id": confirm_id, "command": command,
                                 "handler": handler, "created_at": time.time(), "authority": authority}


def get_pending(session_key: str) -> Optional[dict[str, Any]]:
    """Return a copy of the pending confirm dict for a session, or None."""
    with _lock:
        entry = _pending.get(session_key)
        return dict(entry) if entry else None


def clear(session_key: str) -> None:
    """Drop the pending confirm for ``session_key`` without running it."""
    with _lock:
        _pending.pop(session_key, None)


def _is_stale(entry: dict[str, Any], timeout: float) -> bool:
    return time.time() - float(entry.get("created_at", 0) or 0) > timeout


def clear_if_stale(session_key: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> bool:
    """Drop the pending confirm if older than ``timeout`` seconds; True if dropped."""
    with _lock:
        entry = _pending.get(session_key)
        stale = bool(entry and _is_stale(entry, timeout))
        if stale:
            _pending.pop(session_key, None)
        return stale


def claim(session_key: str, confirm_id: str, timeout: float = DEFAULT_TIMEOUT_SECONDS
          ) -> Optional[Callable[[str], Awaitable[Optional[str]]]]:
    """Take this prompt's pending confirm: a callable that runs it with the choice, or None if it
    was stale, already resolved, or superseded. A button adapter claims BEFORE it renders the
    decision, so a tap that will run nothing is never shown as approved."""
    with _lock:
        entry = _pending.get(session_key)
        if not entry or entry.get("confirm_id") != confirm_id:
            return None
        # Pop before running so duplicate callbacks (button double-click) cannot run it twice.
        _pending.pop(session_key, None)
        if _is_stale(entry, timeout) or not entry.get("handler"):
            return None
    return functools.partial(_run, entry["handler"], entry.get("command", "?"))


async def resolve(session_key: str, confirm_id: str, choice: str,
                  timeout: float = DEFAULT_TIMEOUT_SECONDS) -> Optional[str]:
    """Run the pending handler with ``choice`` ("once" / "always" / "cancel").

    Returns the handler's output string, or None if the confirm was stale, already resolved,
    or the confirm_id doesn't match (superseded prompt).
    """
    run = claim(session_key, confirm_id, timeout)
    return await run(choice) if run else None


async def _run(handler: Callable[[str], Awaitable[Optional[str]]], command: str, choice: str) -> Optional[str]:
    try:
        result = await handler(choice)
    except Exception as exc:
        logger.error("Slash-confirm handler for /%s raised: %s", command, exc, exc_info=True)
        return f"❌ Error handling confirmation: {exc}"
    return result if isinstance(result, str) else None

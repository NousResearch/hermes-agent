"""Gateway-side multi-step interaction state for /generate-image.

The slash command handler runs on the gateway event loop, so it cannot block
while waiting for the user to supply missing fields one at a time.  This
module stores a per-session state machine that the gateway's message
intercept feeds with each user reply, advancing through the missing fields
until the configuration is complete.  The final confirmation then routes
through the existing ``_request_slash_confirm`` primitive.

State shape::

    {
        "session_key": str,
        "fields": dict[str, str],      # collected so far
        "missing": list[str],          # remaining fields to ask, in order
        "created_at": float,
    }

The module is intentionally tiny and self-contained — it mirrors the
shape of ``tools.slash_confirm`` (module-level dict + RLock) so platform
adapters and the gateway intercept can resolve entries without holding a
back-reference to the ``GatewayRunner`` instance.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# session_key → state dict
_pending: Dict[str, Dict[str, Any]] = {}
_lock = threading.RLock()

DEFAULT_TIMEOUT_SECONDS = 600


def register(
    session_key: str,
    fields: Dict[str, str],
    missing: List[str],
) -> None:
    """Register a pending multi-step generate-image interaction.

    ``fields`` holds the values collected so far (may be pre-filled from
    inline args).  ``missing`` is the ordered list of field names still
    needed.  Overwrites any prior pending interaction for the session.
    """
    with _lock:
        _pending[session_key] = {
            "session_key": session_key,
            "fields": dict(fields),
            "missing": list(missing),
            "created_at": time.time(),
        }


def get_pending(session_key: str) -> Optional[Dict[str, Any]]:
    """Return the pending interaction state for a session, or None."""
    with _lock:
        entry = _pending.get(session_key)
        return dict(entry) if entry else None


def clear(session_key: str) -> None:
    """Drop the pending interaction for ``session_key``."""
    with _lock:
        _pending.pop(session_key, None)


def clear_if_stale(session_key: str, timeout: float = DEFAULT_TIMEOUT_SECONDS) -> bool:
    """Drop the pending interaction if older than ``timeout`` seconds."""
    with _lock:
        entry = _pending.get(session_key)
        if not entry:
            return False
        if time.time() - float(entry.get("created_at", 0) or 0) > timeout:
            _pending.pop(session_key, None)
            return True
        return False


def advance(session_key: str, response: str) -> Optional[Dict[str, Any]]:
    """Feed a user reply into the pending interaction.

    Pops the next missing field, stores the response under that field name,
    and returns the updated state dict (with one fewer missing field).
    Returns ``None`` if no pending interaction exists for the session.
    """
    with _lock:
        entry = _pending.get(session_key)
        if not entry:
            return None
        if not entry["missing"]:
            return entry
        field = entry["missing"].pop(0)
        entry["fields"][field] = response
        return dict(entry)

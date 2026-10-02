#!/usr/bin/env python3
"""Bridge desktop-only tools to Hermes-desktop renderer events.

The desktop ``tui_gateway`` installs an emitter via :func:`set_emitter`; elsewhere it
stays ``None`` and tools report "desktop only". Routing keys off ``HERMES_UI_SESSION_ID``
so the event lands on the window that owns the turn (the sink is lock-guarded).

Window-level events (``pane.reveal``, ``preview.open``) route by
``HERMES_UI_SESSION_ID`` — the window/socket that owns the turn.
Session-scoped events (``message.reaction``) route by ``HERMES_SESSION_ID``
so the renderer's gateway-event stream, keyed on the chat session, receives
the frame on the correct transcript transport (#80678).
"""

import json
from typing import Callable, FrozenSet, Optional

from gateway.session_context import get_session_env
from tools.registry import tool_error

# (sid, event, payload) sink, installed by the desktop gateway.
_emit: Optional[Callable[[str, str, dict], None]] = None

# Events that target a specific chat session rather than a window/socket.
# The renderer's gateway-event stream is keyed on the session id, so
# routing these through HERMES_UI_SESSION_ID (the window identity) means
# the frame lands on a stream that doesn't own the transcript and is
# never painted (#80678).
_SESSION_SCOPED_EVENTS: FrozenSet[str] = frozenset({"message.reaction"})


def set_emitter(fn: Optional[Callable[[str, str, dict], None]]) -> None:
    """Install (or clear) the renderer-event sink. Called by the desktop gateway."""
    global _emit
    _emit = fn


def available() -> bool:
    """True when running under the desktop app (an emitter is wired)."""
    return _emit is not None


def user_enabled(setting: str, default: bool) -> bool:
    """Read a desktop Appearance switch from ``display.<setting>``. The renderer mirrors
    these toggles onto the CONNECTED gateway's config, so this is the user's real answer
    for local/SSH/URL/cloud gateways alike; ``check_fn``s use it to withdraw a tool from
    the schema. Unreadable config -> ``default`` so a shipped-on feature does not vanish
    on a transient read error."""
    try:
        from hermes_cli.config import load_config_readonly
        display = load_config_readonly().get("display")
    except Exception:
        return default
    if not isinstance(display, dict) or setting not in display:
        return default
    return bool(display.get(setting))


def emit(event: str, payload: dict) -> bool:
    """Route ``event`` to the window owning the current turn; False when no emitter."""
    if _emit is None:
        return False
    if event in _SESSION_SCOPED_EVENTS:
        sid = get_session_env("HERMES_SESSION_ID", "")
        if not sid:
            sid = get_session_env("HERMES_UI_SESSION_ID", "")
    else:
        sid = get_session_env("HERMES_UI_SESSION_ID", "")
    _emit(sid, event, payload)
    return True


def emit_or_error(event: str, payload: dict, fail_prefix: str, desktop_only: str, result: dict) -> str:
    """Emit ``event``; ``tool_error`` text on failure (``fail_prefix`` + exception, or
    ``desktop_only`` when no emitter), else ``result`` as JSON. Calls ``emit`` via the
    module attribute so tests can patch it."""
    try:
        ok = emit(event, payload)
    except Exception as exc:
        return tool_error(f"{fail_prefix}{exc}")
    return json.dumps(result, ensure_ascii=False) if ok else tool_error(desktop_only)


def passthrough_json(raw) -> str:
    """Desktop answers with a JSON object; pass it through, else wrap the raw text."""
    try:
        return json.dumps(json.loads(raw), ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"text": str(raw)}, ensure_ascii=False)

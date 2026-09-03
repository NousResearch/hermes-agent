#!/usr/bin/env python3
"""Bridge desktop-only tools to Hermes-desktop renderer events.

The preview pane, pane focus, and friends live in the desktop renderer, so
desktop-gated tools reach them through an emitter the desktop ``tui_gateway``
installs at session start via :func:`set_emitter`. Everywhere else it stays
``None`` and the tools report "desktop only".

Window-level events (``pane.reveal``, ``preview.open``) route by
``HERMES_UI_SESSION_ID`` — the window/socket that owns the turn.
Session-scoped events (``message.reaction``) route by
``HERMES_SESSION_ID`` so the renderer's gateway-event stream, keyed on
the chat session, receives the frame on the correct transcript transport.

``_emit``/``write_json`` is ``_stdout_lock``-guarded, so emitting from the
tool's thread is safe.
"""

from typing import Callable, FrozenSet, Optional

from gateway.session_context import get_session_env

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
    """Read one of the desktop's Appearance switches from ``display.<setting>``.

    The renderer owns these toggles and mirrors them onto the CONNECTED
    gateway's config (``config.set``), so this reads the user's real answer
    whether that gateway is local, SSH, URL, or cloud — where an env var would
    only ever describe the process. Tool ``check_fn``s call it to withdraw
    themselves from the schema when the user has switched the feature off:
    Hermes should not be told about a surface it isn't allowed to use.

    An unreadable config falls back to ``default``, which is how a feature that
    ships on stays on rather than disappearing on a transient read error.
    """
    try:
        from hermes_cli.config import load_config_readonly

        display = load_config_readonly().get("display")
    except Exception:
        return default
    if not isinstance(display, dict) or setting not in display:
        return default
    return bool(display.get(setting))


def emit(event: str, payload: dict) -> bool:
    """Route ``event`` to the window that owns the current turn.

    Returns ``False`` when no emitter is wired (i.e. not the desktop app)."""
    fn = _emit
    if fn is None:
        return False
    if event in _SESSION_SCOPED_EVENTS:
        sid = get_session_env("HERMES_SESSION_ID", "")
        if not sid:
            sid = get_session_env("HERMES_UI_SESSION_ID", "")
    else:
        sid = get_session_env("HERMES_UI_SESSION_ID", "")
    fn(sid, event, payload)
    return True

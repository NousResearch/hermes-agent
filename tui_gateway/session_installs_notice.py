"""Other-installs launch notice for interactive TUI sessions. See ``hermes installs``."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

NOTICE_KEY = "installs-notice"
NOTICE_TTL_MS = 15000


def _emit_notice(sid: str, text: str) -> None:
    from .server import _emit

    _emit("notification.show", sid, {
        "text": text, "level": "warn", "kind": "ttl", "ttl_ms": NOTICE_TTL_MS,
        "key": NOTICE_KEY, "id": NOTICE_KEY})


def _interactive_tui_session(session: dict) -> bool:
    """True only for the stdio TUI session a person reads. Web, desktop, and messaging
    clients connect over other transports and stay silent."""
    if session.get("room_plumbing") or session.get("pending_hidden"):
        return False
    try:
        from .server import _stdio_transport
        return session.get("transport") is _stdio_transport
    except Exception:
        return False


def maybe_show(sid: str, session: dict) -> None:
    """Show the other-installs notice once per session. The notice is advisory: every
    failure here is swallowed with a debug log."""
    if session.get("_installs_notice_shown"):
        return
    session["_installs_notice_shown"] = True
    if not _interactive_tui_session(session):
        return
    try:
        from gateway.warning_notifications import render_notification
        from hermes_cli.installs import launch_notice

        notice = launch_notice()
        if notice:
            render_notification(lambda: _emit_notice(sid, notice), platform="tui")
    except Exception:
        logger.debug("installs notice failed", exc_info=True)

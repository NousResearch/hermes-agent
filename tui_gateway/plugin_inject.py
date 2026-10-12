"""Ink TUI / desktop host for ``PluginContext.inject_message``.

Separate from ``PluginManager.set_gateway_message_injector``. A messaging gateway
and this process can both be live; they must not share one slot (last writer
would win). Routing uses the reported ``session_key``, not the ephemeral UI
session id, and lands on that session's prompt queue.

Bodies are rebound onto ``server.py`` globals at install time
(``method_ctx.bind_module``), so ``_sessions`` and the queue helpers are bare names.
"""

from __future__ import annotations

import atexit

from .method_ctx import bind_module

# Identity for the process-owned host. A later owner can replace it; clear is
# identity-conditional so this process cannot drop that replacement.
_TUI_INJECT_OWNER = object()
_atexit_registered = False


def inject_tui_session_message(*, session_key: str, content: str, plugin_id: str = "") -> bool:
    """Queue *content* on the live session whose ``session_key`` matches.

    Returns False when this process has no such session, so the caller can fall
    through to the messaging-gateway slot. Never reroutes to a different session.
    A busy session only queues (a notice must not cancel in-flight work). An idle
    session drains so the queued prompt starts a turn.
    """
    del plugin_id  # accepted so the host matches the gateway injector kwargs
    if not isinstance(session_key, str) or not session_key:
        return False
    if not isinstance(content, str) or not content.strip():
        return False
    with _sessions_lock:
        match = next(
            (
                (sid, session)
                for sid, session in list(_sessions.items())
                if isinstance(session, dict) and session.get("session_key") == session_key
            ),
            None,
        )
    if match is None:
        return False
    sid, session = match
    if session.get("lazy") or session.get("_closing") or session.get("_finalized"):
        return False
    if session.get("history_lock") is None:
        return False
    with session["history_lock"]:
        queued = session.get("queued_prompt") or {}
        keep_transport = queued.get("transport") if isinstance(queued, dict) else None
        running = bool(session.get("running"))
        _enqueue_prompt(session, content, keep_transport)
        session["last_active"] = time.time()
        if running:
            return True
    rid = f"inject-{uuid.uuid4().hex[:8]}"
    threading.Thread(
        target=_drain_queued_prompt, args=(rid, sid, session),
        daemon=True, name=f"tui-inject-{sid[:8]}",
    ).start()
    return True


def install_tui_message_injector(manager=None) -> None:
    """Publish this process's TUI host on *manager*, or the active profile's manager.

    Passing a manager (tests) does not publish process-wide. The no-arg form is
    the Ink TUI / desktop startup path: every profile manager in this process
    gets the host, and managers created later pick it up, without touching the
    messaging-gateway slot.
    """
    from hermes_cli.plugins import get_plugin_manager, publish_tui_message_host

    global _atexit_registered
    if manager is None:
        publish_tui_message_host(_TUI_INJECT_OWNER, inject_tui_session_message)
        manager = get_plugin_manager()
        if not _atexit_registered:
            atexit.register(clear_tui_message_injector)
            _atexit_registered = True
    manager.set_tui_message_injector(_TUI_INJECT_OWNER, inject_tui_session_message)


def clear_tui_message_injector(manager=None) -> None:
    """Drop this process's host. A different owner is left in place."""
    from hermes_cli.plugins import clear_published_tui_message_host, get_plugin_manager

    if manager is None:
        clear_published_tui_message_host(_TUI_INJECT_OWNER)
        manager = get_plugin_manager()
    manager.clear_tui_message_injector(_TUI_INJECT_OWNER)


def register(server) -> None:
    """Publish this module's helpers onto ``server``, rebound to its globals."""
    bind_module(globals(), server, skip=("_",))

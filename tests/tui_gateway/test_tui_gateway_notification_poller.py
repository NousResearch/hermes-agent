"""One live notification poller per TUI session across agent rebuilds (#138806)."""

import threading
import types

from tui_gateway import server


def _session():
    return {"agent": types.SimpleNamespace(), "session_key": "session-key", "history": [],
            "history_lock": threading.Lock(), "history_version": 0, "running": False,
            "attached_images": [], "image_counter": 0, "cols": 80, "slash_worker": None,
            "show_reasoning": False, "tool_progress_mode": "all"}


def test_session_rebuild_stops_previous_notification_poller(monkeypatch):
    """Rebuilding the agent for a live session re-enters _start_session_services; the previous
    poller's stop event must be set before the record's ``_notif_stop`` is replaced (#138806).
    Otherwise the orphaned poller keeps draining the process-global completion queue for the
    session (two live loops per rebuild) until the session itself is finalized — close can only
    signal the newest event."""
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *a, **k: None)
    sess = _session()
    server._sessions["sid_rebuild"] = sess
    try:
        server._start_session_services("sid_rebuild", "session-key", sess)
        first_stop = sess["_notif_stop"]
        assert not first_stop.is_set()  # a fresh session really starts its poller running
        # Identify the first poller by its stop event now: once it exits, the rebuild may prune it
        # and only the replacement would carry the shared thread name.
        first_thread = next(thread for stop, thread in list(server._notification_pollers) if stop is first_stop)

        # Agent rebuild for the SAME session (in-session /model switch, resume, late MCP refresh).
        server._start_session_services("sid_rebuild", "session-key", sess)
        second_stop = sess["_notif_stop"]
        assert second_stop is not first_stop
        assert first_stop.is_set()  # the previous poller was stopped, not orphaned

        # The old loop wakes at most every 0.5s; give it its real deadline to exit.
        first_thread.join(timeout=3.0)
        assert not first_thread.is_alive(), "the stopped previous poller never exited"
        live = [
            thread
            for _stop, thread in list(server._notification_pollers)
            if thread.name == "tui-notif-poller-sid_rebuild" and thread.is_alive()
        ]
        assert len(live) == 1, f"expected exactly one live poller, got {len(live)}"

        # Session close can only signal the newest stop event — which now stops the only live poller.
        second_stop.set()
        live[0].join(timeout=3.0)
        assert not live[0].is_alive()
    finally:
        server._sessions.pop("sid_rebuild", None)
        for stop, thread in list(server._notification_pollers):
            stop.set()
            thread.join(timeout=1.0)

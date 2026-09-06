"""Regression coverage for Desktop-owned Heartbeat scheduling."""

from __future__ import annotations

import io
import threading
import time


def test_due_heartbeat_dispatches_a_desktop_turn_and_publishes_the_new_count(tmp_path, monkeypatch):
    """A due idle Desktop session must fire through the normal turn path exactly once."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))

    from hermes_cli import goals
    from hermes_cli.heartbeat import HeartbeatManager, HeartbeatState, save_heartbeat
    from tui_gateway import server

    goals._DB_CACHE.clear()
    sid, key = "desktop-heartbeat-sid", "desktop-heartbeat-key"
    session = {
        "session_key": key,
        "history_lock": threading.RLock(),
        "running": False,
        "queued_prompt": None,
        "queued_prompts": [],
        "_closing": False,
        "agent": object(),
        "attached_images": [],
    }
    server._sessions[sid] = session
    save_heartbeat(
        key,
        HeartbeatState(
            prompt="Check the desktop Heartbeat path.",
            interval_seconds=60,
            created_at=time.time() - 61,
        ),
    )
    dispatched, events = [], []
    monkeypatch.setattr(
        server,
        "_run_prompt_submit",
        lambda rid, got_sid, got_session, text, **kwargs: dispatched.append((rid, got_sid, got_session, text)),
    )
    monkeypatch.setattr(server, "_emit", lambda event, got_sid, payload: events.append((event, got_sid, payload)))

    try:
        assert server._poll_desktop_heartbeats_once() == 1
        assert len(dispatched) == 1
        assert dispatched[0][1:3] == (sid, session)
        assert "[Heartbeat — recurring instruction" in dispatched[0][3]
        assert session["running"] is True
        assert HeartbeatManager(key).state.fire_count == 1
        assert any(
            event == "session.control.update"
            and got_sid == sid
            and payload["control"]["heartbeat"]["fire_count"] == 1
            for event, got_sid, payload in events
        )
    finally:
        server._sessions.pop(sid, None)
        goals._DB_CACHE.clear()


def test_entry_starts_desktop_heartbeat_driver(monkeypatch):
    from hermes_cli import model_switch_providers
    from tui_gateway import entry, server

    started = {"count": 0}
    monkeypatch.setattr(server, "_start_desktop_heartbeat_driver", lambda: started.__setitem__("count", started["count"] + 1))
    monkeypatch.setattr(server, "_start_backend_heartbeat_refresher", lambda: None)
    monkeypatch.setattr(server, "_schedule_startup_orphan_sweep", lambda: None)
    monkeypatch.setattr(entry, "_install_sidecar_publisher", lambda: None)
    monkeypatch.setattr(entry, "ensure_mcp_discovery_started", lambda: None)
    monkeypatch.setattr(entry, "resolve_skin", lambda: "default")
    monkeypatch.setattr(entry.server, "_ensure_skin_watcher", lambda: None)
    monkeypatch.setattr(entry, "_log_exit", lambda reason: None)
    monkeypatch.setattr(entry, "handle_spurious_eof", lambda *args: False)
    monkeypatch.setattr(entry, "write_json", lambda payload: True)
    monkeypatch.setattr(entry.sys, "stdin", io.StringIO(""))
    monkeypatch.setattr(model_switch_providers, "prewarm_picker_cache_async", lambda: None)

    entry.main()
    assert started["count"] == 1


def test_websocket_starts_desktop_heartbeat_driver(monkeypatch):
    import asyncio

    from tui_gateway import server, ws

    started = {"count": 0}
    monkeypatch.setattr(server, "_start_desktop_heartbeat_driver", lambda: started.__setitem__("count", started["count"] + 1))
    monkeypatch.setattr(server, "_start_backend_heartbeat_refresher", lambda: None)
    monkeypatch.setattr(server, "_schedule_startup_orphan_sweep", lambda: None)
    monkeypatch.setattr(server, "resolve_skin", lambda: "default")
    monkeypatch.setattr(server, "_ensure_skin_watcher", lambda: None)
    monkeypatch.setattr(server, "register_live_transport", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 0)

    class FakeWS:
        async def accept(self):
            pass

        async def send_text(self, line):
            pass

        async def receive_text(self):
            raise ws._WebSocketDisconnect()

        async def close(self):
            pass

    asyncio.run(ws.handle_ws(FakeWS()))
    assert started["count"] == 1

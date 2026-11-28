"""The periodic repair restores lost disconnect cleanup, never evicts live viewers."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tui_gateway import server


@pytest.mark.parametrize("already_scheduled", [False, True])
def test_repair_only_schedules_missing_detached_timer(monkeypatch, already_scheduled):
    sid = "closed-desktop"
    session = {"transport": server._detached_ws_transport, "source": "desktop"}
    pending = {sid: object()} if already_scheduled else {}
    schedule = Mock()
    monkeypatch.setattr(server, "_sessions", {sid: session})
    monkeypatch.setattr(server, "_pending_ws_reaps", pending)
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 20)
    monkeypatch.setattr(server, "_schedule_ws_orphan_reap", schedule)

    server._repair_missing_ws_orphan_reaps()

    if already_scheduled:
        schedule.assert_not_called()
    else:
        schedule.assert_called_once_with(sid)
    assert server._sessions[sid] is session


def test_repair_does_not_treat_idle_live_desktop_socket_as_closed_tab(monkeypatch):
    session = {"transport": SimpleNamespace(_closed=False), "source": "desktop", "running": False}
    schedule = Mock()
    disconnect = Mock()
    monkeypatch.setattr(server, "_sessions", {"still-connected": session})
    monkeypatch.setattr(server, "_pending_ws_reaps", {})
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 20)
    monkeypatch.setattr(server, "_schedule_ws_orphan_reap", schedule)
    monkeypatch.setattr(server, "_close_sessions_for_transport", disconnect)

    server._repair_missing_ws_orphan_reaps()

    disconnect.assert_not_called()
    schedule.assert_not_called()


def test_repair_routes_closed_socket_through_normal_disconnect(monkeypatch):
    transport = SimpleNamespace(_closed=True)
    session = {"transport": transport, "source": "desktop"}
    schedule = Mock()

    def disconnect(closed):
        assert closed is transport
        session["transport"] = server._detached_ws_transport

    monkeypatch.setattr(server, "_sessions", {"lost-disconnect": session})
    monkeypatch.setattr(server, "_pending_ws_reaps", {})
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 20)
    monkeypatch.setattr(server, "_schedule_ws_orphan_reap", schedule)
    monkeypatch.setattr(server, "_close_sessions_for_transport", disconnect)

    server._repair_missing_ws_orphan_reaps()

    schedule.assert_called_once_with("lost-disconnect")

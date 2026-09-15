"""WS orphan cleanup must preserve session-owned completion work."""

from types import SimpleNamespace

from tui_gateway import server


class _Timer:
    created = []

    def __init__(self, delay, callback):
        self.delay = delay
        self.callback = callback
        self.cancelled = False
        self.created.append(self)

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True


class _Agent:
    def __init__(self):
        self._process_owner_task_ids = {"delivery-owner"}
        self.close_calls = 0
        self.kill_calls = 0

    def close(self):
        self.close_calls += 1
        self.kill_calls += 1


def _install_orphan(monkeypatch, *, active_processes):
    _Timer.created = []
    sid = "bot-delivery"
    agent = _Agent()
    session = {
        "agent": agent,
        "running": False,
        "transport": server._detached_ws_transport,
        "session_key": "bot-chat",
    }
    monkeypatch.setattr(server, "_sessions", {sid: session})
    monkeypatch.setattr(server, "_pending_ws_reaps", {})
    monkeypatch.setattr(server.threading, "Timer", _Timer)
    monkeypatch.setattr(server, "_WS_ORPHAN_REAP_GRACE_S", 20.0)
    monkeypatch.setattr("tools.async_delegation.has_live_for_session", lambda **_kwargs: False)
    monkeypatch.setattr(
        "tools.process_registry.process_registry.running_owned_by",
        lambda owner: list(active_processes) if owner == "delivery-owner" else [],
    )

    reaped = []

    def teardown(popped, *, end_reason):
        reaped.append((popped, end_reason))
        popped["agent"].close()
        return True

    monkeypatch.setattr(server, "_teardown_popped_session", teardown)
    server._schedule_ws_orphan_reap(sid)
    return sid, session, agent, reaped


def test_orphan_without_active_owned_work_is_reaped_normally(monkeypatch):
    sid, session, agent, reaped = _install_orphan(monkeypatch, active_processes=[])

    _Timer.created[-1].callback()

    assert sid not in server._sessions
    assert reaped == [(session, "ws_orphan_reap")]
    assert agent.close_calls == 1


def test_owned_background_delivery_defers_then_reaps_after_completion(monkeypatch):
    active = [SimpleNamespace(notify_on_complete=True)]
    sid, session, agent, reaped = _install_orphan(monkeypatch, active_processes=active)

    _Timer.created[-1].callback()

    assert server._sessions[sid] is session
    assert server._pending_ws_reaps[sid] is _Timer.created[-1]
    assert _Timer.created[-1].delay == 20.0
    assert reaped == []
    assert agent.close_calls == 0
    assert agent.kill_calls == 0, "WS disconnect must not SIGTERM active delivery through agent.close()"

    active.clear()
    _Timer.created[-1].callback()

    assert sid not in server._sessions
    assert reaped == [(session, "ws_orphan_reap")]
    assert agent.close_calls == 1

"""Session control must share the close path's registry-before-history lock order."""

from __future__ import annotations

import threading

import pytest

from tests.tui_gateway.test_plugin_session_slash import LiveAgent, _session

pytest_plugins = ("tests.tui_gateway.test_plugin_session_slash",)


class RegistryBarrier:
    """Reject a registry wait while this operation already owns non-reentrant history."""

    def __init__(self, history_lock):
        self.history_lock = history_lock
        self.lock = threading.RLock()

    def __enter__(self):
        assert not self.history_lock.locked(), "history-to-registry wait can deadlock close"
        self.lock.acquire()
        return self

    def __exit__(self, *_exc):
        self.lock.release()


def test_control_claim_and_canonical_write_never_invert_close_locks(gateway, monkeypatch):
    server, db, _home = gateway
    agent = LiveAgent(db, "ordered-control")
    session = _session(server, db, "ordered", agent.session_id, agent)
    monkeypatch.setattr(server, "_sessions_lock", RegistryBarrier(session["history_lock"]))

    def child_probe(*_args):
        assert not session["history_lock"].locked()
        return False

    monkeypatch.setattr(server, "_child_run_active", child_probe)
    target, error = server._claim_live_plugin_session_control("ordered", session)
    assert error is None and target is not None
    try:
        server._plugin_session_ensure(target, session)()
        assert db.get_session(agent.session_id) is not None
    finally:
        assert server._release_live_plugin_session_control(target)
    assert session.get("_session_control_inflight") is None


def test_cold_attach_uses_close_lock_order_without_touching_control_owner(gateway, monkeypatch):
    server, db, _home = gateway
    session = _session(server, db, "ordered-attach", "ordered-attach-key", None)
    token = session["_session_control_inflight"] = object()
    monkeypatch.setattr(server, "_sessions_lock", RegistryBarrier(session["history_lock"]))
    assert not server._attach_built_agent("ordered-attach", session, LiveAgent(db, session["session_key"]))
    assert session["agent"] is None and session["_session_control_inflight"] is token

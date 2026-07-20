import threading

import tui_gateway.server as server


def _clear_runtime_state():
    with server._sessions_lock:
        server._sessions.clear()
    server._pending.clear()
    server._pending_prompt_payloads.clear()


def test_active_work_snapshot_reports_running_session_without_active_session_cap():
    _clear_runtime_state()
    try:
        with server._sessions_lock:
            server._sessions["sid-1"] = {"running": True}

        snapshot = server.active_work_snapshot()

        assert snapshot["active"] is True
        assert snapshot["running_sessions"] == 1
    finally:
        _clear_runtime_state()


def test_active_work_snapshot_reports_agent_build_starting():
    _clear_runtime_state()
    ready = threading.Event()
    try:
        with server._sessions_lock:
            server._sessions["sid-1"] = {
                "agent_build_started": True,
                "agent_ready": ready,
                "running": False,
            }

        snapshot = server.active_work_snapshot()

        assert snapshot["active"] is True
        assert snapshot["starting_sessions"] == 1
    finally:
        _clear_runtime_state()


def test_active_work_snapshot_merges_subagent_registries(monkeypatch):
    import tools.async_delegation as async_delegation
    import tools.delegate_tool as delegate_tool

    _clear_runtime_state()
    monkeypatch.setattr(delegate_tool, "list_active_subagents", lambda: [{"id": "native"}])
    monkeypatch.setattr(async_delegation, "active_count", lambda: 2)

    snapshot = server.active_work_snapshot()

    assert snapshot["active"] is True
    assert snapshot["active_subagents"] == 3


def test_active_work_snapshot_is_idle_when_only_detached_idle_sessions_exist():
    _clear_runtime_state()
    try:
        with server._sessions_lock:
            server._sessions["sid-1"] = {"running": False}

        snapshot = server.active_work_snapshot()

        assert snapshot["active"] is False
        assert snapshot["running_sessions"] == 0
    finally:
        _clear_runtime_state()

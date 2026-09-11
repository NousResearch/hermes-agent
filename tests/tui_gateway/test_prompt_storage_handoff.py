"""Storage refusals must leave a fresh handoff session resumable and retryable."""

import threading
from unittest.mock import Mock

import pytest

from hermes_state import SessionDB
from tui_gateway import server


@pytest.fixture
def draft(monkeypatch, tmp_path):
    db = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_db_error", None)
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *args: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_load_cfg", lambda: {"dashboard": {}})
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    created = server.handle_request({
        "id": "create", "method": "session.create",
        "params": {"source": "desktop", "cwd": str(tmp_path), "title": "First build"},
    })["result"]
    sid, key = created["session_id"], created["stored_session_id"]
    session = server._sessions[sid]
    assert db.get_session(key) is None
    yield db, sid, key, session
    server._release_active_session_slot(session)
    from tools.terminal_tool import clear_task_env_overrides
    clear_task_env_overrides(key)
    db.close()


@pytest.mark.parametrize("code", [5070, 5071, 5072])
def test_storage_refusal_releases_turn_and_allows_repair_retry(draft, monkeypatch, code):
    db, sid, key, session = draft
    start_agent = Mock()
    monkeypatch.setattr(server, "_start_agent_build", start_agent)
    ran = threading.Event()
    monkeypatch.setattr(server, "_run_after_agent_ready", lambda *args: ran.set())

    with monkeypatch.context() as broken:
        if code == 5070:
            # A real SQLite SQLITE_FULL, without filling the machine's disk.
            pages = db._conn.execute("PRAGMA page_count").fetchone()[0]
            db._conn.execute(f"PRAGMA max_page_count = {pages}")
            session["model_override"] = {"model": "x" * 131072}
        elif code == 5071:
            def fail_seed(_session):
                raise OSError("seed storage unavailable")
            broken.setattr(server, "_persist_branch_seed", fail_seed)
        else:
            broken.setattr(server, "_get_db", lambda: None)
            broken.setattr(server, "_db_error", "state.db cannot be opened")
        response = server.handle_request({
            "id": "reject", "method": "prompt.submit",
            "params": {"session_id": sid, "text": "Build my tracker"},
        })
        assert response["error"]["code"] == code
        assert session["running"] is False
        assert session["inflight_turn"] is None
        assert server._turn_started_at(session) is None
        assert session.get("active_session_lease") is None
        assert session.get("_run_thread") is None
        start_agent.assert_not_called()
        assert not ran.is_set()
        assert session["history"] == []

    db._conn.execute("PRAGMA max_page_count = 2147483646")
    session["model_override"] = None
    retry = server.handle_request({
        "id": "retry", "method": "prompt.submit",
        "params": {"session_id": sid, "text": "Build my tracker"},
    })
    assert retry["result"]["status"] == "streaming"
    assert ran.wait(2)
    session["_run_thread"].join(timeout=2)
    start_agent.assert_called_once()
    assert session["running"] is True
    assert session["inflight_turn"]["user"] == "Build my tracker"
    assert db.get_session(key)["id"] == key
    assert list(server._sessions) == [sid]

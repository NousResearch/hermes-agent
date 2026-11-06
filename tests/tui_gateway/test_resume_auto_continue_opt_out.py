"""``session.resume {auto_continue: false}`` never starts a crash-recovery continuation.

A client that opens a stored session only to read it or answer its prompts (a phone
browsing sessions, say) must not kick off a continuation turn, which can run tools.
The turn marker is left untouched, so a later resume that allows it still recovers.
The default (key absent or true) keeps today's behaviour on every cold path.
"""

import threading

import pytest

from hermes_state import SessionDB
from tui_gateway import server


def _interrupted_db(tmp_path):
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    db = SessionDB(home / "state.db")
    db.create_session("crashed", source="tui")
    db.append_message("crashed", "user", "fix the flaky test", timestamp=100.0)
    return db, home


def _mount(monkeypatch, db, home, tmp_path):
    scheduled = []
    built = threading.Event()
    monkeypatch.setattr("hermes_state_registry.acquire", lambda db_path=None, **kwargs: db)
    monkeypatch.setattr(server, "_profile_home", lambda p: home if p else None)
    monkeypatch.setattr(server, "_profile_configured_cwd", lambda _: str(tmp_path))
    monkeypatch.setattr(server, "_default_session_cwd", lambda: str(tmp_path))
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *args: built.set())
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)

    def _record(sid, session, key):
        scheduled.append(key)
        return {"attempt": 1, "interrupted_at": 100.0}

    monkeypatch.setattr(server, "_maybe_schedule_auto_continue", _record)
    return scheduled, built


def _resume(params):
    response = server.handle_request({"id": "resume", "method": "session.resume", "params": params})
    assert response is not None and "error" not in response, response
    return response["result"]


@pytest.mark.parametrize("defer_history", [False, True])
@pytest.mark.parametrize("auto_continue, expected", [(None, ["crashed"]), (True, ["crashed"]), (False, [])])
def test_auto_continue_flag_gates_the_continuation(tmp_path, monkeypatch, defer_history, auto_continue, expected):
    db, home = _interrupted_db(tmp_path)
    scheduled, built = _mount(monkeypatch, db, home, tmp_path)
    params = {"session_id": "crashed", "source": "tui", "defer_history": defer_history}
    if auto_continue is not None:
        params["auto_continue"] = auto_continue
    sid = None
    try:
        result = _resume(params)
        sid = result["session_id"]
        if defer_history:
            assert server._sessions[sid]["resume_history_ready"].wait(5)
            assert built.wait(5)
        assert scheduled == expected
        if not expected:
            assert "auto_continue" not in result, "an opted-out resume must not report a continuation"
    finally:
        if sid is not None:
            server._sessions.pop(sid, None)


def test_params_contract_accepts_the_flag():
    from tui_gateway.contracts.sessions import SessionResumeParams

    assert SessionResumeParams(session_id="crashed").auto_continue is True
    assert SessionResumeParams(session_id="crashed", auto_continue=False).auto_continue is False

"""Command reports survive cold resume without becoming model conversation."""

import asyncio
import copy
from types import SimpleNamespace
import uuid

import pytest


@pytest.fixture
def context(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    from tui_gateway import server

    db = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_hermes_home", tmp_path)
    monkeypatch.setattr(server, "_cfg_cache", None)
    monkeypatch.setattr(server, "_cfg_path", None)
    monkeypatch.setattr(server, "_cfg_mtime", None)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_maybe_schedule_auto_continue", lambda *a, **k: None)
    monkeypatch.setattr(server, "_profile_home", lambda profile: None)
    monkeypatch.setattr(server, "_live_slash_command_output", lambda *a: None)
    monkeypatch.setattr(server, "_bundle_key_for", lambda name: None)
    monkeypatch.setattr(server, "_is_profile_skill_command", lambda *a: False)
    monkeypatch.setattr(server, "_dispatch_quick", lambda *a: None)
    monkeypatch.setattr(server, "_plugin_command_handler", lambda name: object() if name == "report" else None)
    known = set(server._sessions)
    yield server, db
    for sid in set(server._sessions) - known:
        server._sessions.pop(sid, None)
    db.close()


def call(server, method, **params):
    return server.handle_request({"id": str(uuid.uuid4()), "method": method, "params": params})


def test_event_merge_does_not_reorder_compacted_message_tail(context, monkeypatch):
    from tui_gateway.command_display import with_display_events

    _, db = context
    db.create_session("compacted", source="desktop")
    monkeypatch.setattr("hermes_state_display.time", SimpleNamespace(time=lambda: 15.0))
    record = db.append_display_event("compacted", str(uuid.uuid4()), "report", "Receipt")
    messages = [
        {"role": "user", "text": "Fresh summary", "timestamp": 20},
        {"role": "assistant", "text": "Carried original tail", "timestamp": 10},
    ]
    before = copy.deepcopy(messages)
    merged = with_display_events(messages, db, "compacted")
    assert merged[0]["row_id"] == record["id"]
    assert merged[1:] == before
    assert messages == before


@pytest.mark.parametrize("method", ["slash.exec", "command.dispatch"])
def test_command_only_chat_cold_resume_and_rest_are_display_only(context, monkeypatch, tmp_path, method):
    from hermes_cli.web_routers import sessions as routes

    server, db = context
    report = "## Analysis receipt\n### Markets\n- H2H: SILVER\n- Spreads: WATCH\n- Totals: NO\\_BET"
    executed = []

    def run_plugin(handler, arg):
        executed.append(arg)
        return report

    monkeypatch.setattr(server, "_run_plugin_command", run_plugin)
    created = call(server, "session.create", source="desktop", title="Report review", cwd=str(tmp_path))["result"]
    sid, key = created["session_id"], created["stored_session_id"]
    assert db.get_session(key) is None
    history = copy.deepcopy(server._sessions[sid]["history"])
    event_id = str(uuid.uuid4())
    params = {"command": "report existing-job"} if method == "slash.exec" else {"name": "report", "arg": "existing-job"}
    result = call(server, method, session_id=sid, display_event_id=event_id, **params)["result"]
    assert result["output"] == report
    assert result["display_event"]["id"] == f"display:{event_id}"
    assert db.get_session_title(key) == "Report review"
    assert db.get_messages(key) == []
    assert db.get_resume_conversations(key) == ([], [])
    assert server._sessions[sid]["history"] == history
    assert any(row["id"] == key for row in db.list_sessions_rich(min_message_count=1, include_display_events=True))

    # Drop only this test's runtime cache to exercise the real cold-resume path.
    server._sessions.pop(sid)
    resumed = call(server, "session.resume", session_id=key, source="desktop")["result"]
    assert server._sessions[resumed["session_id"]]["history"] == history
    display = resumed["messages"]
    assert len(display) == 1
    assert display[0]["text"] == f"slash:/report\n{report}"
    assert display[0]["display_kind"] == "command_result"
    assert display[0]["row_id"] == f"display:{event_id}"

    monkeypatch.setattr(routes, "_with_db", lambda profile, action, **kw: action(db))
    async def read(include):
        return await routes.get_session_messages(
            key, profile=None, limit=10, offset=0, order="latest",
            include_compacted=False, include_display_events=include)

    assert asyncio.run(read(False))["messages"] == []
    visible = asyncio.run(read(True))["messages"]
    assert len(visible) == 1 and visible[0]["content"] == display[0]["text"]
    assert server._coerce_seed_history(visible) == []
    assert server._coerce_seed_history(display) == []
    assert executed == ["existing-job"]


def test_save_failure_preserves_result_and_other_profile_isolation(context, monkeypatch, tmp_path):
    from hermes_state import SessionDB

    server, launch_db = context
    owner_home = tmp_path / "other-profile"
    owner_home.mkdir()
    owner = SessionDB(owner_home / "state.db")
    executed = []
    monkeypatch.setattr(server, "_run_plugin_command", lambda *a: executed.append(True) or "## Finished report")
    created = call(server, "session.create", source="desktop", cwd=str(tmp_path))["result"]
    sid, key = created["session_id"], created["stored_session_id"]
    server._sessions[sid]["profile_home"] = str(owner_home)
    monkeypatch.setattr("hermes_state_registry.acquire", lambda path: owner)
    monkeypatch.setattr("hermes_state_registry.release_or_close", lambda db: None)
    try:
        success = call(server, "slash.exec", session_id=sid, command="report",
                       display_event_id=str(uuid.uuid4()))["result"]
        assert "display_event" in success
        assert launch_db.get_session(key) is None
        assert owner.get_session(key) is not None
        assert len(owner.get_display_events(key)) == 1
        owner.append_message(key, "user", "Existing question")
        owner.append_message(key, "assistant", "Existing answer")
        model_before = owner.get_resume_conversations(key)[0]
        server._sessions[sid]["history"] = copy.deepcopy(model_before)

        def unavailable(*args, **kwargs):
            raise OSError("sensitive-provider-response")

        monkeypatch.setattr(owner, "append_display_event", unavailable)
        result = call(server, "slash.exec", session_id=sid, command="report",
                      display_event_id=str(uuid.uuid4()))
        assert "error" not in result
        assert result["result"]["output"] == "## Finished report"
        assert "could not be saved" in result["result"]["persistence_error"]
        assert "sensitive-provider-response" not in str(result)
        assert len(executed) == 2
        assert len(owner.get_display_events(key)) == 1
        assert owner.get_resume_conversations(key)[0] == model_before
        assert server._sessions[sid]["history"] == model_before
    finally:
        owner.close()

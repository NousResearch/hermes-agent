"""Real foreign logs through the registered desktop RPCs and profile stores."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest


def test_foreign_rpc_preview_import_and_profile_isolation(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    from tui_gateway import server

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.delenv("HERMES_DESKTOP", raising=False)
    folder = tmp_path / ".claude" / "projects" / "project"
    folder.mkdir(parents=True)
    log = folder / "session.jsonl"
    lines = [{"type": role, "sessionId": "foreign-one", "cwd": str(tmp_path),
              "message": {"role": role, "content": content}}
             for role, content in [("user", "# Files mentioned by the user:\nplan.md\n\n## My request:\nFix the import"), ("assistant", "Here is the fix."),
                                   ("assistant", "And its test."), ("user", "Continue")]]
    log.write_text("\n".join(map(json.dumps, lines)), encoding="utf-8")
    original = log.read_bytes()
    db = SessionDB(tmp_path / ".hermes" / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_profile_home", lambda profile: None)

    def rpc(method, **params):
        result = server._methods[f"session.foreign.{method}"](1, params)
        assert "error" not in result, result
        return result["result"]

    try:
        page = rpc("list", limit=1)
        handle = page["sessions"][0]["id"]
        assert page["sessions"][0]["title"] == "Fix the import"
        assert "path" not in page["sessions"][0]
        preview = rpc("preview", id=handle)
        assert preview["already_imported"] is None
        assert db.session_count() == 0
        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: rpc("import", id=handle), range(2)))
        assert results[0]["session_id"] == results[1]["session_id"]
        sid = results[0]["session_id"]
        stored = db.get_session(sid)
        assert json.loads(stored["origin_json"])["imported_from"]["foreign_session_id"] == "foreign-one"
        history = db.get_messages(sid)
        assert [message["role"] for message in history] == ["user", "assistant", "user"]
        assert len(history) == preview["total"] == stored["message_count"]
        assert rpc("preview", id=handle)["already_imported"] == sid
        assert log.read_bytes() == original
        with SessionDB(tmp_path / "other" / "state.db") as other:
            monkeypatch.setattr(server, "_get_db", lambda: other)
            assert rpc("preview", id=handle)["already_imported"] is None
            assert rpc("import", id=handle)["session_id"] != sid
    finally:
        db.close()


def test_foreign_pages_confine_handles_and_failed_import_rolls_back(tmp_path, monkeypatch):
    from hermes_cli import foreign_sessions_browser as browser
    from hermes_state import SessionDB

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    folder = tmp_path / ".codex" / "sessions"
    folder.mkdir(parents=True)
    for index in range(3):
        (folder / f"rollout-{index}.jsonl").write_text(json.dumps({
            "type": "response_item", "payload": {"type": "message", "role": "user",
            "content": [{"type": "input_text", "text": f"Question {index}"}]}}), encoding="utf-8")
    calls = []
    parse = browser._parse
    monkeypatch.setattr(browser, "_parse", lambda row: (calls.append(row[1]), parse(row))[1])
    page = browser.list_foreign_sessions(limit=1)
    assert len(calls) == 1
    next_page = browser.list_foreign_sessions(offset=page["next_offset"], limit=1)
    assert page["sessions"][0]["id"] != next_page["sessions"][0]["id"]
    with pytest.raises(ValueError):
        browser.resolve_foreign_session(str(tmp_path / "secret.jsonl"))
    # A fabricated well-shaped handle cannot become a file-read request.
    with pytest.raises(ValueError):
        browser.resolve_foreign_session("0" * 64)
    db = SessionDB(tmp_path / ".hermes" / "state.db")
    insert = db._insert_message_rows
    def failing_insert(*args, **kwargs):
        insert(*args, **kwargs)
        raise RuntimeError("interrupted write")
    monkeypatch.setattr(db, "_insert_message_rows", failing_insert)
    try:
        with pytest.raises(RuntimeError, match="interrupted write"):
            browser.import_browser_session(page["sessions"][0]["id"], db, "default")
        assert db.session_count() == 0
        from tui_gateway import server
        monkeypatch.setattr(server, "_get_db", lambda: db)
        monkeypatch.setattr(server, "_profile_home", lambda profile: None)
        with pytest.raises(RuntimeError, match="interrupted write"):
            server.handle_request({"id": 2, "method": "session.foreign.import",
                                   "params": {"id": page["sessions"][0]["id"]}})
        assert db.session_count() == 0
    finally:
        db.close()


@pytest.mark.parametrize("collision", ["foreign_live", "foreign_ended", "local"])
def test_foreign_rpc_refuses_generated_id_collision(tmp_path, monkeypatch, collision):
    from hermes_state import SessionDB
    import hermes_state_portability as portability
    from tui_gateway import server

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    folder = tmp_path / ".claude" / "projects" / "project"
    folder.mkdir(parents=True)
    log = folder / "session.jsonl"
    log.write_text(json.dumps({"type": "user", "sessionId": "foreign-one", "message": {"role": "user", "content": "external"}}), encoding="utf-8")
    original = log.read_bytes()
    home = tmp_path / ".hermes"
    profile = home / "profiles" / "science"
    profile.mkdir(parents=True)
    local, sibling = SessionDB(home / "state.db"), SessionDB(profile / "state.db")
    sid = "fixed-foreign-collision"
    owner = local if collision == "local" else sibling
    try:
        owner.create_session(sid, "cli")
        owner.append_message(sid, "user", "preserve")
        if collision == "foreign_ended":
            owner.end_session(sid, "done")
        def snapshot():
            return [[tuple(row) for row in db._conn.execute("SELECT * FROM " + table + " ORDER BY rowid")]
                    for db in (local, sibling) for table in ("sessions", "messages")]
        before = snapshot()
        monkeypatch.setattr(portability, "new_session_id", lambda **kwargs: sid)
        monkeypatch.setattr(server, "_get_db", lambda: local)
        monkeypatch.setattr(server, "_profile_home", lambda profile: None)
        page = server._methods["session.foreign.list"](1, {"limit": 1})["result"]
        response = server.handle_request({"id": 2, "method": "session.foreign.import", "params": {"id": page["sessions"][0]["id"]}})
        assert response.get("error"), response
        assert snapshot() == before
        assert (local.get_session(sid) is not None) == (collision == "local")
        assert log.read_bytes() == original
    finally:
        local.close()
        sibling.close()


def test_foreign_rpc_existing_origin_replay_precedes_new_id_admission(tmp_path, monkeypatch):
    from hermes_state import SessionDB
    import hermes_state_portability as portability
    from tui_gateway import server

    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    folder = tmp_path / ".claude" / "projects" / "project"
    folder.mkdir(parents=True)
    log = folder / "session.jsonl"
    log.write_text(json.dumps({"type": "user", "sessionId": "replay-foreign", "message": {"role": "user", "content": "content"}}), encoding="utf-8")
    home = tmp_path / ".hermes"
    profile = home / "profiles" / "science"
    profile.mkdir(parents=True)
    local, sibling = SessionDB(home / "state.db"), SessionDB(profile / "state.db")
    try:
        sibling.create_session("blocked-next-id", "cli")
        monkeypatch.setattr(server, "_get_db", lambda: local)
        monkeypatch.setattr(server, "_profile_home", lambda profile: None)
        handle = server._methods["session.foreign.list"](1, {"limit": 1})["result"]["sessions"][0]["id"]
        monkeypatch.setattr(portability, "new_session_id", lambda **kwargs: "first-import-id")
        first = server.handle_request({"id": 2, "method": "session.foreign.import", "params": {"id": handle}})["result"]
        assert first == {"session_id": "first-import-id", "already_imported": False}
        before = [tuple(row) for row in local._conn.execute("SELECT * FROM sessions ORDER BY id")]
        monkeypatch.setattr(portability, "new_session_id", lambda **kwargs: "blocked-next-id")
        replay = server.handle_request({"id": 3, "method": "session.foreign.import", "params": {"id": handle}})["result"]
        assert replay == {"session_id": "first-import-id", "already_imported": True}
        assert [tuple(row) for row in local._conn.execute("SELECT * FROM sessions ORDER BY id")] == before
        assert local.get_session("blocked-next-id") is None
    finally:
        local.close()
        sibling.close()

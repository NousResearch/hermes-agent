"""Closing the app before onboarding finishes does not count: the next launch's ``onboarding.start``
(``fresh_launch``) deletes the unfinished chat and starts over at the name beat, while a second window
or a reconnect in the same launch (no flag) keeps the run."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent import onboarding_record
from agent.onboarding_flow import PENDING, Answered, Facts, Shown
from hermes_cli.config import read_user_config_raw
from tui_gateway import server

FACTS = Facts(os="macos", arch="arm64", machine_kind="Mac", plugins=())


@pytest.fixture
def root(tmp_path, monkeypatch) -> Path:
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(server, "_hermes_home", str(home))
    monkeypatch.setattr(server, "_db", None)
    monkeypatch.setattr("agent.onboarding_facts.read_facts", lambda **kw: FACTS)
    (home / ".env").write_text("OPENROUTER_API_KEY=sk-or-test\n", encoding="utf-8")  # a model to run on
    return home


@pytest.fixture
def client(root, monkeypatch):
    """session.resume and prompt.submit as a recorder; a submit stores its user turn, as the real turn does."""
    calls = []

    def resume(rid, params):
        calls.append(("resume", params))
        return {"result": {"session_id": f"rt-{params['session_id']}"}}

    def submit(rid, params):
        calls.append(("submit", params))
        server._get_db().append_message(params["session_id"].removeprefix("rt-"), "user", params["text"])
        return {"result": {"status": "streaming"}}

    monkeypatch.setitem(server._methods, "session.resume", resume)
    monkeypatch.setitem(server._methods, "prompt.submit", submit)
    from hermes_cli.process_identity import serving_local_desktop
    with serving_local_desktop(True):  # the local Desktop's requests: only it starts a first run
        yield calls


@pytest.fixture
def ended(monkeypatch) -> list[dict]:
    told: list[dict] = []
    monkeypatch.setattr(onboarding_record, "_ended_listeners", [told.append])
    return told


def _start(params: dict | None = None) -> dict:
    response = server.handle_request({"jsonrpc": "2.0", "id": 7, "method": "onboarding.start", "params": params or {}})
    assert "result" in response, response
    return response["result"]


def _run_flag(root: Path) -> bool:
    return read_user_config_raw(root / "config.yaml")["onboarding"]["run"]


def test_a_new_launch_deletes_the_unfinished_chat_and_starts_over_at_the_name_beat(root, client, ended):
    first = _start({"fresh_launch": True})
    old = first["stored_session_id"]
    db = server._get_db()
    old_title = db.get_session(old)["title"]
    # The user got past the name card, then quit the app.
    onboarding_record.advance(db, old, Shown("name"))
    onboarding_record.advance(db, old, Answered("name", {"name": "Sid"}))
    assert onboarding_record.load(db, old)["beats"]["name"] != PENDING

    again = _start({"fresh_launch": True})
    new = again["stored_session_id"]

    assert new != old and again["resumed"] is False and again["run_id"] != first["run_id"]
    assert db.get_session(old) is None
    assert [(e["stored_session_id"], e["outcome"]) for e in ended] == [(old, "deleted")]
    record = onboarding_record.load(db, new)
    assert onboarding_record.active_session_id(db) == new and record["beats"]["name"] == PENDING
    # The fresh chat plays the opening again, under the title the deleted chat had.
    assert [kind for kind, _ in client] == ["resume", "submit", "resume", "submit"]
    assert db.get_session(new)["title"] == old_title
    assert _run_flag(root) is True


def test_a_second_window_or_a_reconnect_keeps_the_run(root, client, ended):
    first = _start({"fresh_launch": True})

    again = _start()

    assert again["stored_session_id"] == first["stored_session_id"] and again["resumed"] is True
    assert again["run_id"] == first["run_id"] and ended == []


def test_a_new_launch_closes_the_unfinished_chat_still_live_here_before_deleting_it(root, client, ended, monkeypatch):
    old = _start({"fresh_launch": True})["stored_session_id"]
    closed = []
    monkeypatch.setitem(server._sessions, "rt-live", {"session_key": old})
    monkeypatch.setattr(server, "_close_session_by_id",
                        lambda sid, **kw: closed.append((sid, server._get_db().get_session(old) is not None)))

    _start({"fresh_launch": True})

    assert closed == [("rt-live", True)]
    assert server._get_db().get_session(old) is None

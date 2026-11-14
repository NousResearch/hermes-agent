"""Discovery output is derived recall text: strict redaction on every model-visible field (#113165)."""

import json

import pytest

from hermes_state import SessionDB
from tools.session_search_tool import session_search

SECRET = "ghp_" + "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456"


@pytest.fixture
def db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    db.create_session("s_secret", source="cli")
    db.append_message("s_secret", role="user", content="please deploy the service")
    db.append_message("s_secret", role="assistant", content="sure", tool_calls=[{
        "id": "c1", "type": "function",
        "function": {"name": "terminal", "arguments": json.dumps({"command": f"export GITHUB_TOKEN={SECRET}"})}}])
    db.append_message("s_secret", role="user", content=f"deployneedle token is {SECRET}")
    db.append_message("s_secret", role="assistant", content="deployed")
    return db


def _dump(entry) -> str:
    return json.dumps(entry, ensure_ascii=False)


@pytest.mark.parametrize("detail", ["compact", "full"])
def test_discovery_never_returns_secret_in_message_bodies(db, monkeypatch, detail):
    from agent import redact
    monkeypatch.setattr(redact, "_redact_enabled", lambda: False)  # user opted out; derived text stays strict
    out = json.loads(session_search(query="deployneedle", detail=detail, db=db))
    assert out["results"], out
    hit = out["results"][0]
    assert any(m.get("anchor") for m in hit["messages"])
    assert SECRET not in _dump(hit)


def test_title_match_discovery_redacts_window(db, monkeypatch):
    db.set_session_title("s_secret", "release runbook")
    out = json.loads(session_search(query="release runbook", db=db))
    hit = out["results"][0]
    assert hit["matched_role"] == "session_title"
    assert SECRET not in _dump(hit)


def test_explicit_session_read_stays_verbatim(db):
    read = json.loads(session_search(session_id="s_secret", db=db))
    assert SECRET in _dump(read["messages"])

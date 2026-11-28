"""Both desktop fork RPCs distinguish inherited titles from explicit user names."""

import importlib
import threading
from types import SimpleNamespace

import pytest

from persistence.conversation.messages import TranscriptRead, TranscriptView
from persistence.conversation.sessions import TitleSource
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_messages, seed_session


@pytest.mark.parametrize("method", ["session.branch", "session.create"])
@pytest.mark.parametrize("explicit_title", [None, "My chosen branch"])
def test_branch_title_provenance(method, explicit_title, monkeypatch, tmp_path):
    server = importlib.import_module("tui_gateway.server")
    db = open_persistence(profile_scope(home=tmp_path))
    history = [
        {"role": "user", "content": "Build a dashboard"},
        {"role": "assistant", "content": "Use a status overview"},
    ]
    seed_session(db, "parent", "desktop", title="Dashboard project")
    seed_messages(db, "parent", *history)
    agent = SimpleNamespace(model="test-model")
    ready = threading.Event()
    ready.set()
    parent = {
        "session_key": "parent", "history": history, "history_lock": threading.Lock(),
        "running": False, "agent": agent, "agent_ready": ready, "agent_error": None,
        "source": "desktop", "cwd": str(tmp_path), "cols": 96,
        "profile_home": None, "created_at": 1.0, "last_active": 1.0,
    }
    monkeypatch.setattr(server, "_sessions", {"parent-runtime": parent})
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a, **k: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda *a, **k: None)
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *a, **k: None)
    monkeypatch.setattr(server, "_build_branch_agent", lambda *a, **k: agent)
    monkeypatch.setattr(server, "_session_info", lambda *a, **k: {})
    monkeypatch.setattr(server, "_project_info_for_cwd", lambda *a, **k: None)
    try:
        if method == "session.branch":
            params = {"session_id": "parent-runtime"}
            if explicit_title is not None:
                params["name"] = explicit_title
        else:
            params = {"source": "desktop", "parent_session_id": "parent", "messages": history,
                      "cwd": str(tmp_path)}
            if explicit_title is not None:
                params["title"] = explicit_title
        response = server.handle_request({"id": "branch-title", "method": method, "params": params})
        assert "result" in response, response
        key = response["result"]["stored_session_id"]
        row = db.sessions.get(key)
        assert row is not None
        assert row.title_source == (TitleSource.USER if explicit_title else TitleSource.BRANCH)
        assert row.title == (explicit_title or "Dashboard project #2")
        assert row.parent_session_id == "parent"
        assert row.model_config["_branched_from"] == "parent"
        copied = db.messages.read_transcript(key, TranscriptRead(view=TranscriptView.ALL))
        assert [m.content for m in copied] == [m["content"] for m in history]
        assert db.sessions.get_title("parent") == "Dashboard project"
    finally:
        db.close()

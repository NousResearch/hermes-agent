"""Both desktop fork RPCs distinguish inherited titles from explicit user names."""

import importlib
import json
import threading
from types import SimpleNamespace

import pytest

from hermes_state import SessionDB


@pytest.mark.parametrize("method", ["session.branch", "session.create"])
@pytest.mark.parametrize("explicit_title", [None, "My chosen branch"])
def test_branch_title_provenance(method, explicit_title, monkeypatch, tmp_path):
    server = importlib.import_module("tui_gateway.server")
    db = SessionDB(db_path=tmp_path / "state.db")
    history = [
        {"role": "user", "content": "Build a dashboard"},
        {"role": "assistant", "content": "Use a status overview"},
    ]
    db.create_session("parent", source="desktop")
    db.set_session_title("parent", "Dashboard project")
    db.append_messages_batch("parent", history)
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
        row = db.get_session(key)
        assert row is not None
        assert row["title_source"] == ("user" if explicit_title else "branch")
        assert row["title"] == (explicit_title or "Dashboard project #2")
        assert row["parent_session_id"] == "parent"
        assert json.loads(row["model_config"])["_branched_from"] == "parent"
        assert [m["content"] for m in db.get_messages(key)] == [m["content"] for m in history]
        assert db.get_session_title("parent") == "Dashboard project"
    finally:
        db.close()

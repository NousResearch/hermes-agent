"""Implicit display directories must not become terminal overrides (#38156)."""

import os
import threading
from pathlib import Path

import pytest

from hermes_state import SessionDB
from tools import terminal_tool
from tui_gateway import server


@pytest.mark.parametrize("backend", ["docker", "ssh", "local"])
def test_only_a_chosen_workspace_registers_a_terminal_override(monkeypatch, tmp_path, backend):
    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text(f"terminal:\n  backend: {backend}\n", encoding="utf-8")
    monkeypatch.setattr(server, "_hermes_home", home)
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    assert server._effective_terminal_backend() == backend
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(server, "_get_db", lambda: None)
    explicit, cwd, _ = server._resolve_create_cwd({}, "tui", None)
    key = f"cwd-provenance-{backend}"
    session = {"session_key": key, "cwd": cwd, "explicit_cwd": explicit, "source": "tui"}
    try:
        server._register_session_cwd(session)
        assert cwd == str(tmp_path) and not explicit
        assert terminal_tool.get_session_cwd(key) is None

        workspace = tmp_path / "chosen"
        workspace.mkdir()
        assert server._set_session_cwd(session, str(workspace)) == str(workspace)
        assert terminal_tool.get_session_cwd(key) == str(workspace)
    finally:
        terminal_tool.clear_task_env_overrides(key)
        terminal_tool.clear_session_cwd(key)


@pytest.mark.parametrize("backend", ["docker", "ssh"])
@pytest.mark.parametrize("explicit", [False, True])
def test_branch_keeps_the_cwd_snapshot_and_remote_persistence(
    monkeypatch, tmp_path, backend, explicit
):
    home = Path(os.environ["HERMES_HOME"])
    (home / "config.yaml").write_text(f"terminal:\n  backend: {backend}\n", encoding="utf-8")
    monkeypatch.setattr(server, "_hermes_home", home)
    monkeypatch.delenv("TERMINAL_ENV", raising=False)
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    assert server._effective_terminal_backend() == backend
    cwd = str(tmp_path)
    session = {"session_key": "parent", "cwd": cwd, "explicit_cwd": explicit, "source": "tui",
               "history": [{"role": "user", "content": "branch this"}], "history_lock": threading.Lock()}
    captured = {}
    persist = server._persist_branch

    def persist_then_move(*args, **kwargs):
        persist(*args, **kwargs)
        session.update(cwd=str(tmp_path / "later"), explicit_cwd=True)

    def build_branch(record, _sid, key, _history, _source):
        captured.update(record, session_key=key)
        server._register_session_cwd(captured)

    monkeypatch.setattr(server, "_persist_branch", persist_then_move)
    monkeypatch.setattr(server, "_build_branch_agent", build_branch)
    monkeypatch.setattr(server, "_session_info", lambda *_a: {})
    monkeypatch.setattr(server, "_session_default_route", lambda _s: ("test-model", ""))
    with SessionDB(db_path=tmp_path / "state.db") as db:
        db.create_session("parent", source="tui", model="test-model")
        monkeypatch.setattr(server, "_get_db", lambda: db)
        try:
            response = server._branch_live("branch", {}, session, omit_messages=True)
            assert "error" not in response, response
            key = response["result"]["stored_session_id"]
            assert captured["cwd"] == cwd and captured["explicit_cwd"] is explicit
            assert terminal_tool.get_session_cwd(key) == (cwd if explicit else None)
            assert db.get_session(key)["cwd"] == (None if backend == "ssh" and not explicit else cwd)
        finally:
            if key := captured.get("session_key"):
                terminal_tool.clear_task_env_overrides(key)
                terminal_tool.clear_session_cwd(key)

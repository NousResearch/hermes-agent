"""Seeded Desktop branches keep generated titles out of memory identity."""

import subprocess

import pytest

from hermes_state import SessionDB
from plugins.memory.honcho.client import HonchoClientConfig


@pytest.mark.parametrize("strategy", ["per-repo", "per-directory", "global"])
def test_seeded_desktop_branch_title_preserves_memory_strategy(monkeypatch, tmp_path, strategy):
    monkeypatch.setattr("hermes_cli.banner.prefetch_update_check", lambda: None)
    from tui_gateway import server

    project = tmp_path / "example-project"
    project.mkdir()
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    db = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_load_cfg", lambda: {})
    monkeypatch.setattr(server, "_profile_home", lambda *a: None)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.setattr(server, "_enable_gateway_prompts", lambda: None)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda *a: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda *a: None)
    monkeypatch.setattr(server, "_project_info_for_cwd", lambda *a: None)
    history = [{"role": "user", "content": "Synthetic parent input"},
               {"role": "assistant", "content": "Synthetic parent response"}]
    try:
        db.create_session("parent", source="desktop", cwd=str(project))
        db.set_auto_title("parent", "Generated parent title", source="llm")
        response = server._methods["session.create"]("create", {
            "source": "desktop", "cwd": str(project),
            "parent_session_id": "parent", "messages": history,
        })
        assert "error" not in response, response
        child = response["result"]["stored_session_id"]
        cfg = HonchoClientConfig(session_strategy=strategy, workspace_id="shared-memory")
        expected = cfg.resolve_session_name(cwd=str(project), session_id=child)
        assert cfg.resolve_session_name(
            cwd=str(project), session_id=child,
            session_title=db.get_session_title(child),
            session_title_source=db.get_session_title_source(child),
        ) == expected
        assert db.get_session_title_source(child) == SessionDB.TITLE_SOURCE_DERIVED
        assert db.message_count(child) == len(history)
        # A subsequent explicit rename must retain user authority.
        db.set_session_title(child, "User chosen title")
        assert db.get_session_title_source(child) == SessionDB.TITLE_SOURCE_USER
        assert cfg.resolve_session_name(
            cwd=str(project), session_id=child,
            session_title=db.get_session_title(child),
            session_title_source=db.get_session_title_source(child),
        ) == cfg.resolve_session_name(cwd=str(project), session_title="User chosen title")
    finally:
        db.close()

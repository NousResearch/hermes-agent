"""Seeded Desktop branches keep generated titles out of memory identity."""

import subprocess

import pytest

from persistence.conversation.messages import TranscriptRead, TranscriptView
from persistence.conversation.sessions import TitleSource
from persistence_assembly import open_persistence, profile_scope
from tests.fixtures.storage_seed import seed_session


def test_seeded_desktop_branch_title_is_derived_until_the_user_renames(monkeypatch, tmp_path):
    monkeypatch.setattr("hermes_cli.banner.prefetch_update_check", lambda: None)
    from tui_gateway import server

    project = tmp_path / "example-project"
    project.mkdir()
    subprocess.run(["git", "init", "-q", str(project)], check=True)
    db = open_persistence(profile_scope(home=tmp_path))
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_load_cfg", dict)
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
        seed_session(db, "parent", "desktop", cwd=str(project))
        db.sessions.set_auto_title("parent", "Generated parent title", source=TitleSource.LLM)
        response = server._methods["session.create"]("create", {
            "source": "desktop", "cwd": str(project),
            "parent_session_id": "parent", "messages": history,
        })
        assert "error" not in response, response
        child = response["result"]["stored_session_id"]
        # A DERIVED title is what memory providers skip when naming sessions (the provenance
        # contract on TitleSource); a generated parent title must not arrive as USER/LLM.
        assert db.sessions.get_title_source(child) == TitleSource.DERIVED
        assert len(db.messages.read_transcript(child, TranscriptRead(view=TranscriptView.ALL))) == len(history)
        # A subsequent explicit rename must retain user authority.
        db.sessions.set_title(child, "User chosen title")
        assert db.sessions.get_title_source(child) == TitleSource.USER
    finally:
        db.close()

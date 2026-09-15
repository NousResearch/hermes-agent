from __future__ import annotations

import contextlib

from hermes_state import SessionDB
from tui_gateway import server


def test_explicit_project_switch_persists_identity_and_invalidates_prompt(monkeypatch, tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "AGENTS.md").write_text("SWITCHED-PROJECT-RULE\n", encoding="utf-8")
    db = SessionDB(db_path=tmp_path / "state.db")
    session_key = "stored-session"
    runtime_sid = "runtime-session"
    db.create_session(session_key, source="desktop", cwd=str(tmp_path))
    agent = type("Agent", (), {
        "session_id": session_key,
        "_cached_system_prompt": "PINNED-SYSTEM",
        "_cached_system_prompt_static": "PINNED-STATIC",
        "_memory_store": None,
    })()
    session = {
        "session_key": session_key,
        "agent": agent,
        "cwd": str(tmp_path),
        "explicit_cwd": False,
        "source": "desktop",
    }
    emitted = []
    git_claims = []
    monkeypatch.setattr(server, "_sessions", {runtime_sid: session})
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_persist_session_git_meta", lambda *args: git_claims.append(args))
    monkeypatch.setattr(server, "_session_db", lambda _session: contextlib.nullcontext(db))
    monkeypatch.setattr(server, "_session_info", lambda _agent, _session: {"cwd": _session["cwd"]})
    monkeypatch.setattr(server, "_emit", lambda *args: emitted.append(args))

    try:
        result = server._apply_project_workspace(
            session_key, "project-1", str(project_root), "Project One",
        )

        row = db.get_session(session_key)
        assert session["cwd"] == str(project_root.resolve())
        assert session["explicit_cwd"] is True
        assert row["project_id"] == "project-1"
        assert row["project_root"] == str(project_root.resolve())
        assert row["project_affinity_generation"] == 1
        assert row["project_context_hash"] == result["context_hash"]
        assert result["project_id"] == "project-1"
        assert result["status"] == "bound"
        assert "context" not in result
        assert agent._cached_system_prompt is None
        assert git_claims == [(session, str(project_root.resolve()), 1)]
        assert emitted[-1][0:2] == ("session.info", runtime_sid)
    finally:
        db.close()

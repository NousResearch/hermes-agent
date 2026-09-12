"""Gateway wiring for generation-scoped Git metadata publication."""

from __future__ import annotations

import tui_gateway.server as server
from hermes_state import SessionDB


class _ImmediateThread:
    def __init__(self, *, target, **_kwargs):
        self._target = target

    def start(self):
        self._target()


def test_cwd_claim_precedes_probe_and_generation_reaches_publish(monkeypatch):
    events = []

    class DB:
        def update_session_cwd(self, session_id, cwd):
            events.append(("claim", session_id, cwd))
            return 17

        def publish_session_git_metadata(
            self, session_id, cwd, generation, branch, root
        ):
            events.append(
                ("publish", session_id, cwd, generation, branch, root)
            )
            return True

    monkeypatch.setattr(server, "_get_db", lambda: DB())
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(
        server.git_probe,
        "branch",
        lambda cwd: events.append(("probe", cwd)) or "feature",
    )
    monkeypatch.setattr(server.git_probe, "common_repo_root", lambda _cwd: "/repo")

    generation = server._persist_session_cwd_and_schedule_git_meta(
        {"session_key": "session"}, "/repo/worktree"
    )

    assert generation == 17
    assert events == [
        ("claim", "session", "/repo/worktree"),
        ("probe", "/repo/worktree"),
        (
            "publish",
            "session",
            "/repo/worktree",
            17,
            "feature",
            "/repo",
        ),
    ]


def test_missing_db_claim_never_starts_git_probe(monkeypatch):
    probed = []
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(
        server.git_probe, "branch", lambda cwd: probed.append(cwd)
    )

    generation = server._persist_session_cwd_and_schedule_git_meta(
        {"session_key": "session"}, "/repo"
    )

    assert generation is None
    assert probed == []


def test_lazy_desktop_row_with_explicit_cwd_is_enriched_on_hydration(monkeypatch, tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    repo = tmp_path / "repo"
    repo.mkdir()
    session = {
        "session_key": "desktop-session",
        "source": "desktop",
        "cwd": str(repo),
        "explicit_cwd": True,
    }
    sid = "live-desktop-session"

    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server.git_probe, "branch", lambda _cwd: "feature/session-metadata")
    monkeypatch.setattr(server.git_probe, "common_repo_root", lambda _cwd: str(repo))
    server._sessions[sid] = session
    try:
        assert server._ensure_session_db_row(session) is True
        assert db.get_session("desktop-session")["git_branch"] is None

        server._hydrate_session_cwd(sid, "desktop-session", db, None)

        row = db.get_session("desktop-session")
        assert row["cwd"] == str(repo)
        assert row["git_branch"] == "feature/session-metadata"
        assert row["git_repo_root"] == str(repo)
    finally:
        server._sessions.pop(sid, None)
        db.close()

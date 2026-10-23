"""A Desktop session's lazily created row must get git_branch/git_repo_root (#108784).

``session.create`` runs ``_hydrate_session_cwd`` before any row exists, so its cwd claim is a no-op; the row lands
later on the first submit via ``_ensure_session_db_row``. Without a probe there, desktop rows kept NULL git metadata
and their sidebar lane fell back to a fake ``main`` label (#108694).
"""

from __future__ import annotations

import subprocess

from hermes_state import SessionDB
from tui_gateway import server


class _ImmediateThread:
    def __init__(self, *, target, **_kwargs):
        self._target = target

    def start(self):
        self._target()


def _master_repo(path):
    subprocess.run(["git", "init", "-q", "-b", "master", str(path)], check=True)
    return path


def _desktop_session(monkeypatch, tmp_path, cwd):
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.setattr(server, "_start_agent_build", lambda *a, **k: None)
    resp = server.handle_request({
        "id": "1", "method": "session.create", "params": {"cols": 80, "source": "desktop", "cwd": str(cwd)},
    })
    sid = resp["result"]["session_id"]
    # Patched only after create: session.create starts real Timers (Thread subclasses).
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    return db, sid, server._sessions[sid], resp["result"]["stored_session_id"]


def test_first_submit_row_records_git_branch_and_root(monkeypatch, tmp_path):
    repo = _master_repo(tmp_path / "repo")
    db, sid, session, key = _desktop_session(monkeypatch, tmp_path, repo)
    try:
        assert db.get_session(key) is None  # lazy: no row until the first submit
        assert server._ensure_session_db_row(session) is True
    finally:
        server._sessions.pop(sid, None)

    row = db.get_session(key)
    assert row["cwd"] == str(repo)
    assert row["git_branch"] == "master"
    assert row["git_repo_root"]


def test_row_git_probe_runs_once_per_session(monkeypatch, tmp_path):
    repo = _master_repo(tmp_path / "repo")
    db, sid, session, key = _desktop_session(monkeypatch, tmp_path, repo)
    probes = []
    real_branch = server.git_probe.branch
    monkeypatch.setattr(server.git_probe, "branch", lambda cwd: probes.append(cwd) or real_branch(cwd))
    try:
        for _ in range(3):  # prompt.submit re-calls the upsert every turn
            assert server._ensure_session_db_row(session) is True
    finally:
        server._sessions.pop(sid, None)

    assert probes == [str(repo)]
    assert db.get_session(key)["git_branch"] == "master"


def test_row_without_cwd_never_probes(monkeypatch, tmp_path):
    probes = []
    db, sid, session, key = _desktop_session(monkeypatch, tmp_path, tmp_path)
    monkeypatch.setattr(server.git_probe, "branch", lambda cwd: probes.append(cwd))
    session.update(cwd="", explicit_cwd=False)
    try:
        assert server._ensure_session_db_row(session) is True
    finally:
        server._sessions.pop(sid, None)

    assert not db.get_session(key)["cwd"]
    assert probes == []

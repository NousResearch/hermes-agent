"""The desktop session lifecycle must persist git_branch for the PR join.

Regression: a desktop chat created in a worktree lane wrote its row with cwd
but never a git_branch (146 of 156 rows in a real profile db). The sidebar's
PR badge joins on ``git_repo_root + git_branch`` (sessionPrKey), so the badge
never rendered even with Show PR on, while the composer statusline — which
probes git live — showed the PR fine.

Two arms in ``server.py`` failed to schedule the async git enrichment:

* create: ``_ensure_session_db_row`` inserted the row AFTER the only
  enrichment call site, so the generation claim found no row and gave up.
* resume: ``_init_session`` adopted an existing row's cwd and skipped
  enrichment entirely, so a branchless row stayed branchless forever.

These tests run the REAL SessionDB against a temp state.db — only the git
subprocess probes and the enrichment thread are stubbed.
"""

from __future__ import annotations

import types

import pytest

import tui_gateway.server as server
from hermes_state import SessionDB


class _ImmediateThread:
    """Run the git-meta enrichment inline; inert stand-in for other threads.

    ``_init_session`` also spawns the notification poller through
    ``threading.Thread`` — that one must NOT run inline (it loops forever)
    and the poller registry calls ``is_alive()`` on it.
    """

    def __init__(self, *, target=None, name=None, **_kwargs):
        self._target = target
        self._name = name or ""

    def start(self):
        if self._name == "git-meta" and self._target is not None:
            self._target()

    def is_alive(self):
        return False


@pytest.fixture
def git_env(tmp_path, monkeypatch):
    """Real db + deterministic git probes + synchronous enrichment thread."""
    db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server.threading, "Thread", _ImmediateThread)
    monkeypatch.setattr(server, "_git_branch_for_cwd", lambda _cwd: "ethie/title-gen")
    monkeypatch.setattr(
        server, "_git_common_repo_root_for_cwd", lambda _cwd: "/repo"
    )
    yield db
    db.close()


def _desktop_session(key: str, cwd: str) -> dict:
    return {
        "session_key": key,
        "cwd": cwd,
        "explicit_cwd": True,
        "profile_home": None,
        "model_override": None,
        "parent_session_id": None,
    }


def test_create_path_persists_git_branch(git_env, monkeypatch, tmp_path):
    """A fresh desktop row must come out of first-submit with a branch."""
    monkeypatch.setattr(server, "_resolve_model", lambda: "test-model")
    monkeypatch.setattr(server, "_session_source", lambda _s: "desktop")

    workdir = tmp_path / "wt"
    workdir.mkdir()
    session = _desktop_session("sess-create", str(workdir))

    server._ensure_session_db_row(session)

    row = git_env.get_session("sess-create")
    assert row is not None
    assert row["cwd"] == str(workdir)
    # The invariant the sidebar join needs: branch and root both present.
    assert row["git_branch"] == "ethie/title-gen"
    assert row["git_repo_root"] == "/repo"


def test_resume_backfills_branchless_row(git_env, monkeypatch, tmp_path):
    """Adopting a persisted cwd must heal a row with no git metadata."""
    # A real directory, so _heal_dead_cwd stays a no-op and the enrichment we
    # observe is the adopt arm's own, not the heal path's.
    workdir = tmp_path / "wt-resume"
    workdir.mkdir()
    # A row the old create path left behind: cwd yes, branch no.
    git_env.create_session("sess-resume", source="desktop", cwd=str(workdir))
    before = git_env.get_session("sess-resume")
    assert before["git_branch"] is None

    monkeypatch.setattr(server, "_SlashWorker", None, raising=False)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *a: None)

    import tools.approval as approval

    monkeypatch.setattr(approval, "register_gateway_notify", lambda k, cb: None)
    monkeypatch.setattr(approval, "load_permanent_allowlist", lambda: None)

    sid = "sid-resume"
    try:
        server._init_session(
            sid,
            "sess-resume",
            types.SimpleNamespace(model="x"),
            history=[],
            cols=80,
        )
        after = git_env.get_session("sess-resume")
        assert after["git_branch"] == "ethie/title-gen"
        assert after["git_repo_root"] == "/repo"
    finally:
        server._sessions.pop(sid, None)


def test_resume_leaves_enriched_row_alone(git_env, monkeypatch, tmp_path):
    """A row that already has its metadata must not be re-enriched on adopt.

    ``_init_session`` still probes git for the LIVE session.info payload (the
    composer statusline) — that is fine and not what this guards. The guard is
    on the persistence scheduler: an enriched row must not claim a new
    generation, and its stored branch must survive the resume.
    """
    workdir = tmp_path / "wt-full"
    workdir.mkdir()
    git_env.create_session("sess-full", source="desktop", cwd=str(workdir))
    gen = git_env.update_session_cwd("sess-full", str(workdir))
    git_env.publish_session_git_metadata(
        "sess-full", str(workdir), gen, "already-there", "/repo"
    )

    scheduled = []
    real = server._persist_session_cwd_and_schedule_git_meta
    monkeypatch.setattr(
        server,
        "_persist_session_cwd_and_schedule_git_meta",
        lambda session, cwd, **kw: scheduled.append(cwd) or real(session, cwd, **kw),
    )
    monkeypatch.setattr(server, "_SlashWorker", None, raising=False)
    monkeypatch.setattr(server, "_wire_callbacks", lambda _sid: None)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_notify_session_boundary", lambda *a: None)

    import tools.approval as approval

    monkeypatch.setattr(approval, "register_gateway_notify", lambda k, cb: None)
    monkeypatch.setattr(approval, "load_permanent_allowlist", lambda: None)

    sid = "sid-full"
    try:
        server._init_session(
            sid,
            "sess-full",
            types.SimpleNamespace(model="x"),
            history=[],
            cols=80,
        )
        assert scheduled == []
        after = git_env.get_session("sess-full")
        assert after["git_branch"] == "already-there"
    finally:
        server._sessions.pop(sid, None)

"""An agent that works inside a linked worktree it created mid-chat is shown there.

The session's own workspace is deliberately NOT re-homed by per-command
``workdir`` activity (that is transient by contract), so a desktop chat that
started in a non-git launch directory kept painting nothing while every tool
call ran in ``<repo>/.worktrees/<slug>``. The gateway now tracks that linked
worktree as a display-only ``agent_worktree`` on the session: adopted from
terminal activity, persisted with the row, restored on resume, and cleared
when the tree is gone or the session's workspace becomes that tree.
"""

from __future__ import annotations

import json
import subprocess

import pytest

import tools.terminal_tool as terminal_tool
import tui_gateway.server as server
from hermes_state import SessionDB


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def repo_with_worktree(tmp_path):
    repo = tmp_path / "proj"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("hi\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    worktree = repo / ".worktrees" / "task-a"
    _git(repo, "worktree", "add", "-b", "task/a", str(worktree))
    from tui_gateway import git_probe
    git_probe.invalidate()
    yield repo, worktree
    git_probe.invalidate()


@pytest.fixture
def session(tmp_path):
    """A desktop chat whose cwd is a plain, non-git launch directory (the reported shape)."""
    launch = tmp_path / "projects"
    launch.mkdir()
    key = "sess-agent-worktree"
    terminal_tool.clear_session_cwd(key)
    yield {"session_key": key, "cwd": str(launch), "source": "desktop", "explicit_cwd": False}
    terminal_tool.clear_session_cwd(key)


@pytest.fixture(autouse=True)
def _no_db(monkeypatch):
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_persist_session_git_meta", lambda *_a: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _s: None)
    monkeypatch.setattr(server, "_is_local_terminal_backend", lambda: True)


def _terminal_call(workdir=None, result=None):
    args = {"command": "pwd"}
    if workdir:
        args["workdir"] = workdir
    return args, json.dumps(result or {"output": "", "exit_code": 0})


# ── adoption ────────────────────────────────────────────────────────────────


def test_workdir_activity_in_a_linked_worktree_is_adopted(session, repo_with_worktree):
    """The reported bug: 405 `workdir=` calls, zero `cd`, nothing on screen."""
    repo, worktree = repo_with_worktree
    args, result = _terminal_call(workdir=str(worktree))

    assert server._observe_terminal_activity(session, args, result) is True
    adopted = session["agent_worktree"]
    assert adopted["cwd"] == str(worktree)
    assert adopted["branch"] == "task/a"
    assert adopted["repoRoot"] == str(repo).replace("\\", "/")
    assert adopted["projectName"] == "proj"
    # Display-only: the session's own workspace is untouched.
    assert session["cwd"] != str(worktree)
    assert session.get("explicit_cwd") is False


def test_a_subdirectory_of_the_worktree_resolves_to_its_root(session, repo_with_worktree):
    _, worktree = repo_with_worktree
    sub = worktree / "apps" / "desktop"
    sub.mkdir(parents=True)
    args, result = _terminal_call(workdir=str(sub))

    assert server._observe_terminal_activity(session, args, result) is True
    assert session["agent_worktree"]["cwd"] == str(worktree)


def test_cd_echo_is_adopted_too(session, repo_with_worktree):
    """A bare `cd` reports the landing dir in the result; that counts as activity."""
    _, worktree = repo_with_worktree
    args, result = _terminal_call(result={"output": "", "exit_code": 0, "cwd": str(worktree)})

    assert server._observe_terminal_activity(session, args, result) is True
    assert session["agent_worktree"]["cwd"] == str(worktree)


def test_the_main_checkout_is_not_an_agent_worktree(session, repo_with_worktree):
    """Visiting the primary checkout is browsing (the existing settle rule already covers main→worktree moves)."""
    repo, _ = repo_with_worktree
    args, result = _terminal_call(workdir=str(repo))

    assert server._observe_terminal_activity(session, args, result) is False
    assert "agent_worktree" not in session


def test_browsing_outside_git_does_not_adopt_or_clear(session, repo_with_worktree, tmp_path):
    _, worktree = repo_with_worktree
    server._observe_terminal_activity(session, *_terminal_call(workdir=str(worktree)))
    scratch = tmp_path / "scratch"
    scratch.mkdir()

    assert server._observe_terminal_activity(session, *_terminal_call(workdir=str(scratch))) is False
    # Sticky: a `cat /tmp/log` between edits must not drop the badge.
    assert session["agent_worktree"]["cwd"] == str(worktree)


def test_settling_in_another_linked_worktree_switches(session, repo_with_worktree):
    repo, worktree = repo_with_worktree
    other = repo / ".worktrees" / "task-b"
    _git(repo, "worktree", "add", "-b", "task/b", str(other))
    from tui_gateway import git_probe
    git_probe.invalidate()
    server._observe_terminal_activity(session, *_terminal_call(workdir=str(worktree)))

    assert server._observe_terminal_activity(session, *_terminal_call(workdir=str(other))) is True
    assert session["agent_worktree"]["branch"] == "task/b"


def test_repeated_activity_in_the_same_worktree_is_not_a_change(session, repo_with_worktree):
    _, worktree = repo_with_worktree
    args, result = _terminal_call(workdir=str(worktree))
    assert server._observe_terminal_activity(session, args, result) is True
    assert server._observe_terminal_activity(session, args, result) is False


def test_the_sessions_own_worktree_is_not_an_agent_worktree(session, repo_with_worktree):
    """A chat explicitly started IN a worktree (Projects picker) already shows it; no duplicate badge."""
    _, worktree = repo_with_worktree
    session.update(cwd=str(worktree), explicit_cwd=True)

    assert server._observe_terminal_activity(session, *_terminal_call(workdir=str(worktree))) is False
    assert "agent_worktree" not in session


def test_a_coding_workspace_binding_wins(session, repo_with_worktree):
    """The durable Projects binding is the workspace identity; the badge never competes with it."""
    _, worktree = repo_with_worktree
    session["coding_workspace"] = {"cwd": str(worktree), "requestId": "r"}

    assert server._observe_terminal_activity(session, *_terminal_call(workdir=str(worktree))) is False
    assert "agent_worktree" not in session


def test_remote_backends_do_not_adopt(session, repo_with_worktree, monkeypatch):
    _, worktree = repo_with_worktree
    monkeypatch.setattr(server, "_is_local_terminal_backend", lambda: False)

    assert server._observe_terminal_activity(session, *_terminal_call(workdir=str(worktree))) is False


def test_non_terminal_tools_are_ignored(session, repo_with_worktree):
    _, worktree = repo_with_worktree
    args = {"path": str(worktree / "README.md")}

    assert server._observe_tool_activity(session, "read_file", args, "{}") is False
    assert "agent_worktree" not in session


# ── settle / validation ─────────────────────────────────────────────────────


def test_settle_clears_a_deleted_worktree(session, repo_with_worktree):
    repo, worktree = repo_with_worktree
    server._observe_terminal_activity(session, *_terminal_call(workdir=str(worktree)))
    _git(repo, "worktree", "remove", "--force", str(worktree))

    assert server._revalidate_agent_worktree(session) is True
    assert session.get("agent_worktree") is None


def test_settle_clears_when_the_session_is_rehomed_onto_it(session, repo_with_worktree):
    """Once the session's own workspace IS that tree (settle-follow / explicit move), the badge is redundant."""
    _, worktree = repo_with_worktree
    server._observe_terminal_activity(session, *_terminal_call(workdir=str(worktree)))
    session.update(cwd=str(worktree), explicit_cwd=True)

    assert server._revalidate_agent_worktree(session) is True
    assert session.get("agent_worktree") is None


def test_settle_refreshes_the_branch(session, repo_with_worktree):
    repo, worktree = repo_with_worktree
    server._observe_terminal_activity(session, *_terminal_call(workdir=str(worktree)))
    _git(worktree, "checkout", "-b", "task/a-renamed")

    assert server._revalidate_agent_worktree(session) is True
    assert session["agent_worktree"]["branch"] == "task/a-renamed"


def test_settled_session_info_carries_the_agent_worktree(session, repo_with_worktree, monkeypatch):
    _, worktree = repo_with_worktree
    emitted: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(server, "_emit", lambda ev, sid, payload=None: emitted.append((ev, sid, payload or {})))
    server._observe_terminal_activity(session, *_terminal_call(workdir=str(worktree)))

    server._emit_settled_session_info("sid-1", session, agent=None)

    payload = emitted[-1][2]
    assert payload["agent_worktree"]["cwd"] == str(worktree)
    assert payload["agent_worktree"]["branch"] == "task/a"
    # The session's own cwd is still the launch dir: the badge is additive.
    assert payload["cwd"] == session["cwd"]


def test_tool_complete_emits_session_info_on_adoption(session, repo_with_worktree, monkeypatch):
    """Mid-turn: the desktop learns about the worktree when the agent starts working there, not at turn end."""
    _, worktree = repo_with_worktree
    emitted: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(server, "_emit", lambda ev, sid, payload=None: emitted.append((ev, sid, payload or {})))
    monkeypatch.setattr(server, "_tool_progress_enabled", lambda sid: False)
    monkeypatch.setattr(server, "_session_verbose", lambda sid: False)
    monkeypatch.setitem(server._sessions, "sid-1", session)
    try:
        args, result = _terminal_call(workdir=str(worktree))
        server._on_tool_complete("sid-1", "tc-1", "terminal", args, result)
    finally:
        server._sessions.pop("sid-1", None)

    infos = [p for ev, sid, p in emitted if ev == "session.info" and sid == "sid-1"]
    assert infos and infos[-1]["agent_worktree"]["cwd"] == str(worktree)


# ── persistence ─────────────────────────────────────────────────────────────


def test_agent_worktree_is_persisted_and_restored_on_resume(tmp_path, monkeypatch, repo_with_worktree):
    repo, worktree = repo_with_worktree
    db = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    launch = tmp_path / "projects"
    launch.mkdir()
    created = server._methods["session.create"](1, {"source": "desktop", "cwd": str(launch)})["result"]
    live = server._sessions[created["session_id"]]
    server._ensure_session_db_row(live)
    try:
        assert server._observe_terminal_activity(live, *_terminal_call(workdir=str(worktree))) is True
        stored = json.loads(db.get_session(created["stored_session_id"])["model_config"])
        assert stored["agent_worktree"]["cwd"] == str(worktree)
        # The session's own row keeps the launch-dir rule: no cwd, no git identity.
        assert db.get_session(created["stored_session_id"])["git_repo_root"] is None
    finally:
        server._sessions.pop(created["session_id"], None)

    resumed = server._methods["session.resume"](2, {"session_id": created["stored_session_id"], "source": "desktop", "lazy": True})["result"]
    try:
        assert resumed["info"]["agent_worktree"]["cwd"] == str(worktree)
        record = server._sessions[resumed["session_id"]]
        assert server._session_info(None, record)["agent_worktree"]["branch"] == "task/a"
    finally:
        server._sessions.pop(resumed["session_id"], None)
        db.close()


def test_a_stale_persisted_worktree_is_dropped_on_resume(tmp_path, monkeypatch, repo_with_worktree):
    repo, worktree = repo_with_worktree
    db = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    launch = tmp_path / "projects"
    launch.mkdir()
    created = server._methods["session.create"](1, {"source": "desktop", "cwd": str(launch)})["result"]
    live = server._sessions[created["session_id"]]
    server._ensure_session_db_row(live)
    server._observe_terminal_activity(live, *_terminal_call(workdir=str(worktree)))
    server._sessions.pop(created["session_id"], None)
    _git(repo, "worktree", "remove", "--force", str(worktree))

    resumed = server._methods["session.resume"](2, {"session_id": created["stored_session_id"], "source": "desktop", "lazy": True})["result"]
    try:
        assert resumed["info"].get("agent_worktree") is None
    finally:
        server._sessions.pop(resumed["session_id"], None)
        db.close()

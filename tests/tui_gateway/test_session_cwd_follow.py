"""A chat's workspace (and so its branch label) is where the chat was put, not where its shell last ``cd``-ed.

A turn can end in a sibling worktree for unrelated reasons, such as a commit on main or a worktree set up for another
task. These tests use real git worktrees: a ``cd`` never relabels the chat, a ``desktop_project`` switch with a path
does, and the end-of-turn ``session.info`` reports the workspace's branch.
"""

from __future__ import annotations

import json
import subprocess

import pytest

from tools import project_tools, terminal_tool
from tools.registry import registry
from tui_gateway import server


def _git(cwd, *args):
    subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True)


@pytest.fixture
def trees(tmp_path):
    """A real repo on ``main`` plus linked worktrees on ``feature`` (this chat's work) and ``handoff`` (made for
    another chat)."""
    repo = tmp_path / "proj"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    (repo / "README.md").write_text("hi\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "init")
    feature, handoff = tmp_path / "proj-feature", tmp_path / "proj-handoff"
    _git(repo, "worktree", "add", "-b", "feature", str(feature))
    _git(repo, "worktree", "add", "-b", "handoff", str(handoff))

    from tui_gateway import git_probe

    git_probe.invalidate()
    yield repo, feature, handoff
    git_probe.invalidate()


@pytest.fixture
def chat(trees, monkeypatch):
    """A live desktop chat whose workspace is the primary checkout, wired through the real gateway callback."""
    repo, _, _ = trees
    emitted: list[tuple[str, str, dict]] = []
    monkeypatch.setattr(server, "_sessions", {})
    monkeypatch.setattr(server, "_get_db", lambda: None)
    monkeypatch.setattr(server, "_persist_session_cwd_and_schedule_git_meta", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_effective_terminal_backend", lambda: "local")
    monkeypatch.setattr(server, "_emit", lambda ev, sid, payload=None: emitted.append((ev, sid, payload or {})))
    monkeypatch.setattr(terminal_tool, "_task_env_overrides", {})
    key = "chat-binding"
    terminal_tool.clear_session_cwd(key)
    session = {"session_key": key, "cwd": str(repo), "source": "desktop", "explicit_cwd": True}
    server._sessions["sid-1"] = session
    server._register_session_cwd(session)
    project_tools.set_project_workspace_callback(server._apply_project_workspace)
    try:
        yield session, emitted
    finally:
        project_tools.set_project_workspace_callback(None)
        terminal_tool.clear_session_cwd(key)


def _turn_ends(session, emitted) -> dict:
    """End a turn; the ``session.info`` the desktop's branch chip follows."""
    emitted.clear()
    server._emit_settled_session_info("sid-1", session, agent=None)
    infos = [payload for event, sid, payload in emitted if (event, sid) == ("session.info", "sid-1")]
    assert len(infos) == 1
    return infos[0]


def test_whichever_tree_a_turn_ends_in_the_chat_keeps_its_workspace(chat, trees):
    """The reported bug: the chip showed the tree the last command ran in, not the chat's."""
    session, emitted = chat
    repo, feature, handoff = trees
    for landed in (feature, handoff, repo):
        terminal_tool.record_session_cwd(session["session_key"], str(landed))
        info = _turn_ends(session, emitted)
        assert (info["cwd"], info["branch"]) == (str(repo), "main")
    assert session["cwd"] == str(repo)


def test_a_declared_move_relabels_the_chat_and_sticks(chat, trees):
    session, emitted = chat
    repo, feature, _ = trees

    moved = json.loads(registry.dispatch(
        "desktop_project", {"action": "switch", "path": str(feature)}, task_id=session["session_key"]))

    assert moved == {"success": True, "path": str(feature)}
    assert session["cwd"] == str(feature)
    # The move re-seats the shell in the new workspace...
    assert terminal_tool.get_session_cwd(session["session_key"]) == str(feature)
    assert _turn_ends(session, emitted)["branch"] == "feature"
    # ...and a later visit to the primary checkout (a coordination commit) does not undo it.
    terminal_tool.record_session_cwd(session["session_key"], str(repo))
    info = _turn_ends(session, emitted)
    assert (info["cwd"], info["branch"]) == (str(feature), "feature")


def test_a_move_to_a_missing_directory_is_refused(chat, trees, tmp_path):
    session, _ = chat
    repo, _, _ = trees

    refused = json.loads(registry.dispatch(
        "desktop_project", {"action": "switch", "path": str(tmp_path / "gone")}, task_id=session["session_key"]))

    assert refused["success"] is False
    assert session["cwd"] == str(repo)

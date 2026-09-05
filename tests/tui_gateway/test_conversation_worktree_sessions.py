"""Conversation-worktree ownership at desktop/TUI session boundaries."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import tui_gateway.server as server


@dataclass(frozen=True)
class _Binding:
    root_session_id: str
    path: Path
    branch: str
    base_commit: str
    repo_common_dir: Path


@pytest.fixture(autouse=True)
def _clean_sessions(monkeypatch):
    with server._sessions_lock:
        server._sessions.clear()
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_resolve_model", lambda: "test/model")
    monkeypatch.setattr(server, "_git_branch_for_cwd", lambda cwd: "stable")
    monkeypatch.setattr(server, "_project_info_for_cwd", lambda cwd: {})
    monkeypatch.setattr(server, "_load_show_reasoning", lambda: False)
    monkeypatch.setattr(server, "_load_tool_progress_mode", lambda: "compact")
    monkeypatch.setattr(
        server,
        "_acquire_conversation_root_lease",
        lambda _binding, *, surface: MagicMock(surface=surface),
    )
    yield
    with server._sessions_lock:
        server._sessions.clear()


def _binding(root: str) -> _Binding:
    return _Binding(
        root_session_id=root,
        path=Path("/repo/.worktrees") / root,
        branch=f"hermes/session/{root}",
        base_commit="a" * 40,
        repo_common_dir=Path("/repo/.git"),
    )


def test_session_create_binds_before_deferred_agent_build(monkeypatch):
    calls: list[tuple[str, str]] = []
    scheduled: list[tuple[str, str]] = []

    def bind(root_session_id: str, *, profile_home=None, db=None):
        calls.append((root_session_id, "interactive"))
        return _binding(root_session_id)

    monkeypatch.setattr(server, "_bind_new_interactive_conversation_worktree", bind)
    monkeypatch.setattr(
        server,
        "_schedule_agent_build",
        lambda sid: scheduled.append((sid, server._sessions[sid]["cwd"])),
    )

    response = server._methods["session.create"](
        "create", {"source": "desktop", "cwd": "/stable"}
    )

    assert "error" not in response
    result = response["result"]
    root = result["stored_session_id"]
    assert calls == [(root, "interactive")]
    assert result["info"]["cwd"] == f"/repo/.worktrees/{root}"
    assert scheduled == [(result["session_id"], f"/repo/.worktrees/{root}")]
    record = server._sessions[result["session_id"]]
    assert record["cwd"] == f"/repo/.worktrees/{root}"
    assert record["explicit_cwd"] is True
    assert record["conversation_worktree"]["root_session_id"] == root


def test_desktop_root_lease_lives_until_session_finalize(monkeypatch):
    lease = MagicMock()
    monkeypatch.setattr(
        server,
        "_acquire_conversation_root_lease",
        lambda _binding, *, surface: lease,
    )
    monkeypatch.setattr(
        server,
        "_bind_new_interactive_conversation_worktree",
        lambda root_session_id, **_kwargs: _binding(root_session_id),
    )
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)

    response = server._methods["session.create"](
        "create-lease", {"source": "desktop", "cwd": "/stable"}
    )
    sid = response["result"]["session_id"]
    session = server._sessions[sid]

    assert session["conversation_root_lease"] is lease
    lease.release.assert_not_called()

    server._finalize_session(session)

    lease.release.assert_called_once_with()


def test_session_create_fails_closed_without_scheduling_agent(monkeypatch):
    scheduled: list[str] = []

    def fail(_root_session_id: str, *, profile_home=None, db=None):
        raise RuntimeError("bootstrap failed")

    monkeypatch.setattr(server, "_bind_new_interactive_conversation_worktree", fail)
    monkeypatch.setattr(server, "_schedule_agent_build", lambda sid: scheduled.append(sid))

    response = server._methods["session.create"]("create", {"source": "desktop"})

    assert response["error"]["code"] == 5000
    assert "conversation worktree" in response["error"]["message"]
    assert scheduled == []
    assert server._sessions == {}


def test_resume_resolves_existing_binding_without_creation(monkeypatch):
    root = "root-existing"
    continuation = "compressed-tip"
    resolve_calls: list[str] = []
    create_calls: list[str] = []

    class _LineageDB:
        def get_session(self, session_id):
            return {"parent_session_id": root} if session_id == continuation else {}

    monkeypatch.setattr(
        server,
        "_resolve_existing_conversation_worktree",
        lambda root_session_id, *, profile_home=None, db=None: (
            resolve_calls.append(root_session_id)
            or (_binding(root_session_id) if root_session_id == root else None)
        ),
    )
    monkeypatch.setattr(
        server,
        "_bind_new_interactive_conversation_worktree",
        lambda root_session_id, *, profile_home=None, db=None: create_calls.append(root_session_id),
    )

    assert server._resolve_conversation_worktree_for_resume(
        continuation, profile_home=None, db=_LineageDB()
    ) == _binding(root)
    assert resolve_calls == [continuation, root]
    assert create_calls == []


def test_branch_binds_a_distinct_root_before_agent_construction(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(
        server,
        "_bind_new_interactive_conversation_worktree",
        lambda root_session_id, *, profile_home=None, db=None: calls.append(root_session_id)
        or _binding(root_session_id),
    )

    binding = server._bind_conversation_worktree_for_new_root("branch-root", profile_home=None)

    assert binding.path == Path("/repo/.worktrees/branch-root")
    assert calls == ["branch-root"]

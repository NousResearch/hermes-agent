"""RPC contracts for explicit conversation-worktree cleanup."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import pytest

from agent.conversation_worktree import CleanupResult, CleanupVerdict
import tui_gateway.server as server


@dataclass
class _Record:
    root_session_id: str = "root"
    worktree_path: str = "/repo/worktrees/root"
    branch: str = "hermes/session/root"
    base_commit: str = "a" * 40
    state: str = "ready"


class _DB:
    def __init__(self):
        self.record = _Record()
        self.deleted: list[str] = []

    def get_conversation_worktree(self, root_session_id):
        return self.record if root_session_id == self.record.root_session_id else None

    def get_session(self, session_id):
        if session_id == "tip":
            return {"parent_session_id": "root"}
        return None

    def delete_session(self, target, *, sessions_dir):
        self.deleted.append(target)
        return True


class _Manager:
    def __init__(self, verdict: CleanupVerdict, result: CleanupResult | None = None):
        self.verdict = verdict
        self.result = result
        self.inspect_calls: list[tuple[str, bool]] = []
        self.remove_calls: list[tuple[str, bool]] = []

    def inspect_cleanup(self, root_session_id, *, active_session_bound=False):
        self.inspect_calls.append((root_session_id, active_session_bound))
        return self.verdict

    def remove_after_explicit_request(self, root_session_id, *, active_session_bound=False):
        self.remove_calls.append((root_session_id, active_session_bound))
        return self.result or CleanupResult(
            removed=self.verdict.allowed, verdict=self.verdict
        )


@pytest.fixture(autouse=True)
def clean_sessions():
    with server._sessions_lock:
        server._sessions.clear()
    yield
    with server._sessions_lock:
        server._sessions.clear()


def call(params):
    return server._methods["session.worktree_cleanup"]("cleanup", params)


def install(monkeypatch, manager, db):
    monkeypatch.setattr(server, "_profile_db", lambda _params: nullcontext(db))
    monkeypatch.setattr(
        server,
        "_conversation_worktree_manager",
        lambda *, profile_home=None, db=None: (manager, db, False),
    )


def test_inspect_returns_every_blocking_reason_and_resolves_root(monkeypatch):
    verdict = CleanupVerdict(
        allowed=False,
        reasons=("dirty", "unintegrated", "unpushed"),
    )
    manager = _Manager(verdict)
    db = _DB()
    install(monkeypatch, manager, db)

    response = call({"session_id": "tip", "action": "inspect"})

    assert "error" not in response
    assert response["result"] == {
        "allowed": False,
        "reasons": ["dirty", "unintegrated", "unpushed"],
        "removed": False,
        "root_session_id": "root",
        "path": "/repo/worktrees/root",
        "branch": "hermes/session/root",
        "base_commit": "a" * 40,
        "state": "ready",
    }
    assert manager.inspect_calls == [("root", False)]
    assert manager.remove_calls == []


def test_inspect_reports_live_root_binding_as_active(monkeypatch):
    verdict = CleanupVerdict(allowed=False, reasons=("active",))
    manager = _Manager(verdict)
    db = _DB()
    install(monkeypatch, manager, db)
    with server._sessions_lock:
        server._sessions["runtime"] = {
            "conversation_worktree": {"root_session_id": "root"},
            "session_key": "tip",
        }

    response = call({"session_id": "root", "action": "inspect"})

    assert "error" not in response
    assert response["result"]["reasons"] == ["active"]
    assert manager.inspect_calls == [("root", True)]


def test_remove_requires_explicit_action_and_returns_verified_result(monkeypatch):
    verdict = CleanupVerdict(allowed=True, reasons=())
    manager = _Manager(verdict)
    db = _DB()
    install(monkeypatch, manager, db)

    inspected = call({"session_id": "root", "action": "inspect"})
    removed = call({"session_id": "root", "action": "remove"})

    assert inspected["result"]["removed"] is False
    assert removed["result"]["removed"] is True
    assert manager.inspect_calls == [("root", False)]
    assert manager.remove_calls == [("root", False)]


def test_remove_failure_returns_stable_reason_phase_and_safe_message(monkeypatch):
    verdict = CleanupVerdict(allowed=False, reasons=("remove_failed",))
    manager = _Manager(
        verdict,
        CleanupResult(
            removed=False,
            verdict=verdict,
            failure_phase="remove",
            failure_message="permission denied",
        ),
    )
    db = _DB()
    install(monkeypatch, manager, db)

    response = call({"session_id": "root", "action": "remove"})

    assert response["result"]["removed"] is False
    assert response["result"]["reasons"] == ["remove_failed"]
    assert response["result"]["failure_phase"] == "remove"
    assert response["result"]["failure_message"] == "permission denied"


def test_unknown_cleanup_action_fails_without_inspection_or_removal(monkeypatch):
    manager = _Manager(CleanupVerdict(allowed=True, reasons=()))
    db = _DB()
    install(monkeypatch, manager, db)

    response = call({"session_id": "root", "action": "archive"})

    assert response["error"]["code"] == 4006
    assert manager.inspect_calls == []
    assert manager.remove_calls == []


def test_close_and_delete_never_imply_worktree_cleanup(monkeypatch, tmp_path):
    db = _DB()

    def cleanup_must_not_be_built(*args, **kwargs):
        raise AssertionError("ordinary session lifecycle called worktree cleanup")

    monkeypatch.setattr(server, "_conversation_worktree_manager", cleanup_must_not_be_built)
    monkeypatch.setattr(server, "_profile_db", lambda _params: nullcontext(db))
    monkeypatch.setattr(server, "get_hermes_home", lambda: tmp_path)

    closed = server._methods["session.close"]("close", {"session_id": "missing"})
    deleted = server._methods["session.delete"]("delete", {"session_id": "root"})

    assert closed["result"] == {"closed": False}
    assert deleted["result"] == {"deleted": "root"}
    assert db.deleted == ["root"]

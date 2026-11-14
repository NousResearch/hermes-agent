"""Live Kanban-comment injection into a running worker.

``tools.kanban_tools.inject_new_comments_from_env`` polls the worker's task
for comments added *after* the run started and folds them into the live turn
through its dedicated Kanban-note channel. A task comment is not a message
from the user and must never use the OUT-OF-BAND steer channel.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
import tools.kanban_tools as kt


class FakeAgent:
    def __init__(self):
        self.notes: list[str] = []

    def kanban_note(self, text: str) -> bool:
        self.notes.append(text)
        return True


class FakeAgentNoNote:
    def __init__(self):
        self.steers: list[str] = []

    def steer(self, text: str) -> bool:
        self.steers.append(text)
        return True


@pytest.fixture
def worker_home(tmp_path, monkeypatch):
    home = tmp_path / "hermes_home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for var in ("HERMES_KANBAN_DB", "HERMES_KANBAN_WORKSPACES_ROOT", "HERMES_KANBAN_HOME", "HERMES_KANBAN_BOARD"):
        monkeypatch.delenv(var, raising=False)
    try:
        import hermes_constants
        hermes_constants._cached_default_hermes_root = None  # type: ignore[attr-defined]
    except Exception:
        pass
    kb._INITIALIZED_PATHS.clear()
    # Reset module-level poll state so tests don't leak into each other.
    kt._comment_watermark.clear()
    kt._comment_poll_last_attempt = 0.0
    return home


def _unthrottle():
    """Bypass the inter-poll rate limit for deterministic tests."""
    kt._comment_poll_last_attempt = 0.0


def test_noop_without_worker_env(worker_home, monkeypatch):
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    agent = FakeAgent()
    assert kt.inject_new_comments_from_env(agent) is False
    assert agent.notes == []


def test_noop_without_kanban_note_method(worker_home, monkeypatch):
    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_missing_note")
    agent = FakeAgentNoNote()

    assert kt.inject_new_comments_from_env(agent) is False
    assert agent.steers == []


def test_seed_then_inject_new_comment(worker_home, monkeypatch):
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="live task")
        kb.add_comment(conn, tid, author="desktop", body="pre-existing note")
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_PROFILE", "worker-bot")
    agent = FakeAgent()

    # First poll seeds the watermark past the existing thread — no injection.
    _unthrottle()
    assert kt.inject_new_comments_from_env(agent) is False
    assert agent.notes == []

    conn = kbc.connect()
    try:
        kb.add_comment(conn, tid, author="desktop", body="actually use the v2 API")
    finally:
        conn.close()

    _unthrottle()
    assert kt.inject_new_comments_from_env(agent) is True
    assert len(agent.notes) == 1
    assert "v2 API" in agent.notes[0]
    assert "not a message from the user" in agent.notes[0]

    # Watermark advanced — a re-poll with no new comments injects nothing.
    _unthrottle()
    assert kt.inject_new_comments_from_env(agent) is False
    assert len(agent.notes) == 1


def test_skips_own_authored_comments(worker_home, monkeypatch):
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="echo guard")
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_PROFILE", "worker-bot")
    agent = FakeAgent()

    _unthrottle()
    kt.inject_new_comments_from_env(agent)  # seed

    conn = kbc.connect()
    try:
        kb.add_comment(conn, tid, author="worker-bot", body="i did a thing")
    finally:
        conn.close()

    _unthrottle()
    assert kt.inject_new_comments_from_env(agent) is False
    assert agent.notes == []


def test_delegated_child_in_worker_process_neither_receives_nor_consumes_notes(worker_home, monkeypatch):
    """A delegate_task child inherits the worker's ``HERMES_KANBAN_TASK``; operator notes
    address the worker, so the child must not be steered by them and must not advance the
    shared watermark (which would make the worker miss them) (#112817)."""
    from agent.delegation_context import delegated_child_context

    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="live task")
    finally:
        conn.close()
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_PROFILE", "worker-bot")

    worker = FakeAgent()
    _unthrottle()
    assert kt.inject_new_comments_from_env(worker) is False  # seed

    conn = kbc.connect()
    try:
        kb.add_comment(conn, tid, author="desktop", body="operator note for the worker")
    finally:
        conn.close()

    child = FakeAgent()
    with delegated_child_context():
        _unthrottle()
        assert kt.inject_new_comments_from_env(child) is False
    assert child.notes == []

    _unthrottle()
    assert kt.inject_new_comments_from_env(worker) is True
    assert "operator note" in worker.notes[0]


def test_skips_own_authored_comments_without_env_profile(worker_home, monkeypatch):
    """Own comments remain invisible when the profile identity comes from HERMES_HOME."""
    conn = kbc.connect()
    try:
        tid = kb.create_task(conn, title="identity echo guard")
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    monkeypatch.delenv("HERMES_PROFILE_NAME", raising=False)
    agent = FakeAgent()

    _unthrottle()
    kt.inject_new_comments_from_env(agent)  # seed

    identity = kt._persisted_identity()
    assert identity != "worker"

    conn = kbc.connect()
    try:
        kb.add_comment(conn, tid, author=identity, body="note to self")
    finally:
        conn.close()

    _unthrottle()
    assert kt.inject_new_comments_from_env(agent) is False
    assert agent.notes == []

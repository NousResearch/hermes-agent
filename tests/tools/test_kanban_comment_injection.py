"""Live kanban-comment injection into a running kanban worker.

``tools.kanban_tools.inject_new_comments_from_env`` polls the worker's task
for comments added *after* the run started and folds them into the live turn
via the agent's kanban-note channel, a sibling of the user-facing OUT-OF-BAND
steer channel that carries no user authority, so a comment on the task board
reaches a running worker without the block, comment, unblock dance or a
restart.

Verifies: no-op off a worker, watermark seeding (history isn't re-injected),
new comments arrive via kanban_note (never steer), honest author attribution
(no "operator" fallback), and own-authored comments are skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_WORKTREE = Path(__file__).resolve().parents[2]
if str(_WORKTREE) not in sys.path:
    sys.path.insert(0, str(_WORKTREE))

from hermes_cli import kanban_db as kb
import tools.kanban_tools as kt


class FakeAgent:
    def __init__(self):
        self.steers: list[str] = []
        self.notes: list[str] = []

    def steer(self, text: str) -> bool:
        self.steers.append(text)
        return True

    def kanban_note(self, text: str) -> bool:
        self.notes.append(text)
        return True


class FakeAgentNoNote:
    """An agent exposing only the legacy steer() channel, no kanban_note()."""

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
    assert agent.steers == []
    assert agent.notes == []


def test_noop_without_kanban_note_method(worker_home, monkeypatch):
    """An agent that only exposes steer() (no kanban_note()) must be skipped.

    This is the trust-boundary guard itself: kanban comments must never
    fall back to the user-steer channel just because that's the only
    channel an older agent object happens to expose.
    """
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="legacy agent")
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_PROFILE", "worker-bot")
    agent = FakeAgentNoNote()

    _unthrottle()
    assert kt.inject_new_comments_from_env(agent) is False
    assert agent.steers == []


def test_seed_then_inject_new_comment(worker_home, monkeypatch):
    conn = kb.connect()
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
    assert agent.steers == []

    conn = kb.connect()
    try:
        kb.add_comment(conn, tid, author="desktop", body="actually use the v2 API")
    finally:
        conn.close()

    _unthrottle()
    assert kt.inject_new_comments_from_env(agent) is True
    assert len(agent.notes) == 1
    assert "v2 API" in agent.notes[0]
    assert "desktop" in agent.notes[0]
    # Must never use the user-steer channel — a kanban comment is not the user.
    assert agent.steers == []

    # Watermark advanced — a re-poll with no new comments injects nothing.
    _unthrottle()
    assert kt.inject_new_comments_from_env(agent) is False
    assert len(agent.notes) == 1


def test_new_comment_never_uses_steer_channel(worker_home, monkeypatch):
    """Regression guard for the trust-boundary fix: comments must ride
    kanban_note(), never steer(), even when both are available.
    """
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="channel check")
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_PROFILE", "worker-bot")
    agent = FakeAgent()

    _unthrottle()
    kt.inject_new_comments_from_env(agent)  # seed

    conn = kb.connect()
    try:
        kb.add_comment(conn, tid, author="reviewer", body="please double check the retry logic")
    finally:
        conn.close()

    _unthrottle()
    assert kt.inject_new_comments_from_env(agent) is True
    assert agent.steers == [], "kanban comments must never ride the user-steer channel"
    assert len(agent.notes) == 1
    assert "reviewer" in agent.notes[0]


def test_missing_author_uses_honest_placeholder_not_operator(worker_home, monkeypatch):
    """A comment with no author must never render as 'operator'. That
    word implies user-level authority no kanban comment actually carries.

    ``add_comment`` itself rejects an empty author, so simulate the
    defensive edge case (a legacy/malformed row with an empty-string
    author) with a direct SQL insert, exactly the shape
    ``list_comments_after`` could hand back from data written before this
    constraint existed.
    """
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="honest attribution")
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_PROFILE", "worker-bot")
    agent = FakeAgent()

    _unthrottle()
    kt.inject_new_comments_from_env(agent)  # seed

    conn = kb.connect()
    try:
        with kb.write_txn(conn):
            conn.execute(
                "INSERT INTO task_comments (task_id, author, body, created_at) "
                "VALUES (?, '', ?, ?)",
                (tid, "unattributed note", 0),
            )
    finally:
        conn.close()

    _unthrottle()
    assert kt.inject_new_comments_from_env(agent) is True
    assert len(agent.notes) == 1
    assert "operator" not in agent.notes[0]
    assert "unknown" in agent.notes[0]


def test_skips_own_authored_comments(worker_home, monkeypatch):
    conn = kb.connect()
    try:
        tid = kb.create_task(conn, title="echo guard")
    finally:
        conn.close()

    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_PROFILE", "worker-bot")
    agent = FakeAgent()

    _unthrottle()
    kt.inject_new_comments_from_env(agent)  # seed

    conn = kb.connect()
    try:
        kb.add_comment(conn, tid, author="worker-bot", body="i did a thing")
    finally:
        conn.close()

    _unthrottle()
    assert kt.inject_new_comments_from_env(agent) is False
    assert agent.notes == []
    assert agent.steers == []

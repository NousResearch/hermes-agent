"""Tests for resolve_board_source() — epoch injection source resolution.

Verifies that after a gateway restart (in-memory registry cleared), the
DB-backed ``kanban_notify_subs`` fallback keeps epoch/notification routing
working. This is the fix for Bug #3/#4 (resolve_board_source unification).

Resolution order in resolve_board_source():
  1. In-memory ``_board_source_registry`` (module singleton, lost on restart)
  2. DB fallback: first row in ``kanban_notify_subs`` for the board

Run:
  cd /home/zml/workspace/hermes-agent
  venv/bin/python -m pytest tests/tools/test_resolve_board_source.py -v
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Fixture: isolated HERMES_HOME + initialised kanban DB
# ---------------------------------------------------------------------------

@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    """Provide a clean HERMES_HOME so the DB is isolated per-test."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    # Ensure a fresh DB is created for this test.
    from hermes_cli import kanban_db as kb
    kb.init_db()
    yield home


@pytest.fixture
def clean_registry():
    """Snapshot and clear the module-level _board_source_registry, restore on exit."""
    from tools import kanban_tools as kt
    registry = kt._board_source_registry
    # Save current state
    saved = registry.snapshot()
    # Clear to simulate a fresh process / post-restart state
    registry.clear()
    yield registry
    # Restore
    registry.clear()
    for board, src in saved.items():
        registry.set(board, src[0], src[1])


# ---------------------------------------------------------------------------
# Unit tests for resolve_board_source()
# ---------------------------------------------------------------------------

def test_in_memory_registry_resolves_before_restart(clean_registry):
    """When the in-memory registry has a source for a board,
    resolve_board_source returns it without touching the DB."""
    from tools import kanban_tools as kt

    kt._board_source_registry.set("my-board", "telegram", "chat-123")
    result = kt.resolve_board_source("my-board")
    assert result == ("telegram", "chat-123")


def test_db_fallback_after_restart_simulated(kanban_home, clean_registry):
    """After a gateway restart, the in-memory registry is empty.
    resolve_board_source must fall back to kanban_notify_subs in the DB
    and still return a valid (platform, chat_id) tuple."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    # Seed a notify subscription in the DB (this is what survives restart)
    conn = kb.connect(board="restart-board")
    try:
        kb.add_notify_sub(
            conn,
            task_id="t_test_001",
            platform="feishu",
            chat_id="oc_feishu_chat",
        )
    finally:
        conn.close()

    # Registry is empty (simulating post-restart state)
    assert kt._board_source_registry.get("restart-board") is None

    result = kt.resolve_board_source("restart-board")
    assert result is not None
    assert result[0] == "feishu"
    assert result[1] == "oc_feishu_chat"


def test_in_memory_takes_priority_over_db(kanban_home, clean_registry):
    """When both the registry and the DB have a source for the same board,
    the in-memory (most-recent) source wins."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    # Seed DB with one source
    conn = kb.connect(board="priority-board")
    try:
        kb.add_notify_sub(
            conn,
            task_id="t_test_002",
            platform="telegram",
            chat_id="old-chat",
        )
    finally:
        conn.close()

    # Set a different, more-recent source in the registry
    kt._board_source_registry.set("priority-board", "feishu", "new-chat")

    result = kt.resolve_board_source("priority-board")
    assert result == ("feishu", "new-chat"), (
        "in-memory registry must take priority over DB fallback"
    )


def test_returns_none_when_no_source_anywhere(kanban_home, clean_registry):
    """When neither the registry nor the DB has a source, returns None."""
    from tools import kanban_tools as kt

    result = kt.resolve_board_source("nonexistent-board")
    assert result is None


def test_db_fallback_skips_rows_with_empty_platform(kanban_home, clean_registry):
    """A notify sub with an empty platform string must not produce a
    bogus (empty, chat_id) tuple — resolve_board_source should return None."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    conn = kb.connect(board="bad-sub-board")
    try:
        kb.add_notify_sub(
            conn,
            task_id="t_test_003",
            platform="",
            chat_id="some-chat",
        )
    finally:
        conn.close()

    result = kt.resolve_board_source("bad-sub-board")
    assert result is None, (
        "notify sub with empty platform should not resolve to a source"
    )


def test_db_fallback_handles_multiple_subs(kanban_home, clean_registry):
    """When a board has multiple notify subs, resolve_board_source uses
    the first one returned by list_notify_subs (deterministic by PK order)."""
    from hermes_cli import kanban_db as kb
    from tools import kanban_tools as kt

    conn = kb.connect(board="multi-sub-board")
    try:
        kb.add_notify_sub(
            conn,
            task_id="t_multi_001",
            platform="telegram",
            chat_id="first-chat",
        )
        kb.add_notify_sub(
            conn,
            task_id="t_multi_002",
            platform="feishu",
            chat_id="second-chat",
        )
    finally:
        conn.close()

    result = kt.resolve_board_source("multi-sub-board")
    assert result is not None
    # First-inserted sub should be returned (PK order)
    assert result[0] == "telegram"
    assert result[1] == "first-chat"


# ---------------------------------------------------------------------------
# Registry thread-safety
# ---------------------------------------------------------------------------

def test_registry_set_and_get_are_thread_safe(clean_registry):
    """Concurrent set() calls on different boards must not corrupt each other."""
    import threading
    from tools import kanban_tools as kt

    errors: list[Exception] = []

    def writer(board: str, platform: str, chat_id: str):
        try:
            for _ in range(100):
                kt._board_source_registry.set(board, platform, chat_id)
                got = kt._board_source_registry.get(board)
                if got != (platform, chat_id):
                    errors.append(AssertionError(
                        f"{board}: expected ({platform}, {chat_id}), got {got}"
                    ))
        except Exception as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=writer, args=("board-a", "telegram", "chat-a")),
        threading.Thread(target=writer, args=("board-b", "feishu", "chat-b")),
        threading.Thread(target=writer, args=("board-c", "discord", "chat-c")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == [], f"thread-safety violations: {errors}"

    # Each board retains its own value
    assert kt._board_source_registry.get("board-a") == ("telegram", "chat-a")
    assert kt._board_source_registry.get("board-b") == ("feishu", "chat-b")
    assert kt._board_source_registry.get("board-c") == ("discord", "chat-c")


def test_registry_clear_removes_single_board(clean_registry):
    """clear(board) removes only that board, not others."""
    from tools import kanban_tools as kt

    kt._board_source_registry.set("board-x", "telegram", "chat-x")
    kt._board_source_registry.set("board-y", "feishu", "chat-y")

    kt._board_source_registry.clear("board-x")

    assert kt._board_source_registry.get("board-x") is None
    assert kt._board_source_registry.get("board-y") == ("feishu", "chat-y")

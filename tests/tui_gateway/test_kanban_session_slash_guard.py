"""Tests for the Kanban terminal-session slash-worker guard (#68779).

A Kanban-dispatched worker session that reaches a terminal state (blocked,
done, archived) must NOT be resumed as a write-capable Desktop/TUI session
with a slash worker — otherwise it becomes an unobservable "ghost writer"
that the Kanban board/dispatcher cannot supervise.
"""

import time
from unittest.mock import patch, MagicMock


def test_is_terminal_kanban_session_blocks_on_blocked(tmp_path, monkeypatch):
    """A session linked to a blocked Kanban task is flagged as terminal."""
    from tui_gateway import server as srv

    # Reset cache between tests
    srv._kanban_task_cache.clear()

    import sqlite3

    db_path = tmp_path / "kanban.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, session_id TEXT, status TEXT)"
    )
    conn.execute(
        "INSERT INTO tasks (id, session_id, status) VALUES (?, ?, ?)",
        ("t-1", "sess-abc-123", "blocked"),
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))

    assert srv._is_terminal_kanban_session("sess-abc-123") is True


def test_is_terminal_kanban_session_allows_running(tmp_path, monkeypatch):
    """A session linked to a running Kanban task is NOT flagged as terminal."""
    from tui_gateway import server as srv

    srv._kanban_task_cache.clear()

    import sqlite3

    db_path = tmp_path / "kanban.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, session_id TEXT, status TEXT)"
    )
    conn.execute(
        "INSERT INTO tasks (id, session_id, status) VALUES (?, ?, ?)",
        ("t-2", "sess-running", "running"),
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))

    assert srv._is_terminal_kanban_session("sess-running") is False


def test_is_terminal_kanban_session_allows_unknown_session(tmp_path, monkeypatch):
    """A session with no Kanban task linkage is NOT flagged as terminal."""
    from tui_gateway import server as srv

    srv._kanban_task_cache.clear()

    import sqlite3

    db_path = tmp_path / "kanban.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE tasks (id TEXT PRIMARY KEY, session_id TEXT, status TEXT)"
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("HERMES_KANBAN_DB", str(db_path))

    assert srv._is_terminal_kanban_session("random-session-id") is False


def test_is_terminal_kanban_session_caches_result():
    """The guard caches its DB lookup to avoid per-spawn SQLite queries."""
    from tui_gateway import server as srv

    srv._kanban_task_cache.clear()

    with patch.object(srv, "_kanban_cache_ttl", 60.0):
        # First call queries the DB
        with patch("hermes_cli.kanban_db.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchone.return_value = ("blocked",)
            mock_connect.return_value = mock_conn
            result = srv._is_terminal_kanban_session("cached-session")
            assert result is True
            assert mock_connect.call_count == 1

        # Second call within TTL uses the cache
        with patch("hermes_cli.kanban_db.connect") as mock_connect2:
            result = srv._is_terminal_kanban_session("cached-session")
            assert result is True
            assert mock_connect2.call_count == 0


def test_is_terminal_kanban_session_cache_expires():
    """After the TTL expires, the guard re-queries the DB."""
    from tui_gateway import server as srv

    srv._kanban_task_cache.clear()

    with patch.object(srv, "_kanban_cache_ttl", 0.0):
        with patch("hermes_cli.kanban_db.connect") as mock_connect:
            mock_conn = MagicMock()
            mock_conn.execute.return_value.fetchone.return_value = ("blocked",)
            mock_connect.return_value = mock_conn

            srv._is_terminal_kanban_session("expiring-session")
            assert mock_connect.call_count == 1

            srv._is_terminal_kanban_session("expiring-session")
            # TTL=0 means every call re-queries
            assert mock_connect.call_count == 2


def test_is_terminal_kanban_session_db_error_falls_back():
    """If the Kanban DB is unavailable, the guard allows the session through."""
    from tui_gateway import server as srv

    srv._kanban_task_cache.clear()

    with patch("hermes_cli.kanban_db.connect", side_effect=Exception("db error")):
        assert srv._is_terminal_kanban_session("any-session") is False

"""``session.active_list`` cross-process (foreign) liveness rows — #85302.

Sessions that exist only in state.db (cron runs, CLI one-shots, messaging
turns written by other processes) never enter the gateway's ``_sessions``, so
``active_list`` must surface recently-active rows flagged ``foreign`` for
clients to paint from the same poll.
"""

import threading
import time
import types

from tui_gateway import server


def _session(agent=None, **extra):
    return {
        "agent": agent if agent is not None else types.SimpleNamespace(),
        "session_key": "session-key",
        "history": [],
        "history_lock": threading.Lock(),
        "history_version": 0,
        "running": False,
        "attached_images": [],
        "image_counter": 0,
        "cols": 80,
        "slash_worker": None,
        "show_reasoning": False,
        "tool_progress_mode": "all",
        **extra,
    }


class _RestoreSessions:
    """Snapshot and clear ``server._sessions`` / ``_active_child_runs`` around a poll."""

    def __enter__(self):
        self.previous_sessions = dict(server._sessions)
        self.previous_children = dict(server._active_child_runs)
        server._sessions.clear()
        server._active_child_runs.clear()

    def __exit__(self, *exc_info):
        server._sessions.clear()
        server._sessions.update(self.previous_sessions)
        server._active_child_runs.clear()
        server._active_child_runs.update(self.previous_children)


def test_session_active_list_reports_foreign_db_rows(monkeypatch):
    """Cross-process liveness (#85302): sessions that exist only in state.db
    (cron runs, CLI one-shots, messaging turns written by other processes)
    never enter ``_sessions``, so ``session.active_list`` must surface
    recently-active rows flagged ``foreign`` for clients to paint from the
    same poll. Rows outside the 300s recency window are not reported, and a
    failed DB probe keeps the in-memory answer (logged, not swallowed)."""

    class _DB:
        def get_session_title(self, key):
            return ""

        def list_sessions_rich(self, **kwargs):
            now = time.time()
            return [
                {
                    "id": "cron_abc_20260812",
                    "source": "cron",
                    "model": "m",
                    "title": "Nightly job",
                    "started_at": now - 100,
                    "last_active": now - 10,
                    "last_activity_description": "executing tool",
                    "message_count": 4,
                    "ended_at": None,
                },
                {
                    "id": "old_cli_session",
                    "source": "cli",
                    "model": "m",
                    "title": "Stale",
                    "started_at": now - 4000,
                    "last_active": now - 4000,
                    "last_activity_description": "",
                    "message_count": 9,
                    "ended_at": None,
                },
                {
                    # Ended rows are not live even inside the window.
                    "id": "ended_gw_session",
                    "source": "gateway",
                    "model": "m",
                    "title": "Ended",
                    "started_at": now - 50,
                    "last_active": now - 5,
                    "last_activity_description": "",
                    "message_count": 2,
                    "ended_at": now - 1,
                },
            ]

    monkeypatch.setattr(server, "_get_db", lambda: _DB())
    with _RestoreSessions():
        resp = server.handle_request(
            {
                "id": "1",
                "method": "session.active_list",
                "params": {},
            }
        )

    rows = {row["id"]: row for row in resp["result"]["sessions"]}
    foreign = rows["cron_abc_20260812"]
    assert foreign["foreign"] is True
    assert foreign["status"] == "working"
    assert foreign["session_key"] == "cron_abc_20260812"
    assert foreign["description"] == "executing tool"
    # A row outside the 300s recency window is NOT reported.
    assert "old_cli_session" not in rows
    # A row with ended_at is NOT reported regardless of recency.
    assert "ended_gw_session" not in rows


def test_session_active_list_foreign_probe_failure_keeps_in_memory_answer(monkeypatch):
    """The foreign-row DB probe is best-effort: a failed probe must keep the
    in-memory answer (the AI-review finding on the blanket except-pass — the
    failure is logged, never silently swallowed, and never breaks the poll)."""

    class _BrokenDB:
        def list_sessions_rich(self, **kwargs):
            raise RuntimeError("state.db is locked")

    previous_sessions = dict(server._sessions)
    server._sessions.clear()
    server._sessions["sid-live"] = _session(
        agent=types.SimpleNamespace(model="model-live"),
        history=[{"role": "user", "content": "here"}],
        session_key="key-live",
        created_at=10.0,
        last_active=20.0,
    )
    monkeypatch.setattr(server, "_get_db", lambda: _BrokenDB())
    try:
        resp = server.handle_request(
            {"id": "1", "method": "session.active_list", "params": {}}
        )
    finally:
        server._sessions.clear()
        server._sessions.update(previous_sessions)

    assert [row["id"] for row in resp["result"]["sessions"]] == ["sid-live"]


def test_session_active_list_reports_in_flight_subagent_children(monkeypatch):
    """A delegated child with a run in flight but no watch window never enters
    ``_sessions``; the liveness registry row (keyed on the owning profile home)
    is reported as a foreign working row."""

    monkeypatch.setattr(server, "_get_db", lambda: None)
    with _RestoreSessions():
        server._active_child_runs[(None, "child-key")] = time.time()
        resp = server.handle_request(
            {"id": "1", "method": "session.active_list", "params": {}}
        )

    rows = {row["id"]: row for row in resp["result"]["sessions"]}
    assert rows["child-key"]["foreign"] is True
    assert rows["child-key"]["status"] == "working"
    assert rows["child-key"]["description"] == "subagent running"

"""Tests for slash_worker lifecycle: detach-time worker close and session
close logging.

These tests cover the new fixes in this PR. Related work by other contributors:
- #53303 / PR #53308 (drain thread join) by @pasevin
- #38095 / PR #41473 (worker registry + idempotent close) by @Fewmanism
- #48643 / PR #48656 (startup reaper + zombie parent detection) by @Nigmat-future
"""

from __future__ import annotations

from unittest.mock import MagicMock

from tui_gateway import server


def test_close_sessions_for_transport_closes_worker_on_detach():
    """When a session is detached (WS disconnect), its slash_worker must be
    closed immediately rather than lingering until the 6h TTL reaper.

    The Desktop app uses one WebSocket for all sessions. When the user switches
    sessions, the old session's transport is detached but the slash_worker
    subprocess (~13 MB) stays alive. Over a few hours this accumulates dozens
    of idle workers, contributing to GIL pressure and memory bloat.
    """
    fake_worker = MagicMock()
    fake_worker._closed = False

    fake_transport = MagicMock()
    fake_transport._closed = True  # _transport_is_dead returns True

    sid = "detach-test-sid"
    session = {
        "transport": fake_transport,
        "slash_worker": fake_worker,
        "close_on_disconnect": False,
        "session_key": "detach-test",
    }
    server._sessions[sid] = session

    reaped, detached = server._close_sessions_for_transport(
        fake_transport, end_reason="ws_disconnect"
    )

    assert detached == 1
    assert reaped == 0
    # The worker should have been closed immediately
    fake_worker.close.assert_called_once()
    # The session's slash_worker should now be None (lazy recreation on next use)
    assert session["slash_worker"] is None

    server._sessions.clear()


def test_close_sessions_for_transport_preserves_worker_on_reap():
    """Sessions with close_on_disconnect=True go through _close_session_by_id,
    which tears down the session fully via _teardown_session (which closes the
    worker inside _finalize_session). The detach-time close should NOT also
    fire for these — it would be a redundant close (harmless due to the
    _closed guard, but we assert the path is clean)."""
    fake_transport = MagicMock()
    sid = "reap-test-sid"
    session = {
        "transport": fake_transport,
        "slash_worker": None,
        "close_on_disconnect": True,
        "session_key": "reap-test",
        "_finalized": False,
        "agent": None,
        "running": False,
    }
    server._sessions[sid] = session

    reaped, detached = server._close_sessions_for_transport(
        fake_transport, end_reason="ws_disconnect"
    )

    assert reaped == 1
    assert detached == 0
    assert sid not in server._sessions

    server._sessions.clear()


def test_close_sessions_for_transport_handles_worker_close_exception():
    """If worker.close() raises, the detach path must still complete — the
    session is pointed at the detached transport and the orphan reaper is
    scheduled."""
    fake_worker = MagicMock()
    fake_worker._closed = False
    fake_worker.close.side_effect = RuntimeError("boom")

    fake_transport = MagicMock()
    fake_transport._closed = True

    sid = "exception-test-sid"
    session = {
        "transport": fake_transport,
        "slash_worker": fake_worker,
        "close_on_disconnect": False,
        "session_key": "exception-test",
    }
    server._sessions[sid] = session

    # Must not raise
    reaped, detached = server._close_sessions_for_transport(
        fake_transport, end_reason="ws_disconnect"
    )

    assert detached == 1
    # worker.close() was called (even though it raised)
    fake_worker.close.assert_called_once()
    # slash_worker is still set to None despite the exception
    assert session["slash_worker"] is None

    server._sessions.clear()


def test_close_session_by_id_logs_end_reason(caplog):
    """_close_session_by_id should log the session id and end_reason at INFO
    level for observability — previously the teardown path was completely
    silent, making it impossible to diagnose why sessions were or weren't
    being reaped."""
    import logging

    sid = "log-test-sid"
    session = {
        "transport": None,
        "slash_worker": None,
        "close_on_disconnect": False,
        "session_key": "log-test",
        "_finalized": False,
        "agent": None,
        "running": False,
    }
    server._sessions[sid] = session

    with caplog.at_level(logging.INFO, logger="tui_gateway.server"):
        result = server._close_session_by_id(sid, end_reason="ws_orphan_reap")

    assert result is True
    assert any(
        "session closed" in record.message and "ws_orphan_reap" in record.message
        for record in caplog.records
    ), f"expected 'session closed ... ws_orphan_reap' in logs, got: {[r.message for r in caplog.records]}"

    server._sessions.clear()

"""Tests for the usage.* JSON-RPC methods on the tui_gateway server."""

from __future__ import annotations

import time

import pytest

import tui_gateway.server as server

from hermes_state import SessionDB


def _call(method, params=None):
    handler = server._methods[method]
    resp = handler(1, params or {})
    assert "error" not in resp, resp.get("error")
    return resp["result"]


@pytest.fixture()
def usage_db(tmp_path, monkeypatch):
    """Point the gateway at a temp SessionDB with one known session."""
    db_path = tmp_path / "usage_rpc.db"
    session_db = SessionDB(db_path=db_path)
    now = time.time()
    session_db.create_session(
        session_id="rpc1", source="cli",
        model="deepseek/deepseek-v4-flash-0731", user_id="u1",
    )
    session_db._conn.execute(
        "UPDATE sessions SET started_at = ? WHERE id = 'rpc1'", (now - 3600,))
    session_db.end_session("rpc1", end_reason="user_exit")
    session_db.update_token_counts("rpc1", input_tokens=12345, output_tokens=6789)
    session_db._conn.commit()

    monkeypatch.setattr(server, "_db", session_db)
    monkeypatch.setattr(server, "_db_error", None)
    yield session_db
    session_db.close()
    monkeypatch.setattr(server, "_db", None)


def test_usage_overview_registered():
    assert "usage.overview" in server._methods


def test_usage_overview_returns_engine_report(usage_db):
    report = _call("usage.overview", {"days": 30})
    assert report["empty"] is False
    assert report["overview"]["total_sessions"] >= 1
    assert report["overview"]["total_input_tokens"] >= 12345
    # The desktop surface's two new inputs are present.
    assert "daily_series" in report
    assert "cost_buckets" in report["overview"]


def test_usage_overview_daily_series_window(usage_db):
    report = _call("usage.overview", {"days": 7})
    assert len(report["daily_series"]) == 7


def test_usage_overview_bad_days_rejected():
    resp = server._methods["usage.overview"](1, {"days": 99999})
    assert "error" in resp
    assert resp["error"]["code"] == 5071

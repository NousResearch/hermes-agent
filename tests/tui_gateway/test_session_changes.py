from __future__ import annotations

import logging
from pathlib import Path

import pytest

from hermes_state import SessionDB
from tui_gateway import server


def _call(method: str, params: dict) -> dict:
    return server._methods[method](1, params)


@pytest.fixture()
def db(tmp_path: Path, monkeypatch):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    monkeypatch.setattr(server, "_get_db", lambda: session_db)
    yield session_db
    session_db.close()


def test_session_changes_returns_ordered_rows_after_cursor_and_uses_index(db):
    db.create_session("s1", source="desktop")
    first_id = db.append_message("s1", "user", "one", timestamp=1.0)
    second_id = db.append_message("s1", "assistant", "two", timestamp=2.0)
    third_id = db.append_message("s1", "user", "three", timestamp=3.0)
    db.create_session("other", source="desktop")
    db.append_message("other", "user", "ignore me", timestamp=4.0)

    envelope = _call(
        "session.changes",
        {"session_id": "s1", "since_message_id": first_id},
    )

    assert "error" not in envelope
    result = envelope["result"]
    assert [message["id"] for message in result["messages"]] == [second_id, third_id]
    assert [message["text"] for message in result["messages"]] == ["two", "three"]
    assert [message["timestamp"] for message in result["messages"]] == [2.0, 3.0]
    assert result["last_id"] == third_id

    plan_rows = db._conn.execute(
        """
        EXPLAIN QUERY PLAN
        SELECT * FROM messages
        WHERE session_id = ? AND id > ?
        ORDER BY id
        """,
        ("s1", first_id),
    ).fetchall()
    plan = "\n".join(str(tuple(row)) for row in plan_rows)
    assert "USING INDEX idx_messages_session_id" in plan


def test_session_changes_unknown_session_returns_clean_json_rpc_error(db):
    envelope = _call(
        "session.changes",
        {"session_id": "missing", "since_message_id": 0},
    )

    assert envelope["error"]["code"] == 4044
    assert "session not found" in envelope["error"]["message"]
    assert "traceback" not in str(envelope).lower()


def test_session_changes_disabled_returns_feature_error_without_db_read(monkeypatch):
    monkeypatch.setattr(
        server,
        "_load_cfg",
        lambda: {"dashboard": {"session_sync": {"enabled": False}}},
    )
    monkeypatch.setattr(
        server,
        "_get_db",
        lambda: pytest.fail("disabled session.changes must not open state.db"),
    )

    envelope = _call(
        "session.changes",
        {"session_id": "s1", "since_message_id": 0},
    )

    assert envelope["error"]["code"] == server._SESSION_CHANGES_DISABLED_ERROR
    assert "disabled" in envelope["error"]["message"]


def test_session_sync_config_reads_all_knobs_from_dashboard_block():
    cfg = server._load_session_sync_config(
        {
            "dashboard": {
                "session_sync": {
                    "enabled": True,
                    "t_silence": 12.5,
                    "poll_interval": 3.25,
                    "refocus_debounce": 1.75,
                }
            }
        }
    )

    assert cfg == {
        "enabled": True,
        "t_silence": 12.5,
        "poll_interval": 3.25,
        "refocus_debounce": 1.75,
    }


@pytest.mark.asyncio
async def test_status_omits_session_changes_capability_when_disabled(monkeypatch):
    import hermes_cli.web_server as web_server

    config = {
        "dashboard": {
            "session_sync": {
                "enabled": False,
                "t_silence": 8.0,
                "poll_interval": 4.0,
                "refocus_debounce": 2.0,
            }
        }
    }

    async def _active_sessions():
        return 0

    monkeypatch.setattr(web_server, "load_config", lambda: config)
    monkeypatch.setattr(web_server, "check_config_version", lambda: (1, 1))
    monkeypatch.setattr(web_server, "get_running_pid_cached", lambda: None)
    monkeypatch.setattr(web_server, "read_runtime_status", lambda: None)
    monkeypatch.setattr(web_server, "_status_active_sessions", _active_sessions)
    monkeypatch.setattr(web_server, "_resolve_restart_drain_timeout", lambda: 0)
    monkeypatch.setattr(
        web_server,
        "_collect_profile_gateway_topology",
        lambda: {"profiles": [], "gateway_mode": "single", "gateways": []},
    )
    monkeypatch.setattr(
        web_server,
        "_dashboard_local_update_managed_externally",
        lambda: False,
    )
    monkeypatch.setattr(web_server, "_GATEWAY_HEALTH_URL", "")
    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)

    status = await web_server.get_status()

    assert "session_changes" not in status["capabilities"]
    assert status["session_sync"] == {
        "t_silence": 8.0,
        "poll_interval": 4.0,
        "refocus_debounce": 2.0,
    }


@pytest.mark.asyncio
async def test_status_advertises_session_changes_capability_when_enabled(monkeypatch):
    import hermes_cli.web_server as web_server

    async def _active_sessions():
        return 0

    monkeypatch.setattr(
        web_server,
        "load_config",
        lambda: {"dashboard": {"session_sync": {"enabled": True}}},
    )
    monkeypatch.setattr(web_server, "check_config_version", lambda: (1, 1))
    monkeypatch.setattr(web_server, "get_running_pid_cached", lambda: None)
    monkeypatch.setattr(web_server, "read_runtime_status", lambda: None)
    monkeypatch.setattr(web_server, "_status_active_sessions", _active_sessions)
    monkeypatch.setattr(web_server, "_resolve_restart_drain_timeout", lambda: 0)
    monkeypatch.setattr(
        web_server,
        "_collect_profile_gateway_topology",
        lambda: {"profiles": [], "gateway_mode": "single", "gateways": []},
    )
    monkeypatch.setattr(
        web_server,
        "_dashboard_local_update_managed_externally",
        lambda: False,
    )
    monkeypatch.setattr(web_server, "_GATEWAY_HEALTH_URL", "")
    monkeypatch.setattr(web_server.app.state, "auth_required", True, raising=False)

    status = await web_server.get_status()

    assert status["capabilities"]["session_changes"] is True


def test_dashboard_startup_state_db_log_is_absolute(tmp_path, monkeypatch, caplog):
    import hermes_cli.web_server as web_server

    monkeypatch.setattr(web_server, "get_hermes_home", lambda: tmp_path)

    with caplog.at_level(logging.INFO, logger="hermes_cli.web_server"):
        path = web_server._log_dashboard_state_db_path()

    assert path.is_absolute()
    assert path.name == "state.db"
    assert any(
        "Dashboard state.db path:" in record.getMessage()
        and str(path) in record.getMessage()
        for record in caplog.records
    )

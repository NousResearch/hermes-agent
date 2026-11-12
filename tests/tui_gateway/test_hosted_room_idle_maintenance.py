"""A gateway that seats no Bot from another gateway writes nothing to its room store on idle cycles."""

import sqlite3

import pytest

from hermes_agent.gateway import hosted_rooms
from tests.gateway.test_session_group_peers import gateway as gateway_fixture

gateway = gateway_fixture

_WRITES = ("BEGIN IMMEDIATE", "INSERT", "UPDATE", "DELETE", "REPLACE")


@pytest.fixture
def writes(monkeypatch):
    """Every write statement any SQLite connection runs from here on."""
    seen, connect = [], sqlite3.connect

    def traced(*args, **kwargs):
        conn = connect(*args, **kwargs)
        conn.set_trace_callback(
            lambda sql: seen.append(sql) if sql.lstrip().upper().startswith(_WRITES) else None)
        return conn
    monkeypatch.setattr(sqlite3, "connect", traced)
    return seen


def test_idle_revoke_maintenance_takes_no_write_transaction(gateway, writes):
    service = gateway.service
    hosted_rooms.list_rooms(service.db_path)  # the store exists, as on any running gateway
    writes.clear()
    assert not service.peer_routes
    for _ in range(10):
        assert service.retry_owed_revokes() == 0
    assert writes == []


def test_reading_due_revokes_writes_only_to_drop_an_expired_one(gateway, writes):
    db = gateway.service.db_path
    hosted_rooms.owe_room_grant_revoke(
        db, room_id="ended", member_id="peer", target_url="http://127.0.0.1:9", grant="g", expires_at=10**10)
    writes.clear()
    for _ in range(10):
        assert len(hosted_rooms.due_room_grant_revokes(db, limit=5, now=1.0)) == 0  # not due yet
    assert writes == []
    assert hosted_rooms.due_room_grant_revokes(db, limit=5, now=10**10 + 1) == []
    assert any(sql.upper().startswith("DELETE") for sql in writes)
    assert not hosted_rooms.owes_room_grant_revokes(db)

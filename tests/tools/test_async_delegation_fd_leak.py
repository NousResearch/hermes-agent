"""Regression: the async-delegation ledger must close every SQLite connection.

Sibling of the cron execution-ledger leak (#69567 / PR #69594). A ledger
connection that commits or rolls back but is never closed leaks the
db/-wal/-shm file descriptors on every dispatch, completion, and
delivery-claim. These tests fail if the deterministic ``close()`` is ever
removed again.
"""

import queue
import sqlite3

import pytest

from hermes_cli.sqlite_safe_read import has_live_connection
from persistence import PermanentStoreError
from tools import async_delegation as ad
from persistence_assembly import profile_scope


def _point_ledger(monkeypatch, tmp_path):
    monkeypatch.setattr(ad, "_scope", lambda: profile_scope(home=tmp_path))
    return ad


def _track_handles(monkeypatch):
    import persistence_assembly

    opened, closed = [], []
    real_open = persistence_assembly.open_persistence

    def tracking_open(scope, **kwargs):
        handle = real_open(scope, **kwargs)
        opened.append(id(handle))
        real_close = handle.close

        def close():
            closed.append(id(handle))
            real_close()
        handle.close = close
        return handle

    monkeypatch.setattr(persistence_assembly, "open_persistence", tracking_open)
    return opened, closed


def _track_connections(monkeypatch, factory_for=lambda factory: factory):
    """Record every state.db connection opened; ``factory_for`` may swap the connection class."""
    opened = []
    real_connect = sqlite3.connect

    def tracking_connect(database, *args, factory=sqlite3.Connection, **kwargs):
        if str(database) == ":memory:":
            return real_connect(database, *args, factory=factory, **kwargs)
        conn = real_connect(database, *args, factory=factory_for(factory), **kwargs)
        opened.append(conn)
        return conn

    monkeypatch.setattr(sqlite3, "connect", tracking_connect)
    return opened


def _still_open(connections):
    still_open = []
    for conn in connections:
        try:
            conn.total_changes
        except sqlite3.ProgrammingError:  # closed
            continue
        still_open.append(conn)
    return still_open


def test_ledger_operations_close_every_connection(monkeypatch, tmp_path):
    """Public durable-ledger reads/writes must close every storage handle and connection opened."""
    _point_ledger(monkeypatch, tmp_path)
    opened, closed = _track_handles(monkeypatch)

    ad.get_durable_delegation("nope")
    ad.recover_abandoned_delegations()
    ad.restore_undelivered_completions(queue.Queue())
    ad.claim_completion_delivery("nope", "claim-1")

    assert opened, "expected at least one storage handle to be opened"
    assert len(opened) == len(closed)
    assert set(opened) == set(closed)

    opened = _track_connections(monkeypatch)
    ad.mark_completion_delivered("nope")

    assert opened, "expected at least one connection to be opened"
    assert _still_open(opened) == []
    assert not has_live_connection(tmp_path / "state.db")


def _fail_ddl(sql):
    if "CREATE TABLE" in sql:
        raise sqlite3.OperationalError("simulated schema init failure")


class _FailingSchemaCursor:
    """Cursor proxy that fails the canonical DDL, which the store's schema init runs
    through ``conn.cursor()``."""

    def __init__(self, real):
        object.__setattr__(self, "_real", real)

    def execute(self, sql, *args, **kwargs):
        _fail_ddl(sql)
        return self._real.execute(sql, *args, **kwargs)

    def executescript(self, script):
        _fail_ddl(script)
        return self._real.executescript(script)

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_schema_init_failure_still_closes_connection(monkeypatch, tmp_path):
    """A PRAGMA/DDL failure after connect() must still close the state.db connection.

    Only state.db opens are tracked: the column reconciler builds its reference
    shape from SCHEMA_SQL in a private ``:memory:`` database, which it closes itself
    and which holds no file descriptor."""
    _point_ledger(monkeypatch, tmp_path)

    def failing_schema(factory):
        class _FailingSchemaConnection(factory):
            def execute(self, sql, *args, **kwargs):
                _fail_ddl(sql)
                return super().execute(sql, *args, **kwargs)

            def executescript(self, script):
                _fail_ddl(script)
                return super().executescript(script)

            def cursor(self, *args, **kwargs):
                return _FailingSchemaCursor(super().cursor(*args, **kwargs))
        return _FailingSchemaConnection

    opened = _track_connections(monkeypatch, failing_schema)

    with pytest.raises(PermanentStoreError, match="simulated schema init failure") as raised:
        ad.mark_completion_delivered("nope")

    assert isinstance(raised.value.__cause__, sqlite3.OperationalError)
    assert len(opened) == 1
    assert _still_open(opened) == []
    assert not has_live_connection(tmp_path / "state.db")

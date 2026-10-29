"""Regression: the async-delegation ledger must close every SQLite connection.

Sibling of the cron execution-ledger leak (#69567 / PR #69594). The durable
delegation ledger used ``with _connect() as conn:`` where the connection
context manager commits/rolls back but never closes, leaking the db/-wal/-shm
file descriptors on every dispatch, completion, and delivery-claim. These tests
fail if the deterministic ``close()`` is ever removed again.
"""

import queue
import sqlite3

import pytest

from tools import async_delegation as ad


class _TrackingConnection:
    """Delegates to a real sqlite3.Connection while recording close() calls.

    sqlite3.Connection is a static C type: it has no per-instance __dict__ and
    its methods can't be monkeypatched, so open/close tracking is done via a
    delegating wrapper returned in place of the real connection.
    """

    def __init__(self, real, closed_ids):
        object.__setattr__(self, "_real", real)
        object.__setattr__(self, "_closed_ids", closed_ids)

    def close(self):
        self._closed_ids.append(id(self._real))
        self._real.close()

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._real.__exit__(exc_type, exc, tb)

    def __getattr__(self, name):
        return getattr(self._real, name)

    def __setattr__(self, name, value):
        setattr(self._real, name, value)


def _point_ledger(monkeypatch, tmp_path):
    monkeypatch.setattr(ad, "_db_path", lambda: tmp_path / "state.db")
    return ad


def _track_connections(monkeypatch):
    opened, closed = [], []
    real_connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened.append(id(conn))
        return _TrackingConnection(conn, closed)

    monkeypatch.setattr(ad.sqlite3, "connect", tracking_connect)
    return opened, closed


def test_ledger_operations_close_every_connection(monkeypatch, tmp_path):
    """Public durable-ledger reads/writes must close every connection opened."""
    _point_ledger(monkeypatch, tmp_path)
    opened, closed = _track_connections(monkeypatch)

    ad.get_durable_delegation("nope")
    ad.recover_abandoned_delegations()
    ad.restore_undelivered_completions(queue.Queue())
    ad.mark_completion_delivered("nope")
    ad.claim_completion_delivery("nope", "claim-1")

    assert opened, "expected at least one connection to be opened"
    assert len(opened) == len(closed)
    assert set(opened) == set(closed)


class _FailingSchemaCursor:
    """Cursor proxy that fails the canonical DDL replay. ``apply_declared_schema``
    replays SCHEMA_SQL statement by statement through ``conn.cursor()`` (no
    ``executescript`` implicit COMMIT), so the injection sits on the cursor."""

    def __init__(self, real):
        object.__setattr__(self, "_real", real)

    def execute(self, sql, *args, **kwargs):
        if "CREATE TABLE" in sql:
            raise sqlite3.OperationalError("simulated schema init failure")
        return self._real.execute(sql, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._real, name)


def test_schema_init_failure_still_closes_connection(monkeypatch, tmp_path):
    """A PRAGMA/DDL failure after connect() must still close the state.db connection.

    Only state.db opens are tracked: the column reconciler builds its reference
    shape from SCHEMA_SQL in a private ``:memory:`` database, which it closes itself
    and which holds no file descriptor."""
    _point_ledger(monkeypatch, tmp_path)
    opened, closed = [], []
    real_connect = sqlite3.connect

    class _FailingSchemaConnection(_TrackingConnection):
        def execute(self, sql, *args, **kwargs):
            if "CREATE TABLE" in sql:
                raise sqlite3.OperationalError("simulated schema init failure")
            return self._real.execute(sql, *args, **kwargs)

        def executescript(self, script):
            if "CREATE TABLE" in script:
                raise sqlite3.OperationalError("simulated schema init failure")
            return self._real.executescript(script)

        def cursor(self, *args, **kwargs):
            return _FailingSchemaCursor(self._real.cursor(*args, **kwargs))

    def tracking_connect(database, *args, **kwargs):
        conn = real_connect(database, *args, **kwargs)
        if str(database) == ":memory:":
            return conn
        opened.append(id(conn))
        return _FailingSchemaConnection(conn, closed)

    monkeypatch.setattr(ad.sqlite3, "connect", tracking_connect)

    with pytest.raises(sqlite3.OperationalError, match="simulated schema init failure"):
        with ad._transaction():
            pass

    assert len(opened) == 1
    assert closed == opened

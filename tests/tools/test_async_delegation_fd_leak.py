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


def _track_connections(monkeypatch, tmp_path):
    """Track opens/closes of connections to the LEDGER path only.

    The canonical-shape probe (#109786) opens its own short-lived connections (in-memory
    reference + mode=ro probe) that are not part of the ledger's fd contract.
    """
    opened, closed = [], []
    real_connect = sqlite3.connect
    ledger_path = str(tmp_path / "state.db")

    def _is_ledger(args):
        import os as _os

        for a in args:
            try:
                if isinstance(a, (str, _os.PathLike)) and _os.fspath(a) == ledger_path:
                    return True
            except TypeError:
                continue
        return False

    def tracking_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        if _is_ledger(args):
            opened.append(id(conn))
            return _TrackingConnection(conn, closed)
        return conn

    monkeypatch.setattr(ad.sqlite3, "connect", tracking_connect)
    return opened, closed


def test_ledger_operations_close_every_connection(monkeypatch, tmp_path):
    """Public durable-ledger reads/writes must close every ledger connection opened."""
    _point_ledger(monkeypatch, tmp_path)
    opened, closed = _track_connections(monkeypatch, tmp_path)

    ad.get_durable_delegation("nope")
    ad.recover_abandoned_delegations()
    ad.restore_undelivered_completions(queue.Queue())
    ad.mark_completion_delivered("nope")
    ad.claim_completion_delivery("nope", "claim-1")

    assert opened, "expected at least one ledger connection to be opened"
    assert len(opened) == len(closed)
    assert set(opened) == set(closed)


def _fail_large_schema_replay(self, script):
    # The schema initializer replays the full canonical schema as one
    # large multi-statement script (single durable-shape authority,
    # #94691). Simulate the DDL failure on that path so the
    # connect-close-on-init-failure contract stays pinned.
    if len(script) > 1000:
        raise sqlite3.OperationalError("simulated schema init failure")
    return self._real.executescript(script)


def test_schema_init_failure_still_closes_connection(monkeypatch, tmp_path):
    """A PRAGMA/DDL failure after connect() must still close the connection."""
    _point_ledger(monkeypatch, tmp_path)
    opened, closed = [], []
    real_connect = sqlite3.connect
    ledger_path = str(tmp_path / "state.db")

    class _FailingSchemaConnection(_TrackingConnection):
        def execute(self, sql, *args, **kwargs):
            if "CREATE TABLE" in sql:
                raise sqlite3.OperationalError("simulated schema init failure")
            return self._real.execute(sql, *args, **kwargs)

    # The canonical-shape probe (#109786) legitimately opens its own short-lived
    # connections (an in-memory reference DB and a mode=ro probe of the ledger).
    # Track only connections to the LEDGER path: the leak contract is about the
    # ledger's own handles, and the failing one of those must still close.
    def tracking_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        import os as _os

        is_ledger = any(
            isinstance(a, (str, _os.PathLike)) and _os.fspath(a) == ledger_path
            for a in args
        )
        if is_ledger:
            opened.append(id(conn))
            return _FailingSchemaConnection(conn, closed)
        return conn

    monkeypatch.setattr(ad.sqlite3, "connect", tracking_connect)

    _FailingSchemaConnection.executescript = _fail_large_schema_replay

    with pytest.raises(sqlite3.OperationalError):
        with ad._transaction():
            pass

    assert len(opened) == 1
    assert len(closed) == 1

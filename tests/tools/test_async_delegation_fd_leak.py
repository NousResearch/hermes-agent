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


def test_schema_init_failure_still_closes_connection(monkeypatch, tmp_path):
    """A PRAGMA/DDL failure after connect() must still close the connection."""
    _point_ledger(monkeypatch, tmp_path)
    opened, closed = [], []
    real_connect = sqlite3.connect

    class _FailingSchemaConnection(_TrackingConnection):
        def execute(self, sql, *args, **kwargs):
            if "CREATE TABLE" in sql:
                raise sqlite3.OperationalError("simulated schema init failure")
            return self._real.execute(sql, *args, **kwargs)

    def tracking_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened.append(id(conn))
        return _FailingSchemaConnection(conn, closed)

    monkeypatch.setattr(ad.sqlite3, "connect", tracking_connect)

    with pytest.raises(sqlite3.OperationalError):
        with ad._transaction():
            pass

    assert len(opened) == 1
    assert len(closed) == 1


def test_survives_transient_state_db_write_contention(monkeypatch, tmp_path):
    """Transient lock contention on state.db must be ridden out — the
    helper survives a brief hold by a competing writer instead of
    raising ``database is locked`` mid-dispatch.

    Repro pattern: an older Hermes install mid-FTS-maintenance, or a
    sibling CLI turn append, holds the WAL write lock for a few hundred
    ms. Before A6b this surfaced as a mid-turn OperationalError; the
    shared ``state_db_begin_immediate`` primitive now waits it out via
    the same jitter schedule ``SessionDB._execute_write`` uses.
    """
    import threading

    _point_ledger(monkeypatch, tmp_path)

    # Force the WAL into existence so the contention is real.
    # Let ``_connect()`` create the schema with the right columns; we just
    # need to seed a dummy row + force WAL so a second opener actually has
    # to fight the lock.
    ad._connect().close()
    seed = sqlite3.connect(str(tmp_path / "state.db"), timeout=10, isolation_level=None)
    try:
        # At least one row exists so a competing transaction has work to do.
        seed.execute("INSERT OR IGNORE INTO async_delegations (delegation_id) VALUES ('_seed')")
        seed.execute("PRAGMA journal_mode=WAL")
    finally:
        seed.close()

    competitor_holder = []
    release = threading.Event()

    def _hold_lock():
        c = sqlite3.connect(str(tmp_path / "state.db"), timeout=10, isolation_level=None)
        competitor_holder.append(c)
        c.execute("BEGIN IMMEDIATE")
        release.wait()
        try:
            c.execute("COMMIT")
        except Exception:
            try:
                c.execute("ROLLBACK")
            except Exception:
                pass
        c.close()

    competitor = threading.Thread(target=_hold_lock, daemon=True)
    competitor.start()

    # Wait until the competitor actually holds the write lock
    deadline = __import__("time").monotonic() + 2.0
    while not competitor_holder:
        if __import__("time").monotonic() > deadline:
            pytest.fail("competitor never acquired the lock")
        __import__("time").sleep(0.01)
    __import__("time").sleep(0.05)

    # Schedule the competitor to release after 250 ms — well within the
    # primitive's 20 s default patience budget but long enough to force
    # at least one retry cycle.
    def _release_after():
        __import__("time").sleep(0.25)
        release.set()
    threading.Thread(target=_release_after, daemon=True).start()

    # The dispatch must succeed by riding out the hold.
    ad._persist_dispatch({
        "delegation_id": "deleg_contended",
        "session_key": "test:sess",
        "origin_ui_session_id": "",
        "parent_session_id": None,
        "dispatched_at": 1.0,
        "origin_session_id": "",
        "goal": "contended",
        "context": "ctx",
    })

    row = ad.get_durable_delegation("deleg_contended")
    assert row is not None, row
    assert row["origin_session"] == "test:sess"

    competitor.join(timeout=2.0)

"""The async-delegation ledger reconciles state.db's schema once per database file, not per connection.

``_connect`` opens a fresh short-lived connection for every ledger call, and its initializer replayed the
whole canonical ``SCHEMA_SQL`` plus the column reconcile (~30 ``PRAGMA table_info``) each time. The orphan
sweep runs every 30 s and calls ``_connect`` at least twice per pass (``recover_abandoned_delegations`` and
``sweep_orphaned_completions``), so on a slow filesystem a field deployment saw this thread in two of three
watchdog dumps. Reconciling once per (file identity, ``PRAGMA schema_version``) is enough for
ordinary opens and DDL changes: a replacement file can share the old file's schema counter.
After #135351 (its reconcile-once half).
"""

import os
import sqlite3
import time
from contextlib import closing

import pytest

from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from tools import async_delegation as ad


@pytest.fixture(autouse=True)
def home(tmp_path):
    ad._reset_for_tests()
    token = set_hermes_home_override(str(tmp_path))
    yield tmp_path
    reset_hermes_home_override(token)
    ad._reset_for_tests()


@pytest.fixture
def reconcile_calls(monkeypatch):
    import hermes_state_schema

    calls = []
    real = hermes_state_schema.reconcile_state_schema

    def counting(conn):
        calls.append(1)
        return real(conn)

    monkeypatch.setattr(hermes_state_schema, "reconcile_state_schema", counting)
    return calls


def _sweep_once():
    ad.recover_abandoned_delegations()
    ad.sweep_orphaned_completions(None, now=time.time())


def test_repeated_orphan_sweeps_reconcile_the_schema_once(reconcile_calls):
    for _ in range(4):
        _sweep_once()
    assert len(reconcile_calls) == 1


def test_a_replaced_database_is_reconciled_again(home, reconcile_calls):
    _sweep_once()
    for suffix in ("", "-wal", "-shm", "-journal"):
        (home / f"state.db{suffix}").unlink(missing_ok=True)
    _sweep_once()
    assert len(reconcile_calls) == 2
    # ... and the ledger table really exists on the new file.
    ad._persist_dispatch({"delegation_id": "d-1", "session_key": "", "origin_ui_session_id": "ui",
                          "parent_session_id": "p", "dispatched_at": time.time()})
    assert ad.get_durable_delegation("d-1") is not None


def test_a_recreated_database_on_a_recycled_inode_still_gets_its_ledger(home, reconcile_calls):
    """The filesystem may hand a recreated state.db the deleted file's inode, so no file-identity key can
    tell the two apart; the schema_version header can (the fresh file reads 0)."""
    _sweep_once()
    for suffix in ("", "-wal", "-shm", "-journal"):
        (home / f"state.db{suffix}").unlink(missing_ok=True)
    (home / "state.db").touch()  # an empty file at the same path, whatever inode it got
    ad._persist_dispatch({"delegation_id": "d-2", "session_key": "", "origin_ui_session_id": "ui",
                          "parent_session_id": "p", "dispatched_at": time.time()})
    assert ad.get_durable_delegation("d-2") is not None
    assert len(reconcile_calls) == 2


def test_a_restored_older_database_gets_its_missing_columns(home, reconcile_calls):
    """A state.db swapped in from an older backup has the ledger table but not every column: its different
    schema_version makes it reconcile again."""
    import sqlite3
    _sweep_once()
    for suffix in ("", "-wal", "-shm", "-journal"):
        (home / f"state.db{suffix}").unlink(missing_ok=True)
    with sqlite3.connect(str(home / "state.db")) as old:
        old.execute("CREATE TABLE async_delegations (delegation_id TEXT PRIMARY KEY)")
    ad._persist_dispatch({"delegation_id": "d-3", "session_key": "", "origin_ui_session_id": "ui",
                          "parent_session_id": "p", "dispatched_at": time.time()})
    assert ad.get_durable_delegation("d-3") is not None


def test_each_database_is_reconciled_on_its_own(tmp_path, reconcile_calls):
    _sweep_once()
    other = tmp_path / "other-home"
    other.mkdir()
    token = set_hermes_home_override(str(other))
    try:
        _sweep_once()
        _sweep_once()
    finally:
        reset_hermes_home_override(token)
    assert len(reconcile_calls) == 2


@pytest.fixture
def same_version_replacement(home):
    with closing(ad._connect()) as conn:
        version = conn.execute("PRAGMA schema_version").fetchone()[0]
    replacement = home / "replacement.db"
    with closing(sqlite3.connect(replacement)) as conn:
        conn.execute("CREATE TABLE async_delegations (delegation_id TEXT PRIMARY KEY)")
        # A DDL counter collision from ordinary migrations, not a forced PRAGMA value.
        for i in range(version - 1):
            conn.execute(f"CREATE TABLE historical_migration_{i} (id INTEGER)")
        assert conn.execute("PRAGMA schema_version").fetchone()[0] == version
    return replacement


def _assert_dispatch_persisted():
    ad._persist_dispatch({"delegation_id": "replacement", "session_key": "session",
                          "dispatched_at": time.time()})
    row = ad.get_durable_delegation("replacement")
    assert row is not None
    assert row["origin_session"] == "session"


def test_same_version_replacement_gets_missing_columns(home, reconcile_calls, same_version_replacement):
    os.replace(same_version_replacement, home / "state.db")
    _assert_dispatch_persisted()
    _sweep_once()
    assert len(reconcile_calls) == 2


@pytest.mark.parametrize("error", [FileNotFoundError, PermissionError])
def test_unstatable_replacement_still_reconciles(home, same_version_replacement, monkeypatch, error):
    os.replace(same_version_replacement, home / "state.db")
    real_stat = os.stat

    def failing_stat(path, *args, **kwargs):
        if os.fspath(path) == str(home / "state.db"):
            raise error("file identity unavailable")
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", failing_stat)
    _assert_dispatch_persisted()


def test_replacement_between_stat_and_open_reconciles(home, same_version_replacement, monkeypatch):
    from hermes_cli import sqlite_util

    real_open = sqlite_util.open_db

    def replacing_open(*args, **kwargs):
        monkeypatch.setattr(sqlite_util, "open_db", real_open)
        os.replace(same_version_replacement, home / "state.db")
        return real_open(*args, **kwargs)

    monkeypatch.setattr(sqlite_util, "open_db", replacing_open)
    _assert_dispatch_persisted()


@pytest.mark.platforms("posix")
def test_replacement_during_reconcile_is_not_marked_ready(home, same_version_replacement, monkeypatch):
    """The opened connection still refers to the old inode when its path is replaced."""
    import hermes_state_schema

    ad._SCHEMA_READY.clear()
    real_reconcile = hermes_state_schema.reconcile_state_schema

    def replacing_reconcile(conn):
        real_reconcile(conn)
        os.replace(same_version_replacement, home / "state.db")

    with monkeypatch.context() as patch:
        patch.setattr(hermes_state_schema, "reconcile_state_schema", replacing_reconcile)
        ad._connect().close()
    _assert_dispatch_persisted()


@pytest.mark.parametrize("database", [":memory:", ""])
def test_unnamed_databases_always_reconcile(database, reconcile_calls):
    for _ in range(2):
        with closing(sqlite3.connect(database)) as conn:
            ad._initialize_schema(conn)
            conn.execute("SELECT origin_session FROM async_delegations")
    assert len(reconcile_calls) == 2


def test_data_writes_do_not_reconcile_again(reconcile_calls):
    for _ in range(3):
        _assert_dispatch_persisted()
    assert len(reconcile_calls) == 1


def test_same_file_schema_changes_reconcile_again(home, reconcile_calls):
    _sweep_once()
    with closing(sqlite3.connect(home / "state.db")) as conn:
        conn.execute("DROP TABLE async_delegations")
    _assert_dispatch_persisted()
    assert len(reconcile_calls) == 2

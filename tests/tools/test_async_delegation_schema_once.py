"""The async-delegation ledger reconciles state.db's schema once per database file, not per connection.

``_connect`` opens a fresh short-lived connection for every ledger call, and its initializer replayed the
whole canonical ``SCHEMA_SQL`` plus the column reconcile (~30 ``PRAGMA table_info``) each time. The orphan
sweep runs every 30 s and calls ``_connect`` at least twice per pass (``recover_abandoned_delegations`` and
``sweep_orphaned_completions``), so on a slow filesystem a field deployment saw this thread in two of three
watchdog dumps. Reconciling once per file (path + inode) is enough: the DDL is idempotent, and a database
replaced at the same path is reconciled again. Ported from #135351 (the reconcile-once half).
"""

import time

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

"""Offline owner-fingerprint failure contracts for the PostgreSQL async ledger."""

from __future__ import annotations

import builtins
from contextlib import contextmanager

import pytest

from gateway import status
from tools.async_delegation_ledger_postgresql import PostgreSQLAsyncDelegationLedger


@pytest.fixture
def ledger():
    # These methods do not require a connection; avoid PostgreSQL in this suite.
    return object.__new__(PostgreSQLAsyncDelegationLedger)


def _fail_status_import(monkeypatch):
    original_import = builtins.__import__
    failure = ImportError("gateway status unavailable")

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "gateway.status" and fromlist:
            raise failure
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    return failure


def test_dispatch_does_not_write_without_owner_fingerprint_dependency(ledger, monkeypatch):
    def unexpected_transaction():
        pytest.fail("dispatch opened a transaction before resolving owner fingerprint")

    monkeypatch.setattr(ledger, "_transaction", unexpected_transaction)
    failure = _fail_status_import(monkeypatch)
    with pytest.raises(ImportError) as caught:
        ledger.persist_dispatch({"delegation_id": "d1", "dispatched_at": 1.0})
    assert caught.value is failure


def test_owner_import_failure_is_not_a_dead_owner(ledger, monkeypatch):
    failure = _fail_status_import(monkeypatch)
    with pytest.raises(ImportError) as caught:
        ledger._owner_alive(123, 456)
    assert caught.value is failure


def test_missing_pid_is_not_alive_with_working_status_dependency(ledger):
    assert ledger._owner_alive(None, None) is False


def test_dispatch_persists_supported_or_unsupported_fingerprint(ledger, monkeypatch):
    recorded = []

    class Cursor:
        def execute(self, sql, params):
            recorded.append(params)

    @contextmanager
    def transaction():
        yield Cursor()

    monkeypatch.setattr(ledger, "_transaction", transaction)
    record = {"delegation_id": "d1", "dispatched_at": 1.0}
    for fingerprint in (123456, None):
        monkeypatch.setattr(status, "get_process_start_time", lambda pid: fingerprint)
        ledger.persist_dispatch(record)
        assert recorded[-1][-3] == fingerprint

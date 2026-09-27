"""Seam regression for tools/async_delegation.py routing (no live PostgreSQL).

Verifies a SQLite home keeps the legacy state.db path (selector returns None and
the module functions still write state.db), and a selected-PG home with an
unreachable DSN fails typed WITHOUT opening or creating state.db.
"""
from __future__ import annotations

import time

import pytest

import tools.async_delegation as ad
from tools.async_delegation_ledger_adapter import (
    _SELECTED_LEDGER_CACHE,
    selected_async_delegation_ledger,
)


def _record(delegation_id: str = "d1") -> dict:
    return {"delegation_id": delegation_id, "session_key": "s",
            "origin_ui_session_id": "", "parent_session_id": None,
            "dispatched_at": time.time(), "goal": "g", "origin_session_id": ""}


def _completion(delegation_id: str):
    return ({"type": "async_delegation", "delegation_id": delegation_id,
             "status": "completed", "completed_at": time.time()},
            {"status": "completed", "summary": "ok"})


def _pg_home(tmp_path, monkeypatch, *, dsn_value: str):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text(
        "state_store:\n  backend: postgresql\n  postgresql:\n"
        "    dsn_env: OWNED_ASYNC_DSN\n    connect_timeout_seconds: 1\n    pool_max_size: 2\n",
        encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("OWNED_ASYNC_DSN", dsn_value)
    _SELECTED_LEDGER_CACHE.clear()
    return home


def test_sqlite_home_keeps_legacy_state_db_path(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    _SELECTED_LEDGER_CACHE.clear()

    assert selected_async_delegation_ledger() is None

    ad._persist_dispatch(_record("sqlite-1"))
    assert (home / "state.db").exists()
    assert ad.get_durable_delegation("sqlite-1")["state"] == "running"

    # Full claim-transition surface over the unchanged SQLite path.
    ad._persist_completion(*_completion("sqlite-1"))
    assert ad.claim_completion_delivery("sqlite-1", "c1") is True
    assert ad.claim_completion_delivery("sqlite-1", "c2") is False  # held
    assert ad.complete_completion_delivery("sqlite-1", "c1") is True
    assert ad.get_durable_delegation("sqlite-1")["delivery_state"] == "delivered"
    assert ad.mark_completion_delivered("sqlite-1") is False  # already delivered


def test_unreachable_pg_fails_typed_without_state_db(tmp_path, monkeypatch):
    from state_store_runtime_readiness import (
        PostgreSQLRuntimeActivationError, trap_state_db_opens)

    home = _pg_home(tmp_path, monkeypatch, dsn_value="postgresql://fixture/only")
    with trap_state_db_opens(home) as events:
        with pytest.raises(PostgreSQLRuntimeActivationError):
            ad._persist_dispatch(_record("pg-fail"))
    assert events == []
    assert not (home / "state.db").exists()

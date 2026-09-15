"""Tests for the organization.* RPC surface (pin / batch / archive / undo)."""

from __future__ import annotations

import pytest

import tui_gateway.server as server
from tui_gateway import methods_organization as org


def _call(method, params=None):
    handler = server._methods[method]
    return handler(1, params or {})


def _ok(method, params=None):
    resp = _call(method, params)
    assert "error" not in resp, resp.get("error")
    return resp["result"]


@pytest.fixture()
def org_db(tmp_path, monkeypatch):
    """A real SessionDB with three sessions, and the undo log pinned to tmp."""
    from hermes_state import SessionDB

    db = SessionDB(db_path=tmp_path / "state.db")
    for sid in ("s-1", "s-2", "s-3"):
        db.create_session(sid, "cli", cwd=str(tmp_path / sid))
        db.append_message(sid, "user", f"hello {sid}")
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(org, "_undo_path", lambda: tmp_path / "organization_undo.json")
    yield db
    db.close()


def test_organization_methods_registered():
    for m in ("organization.pin", "organization.archive",
              "organization.history", "organization.undo"):
        assert m in server._methods


def test_batch_pin_reports_applied_and_records_history(org_db):
    result = _ok("organization.pin", {"session_ids": ["s-1", "s-2"], "pinned": True})
    assert set(result["applied"]) == {"s-1", "s-2"}
    assert result["failed"] == []
    assert result["partial"] is False
    for sid in ("s-1", "s-2"):
        assert org_db.get_session(sid)["pinned"] == 1
    assert org_db.get_session("s-3")["pinned"] == 0

    history = _ok("organization.history")["batches"]
    assert len(history) == 1
    assert history[0]["action"] == "pinned"
    assert history[0]["count"] == 2
    assert history[0]["undone"] is False


def test_batch_reports_missing_sessions_as_failures(org_db):
    result = _ok("organization.pin",
                 {"session_ids": ["s-1", "ghost"], "pinned": True})
    assert result["applied"] == ["s-1"]
    assert result["partial"] is True
    assert result["failed"][0]["session_id"] == "ghost"


def test_all_missing_is_an_error_not_an_empty_batch(org_db):
    resp = _call("organization.pin", {"session_ids": ["ghost"], "pinned": True})
    assert resp["error"]["code"] == 5083


def test_empty_or_invalid_ids_rejected(org_db):
    assert _call("organization.pin", {"session_ids": []})["error"]["code"] == 5082
    assert _call("organization.pin", {})["error"]["code"] == 5082


def test_undo_restores_previous_state(org_db):
    # s-2 starts pinned; the batch pins both; undo must restore the mix.
    org_db.set_session_pinned("s-2", True)
    batch = _ok("organization.pin", {"session_ids": ["s-1", "s-2"], "pinned": True})
    assert org_db.get_session("s-1")["pinned"] == 1

    undone = _ok("organization.undo", {"batch_id": batch["batch_id"]})
    assert set(undone["restored"]) == {"s-1", "s-2"}
    assert org_db.get_session("s-1")["pinned"] == 0
    assert org_db.get_session("s-2")["pinned"] == 1  # was pinned BEFORE the batch

    history = _ok("organization.history")["batches"]
    assert history[0]["undone"] is True


def test_undo_newest_without_id(org_db):
    _ok("organization.pin", {"session_ids": ["s-1"], "pinned": True})
    _ok("organization.archive", {"session_ids": ["s-3"], "archived": True})
    undone = _ok("organization.undo", {})  # newest = the archive batch
    assert undone["batch_id"] != _ok("organization.history")["batches"][0]["id"]
    assert org_db.get_session("s-3")["archived"] == 0
    assert org_db.get_session("s-1")["pinned"] == 1  # older batch untouched


def test_undo_twice_is_not_a_second_reversal(org_db):
    batch = _ok("organization.pin", {"session_ids": ["s-1"], "pinned": True})
    _ok("organization.undo", {"batch_id": batch["batch_id"]})
    # The batch is marked undone, so "undo newest" finds nothing.
    resp = _call("organization.undo", {})
    assert resp["error"]["code"] == 5083


def test_undo_unknown_batch_id(org_db):
    resp = _call("organization.undo", {"batch_id": "nope"})
    assert resp["error"]["code"] == 5083


def test_batch_size_cap(org_db):
    ids = [f"s-{i}" for i in range(org._MAX_SESSIONS_PER_BATCH + 1)]
    resp = _call("organization.pin", {"session_ids": ids, "pinned": True})
    assert resp["error"]["code"] == 5082


def test_history_log_capped(org_db, monkeypatch):
    monkeypatch.setattr(org, "_MAX_BATCHES", 3)
    for i in range(5):
        _ok("organization.pin", {"session_ids": [f"s-{(i % 3) + 1}"], "pinned": True})
    history = _ok("organization.history")["batches"]
    assert len(history) == 3

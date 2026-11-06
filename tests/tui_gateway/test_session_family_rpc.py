"""``session.family``: the read-only RPC the Ink `/tree` overlay draws from.

It is the same builder the classic CLI's picker uses, so these tests pin the wire shape, the
view flags, the refusal for a missing session, and the two promises the lane makes about it:
it never writes, and it never reaches outside the request's own profile store.
"""

from __future__ import annotations

import pytest

from hermes_state import SessionDB
from tui_gateway import server
from tui_gateway.contracts import registry


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    session_db.create_session("root", source="cli")
    session_db.append_message("root", "user", "one")
    session_db.end_session("root", "compression")
    session_db.create_session("cont", source="cli", parent_session_id="root")
    session_db.create_session("branch", source="cli", parent_session_id="cont",
                              model_config={"_branched_from": "cont"})
    session_db.create_session("sub", source="tool", parent_session_id="branch",
                              model_config={"_delegate_from": "branch"})
    try:
        yield session_db
    finally:
        session_db.close()


def _family(monkeypatch, db, **params):
    monkeypatch.setattr(server, "_get_db", lambda: db)
    return server.handle_request(
        {"id": "1", "method": "session.family", "params": {"session_id": "branch", **params}})


def test_the_method_is_in_the_contract(db) -> None:
    assert "session.family" in registry.METHODS
    assert registry.METHODS["session.family"].params.__name__ == "SessionFamilyParams"


def test_it_returns_the_family_walk(monkeypatch, db) -> None:
    resp = _family(monkeypatch, db)

    result = resp["result"]
    assert result["root_id"] == "root"
    assert result["truncated"] is False
    assert [node["session_id"] for node in result["nodes"]] == ["root", "branch"]
    assert [node["kind"] for node in result["nodes"]] == ["root", "branch"]
    assert result["nodes"][1]["is_current"] is True


def test_the_view_flags_reach_the_shared_builder(monkeypatch, db) -> None:
    everything = _family(monkeypatch, db, show_compressions=True, show_subagents=True)["result"]

    assert [node["session_id"] for node in everything["nodes"]] == [
        "root", "cont", "branch", "sub"]
    assert [node["depth"] for node in everything["nodes"]] == [0, 1, 2, 3]


def test_folding_a_row_is_applied_owner_side(monkeypatch, db) -> None:
    resp = _family(monkeypatch, db, show_compressions=True, show_subagents=True, folded=["branch"])

    assert [node["session_id"] for node in resp["result"]["nodes"]] == ["root", "cont", "branch"]
    assert resp["result"]["nodes"][-1]["has_children"] is True


def test_a_missing_session_id_is_a_client_error(monkeypatch, db) -> None:
    monkeypatch.setattr(server, "_get_db", lambda: db)

    resp = server.handle_request({"id": "1", "method": "session.family", "params": {}})

    assert resp["error"]["code"] == 4004


def test_a_session_from_another_profile_is_not_reachable(monkeypatch, db, tmp_path) -> None:
    other = SessionDB(db_path=tmp_path / "other" / "state.db")
    try:
        other.create_session("elsewhere", source="cli")
        # Asking THIS store for the other profile's session resolves to nothing ...
        assert _family(monkeypatch, db, session_id="elsewhere")["result"]["nodes"] == []
        # ... and the other store's own family holds its rows only.
        resp = _family(monkeypatch, other, session_id="elsewhere")
        assert [node["session_id"] for node in resp["result"]["nodes"]] == ["elsewhere"]
    finally:
        other.close()


def test_it_never_writes(monkeypatch, db) -> None:
    def _refuse(*_args, **_kwargs):
        raise AssertionError("session.family must not write")

    monkeypatch.setattr(db, "_execute_write", _refuse)
    monkeypatch.setattr(db, "flush_for_read", _refuse)

    assert _family(monkeypatch, db)["result"]["nodes"]

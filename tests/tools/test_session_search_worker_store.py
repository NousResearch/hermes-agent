"""Recall in a gateway-managed worker (#141705).

The worker's agent is constructed with ``session_db=RuntimeSessionStore`` (``agent/managed_worker.py``),
a facade over the owner's RPC that mirrors the write and lineage surface but not the FTS read API.
Handing it to ``session_search`` made every recall inside a worker answer
``'RuntimeSessionStore' object has no attribute 'search_messages'``; recall now opens its own read-only
handle on the profile's state DB, the same one the no-db path uses.
"""
import json

import pytest

from agent.runtime_session_store import RuntimeSessionStore
from hermes_state import SessionDB
from tools.session_search_tool import session_search


@pytest.fixture
def home(tmp_path, monkeypatch):
    """The worker's profile home: ``get_hermes_home()`` resolves to it, so its state.db is the DB."""
    import hermes_constants

    monkeypatch.setattr(hermes_constants, "get_hermes_home", lambda: tmp_path)
    return tmp_path


def _seed(path, session_id, content):
    db = SessionDB(path / "state.db")
    db.create_session(session_id, source="cli")
    db.append_message(session_id, role="user", content=content)
    db.append_message(session_id, role="assistant", content="noted")
    return db


def _worker_store(home):
    """The real facade, with a no-op owner RPC: only the capability matters here."""
    return RuntimeSessionStore(lambda method, **params: {}, {"session_id": "worker-s"}, home / "outbox")


def test_worker_recall_reads_the_state_db_instead_of_the_store(home):
    db = _seed(home, "worker-s", "the zanzibar rollout is blocked on the store")
    try:
        store = _worker_store(home)
        # The capability the worker's handle lacks — the AttributeError the report quotes.
        assert not hasattr(store, "search_messages")
        payload = json.loads(session_search(query="zanzibar", db=store))
    finally:
        db.close()
    assert payload["success"] is True, payload
    assert [r["session_id"] for r in payload["results"]] == ["worker-s"]


def test_a_readable_handle_is_still_used_as_given(home, tmp_path):
    """A real SessionDB passed by a caller is not swapped for the home DB."""
    home_db = _seed(home, "home-session", "zanzibar appears only in the home transcript")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    passed = _seed(elsewhere, "passed-in", "zanzibar from the caller's handle")
    try:
        payload = json.loads(session_search(query="zanzibar", db=passed))
    finally:
        passed.close()
        home_db.close()
    assert [r["session_id"] for r in payload["results"]] == ["passed-in"]

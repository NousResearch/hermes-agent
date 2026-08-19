"""RPC-level tests for the generic hidden-session surface (tui_gateway).

Covers the two seams Bot Mode's "sessions are always hidden" policy leans on:

* ``session.set_hidden`` resolves a DURABLE stored session id when no live
  runtime session matches — plugins reconciling sessions they own (Bot Mode's
  hide sweep) hold stored ids for chats that aren't live right now. The old
  live-only lookup failed those with 4001 and the sweep silently no-opped.
* ``session.list`` honors ``include_hidden`` so owning surfaces (the Bots
  pane's per-profile browser) can still enumerate the rows they hid, while
  every default caller keeps the hidden rows dropped.
"""

import pytest

import tui_gateway.server as srv
import tui_gateway.methods_session  # noqa: F401  (registers the RPC methods)
from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path, monkeypatch):
    database = SessionDB(tmp_path / "state.db")
    monkeypatch.setattr(srv, "_get_db", lambda: database)
    try:
        yield database
    finally:
        database.close()


def _call(method: str, params: dict) -> dict:
    return srv._methods[method](1, params)


def _seed(db, sid: str) -> None:
    db.create_session(sid, source="desktop")
    db._conn.execute("UPDATE sessions SET message_count = 1 WHERE id = ?", (sid,))
    db._conn.commit()


def test_set_hidden_resolves_stored_id_without_live_session(db):
    """A stored (non-live) session id must be hideable — the sweep path."""
    _seed(db, "stored-chat")
    assert srv._find_live_session_by_key("stored-chat") is None

    envelope = _call(
        "session.set_hidden", {"session_id": "stored-chat", "hidden": True}
    )
    assert "error" not in envelope, envelope
    assert envelope["result"]["hidden"] is True
    assert db.get_session("stored-chat")["hidden"] == 1

    # And back — unhide through the same durable path.
    envelope = _call(
        "session.set_hidden", {"session_id": "stored-chat", "hidden": False}
    )
    assert "error" not in envelope, envelope
    assert db.get_session("stored-chat")["hidden"] == 0


def test_set_hidden_unknown_id_still_errors(db):
    envelope = _call(
        "session.set_hidden", {"session_id": "no-such-session", "hidden": True}
    )
    assert envelope.get("error"), envelope


def test_session_list_include_hidden(db):
    _seed(db, "plain-chat")
    _seed(db, "bot-chat")
    assert db.set_session_hidden("bot-chat", True) is True

    default_rows = _call("session.list", {})["result"]["sessions"]
    assert {s["id"] for s in default_rows} == {"plain-chat"}

    all_rows = _call("session.list", {"include_hidden": True})["result"]["sessions"]
    assert {s["id"] for s in all_rows} == {"plain-chat", "bot-chat"}


def test_compute_host_compress_wait_budget_resolution(monkeypatch):
    """#88988: the /compress host wait must scale with the compression
    timeout + retry rounds instead of the fixed 120s that failed the RPC
    while the host finished at ~134s."""
    # The helper lives in tui_gateway.server: register() rebuilds every
    # methods_* handler with server's globals, so module helpers referenced
    # from those handlers must resolve there.
    from tui_gateway import server

    import hermes_cli.config as hc_config

    # Shipped defaults: 120s per attempt x 3 rounds + 30s grace = 390s.
    monkeypatch.setattr(hc_config, "load_config", lambda: {})
    assert server._compute_host_compress_wait_s() == 390.0

    # Config-driven: 60s per attempt x 2 rounds + 30s grace = 150s.
    monkeypatch.setattr(
        hc_config,
        "load_config",
        lambda: {
            "auxiliary": {"compression": {"timeout": 60}},
            "compression": {"max_attempts": 2},
        },
    )
    assert server._compute_host_compress_wait_s() == 150.0

    # Broken config fails open to the default.
    monkeypatch.setattr(hc_config, "load_config", lambda: {"auxiliary": "nonsense"})
    assert server._compute_host_compress_wait_s() == 390.0

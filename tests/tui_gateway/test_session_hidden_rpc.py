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


def _seed(
    db,
    sid: str,
    *,
    source: str = "desktop",
    title: str | None = None,
) -> None:
    db.create_session(sid, source=source)
    if title is not None:
        db.set_session_title(sid, title)
    db._conn.execute("UPDATE sessions SET message_count = 1 WHERE id = ?", (sid,))
    db._conn.commit()


def test_set_hidden_resolves_stored_id_without_live_session(db):
    """A stored (non-live) session id must be hideable — the sweep path."""
    _seed(db, "stored-chat")
    assert srv._find_live_session_by_key("stored-chat") is None

    envelope = _call("session.set_hidden", {"session_id": "stored-chat", "hidden": True})
    assert "error" not in envelope, envelope
    assert envelope["result"]["hidden"] is True
    assert db.get_session("stored-chat")["hidden"] == 1

    # And back — unhide through the same durable path.
    envelope = _call("session.set_hidden", {"session_id": "stored-chat", "hidden": False})
    assert "error" not in envelope, envelope
    assert db.get_session("stored-chat")["hidden"] == 0


def test_set_hidden_unknown_id_still_errors(db):
    envelope = _call("session.set_hidden", {"session_id": "no-such-session", "hidden": True})
    assert envelope.get("error"), envelope


def test_session_list_include_hidden(db):
    _seed(db, "plain-chat")
    _seed(db, "bot-chat")
    assert db.set_session_hidden("bot-chat", True) is True

    default_rows = _call("session.list", {})["result"]["sessions"]
    assert {s["id"] for s in default_rows} == {"plain-chat"}

    all_rows = _call("session.list", {"include_hidden": True})["result"]["sessions"]
    assert {s["id"] for s in all_rows} == {"plain-chat", "bot-chat"}


def test_hosted_room_lookup_is_source_scoped_resurrected_and_pinned(db):
    title = "Group: room-1"
    _seed(db, "ordinary", source="desktop", title=title)
    wrong_source = _call(
        "session.list",
        {"title": title, "source": "bot_room", "include_hidden": True},
    )
    assert wrong_source["result"]["sessions"] == []

    assert db.set_session_title("ordinary", "Ordinary chat")
    _seed(db, "canonical-room", source="bot_room", title=title)
    assert db.set_session_archived("canonical-room", True)

    envelope = _call(
        "session.list",
        {
            "title": title,
            "source": "bot_room",
            "include_hidden": True,
        },
    )

    assert envelope["result"]["sessions"][0]["id"] == "canonical-room"
    canonical = db.get_session("canonical-room")
    assert canonical["archived"] == 0
    assert canonical["pinned"] == 1
    ordinary = db.get_session("ordinary")
    assert ordinary["archived"] == 0
    assert ordinary["pinned"] == 0


def test_first_hosted_room_persist_is_pinned_but_ordinary_session_is_not(
    db,
    monkeypatch,
):
    monkeypatch.setattr(srv, "_resolve_model", lambda: "test/model")

    for session_key, source in (
        ("room-first-prompt", "bot_room"),
        ("plain-first-prompt", "desktop"),
    ):
        srv._ensure_session_db_row({
            "session_key": session_key,
            "source": source,
            "model_override": {"model": "test/model"},
            "profile_home": None,
            "explicit_cwd": False,
            "cwd": None,
        })

    assert db.get_session("room-first-prompt")["pinned"] == 1
    assert db.get_session("plain-first-prompt")["pinned"] == 0

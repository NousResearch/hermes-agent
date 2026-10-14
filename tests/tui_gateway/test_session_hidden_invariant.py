"""Tests: the ``hidden`` invariant (#125534) — a client flag only hides internal plumbing.

``sessions.hidden`` is reserved for internal session plumbing: canonical Bot Chats,
"Agent Inbox" handoffs, "Group:"/room-plumbing seats and deny-listed internal sources.
An ordinary user conversation must never be persisted ``hidden=1`` from a client flag —
once hidden it disappears from the default ``session.list`` and from search while its
transcript stays intact, and a reconnect/resume path re-applies the pending hide.

Contract under test, driven through the real ``server.handle_request`` RPC path:
- ``session.create`` + ``session.set_hidden`` with ``hidden: true`` on an ordinary
  desktop chat never leaves a durable ``hidden=1`` row (lazy apply refuses it).
- ``session.set_hidden`` on a stored ordinary row is refused outright (4027).
- The same seams keep hiding the plumbing identities (version skew: an older Desktop
  sending ``hidden: true`` for a canonical Bot Chat or a room seat keeps working).
- The durable-row check authorizes a compression lineage by ANY of its rows, so the
  registry title may live on the root while the tip is untitled.
"""

import pytest

from hermes_state import SessionDB
from tui_gateway import server


@pytest.fixture
def home(tmp_path, monkeypatch):
    """Temp HERMES_HOME; the gateway's real db seam opens its state.db."""
    h = tmp_path / ".hermes"
    h.mkdir(parents=True)
    monkeypatch.setenv("HERMES_HOME", str(h))
    monkeypatch.setattr(server, "_schedule_agent_build", lambda _sid: None)
    monkeypatch.setattr(server, "_schedule_session_cap_enforcement", lambda: None)
    monkeypatch.setattr(server, "_register_session_cwd", lambda _session: None)
    return h


def _rpc(method, **params):
    return server.handle_request({"id": method, "method": method, "params": params})


def _create(**params):
    response = _rpc("session.create", **{"source": "desktop", **params})
    assert "result" in response, response
    return response["result"]


def _row(home, stored_id):
    db = SessionDB(db_path=home / "state.db")
    try:
        return db.get_session(stored_id)
    finally:
        db.close()


class _LiveSession:
    """Pop the live runtime from ``server._sessions`` on exit so tests stay isolated."""

    def __init__(self, sid):
        self.sid = sid

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        server._sessions.pop(self.sid, None)
        return False


def test_create_hidden_flag_never_hides_ordinary_chat(home):
    # The #105162-class trigger: a desktop create carrying hidden:true for a plain
    # user conversation. The lazy row write (here via session.title, the same seam
    # the canonical Bot Chat flow uses to materialize the row) must drop the flag.
    created = _create(title="My side chat", hidden=True)
    sid, stored = created["session_id"], created["stored_session_id"]

    with _LiveSession(sid):
        response = _rpc("session.title", session_id=sid, title="My side chat")
        assert "result" in response, response

        row = _row(home, stored)
        assert row is not None, "session.title must have materialized the row"
        assert not row["hidden"], "an ordinary chat must never be persisted hidden=1 (#125534)"


def test_set_hidden_refuses_ordinary_stored_chat(home):
    created = _create()
    sid, stored = created["session_id"], created["stored_session_id"]

    with _LiveSession(sid):
        response = _rpc("session.title", session_id=sid, title="Weekly planning")
        assert "result" in response, response

    # No live runtime: the stored tier of session.set_hidden must refuse the hide.
    response = _rpc("session.set_hidden", session_id=stored, hidden=True)
    assert response.get("error", {}).get("code") == 4027, response
    assert not _row(home, stored)["hidden"]


def test_set_hidden_before_row_is_deferred_then_refused_for_ordinary_chat(home):
    # set_hidden on a live runtime whose row does not exist is accepted and deferred
    # (an older client hiding a not-yet-titled plumbing session must keep working) —
    # but the lazy row write re-authorizes and drops the flag for an ordinary chat,
    # so the hide cannot resurface on reconnect/resume (#125534 step 6).
    created = _create()
    sid, stored = created["session_id"], created["stored_session_id"]

    with _LiveSession(sid):
        response = _rpc("session.set_hidden", session_id=sid, hidden=True)
        assert "result" in response, response
        assert server._sessions[sid]["pending_hidden"] is True

        response = _rpc("session.title", session_id=sid, title="Notes")
        assert "result" in response, response

        assert server._sessions[sid].get("pending_hidden") is not True
        row = _row(home, stored)
        assert row is not None and not row["hidden"]


def test_canonical_bot_chat_born_hidden_still_works(home):
    # Version skew: the Bot Mode create (title "Bot Chat" + hidden:true, row then
    # materialized via session.title) must keep landing hidden=1, out of the
    # default list and resolvable by registry title.
    created = _create(title="Bot Chat", hidden=True)
    sid, stored = created["session_id"], created["stored_session_id"]

    with _LiveSession(sid):
        response = _rpc("session.title", session_id=sid, title="Bot Chat")
        assert "result" in response, response

    row = _row(home, stored)
    assert row is not None and row["hidden"], "canonical Bot Chats are born hidden"

    db = SessionDB(db_path=home / "state.db")
    try:
        db.append_message(stored, "user", "hello bot", timestamp=1000)
    finally:
        db.close()

    listed = _rpc("session.list", limit=50)["result"]["sessions"]
    assert stored not in {s["id"] for s in listed}

    listed = _rpc("session.list", limit=50, include_hidden=True)["result"]["sessions"]
    assert stored in {s["id"] for s in listed}

    # The click path's exact-title lookup resolves the hidden registry row.
    by_title = _rpc("session.list", title="Bot Chat")["result"]["sessions"]
    assert [s["id"] for s in by_title] == [stored]


def test_room_plumbing_group_seat_born_hidden_still_works(home):
    # Group member seats (desktop group-turns / hosted room driver) create with a
    # "Group:" title + room_plumbing; the hidden flag must keep applying.
    created = _create(title="Group: room-7", hidden=True, room_plumbing=True,
                      source="bot_room")
    sid, stored = created["session_id"], created["stored_session_id"]

    with _LiveSession(sid):
        response = _rpc("session.title", session_id=sid, title="Group: room-7")
        assert "result" in response, response

    row = _row(home, stored)
    assert row is not None and row["hidden"]
    assert row["source"] == "bot_room"


def test_hide_authorization_spans_the_compression_lineage(home):
    # The registry title may live on the compression ROOT while the tip is untitled:
    # hiding by the tip id must stay authorized (issue #125534's lineage requirement).
    db = SessionDB(db_path=home / "state.db")
    try:
        db.create_session("root", source="desktop")
        assert db.set_session_title("root", "Bot Chat")
        db.end_session("root", "compression")
        db.create_session("tip", source="desktop", parent_session_id="root")
        assert not (db.get_session("tip")["title"] or "").strip()

        assert db.session_hide_authorized("tip")
        assert db.set_session_hidden("tip", True)
        assert db.get_session("root")["hidden"] == 1
        assert db.get_session("tip")["hidden"] == 1

        # An ordinary lineage is never authorized.
        db.create_session("plain", source="desktop")
        assert db.set_session_title("plain", "Scratch")
        assert not db.session_hide_authorized("plain")
    finally:
        db.close()
